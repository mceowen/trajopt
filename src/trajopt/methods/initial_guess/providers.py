"""Construct initial trajectories from physical models and trajectory sources."""
from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import numpy as np

from trajopt.methods.common import discretize
from trajopt.trajectories import TrajectoryData, TrajectorySamples


@dataclass(frozen=True)
class InitialGuessContext:
    """Physical segment, target mesh, provider options, and preceding samples."""
    problem: object
    mesh: object
    options: object
    previous: TrajectorySamples | None = None
    integrate: Callable = discretize.propagate_rk4


def _constraint_of_type(phase, type_name):
    return next((c for c in phase.constraints.values() if c.type == type_name), None)


def _times(context):
    phase, cfg = context.problem, context.options
    initial = _constraint_of_type(phase, 'initial_time')
    final = _constraint_of_type(phase, 'final_time')
    start = float(initial.value) * phase.nondim.time_scale if initial is not None else float(cfg.get('t_start', 0.))
    if initial is not None and final is not None and final.is_fixed:
        stop = float(final.fixed_value) * phase.nondim.time_scale
    elif 'duration' in cfg:
        stop = start + float(cfg.duration)
    elif 't_stop' in cfg:
        stop = float(cfg.t_stop)
    else:
        raise ValueError(f"phase '{phase.name}': guess needs duration or t_stop")
    if not np.isfinite([start, stop]).all() or stop <= start:
        raise ValueError(f"phase '{phase.name}': guess must have a positive finite duration")
    return start, stop


def _endpoint_states(context):
    phase, cfg = context.problem, context.options
    n_x, d2nd = phase.index_map.n.state, phase.nondim.M.state.d2nd
    if 'x_start' in cfg:
        x0 = d2nd @ np.atleast_1d(cfg.x_start)
    else:
        constraint = _constraint_of_type(phase, 'initial_state')
        if constraint is None:
            raise ValueError(f"phase '{phase.name}': guess needs x_start or an initial_state constraint")
        x0 = np.zeros(n_x)
        x0[np.asarray(constraint.idx, dtype=int)] = constraint.value
    xf = x0.copy()
    if 'x_stop' in cfg:
        xf = d2nd @ np.atleast_1d(cfg.x_stop)
    else:
        constraint = _constraint_of_type(phase, 'final_state')
        if constraint is not None:
            xf[np.asarray(constraint.idx, dtype=int)] = constraint.value
    return x0, xf


def straight_line(context):
    """Linear states/controls with zero accumulators."""
    phase, cfg = context.problem, context.options
    start, stop = _times(context)
    x0, xf = _endpoint_states(context)
    return TrajectoryData.from_samples(
        t=[start, stop], x=np.array([x0, xf]) @ phase.nondim.M.state.nd2d.T,
        u=np.array([cfg.u_start, cfg.u_stop]),
        aux=np.zeros((2, len(phase.index_map.indices.z.augmented))))


def propagation(context):
    """RK4 with ramped controls and constant dt/dtau."""
    phase, cfg = context.problem, context.options
    index, idx = phase.index_map, phase.index_map.indices
    start, stop = _times(context)
    x0, _ = _endpoint_states(context)
    u0 = phase.nondim.M.control.d2nd @ np.atleast_1d(cfg.u_start)
    uf = phase.nondim.M.control.d2nd @ np.atleast_1d(cfg.u_stop)
    N = index.N.all
    z0 = np.zeros(index.n.z)
    z0[idx.z.state], z0[idx.z.time] = x0, start / phase.nondim.time_scale
    tau_ref = jnp.linspace(0., 1., N)
    u_ref = jnp.asarray(np.linspace(0, 1, N)[:, None] * (uf-u0) + u0)
    ctrl, dilation = jnp.array(idx.nu.control), jnp.array(idx.nu.dilation_factor)
    sigma = jnp.asarray((stop-start) / phase.nondim.time_scale)

    def nu_fn(z, tau):
        k = jnp.clip(jnp.searchsorted(tau_ref, tau, side='right') - 1, 0, N-2)
        a = (tau-tau_ref[k]) / (tau_ref[k+1]-tau_ref[k])
        return jnp.zeros(index.n.nu).at[ctrl].set((1-a)*u_ref[k]+a*u_ref[k+1]).at[dilation].set(sigma)

    dynamics = _constraint_of_type(phase, 'dynamics')
    if dynamics is None:
        raise ValueError(f"phase '{phase.name}': propagation needs dynamics")
    tau, z, nu = context.integrate(
        z0, 0., 1., nu_fn, dynamics.fcn_znu, phase.params, n_steps=10*(N-1))
    from trajopt.methods.initial_guess.preparation import dimensional_samples
    values = dimensional_samples(phase, z, nu)
    # Restore exact affine time after integration.
    return TrajectoryData.from_samples(
        t=start+(stop-start)*tau, x=values.x, u=values.u, tau=tau,
        s=values.s, aux=values.aux)
