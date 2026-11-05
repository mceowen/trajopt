"""Create physical initial guesses and sample them on the transcription mesh."""
import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import cumulative_trapezoid

from trajopt.common.tools import AttrDict, deep_merge
from trajopt.common.trajectory import (
    TrajectoryData,
    TrajectorySamples,
    TrajectorySource,
)
from trajopt.methods.common import discretize


@dataclass(frozen=True)
class InitialGuessContext:
    """Physical phase, target mesh, provider options, and preceding samples."""
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
    values = dimensional_samples(phase, z, nu)
    # Restore exact affine time after integration.
    return TrajectoryData.from_samples(
        t=start+(stop-start)*tau, x=values.x, u=values.u, tau=tau,
        s=values.s, aux=values.aux)


def guess_options(phase, method_phase):
    options = method_phase.hyperparams.get('initial_guess', AttrDict())
    defaults = deep_merge(options.get('default', AttrDict()), options.get(phase.name, AttrDict()))
    return deep_merge(copy.deepcopy(defaults), copy.deepcopy(phase.guess))


def dimensional_samples(phase, z, nu):
    x, t, aux, u, s = phase.index_map.unpack_znu(np.asarray(z), np.asarray(nu))
    return TrajectorySamples(t[:, 0] * phase.nondim.time_scale,
                        x @ phase.nondim.M.state.nd2d.T,
                        u @ phase.nondim.M.control.nd2d.T,
                        s[:, 0] * phase.nondim.time_scale, aux.copy())


def _context(method_phase, previous):
    phase = method_phase.problem
    options = guess_options(phase, method_phase)
    for field, value in options.items():
        if isinstance(value, str) and value.startswith('fcns.'):
            options[field] = phase.fcns[value[len('fcns.'):]](phase.params)
    for field in ('x_start', 't_start'):
        value = options.get(field)
        if isinstance(value, str):
            if value != 'previous':
                raise ValueError(f"phase '{phase.name}': unknown guess.{field} value {value!r}")
            if previous is None:
                raise ValueError(f"phase '{phase.name}': guess.{field}='previous' needs a preceding phase")
            options[field] = previous.x[-1].copy() if field == 'x_start' else float(previous.t[-1])
    return InitialGuessContext(phase, method_phase.mesh, options, previous,
                               method_phase.fcns.discretize.propagate_rk4)


def _configured_guess(method_phase, context):
    fcns = method_phase.fcns.initial_guess
    provider = fcns.get(context.problem.name, fcns.get('default', AttrDict())).get('guess')
    if not callable(provider):
        raise ValueError(f"phase '{context.problem.name}': initial guess provider must be callable")
    return provider(context)


def _dense_grid(target, native):
    nodes = np.sort(np.concatenate((target, native if native is not None else [],
                                    np.linspace(0, 1, 10*(len(target)-1)+1))))
    # Merge roundoff-separated nodes, retaining exact target coordinates.
    nodes = nodes[np.r_[True, np.diff(nodes) > 1e-12]]
    for t in target:
        nodes[np.argmin(np.abs(nodes-t))] = t
    return nodes


def _source_coordinates(tau, native):
    """Align sample coordinates with nearby source nodes.

    At a discontinuity, roundoff can select the wrong side of a source node.
    Use the exact source coordinate to preserve its boundary convention.
    Leave the stored target grid unchanged.
    """
    if native is None:
        return tau
    native = np.unique(np.asarray(native, dtype=float))
    if not len(native):
        return tau
    following = np.clip(np.searchsorted(native, tau), 0, len(native)-1)
    preceding = np.maximum(following-1, 0)
    closest = np.where(np.abs(tau-native[preceding]) <= np.abs(tau-native[following]),
                       preceding, following)
    return np.where(np.abs(tau-native[closest]) <= 1e-12, native[closest], tau)


def prepare_initial_guess(method_phase, supplied=None, previous=None):
    """Prepare a guess without modifying the current iterate."""
    phase = method_phase.problem
    if supplied is None:
        context = _context(method_phase, previous)
        supplied = _configured_guess(method_phase, context)
    if not isinstance(supplied, TrajectorySource):
        raise TypeError(f"phase '{phase.name}': initial guess must implement sample(tau)")
    if isinstance(supplied, TrajectoryData):
        from trajopt.methods.common.utils.trajectory_conversion import (
            auxiliary_metadata,
            phase_layout,
        )
        if supplied.layout and supplied.layout != phase_layout(phase):
            raise ValueError(f"phase '{phase.name}': incompatible channel identities, ordering, or units")
        if supplied.aux_metadata and supplied.aux_metadata != auxiliary_metadata(phase):
            raise ValueError(f"phase '{phase.name}': incompatible auxiliary definitions; transfer physical values")

    target = method_phase.mesh.tau
    dense_tau = _dense_grid(target, supplied.native_tau)
    values = supplied.sample(_source_coordinates(dense_tau, supplied.native_tau))
    if not isinstance(values, TrajectorySamples) or len(values.t) != len(dense_tau):
        raise TypeError('A trajectory source must return TrajectorySamples with one row per query')
    n = phase.index_map.n
    if values.x.shape[1] != n.state or values.u.shape[1] != n.control:
        raise ValueError(f"phase '{phase.name}': expected {n.state} states and {n.control} controls")
    if np.any(np.diff(values.t) <= 0):
        raise ValueError(f"phase '{phase.name}': guess times must strictly increase")
    duration = values.t[-1]-values.t[0]
    s = values.s
    if s is None:
        if not np.allclose(values.t, values.t[0]+dense_tau*duration, rtol=1e-8, atol=1e-9):
            raise ValueError("A non-affine time map requires explicit s=dt/dtau")
        s = np.full(len(dense_tau), duration)
    if np.any(s <= 0):
        raise ValueError("Guess time dilation must be positive")
    initial_time = method_phase.find_constraint('initial_time')
    if initial_time is not None:
        start = float(initial_time.value) * phase.nondim.time_scale
        # The anchored dt[0] prevents correcting a mismatched start epoch.
        if not np.isclose(values.t[0], start, rtol=1e-10, atol=1e-10):
            raise ValueError("Guess start time must match the initial_time constraint")
    if not method_phase.free_final_time:
        start = float(method_phase.find_constraint('initial_time').value) * phase.nondim.time_scale
        stop = float(method_phase.find_constraint('final_time').fixed_value) * phase.nondim.time_scale
        if not (np.allclose(values.t, start+dense_tau*(stop-start)) and np.allclose(s, stop-start)):
            raise ValueError("A fixed-time problem requires its prescribed affine time map")
    n_aux = len(phase.index_map.indices.z.augmented)
    aux = values.aux
    if aux is None:
        aux = np.zeros((len(dense_tau), n_aux))
    if aux.shape != (len(dense_tau), n_aux):
        raise ValueError(f"phase '{phase.name}': expected {n_aux} auxiliary accumulator columns")
    z, nu = phase.index_map.pack_znu(values.x @ phase.nondim.M.state.d2nd.T,
                                    values.t[:, None]/phase.nondim.time_scale, aux,
                                    values.u @ phase.nondim.M.control.d2nd.T,
                                    s[:, None]/phase.nondim.time_scale)
    if values.aux is None and n_aux:
        dynamics = next(c for c in phase.constraints.values() if c.type == 'dynamics')
        derivatives = np.asarray(jax.jit(jax.vmap(dynamics.fcn_znu, in_axes=(0, 0, None)))(z, nu, phase.params))
        z[:, phase.index_map.indices.z.augmented] = cumulative_trapezoid(
            derivatives[:, phase.index_map.indices.z.augmented], dense_tau, axis=0, initial=0)
    if not np.isfinite(z).all() or not np.isfinite(nu).all():
        raise ValueError(f"phase '{phase.name}': imported guess contains nonfinite values")
    indices = np.searchsorted(dense_tau, target)
    z_nodes, nu_nodes = z[indices].copy(), nu[indices].copy()
    t = z_nodes[:, phase.index_map.indices.z.time].ravel()
    return AttrDict(t=t, dt=np.diff(t[:, None], axis=0), z=z_nodes, nu=nu_nodes,
                    z_dense=z, nu_dense=nu)


def phase_guesses(supplied, names):
    names = list(names)
    if supplied is None:
        return {}
    if not isinstance(supplied, Mapping) and isinstance(supplied, TrajectorySource) and len(names) == 1:
        return {names[0]: supplied}
    if not isinstance(supplied, Mapping):
        raise TypeError('A multiphase initial_guess must map phase names to trajectory sources')
    unknown = set(supplied)-set(names)
    if unknown:
        raise ValueError(f'Unknown initial-guess phases: {sorted(unknown)}')
    if any(not isinstance(value, TrajectorySource) for value in supplied.values()):
        raise TypeError('Each supplied phase initial_guess must implement sample(tau)')
    return supplied
