from typing import TYPE_CHECKING

import numpy as np
import jax.numpy as jnp
from trajopt.methods.scp import integrators
from trajopt.methods.scp import pseudospectral
from trajopt.methods.scp.scp_costs import scp_cost_types


def set_initial_guess(segment, scp_segment):
    guess_type = getattr(segment.guess, "type", "propagation")

    if guess_type == "propagation":
        nonlinear_initial_guess(segment, scp_segment)
    elif guess_type == "straight_line":
        straight_line_initial_guess(segment, scp_segment)

    scp_segment.cost_init = scp_cost_types.compute_nonconvex_terminal_costs(
        scp_segment.initial_guess.z, scp_segment.initial_guess.nu, segment, scp_segment
    )


def straight_line_initial_guess(segment, scp_segment):
    index_map = segment.index_map
    init = scp_segment.initial_guess
    N    = index_map.N.all
    cfg  = segment.guess

    x0 = segment.nondim.M.state.d2nd   @ np.atleast_1d(cfg.x_start)
    xf = segment.nondim.M.state.d2nd   @ np.atleast_1d(cfg.x_stop)
    u0 = segment.nondim.M.control.d2nd @ np.atleast_1d(cfg.u_start)
    uf = segment.nondim.M.control.d2nd @ np.atleast_1d(cfg.u_stop)

    t = np.asarray(init.t).reshape(-1)
    if getattr(scp_segment.flags, 'discretize', 'ms') == 'ps':
        _, etau, _, _ = pseudospectral.flipped_radau_differential_operator(N - 1)
        tau = (etau + 1.0) / 2.0
        t   = t[0] + tau * (t[-1] - t[0])
    else:
        tau = np.linspace(0.0, 1.0, N)

    Ts    = float(t[-1] - t[0])
    alpha = tau.reshape(-1, 1)

    x = (1 - alpha) * x0 + alpha * xf
    u = (1 - alpha) * u0 + alpha * uf

    beta = np.zeros((N, len(index_map.indices.z.augmented)))
    s    = np.full((N, 1), Ts)
    z, nu = index_map.pack_znu(x, t.reshape(-1, 1), beta, u, s)

    init.t        = t
    init.dt       = np.diff(t.reshape(-1, 1), axis=0)
    init.z        = z
    init.nu       = nu
    init.z_dense  = z
    init.nu_dense = nu


def nonlinear_initial_guess(segment, scp_segment):
    init     = scp_segment.initial_guess
    idx      = segment.index_map.indices
    N        = segment.index_map.N.all
    n_z      = segment.index_map.n.z
    n_nu     = segment.index_map.n.nu
    dynamics = segment.constraints.dynamics.fcn_znu
    params   = segment.params

    cfg     = segment.guess
    if hasattr(cfg, 'x_start'):
        x0 = segment.nondim.M.state.d2nd @ np.atleast_1d(cfg.x_start)
    else:
        x0 = segment.constraints.initial_state.value
    u_start = segment.nondim.M.control.d2nd @ cfg.u_start
    u_stop  = segment.nondim.M.control.d2nd @ cfg.u_stop

    t = np.asarray(init.t).reshape(-1)
    if getattr(scp_segment.flags, 'discretize', 'ms') == 'ps':
        _, etau, _, _ = pseudospectral.flipped_radau_differential_operator(N - 1)
        tau = (etau + 1.0) / 2.0
        t   = t[0] + tau * (t[-1] - t[0])
    else:
        tau = np.linspace(0.0, 1.0, N)
    Ts     = float(t[-1] - t[0])

    z0 = np.zeros(n_z)
    z0[idx.z.state] = x0
    z0[idx.z.time]  = t[0]

    tau_ref  = jnp.linspace(0.0, 1.0, N)
    u_ref    = jnp.asarray(np.linspace(0, 1, N).reshape(-1, 1) * (u_stop - u_start) + u_start)
    ctrl_sl  = jnp.array(idx.nu.control)
    dil_sl   = jnp.array(idx.nu.dilation_factor)
    sigma    = jnp.asarray(Ts)

    def nu_fn(z, tau):
        k = jnp.clip(jnp.searchsorted(tau_ref, tau, side='right') - 1, 0, N - 2)
        a = (tau - tau_ref[k]) / (tau_ref[k + 1] - tau_ref[k])
        u = (1 - a) * u_ref[k] + a * u_ref[k + 1]
        return jnp.zeros(n_nu).at[ctrl_sl].set(u).at[dil_sl].set(sigma)

    n_sub   = 10
    n_total = n_sub * (N - 1)
    _, z_dense, nu_dense = integrators.propagate_rk4(
        z0, 0.0, 1.0, nu_fn, dynamics, params, n_steps=n_total,
    )

    node_idx = np.clip(np.round(tau * n_total).astype(int), 0, n_total)

    init.t        = t
    init.dt       = np.diff(t.reshape(-1, 1), axis=0)
    init.z        = z_dense[node_idx]
    init.nu       = nu_dense[node_idx]
    init.z_dense  = z_dense
    init.nu_dense = nu_dense