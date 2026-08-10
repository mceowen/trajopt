from typing import TYPE_CHECKING

import numpy as np
import jax.numpy as jnp
from trajopt.methods.common import integrators
from trajopt.methods.common import pseudospectral
from trajopt.methods.scp.scp_costs import scp_cost_types


def resolve_guess_type(segment, method_segment):
    """The segment's guess type, else the method config's, else propagation."""
    seg_type = getattr(segment.guess, "type", None)
    if seg_type is not None:
        return seg_type

    method_guess = getattr(method_segment.method_config, "guess", None)
    if method_guess is not None:
        return getattr(method_guess, "type", "propagation")

    return "propagation"


def set_initial_guess(segment, method_segment):
    guess_type = resolve_guess_type(segment, method_segment)

    if guess_type == "propagation":
        nonlinear_initial_guess(segment, method_segment)
    elif guess_type == "straight_line":
        straight_line_initial_guess(segment, method_segment)
    else:
        raise ValueError(
            f"segment '{segment.name}': unknown guess type '{guess_type}' "
            "(expected 'propagation' or 'straight_line')"
        )

    method_segment.cost_init = scp_cost_types.compute_nonconvex_terminal_costs(
        method_segment.initial_guess.z, method_segment.initial_guess.nu, segment, method_segment
    )


def _constraint_of_type(segment, type_name):
    return next((c for c in segment.constraints.values() if c.type == type_name), None)


def _endpoint_states(segment):
    """Nondimensional endpoints from guess.x_start/x_stop, else the boundary constraints."""
    cfg  = segment.guess
    n_x  = segment.index_map.n.state
    d2nd = segment.nondim.M.state.d2nd

    if hasattr(cfg, "x_start"):
        x0 = d2nd @ np.atleast_1d(cfg.x_start)
    else:
        cnstr = _constraint_of_type(segment, "initial_state")
        if cnstr is None:
            raise ValueError(
                f"segment '{segment.name}': a straight-line guess needs either "
                "guess.x_start or an initial_state constraint"
            )
        x0 = np.zeros(n_x)
        x0[np.asarray(cnstr.idx, dtype=int)] = cnstr.value

    if hasattr(cfg, "x_stop"):
        xf = d2nd @ np.atleast_1d(cfg.x_stop)
    else:
        xf = x0.copy()
        cnstr = _constraint_of_type(segment, "final_state")
        if cnstr is not None:
            xf[np.asarray(cnstr.idx, dtype=int)] = cnstr.value

    return x0, xf


def straight_line_initial_guess(segment, method_segment):
    index_map = segment.index_map
    init = method_segment.initial_guess
    N    = index_map.N.all
    cfg  = segment.guess

    x0, xf = _endpoint_states(segment)
    u0 = segment.nondim.M.control.d2nd @ np.atleast_1d(cfg.u_start)
    uf = segment.nondim.M.control.d2nd @ np.atleast_1d(cfg.u_stop)

    t = np.asarray(init.t).reshape(-1)
    if getattr(method_segment.flags, 'discretize', 'ms') == 'ps':
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


def nonlinear_initial_guess(segment, method_segment):
    init     = method_segment.initial_guess
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
    if getattr(method_segment.flags, 'discretize', 'ms') == 'ps':
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