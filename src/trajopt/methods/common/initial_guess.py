import numpy as np
import jax.numpy as jnp
from trajopt.methods.common import integrators
from trajopt.methods.common import pseudospectral


CHAIN_FROM_PREVIOUS = "previous"


def resolve_guess_type(phase, method_phase):
    """The phase's guess type, else the method config's, else propagation."""
    seg_type = getattr(phase.guess, "type", None)
    if seg_type is not None:
        return seg_type

    method_guess = getattr(method_phase.method_config, "guess", None)
    if method_guess is not None:
        return getattr(method_guess, "type", "propagation")

    return "propagation"


def guess_endpoint(method_phase):
    """Final state of a phase's initial guess, in dimensional units."""
    phase = method_phase.phase
    z = np.asarray(method_phase.initial_guess.z)
    x_nd = z[:, phase.index_map.indices.z.state]
    return x_nd[-1] @ np.asarray(phase.nondim.M.state.nd2d).T


def set_initial_guess(phase, method_phase):
    guess_type = resolve_guess_type(phase, method_phase)

    if guess_type == "propagation":
        nonlinear_initial_guess(phase, method_phase)
    elif guess_type == "straight_line":
        straight_line_initial_guess(phase, method_phase)
    else:
        raise ValueError(
            f"phase '{phase.name}': unknown guess type '{guess_type}' "
            "(expected 'propagation' or 'straight_line')"
        )

    method_phase.cost_init = method_phase.cost_type_module.compute_nonconvex_terminal_costs(
        method_phase.initial_guess.z, method_phase.initial_guess.nu, phase, method_phase
    )


def _constraint_of_type(phase, type_name):
    return next((c for c in phase.constraints.values() if c.type == type_name), None)


def _endpoint_states(phase):
    """Nondimensional endpoints from guess.x_start/x_stop, else the boundary constraints."""
    cfg  = phase.guess
    n_x  = phase.index_map.n.state
    d2nd = phase.nondim.M.state.d2nd

    if hasattr(cfg, "x_start"):
        x0 = d2nd @ np.atleast_1d(cfg.x_start)
    else:
        cnstr = _constraint_of_type(phase, "initial_state")
        if cnstr is None:
            raise ValueError(
                f"phase '{phase.name}': a straight-line guess needs either "
                "guess.x_start or an initial_state constraint"
            )
        x0 = np.zeros(n_x)
        x0[np.asarray(cnstr.idx, dtype=int)] = cnstr.value

    if hasattr(cfg, "x_stop"):
        xf = d2nd @ np.atleast_1d(cfg.x_stop)
    else:
        xf = x0.copy()
        cnstr = _constraint_of_type(phase, "final_state")
        if cnstr is not None:
            xf[np.asarray(cnstr.idx, dtype=int)] = cnstr.value

    return x0, xf


def straight_line_initial_guess(phase, method_phase):
    index_map = phase.index_map
    init = method_phase.initial_guess
    N    = index_map.N.all
    cfg  = phase.guess

    x0, xf = _endpoint_states(phase)
    u0 = phase.nondim.M.control.d2nd @ np.atleast_1d(cfg.u_start)
    uf = phase.nondim.M.control.d2nd @ np.atleast_1d(cfg.u_stop)

    t = np.asarray(init.t).reshape(-1)
    if getattr(method_phase.flags, 'discretize', 'ms') == 'ps':
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


def nonlinear_initial_guess(phase, method_phase):
    init     = method_phase.initial_guess
    idx      = phase.index_map.indices
    N        = phase.index_map.N.all
    n_z      = phase.index_map.n.z
    n_nu     = phase.index_map.n.nu
    dynamics = phase.constraints.dynamics.fcn_znu
    params   = phase.params

    cfg     = phase.guess
    if hasattr(cfg, 'x_start'):
        x0 = phase.nondim.M.state.d2nd @ np.atleast_1d(cfg.x_start)
    else:
        x0 = phase.constraints.initial_state.value
    u_start = phase.nondim.M.control.d2nd @ cfg.u_start
    u_stop  = phase.nondim.M.control.d2nd @ cfg.u_stop

    t = np.asarray(init.t).reshape(-1)
    if getattr(method_phase.flags, 'discretize', 'ms') == 'ps':
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