import numpy as np
import jax.numpy as jnp
from trajopt.utils.tools import AttrDict, deep_merge


CHAIN_FROM_PREVIOUS = "previous"


def _method_guess(method_config, phase_name):
    # default + phase override, same convention as a constraint's penalty.default
    method_guess = method_config.get("hyperparams", AttrDict()).get("initial_guess", AttrDict())
    default = method_guess.get("default", AttrDict())
    phase_guess = method_guess.get(phase_name)
    return deep_merge(default, phase_guess) if phase_guess is not None else default


def guess_field(phase, subproblem, name, default=None):
    # phase.guess wins over hyperparams.initial_guess; 'fcns.<name>' calls a formulation fcn
    value = getattr(phase.guess, name, None)
    if value is None:
        value = getattr(_method_guess(subproblem.method_config, phase.name), name, default)
    if isinstance(value, str) and value.startswith("fcns."):
        return phase.index_map.call_fcn(subproblem, value[len("fcns."):])
    return value


def _guess_fcn(subproblem, phase_name):
    # phase's own guess fcn if set, else fcns.initial_guess.default.guess
    module_fcns = subproblem.fcns.initial_guess
    phase_fcns = module_fcns.get(phase_name, module_fcns.get("default"))
    return phase_fcns.guess


def guess_endpoint(subproblem):
    """Final state of a phase's initial guess, in dimensional units."""
    phase = subproblem.phase
    z = np.asarray(subproblem.initial_guess.z)
    x_nd = z[:, phase.index_map.indices.z.state]
    return x_nd[-1] @ np.asarray(phase.nondim.M.state.nd2d).T


def set_initial_guess(phase, subproblem):
    guess_fcn = _guess_fcn(subproblem, phase.name)
    guess_fcn(phase, subproblem)

    subproblem.cost_init = subproblem.cost_type_module.compute_nonconvex_terminal_costs(
        subproblem.initial_guess.z, subproblem.initial_guess.nu, phase, subproblem
    )


def _constraint_of_type(phase, type_name):
    return next((c for c in phase.constraints.values() if c.type == type_name), None)


def _endpoint_states(phase, subproblem):
    """Nondimensional endpoints from guess.x_start/x_stop, else the boundary constraints."""
    n_x  = phase.index_map.n.state
    d2nd = phase.nondim.M.state.d2nd

    x_start = guess_field(phase, subproblem, "x_start")
    if x_start is not None:
        x0 = d2nd @ np.atleast_1d(x_start)
    else:
        cnstr = _constraint_of_type(phase, "initial_state")
        if cnstr is None:
            raise ValueError(
                f"phase '{phase.name}': a straight-line guess needs either "
                "guess.x_start or an initial_state constraint"
            )
        x0 = np.zeros(n_x)
        x0[np.asarray(cnstr.idx, dtype=int)] = cnstr.value

    x_stop = guess_field(phase, subproblem, "x_stop")
    if x_stop is not None:
        xf = d2nd @ np.atleast_1d(x_stop)
    else:
        xf = x0.copy()
        cnstr = _constraint_of_type(phase, "final_state")
        if cnstr is not None:
            xf[np.asarray(cnstr.idx, dtype=int)] = cnstr.value

    return x0, xf


def straight_line_initial_guess(phase, subproblem):
    index_map = phase.index_map
    init = subproblem.initial_guess
    N    = index_map.N.all

    x0, xf = _endpoint_states(phase, subproblem)
    u_start = guess_field(phase, subproblem, "u_start")
    u_stop  = guess_field(phase, subproblem, "u_stop")
    u0 = phase.nondim.M.control.d2nd @ np.atleast_1d(u_start)
    uf = phase.nondim.M.control.d2nd @ np.atleast_1d(u_stop)

    t = np.asarray(init.t).reshape(-1)
    if getattr(subproblem.hyperparams.discretize, 'mode', 'ms') == 'ps':
        _, etau, _, _ = subproblem.fcns.discretize.differential_operator(N - 1)
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


def nonlinear_initial_guess(phase, subproblem):
    init     = subproblem.initial_guess
    idx      = phase.index_map.indices
    N        = phase.index_map.N.all
    n_z      = phase.index_map.n.z
    n_nu     = phase.index_map.n.nu
    dynamics = phase.constraints.dynamics.fcn_znu
    params   = phase.params

    x_start = guess_field(phase, subproblem, 'x_start')
    if x_start is not None:
        x0 = phase.nondim.M.state.d2nd @ np.atleast_1d(x_start)
    else:
        x0 = phase.constraints.initial_state.value
    u_start = phase.nondim.M.control.d2nd @ guess_field(phase, subproblem, 'u_start')
    u_stop  = phase.nondim.M.control.d2nd @ guess_field(phase, subproblem, 'u_stop')

    t = np.asarray(init.t).reshape(-1)
    if getattr(subproblem.hyperparams.discretize, 'mode', 'ms') == 'ps':
        _, etau, _, _ = subproblem.fcns.discretize.differential_operator(N - 1)
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
    _, z_dense, nu_dense = subproblem.fcns.discretize.propagate_rk4(
        z0, 0.0, 1.0, nu_fn, dynamics, params, n_steps=n_total,
    )

    node_idx = np.clip(np.round(tau * n_total).astype(int), 0, n_total)

    init.t        = t
    init.dt       = np.diff(t.reshape(-1, 1), axis=0)
    init.z        = z_dense[node_idx]
    init.nu       = nu_dense[node_idx]
    init.z_dense  = z_dense
    init.nu_dense = nu_dense