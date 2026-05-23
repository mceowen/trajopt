import numpy as np
import jax.numpy as jnp
from trajopt.methods.scp import discretize
from trajopt.methods.scp import integrators


def set_initial_guess(problem, method):
    guess_type = getattr(method.config.method.guess, "type", "propagation")

    if guess_type == "propagation":
        nonlinear_initial_guess(problem, method)
    elif guess_type == "straight_line":
        straight_line_initial_guess(problem, method)

    method.cost_init = discretize.compute_nonconvex_costs(
        method.initial_guess.z, method.initial_guess.nu, problem, method
    )


def straight_line_initial_guess(problem, method):
    imap = problem.index_map
    init = method.initial_guess
    N    = method.index_map.N.time_grid
    cfg  = method.config.method.guess

    x0 = problem.nondim.M.state.d2nd   @ np.atleast_1d(cfg.x_start)
    xf = problem.nondim.M.state.d2nd   @ np.atleast_1d(cfg.x_stop)
    u0 = problem.nondim.M.control.d2nd @ np.atleast_1d(cfg.u_start)
    uf = problem.nondim.M.control.d2nd @ np.atleast_1d(cfg.u_stop)

    t = np.asarray(init.t).reshape(-1)
    if getattr(method.flags, 'discretize', 'ms') == 'ps':
        _, etau, _, _ = discretize.compute_ps_differentiation_matrix(N - 1)
        tau = (etau + 1.0) / 2.0
        t   = t[0] + tau * (t[-1] - t[0])

    Ts    = float(t[-1] - t[0])
    alpha = np.linspace(0, 1, N).reshape(-1, 1)

    x = (1 - alpha) * x0 + alpha * xf
    u = (1 - alpha) * u0 + alpha * uf

    beta = np.zeros((N, imap.n.ctcs))
    s    = np.full((N, 1), Ts)
    z, nu = imap.pack_znu(x, t.reshape(-1, 1), beta, u, s)

    init.t        = t
    init.dt       = np.diff(t.reshape(-1, 1), axis=0)
    init.z        = z
    init.nu       = nu
    init.z_dense  = z
    init.nu_dense = nu


def nonlinear_initial_guess(problem, method):
    init     = method.initial_guess
    idx      = problem.index_map.indices
    N        = method.index_map.N.time_grid
    n_z      = problem.index_map.n.z
    n_nu     = problem.index_map.n.nu
    dynamics = problem.constraints.get(type="dynamics")[0].fcn
    params   = problem.params

    x0      = problem.constraints.get(type="initial_state")[0].value
    u_start = problem.nondim.M.control.d2nd @ method.config.method.guess.u_start
    u_stop  = problem.nondim.M.control.d2nd @ method.config.method.guess.u_stop

    t = np.asarray(init.t).reshape(-1)
    if getattr(method.flags, 'discretize', 'ms') == 'ps':
        _, etau, _, _ = discretize.compute_ps_differentiation_matrix(N - 1)
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

    n_sub   = 100
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