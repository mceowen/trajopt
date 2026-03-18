import numpy as np
from trajopt.library.methods import discretize
from trajopt.library.methods import integrators
from scipy.interpolate import interp1d
from trajopt.utils.tools import AttrDict


def set_initial_guess(problem,method):

        if (getattr(method.guess, "type", "propagation") == "propagation") or method.flags.buff_dyn == "term":
            nonlinear_initial_guess(problem, method)
        else:
            straight_line_initial_guess(problem, method)

        method.initial_guess.t = np.asarray(method.initial_guess.t).reshape(-1)
        method.initial_guess.dt = np.diff(method.initial_guess.t.reshape(-1, 1), axis=0)

        augment_initial_guess(problem, method)

        method.cost_init = discretize.compute_nonconvex_costs(method.initial_guess.z, method.initial_guess.nu, problem, method)
    

##################################################################################################################################

def _ensure_initial_guess(method):
    if not hasattr(method, "initial_guess") or method.initial_guess is None:
        method.initial_guess = AttrDict({})

    for key in ("t", "dt", "x", "u", "z", "nu"):
        if not hasattr(method.initial_guess, key):
            setattr(method.initial_guess, key, None)

    if method.initial_guess.t is not None:
        method.initial_guess.t = np.asarray(method.initial_guess.t).reshape(-1)
        method.initial_guess.dt = np.diff(method.initial_guess.t.reshape(-1, 1), axis=0)

    return method.initial_guess


# TODO(Skye): Possibly add actual computation of beta using index_map helpers
def augment_initial_guess(problem, method):
    idx     = problem.index_map.indices
    N       = method.index_map.N.time_grid
    init    = _ensure_initial_guess(method)

    if init.t is None or init.x is None or init.u is None:
        raise ValueError("method.initial_guess.t, x, and u must be set before calling augment_initial_guess")

    t_init                      = np.asarray(init.t).reshape(-1)

    x_init                      = np.asarray(init.x)
    u_init                      = np.asarray(init.u)

    z_init                      = np.zeros((N, problem.index_map.n.z))
    z_init[:, idx.z.state]      = x_init
    z_init[:, idx.z.time]       = t_init.reshape(-1, 1)
    z_init[:, idx.z.ctcs]       = 0.0 

    nu_init                     = np.zeros((N, problem.index_map.n.nu))
    nu_init[:, idx.nu.control]  = u_init

    delta_tau                   = 1.0 / (N - 1)
    dt_init                     = np.zeros((N, 1))
    dt_ref                      = np.diff(t_init.reshape(-1, 1), axis=0)
    init.dt                     = dt_ref
    dt_init[:-1, 0]             = dt_ref[:, 0]
    dt_init[-1, 0]              = dt_ref[-1, 0]
    nu_init[:, idx.nu.dilation_factor] = dt_init / delta_tau

    init.z  = z_init
    init.nu = nu_init

    return z_init, nu_init

def straight_line_initial_guess(problem, method):

    init                        = _ensure_initial_guess(method)

    line_init_u_init            = method.guess["line_guess_u_init"] @ method.nondim.M.control["d2nd"]
    t_init                      = np.asarray(init.t).reshape(-1)

    
    init_state_constraint       = problem.constraints.get(type="equality_bc", boundary="init")[0]
    terminal_state_constraint   = problem.constraints.get(type="equality_bc", boundary="final")[0]

    if len(init_state_constraint.idx) == problem.index_map.n.state:
        xi_full = init_state_constraint.x
    else:
        xi_guess = getattr(init_state_constraint, 'value_guess', None)
        if xi_guess is not None:
            xi_full = xi_guess
        else:
            raise ValueError("Initial_state.xi_guess must be provided for straight_line_initial_guess if initial_state is not fully defined")

    if len(terminal_state_constraint.idx) == problem.index_map.n.state:
        xf_full = terminal_state_constraint.value
    else:
        xf_guess = getattr(terminal_state_constraint, 'value_guess', None)
        if xf_guess is not None:
            xf_full = xf_guess
        else:
            raise ValueError("Final_state.xf_guess must be provided for straight_line_initial_guess if final_state is not fully defined")

    # Initial state
    t_start_end_pts     = np.array([t_init[0], t_init[-1]])
    x_start_end_pts     = np.vstack([xi_full, xf_full])
    x_init_interp_func  = interp1d(t_start_end_pts, x_start_end_pts, axis=0)
    x_init              = x_init_interp_func(t_init)

    # Initial control
    u_init_interp_func  = interp1d(t_init, line_init_u_init, axis=0)
    u_init              = u_init_interp_func(t_init)

    init.t = t_init
    init.x = x_init
    init.u = u_init

    return t_init, x_init, u_init

def nonlinear_initial_guess(problem, method):
    init                = _ensure_initial_guess(method)

    x0                  = problem.constraints.get(type="equality_bc", boundary="init")[0].value
    dynamics_cnstr      = problem.constraints.get(type="dynamics")[0]

    # ---- Control initialization ----
    nl_init_u_start     = method.nondim.M.control.d2nd @ method.guess["nl_guess_u_start"]
    nl_init_u_stop      = method.nondim.M.control.d2nd @ method.guess["nl_guess_u_stop"]

    t_init              = np.asarray(init.t).reshape(-1)
    t_nl                = np.linspace(t_init[0], t_init[-1], 10000)

    # Linearly interpolate control between start and stop values
    t_start_end_pts     = np.array([t_init[0], t_init[-1]])
    u_start_end_pts     = np.vstack([nl_init_u_start, nl_init_u_stop])
    u_init_interp_func  = interp1d(t_start_end_pts, u_start_end_pts, axis=0)
    u_init              = u_init_interp_func(t_init)

    # Propagate physical dynamics dx/dt = f(t, x, u) directly over real time
    if dynamics_cnstr.backend == "jax":
        t_nl_out, x_nl, _ = integrators.propagate_jax_rk4_dense(
            x0,
            u_init,
            t_init,
            t_nl,
            problem,
            method,
            compiled_attr_name="propagate_rk4_physical_jit",
        )
    else:
        t_nl_out, x_nl, _ = integrators.propagate_scipy_rk45(
            x0,
            u_init,
            t_init,
            t_nl,
            problem,
            method,
            dynamics=dynamics_cnstr.fcn_base,
        )

    x_interp_func = interp1d(t_nl_out, x_nl, axis=0)
    x_init        = x_interp_func(t_init)

    init.t = t_init
    init.x = x_init
    init.u = u_init

    return t_init, x_init, u_init