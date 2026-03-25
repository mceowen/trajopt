import numpy as np
from trajopt.library.methods import integrators
from scipy.interpolate import interp1d

def straight_line_initial_guess(problem, method):

    line_guess_u_init = method.guess["line_guess_u_init"] @ method.nondim.M["ctrl"]["d2nd"]
    t_init            = np.cumsum(np.concatenate(([[0]], method.dt_init)))

    
    init_state_constraint = problem.constraints.get(type="equality_bc", boundary="init")[0]
    terminal_state_constraint = problem.constraints.get(type="equality_bc", boundary="final")[0]

    if len(init_state_constraint.idx) == problem.index_map.n.state:
        zi_full = init_state_constraint.value
    else:
        zi_guess = getattr(init_state_constraint, 'value_guess', None)
        if zi_guess is not None:
            zi_full = zi_guess
        else:
            raise ValueError("Initial_state.zi_guess must be provided for straight_line_initial_guess if initial_state is not fully defined")

    if len(terminal_state_constraint.idx) == problem.index_map.n.state:
        zf_full = terminal_state_constraint.value
    else:
        zf_guess = getattr(terminal_state_constraint, 'value_guess', None)
        if zf_guess is not None:
            zf_full = zf_guess
        else:
            raise ValueError("Final_state.zf_guess must be provided for straight_line_initial_guess if final_state is not fully defined")

    # Initial state
    t_start_end_pts = np.array([t_init[0], t_init[-1]])
    z_start_end_pts = np.vstack([zi_full, zf_full])
    z_init_interp_func = interp1d(t_start_end_pts, z_start_end_pts, axis=0)
    z_init = z_init_interp_func(t_init)

    # Initial control
    nu_init_interp_func = interp1d(t_init, line_guess_u_init, axis=0)
    nu_init = nu_init_interp_func(t_init)

    return t_init, z_init, nu_init

def nonlinear_initial_guess(problem, method):
    
    z0 = problem.constraints.get(type="equality_bc", boundary="init")[0].value

    # ---- Control initialization ----
    n_nu = problem.index_map.n['control']
    N = method.index_map.N['N']

    nl_guess_u_start = method.nondim.M.ctrl.d2nd @ method.guess["nl_guess_u_start"]
    nl_guess_u_stop  = method.nondim.M.ctrl.d2nd @ method.guess["nl_guess_u_stop"]

    t_init = np.cumsum(np.concatenate(([[0]], method.dt_init)))
    t_nl   = np.linspace(t_init[0], t_init[-1], 10000)

    # linearly interpolate the control between the start and stop values using scipy.interpolate.interp1d
    t_start_end_pts = np.array([t_init[0], t_init[-1]])
    nu_start_end_pts = np.vstack([nl_guess_u_start, nl_guess_u_stop])
    nu_init_interp_func = interp1d(t_start_end_pts, nu_start_end_pts, axis=0)
    nu_init = nu_init_interp_func(t_init)

    dynamics_cnstr = problem.constraints.get(type="dynamics")[0]
    
    if dynamics_cnstr.backend == "jax":
        t_nl, z_nl, nu_nl = integrators.propagate_jax_rk4_dense(z0, nu_init, t_init, t_nl, problem, method)

        z_interp_func = interp1d(t_nl, z_nl, axis=0)
        z_init        = z_interp_func(t_init)
    
    else:
        t_nl, z_nl, nu_nl = integrators.propagate_scipy_rk45(z0, nu_init, t_init, t_nl, problem, method)

    return t_init, z_init, nu_init


def ctcs_initial_guess(problem, method):
    """
    Initialize the guess for the constrained trajectory control system (CTCS).

    Parameters:
    params (dict): Dictionary containing parameters.

    Returns:
    dict: Updated params with initial guesses for the state vector.
    """
    
    # Extend z_init with zeros for the inequality constraints

    ctcs_init = np.zeros((method.z_init.shape[0], problem.index_map.n['ctcs']))

    return np.hstack([method.z_init, ctcs_init])