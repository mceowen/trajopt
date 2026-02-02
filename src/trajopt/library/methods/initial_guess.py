import numpy as np
from scipy.integrate import solve_ivp
import time
from trajopt.library.methods.integrators import jit_rk4_jax_dense, propagate_rk4_dense

def straight_line_initial_guess(problem, method):
    """
    Generate a straight line initial guess for trajectory and control.

    Parameters:
    params (dict): Dictionary containing parameters.

    Returns:
    dict: Updated params with initial guesses for trajectory and control.
    """


    # Initialization trajectory
    method.dt_init   = (method.guess["T_init"] / (method.N - 1)) * np.ones(method.N - 1)
    method.Ts_init   = method.guess["T_init"] / method.nondim.nt
    method.dt_init  = method.dt_init / method.nondim.nt
    t_init             = np.cumsum(np.concatenate(([0], method.dt_init)))

    # Initial state
    z_init             = np.array([np.linspace(method.guess["zi_guess"][i], method.guess["zf_guess"][i], method.N) for i in range(problem.n)]).T

    # Initial control
    nu_init             = np.zeros((method.N,problem.m))

    # Create initial state and control vector
    method.t_init   = t_init
    method.z_init   = z_init
    method.nu_init   = nu_init

def waypoint_initial_guess(problem, method):
    """
    Generate an initial guess for trajectory and control using waypoints.

    Parameters:
    params (dict): Dictionary containing parameters.

    Returns:
    dict: Updated params with initial guesses for trajectory and control.
    """

    # Initialization trajectory
    method.dt_init   = (method.T_init / (method.N - 1)) * np.ones(method.N - 1)
    method.Ts_init   = method.T_init / method.nondim.nt
    method.dt_init  = method.dt_init / method.nondim.nt
    t_init             = np.cumsum(np.concatenate(([0], method.dt_init)))

    # Waypoint
    if "z_waypt" not in params:
        method.z_waypt = np.zeros(problem.n)
        
        # Loop through initial conditions
        for i_zi in range(problem.n_init):
            i_state = problem.zi_idx[i_zi]

            # Loop through terminal conditions
            for i_zf in range(problem.n_term):
                if problem.zi_idx[i_zi] == problem.zf_idx[i_zf]:
                    if problem.zi[i_zi] != problem.zf[i_zf]:
                        method.z_waypt[i_state] = (problem.zf[i_zf] - problem.zi[i_zi]) / 2
                    else:
                        method.z_waypt[i_state] = problem.zi[i_zi]

    N1 = method.N // 2
    idx1 = np.arange(1, N1 + 1)

    N2 = method.N - N1
    idx2 = np.arange(N1, method.N)

    # Initialize
    z_init = np.zeros((method.N,problem.n))

    # Initial state
    for i_state in range(min(problem.n, len(method.z_waypt))):
        if i_state in problem.zi_idx and i_state in problem.zf_idx:
            i_init = np.where(problem.zi_idx == i_state)[0][0]
            i_term = np.where(problem.zf_idx == i_state)[0][0]

            z_init[idx1-1, i_state]    = np.linspace(problem.zi[i_init], method.z_waypt[i_state], N1)
            z_init[idx2, i_state]      = np.linspace(method.z_waypt[i_state], problem.zf[i_term], N2)

    # Initial control
    nu_init             = method.line_guess_u_init

    # Create initial state and control vector
    method.t_init   = t_init
    method.z_init   = z_init
    method.nu_init   = nu_init


def nonlinear_initial_guess(nu_range, problem, method):
    """
    Generate a nonlinear initial guess for trajectory and control.

    Parameters:
        nu_range (ndarray): 2×1 or 2×2 array giving control input range (min/max per axis)
        params (dict): Parameter dictionary with fields "N", "T_init", "nondim", "z0s", etc.

    Returns:
        dict: Updated params with initial guesses (z_init: N×n, nu_init: N×m)
    """

    # ---- Time grid initialization ----
    method.dt_init = (method.guess["T_init"] / (method.N - 1)) * np.ones(method.N - 1)
    method.Ts_init  = method.guess["T_init"] / method.nondim.nt
    method.dt_init = method.dt_init / method.nondim.nt
    t_init = np.cumsum(np.concatenate(([0], method.dt_init)))

    # ---- Control initialization ----
    # nu_range is expected to have shape (m, 2): [[u1_min, u1_max], [u2_min, u2_max], ...]
    m = nu_range.shape[1]
    N = method.N

    # Create N×m matrix: each column is a linspace for that control dimension
    nu_init = np.zeros((N, m))
    for i in range(m):
        umin, umax = nu_range[0,i], nu_range[1,i]
        nu_init[:, i] = np.linspace(umin, umax, N)

    # ---- Propagate initial trajectory ----
    # Choose integrator based on jax_dyn flag
    use_jax = method.flags.get("jax_dyn", 0)
    z0 = problem.constraints.get('name', 'initial_state')[0].x
    
    if use_jax:
        # Setup JAX integrator if not already done
        if not hasattr(method, 'propagate_rk4_dense_jit'):
            jit_rk4_jax_dense(problem, method)
        
        # Propagate at knot points (use t_init as both reference and evaluation grid)
        z_init = propagate_rk4_dense(z0, nu_init, t_init, t_init, problem, method)
    else:
        # Wrapper that does FOH interpolation before calling dynamics
        dynamics_fcn = problem.constraints.get('name', 'dynamics')[0].fcn
        
        def dynamics_wrapper_scipy(t, z):
            # FOH interpolation to get control at time t
            u = np.array([np.interp(t, t_init, nu_init[:, i]) for i in range(m)])
            return dynamics_fcn(t, z, u)
        
        odesettings = {"atol": 1e-12, "rtol": 1e-12}
        sol = solve_ivp(
            dynamics_wrapper_scipy,
            [t_init[0], t_init[-1]],
            z0,
            t_eval=t_init,
            **odesettings
        )
        # sol.y is (n, N) → transpose to (N, n)
        z_init = sol.y.T

    # ---- Store results ----
    print("initial nondim state: ")
    print(problem.constraints.get('name', 'initial_state')[0].x)

    method.t_init = t_init
    method.z_init = z_init
    method.nu_init = nu_init


def ctcs_initial_guess(problem, method):
    """
    Initialize the guess for the constrained trajectory control system (CTCS).

    Parameters:
    params (dict): Dictionary containing parameters.

    Returns:
    dict: Updated params with initial guesses for the state vector.
    """
    
    # Extend z_init with zeros for the inequality constraints

    ctcs_init = np.zeros((method.z_init.shape[0], problem.n_ctcs))

    method.z_init = np.hstack([method.z_init, ctcs_init])