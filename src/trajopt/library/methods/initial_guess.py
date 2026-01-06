import numpy as np
from scipy.integrate import solve_ivp
import importlib
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import time
import pprint

def rk4_propagate_jax(dynamics, z0, nu_ref, t_ref, problem, method):

    time0 = time.perf_counter()

    # convert to JAX arrays
    z0_jax = jnp.array(z0)
    nu_ref_jax = jnp.array(nu_ref)
    t_ref_jax = jnp.array(t_ref)
    
    N = len(t_ref_jax)
    n = len(z0_jax)
    
    def rk4_step(zi, ti, dt, ui, ui_next):
        k1 = dynamics(ti, zi, ui)
        u2 = 0.5 * ui + 0.5 * ui_next
        k2 = dynamics(ti + 0.5*dt, zi + 0.5*dt*k1, u2)
        k3 = dynamics(ti + 0.5*dt, zi + 0.5*dt*k2, u2)
        k4 = dynamics(ti + dt, zi + dt*k3, ui_next)
    
        zi_next = zi + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return zi_next
    
    # Define loop body for scan
    def scan_fn(zi, i):
        dt = t_ref_jax[i+1] - t_ref_jax[i]
        ti = t_ref_jax[i]
        ui = nu_ref_jax[i]
        ui_next = nu_ref_jax[i+1]
        
        zi_next = rk4_step(zi, ti, dt, ui, ui_next)
        return zi_next, zi_next
    
    # Wrap scan in a function, then JIT-compile it
    def scan_wrapper(z0, xs):
        _, z_propagated = jax.lax.scan(scan_fn, z0, xs)
        return z_propagated
    
    scan_jit = jax.jit(scan_wrapper)
    z_propagated = scan_jit(z0_jax, jnp.arange(N-1))
    
    z_jax = jnp.vstack([z0_jax[None, :], z_propagated])
    z_numpy = np.array(z_jax)

    init_guess_time = time.perf_counter() - time0
    print(f"Initial guess time: {init_guess_time} seconds")

    return z_numpy

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
    
    if use_jax:
        z_init = rk4_propagate_jax(
            problem.constraints.get('name', 'dynamics')[0].fcn,
            problem.constraints.get('name', 'initial_state')[0].x,
            nu_init,
            t_init,
            problem,
            method
        )
    else:
        # Wrapper that does FOH interpolation before calling dynamics
        def dynamics_wrapper_scipy(t, z):
            # FOH interpolation to get control at time t
            u = np.array([np.interp(t, t_init, nu_init[:, i]) for i in range(m)])
            return problem.constraints.get('name', 'dynamics')[0].fcn(t, z, u)
        
        odesettings = {"atol": 1e-12, "rtol": 1e-12}
        sol = solve_ivp(
            dynamics_wrapper_scipy,
            [t_init[0], t_init[-1]],
            problem.zi,
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

# Example usage
if __name__ == "__main__":
    # Define dummy data for testing
    params = {
        "T_init": 10,
        "N": 50,
        "n": 4,
        "m": 2,
        "zi": np.array([0, 0, 0, 0]),
        "zf": np.array([1, 1, 1, 1]),
        "nondim": {"nt": 1}
    }
    
    updated_params = straight_line_initial_guess(params)
    print(updated_params)


    # Define dummy data for testing
    params = {
        "T_init": 10,
        "N": 50,
        "n": 4,
        "m": 2,
        "zi": np.array([0, 0, 0, 0]),
        "zf": np.array([1, 1, 1, 1]),
        "zi_idx": np.array([0, 1, 2, 3]),
        "zf_idx": np.array([0, 1, 2, 3]),
        "n_init": 4,
        "n_term": 4,
        "nondim": {"nt": 1}
    }
    
    updated_params = waypoint_initial_guess(params)
    print(updated_params)


    # Define dummy data for testing
    nu_range = np.array([[0, 1], [1, 2]])
    params = {
        "T_init": 10,
        "N": 50,
        "nondim": {"nt": 1},
        "z0s": np.zeros(4)
    }
    
    updated_params = nonlinear_initial_guess(nu_range, params)
    print(updated_params)


    # Define dummy data for testing
    params = {
        "z_init": np.array([[1, 2, 3], [4, 5, 6]]),
        "n_ineq": 2,
        "N": 3
    }
    
    updated_params = ctcs_initial_guess(params)
    print(updated_params)