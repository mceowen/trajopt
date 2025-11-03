import numpy as np
from scipy.integrate import solve_ivp
import importlib

def straight_line_initial_guess(problem):
    """
    Generate a straight line initial guess for trajectory and control.

    Parameters:
    params (dict): Dictionary containing parameters.

    Returns:
    dict: Updated params with initial guesses for trajectory and control.
    """
    mission = problem.mission
    model = problem.model
    method = problem.method


    # Initialization trajectory
    method.dt_init   = (method.T_init / (method.N - 1)) * np.ones(method.N - 1)
    method.Ts_init   = method.T_init / method.nondim["nt"]
    method.dts_init  = method.dt_init / method.nondim["nt"]
    ts_init             = np.cumsum(np.concatenate(([0], method.dts_init)))

    # Initial state
    zs_init             = np.array([np.linspace(mission.zi[i], mission.zf[i], method.N) for i in range(model.n)]).T

    # Initial control
    us_init             = np.zeros((method.N,model.m))

    # Create initial state and control vector
    method.ts_init   = ts_init
    method.zs_init   = zs_init
    method.us_init   = us_init


def waypoint_initial_guess(problem):
    """
    Generate an initial guess for trajectory and control using waypoints.

    Parameters:
    params (dict): Dictionary containing parameters.

    Returns:
    dict: Updated params with initial guesses for trajectory and control.
    """

    mission = problem.mission
    model = problem.model
    method = problem.method

    # Initialization trajectory
    method.dt_init   = (method.T_init / (method.N - 1)) * np.ones(method.N - 1)
    method.Ts_init   = method.T_init / method.nondim["nt"]
    method.dts_init  = method.dt_init / method.nondim["nt"]
    ts_init             = np.cumsum(np.concatenate(([0], method.dts_init)))

    # Waypoint
    if "z_waypt" not in params:
        method.z_waypt = np.zeros(model.n)
        
        # Loop through initial conditions
        for i_zi in range(mission.n_init):
            i_state = mission.zi_idx[i_zi]

            # Loop through terminal conditions
            for i_zf in range(mission.n_term):
                if mission.zi_idx[i_zi] == mission.zf_idx[i_zf]:
                    if mission.zi[i_zi] != mission.zf[i_zf]:
                        method.z_waypt[i_state] = (mission.zf[i_zf] - mission.zi[i_zi]) / 2
                    else:
                        method.z_waypt[i_state] = mission.zi[i_zi]

    N1 = method.N // 2
    idx1 = np.arange(1, N1 + 1)

    N2 = method.N - N1
    idx2 = np.arange(N1, method.N)

    # Initialize
    zs_init = np.zeros((method.N,model.n))

    # Initial state
    for i_state in range(min(model.n, len(method.z_waypt))):
        if i_state in mission.zi_idx and i_state in mission.zf_idx:
            i_init = np.where(mission.zi_idx == i_state)[0][0]
            i_term = np.where(mission.zf_idx == i_state)[0][0]

            zs_init[idx1-1, i_state]    = np.linspace(mission.zi[i_init], method.z_waypt[i_state], N1)
            zs_init[idx2, i_state]      = np.linspace(method.z_waypt[i_state], mission.zf[i_term], N2)

    # Initial control
    us_init             = np.zeros((method.N, model.m))

    # Create initial state and control vector
    method.ts_init   = ts_init
    method.zs_init   = zs_init
    method.us_init   = us_init


def nonlinear_initial_guess(us_range, problem):
    """
    Generate a nonlinear initial guess for trajectory and control.

    Parameters:
        us_range (ndarray): 2×1 or 2×2 array giving control input range (min/max per axis)
        params (dict): Parameter dictionary with fields "N", "T_init", "nondim", "z0s", etc.

    Returns:
        dict: Updated params with initial guesses (zs_init: N×n, us_init: N×m)
    """

    mission = problem.mission
    model = problem.model
    method = problem.method

    # ---- Time grid initialization ----
    method.dt_init = (method.T_init / (method.N - 1)) * np.ones(method.N - 1)
    method.Ts_init  = method.T_init / method.nondim["nt"]
    method.dts_init = method.dt_init / method.nondim["nt"]
    ts_init = np.cumsum(np.concatenate(([0], method.dts_init)))

    # ---- Control initialization ----
    # us_range is expected to have shape (m, 2): [[u1_min, u1_max], [u2_min, u2_max], ...]
    m = us_range.shape[1]
    N = method.N

    # Create N×m matrix: each column is a linspace for that control dimension
    us_init = np.zeros((N, m))
    for i in range(m):
        umin, umax = us_range[0,i], us_range[1,i]
        us_init[:, i] = np.linspace(umin, umax, N)

    # ---- Propagate initial trajectory ----
    odesettings = {"atol": 1e-12, "rtol": 1e-12}
    sol = solve_ivp(
        model.dynamics,
        [ts_init[0], ts_init[-1]],
        mission.zi,
        args=(us_init, ts_init),
        t_eval=ts_init,
        **odesettings
    )

    # sol.y is (n, N) → transpose to (N, n)
    zs_init = sol.y.T

    # ---- Store results ----
    method.ts_init = ts_init
    method.zs_init = zs_init
    method.us_init = us_init


def ctcs_initial_guess(problem):
    """
    Initialize the guess for the constrained trajectory control system (CTCS).

    Parameters:
    params (dict): Dictionary containing parameters.

    Returns:
    dict: Updated params with initial guesses for the state vector.
    """
    
    mission = problem.mission
    method = problem.method
    # Extend zs_init with zeros for the inequality constraints
    method.zs_init = np.hstack([method.zs_init, np.zeros((method.zs_init.shape[0], mission.n_ineq))])

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
    us_range = np.array([[0, 1], [1, 2]])
    params = {
        "T_init": 10,
        "N": 50,
        "nondim": {"nt": 1},
        "z0s": np.zeros(4)
    }
    
    updated_params = nonlinear_initial_guess(us_range, params)
    print(updated_params)


    # Define dummy data for testing
    params = {
        "zs_init": np.array([[1, 2, 3], [4, 5, 6]]),
        "n_ineq": 2,
        "N": 3
    }
    
    updated_params = ctcs_initial_guess(params)
    print(updated_params)