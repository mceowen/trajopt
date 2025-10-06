import numpy as np
from scipy.integrate import solve_ivp
import importlib

#def system_dynamics(t, x, us_init, params, ts_init):

def get_system_dynamics(params):
    """
    Dynamically load the correct system_dynamics() function
    depending on the provided module flag.
    """
    # Map flag names to module paths
    base_prefix = "trajopt.problem_models"
    full_module = f"{base_prefix}.{params['model_type']}"

    try:
        module = importlib.import_module(full_module)
    except ModuleNotFoundError as e:
        raise ImportError(f"Could not import dynamics module '{full_module}'") from e

    if not hasattr(module, "system_dynamics"):
        raise AttributeError(f"Module '{full_module}' has no function 'system_dynamics'")

    return module.system_dynamics

def straight_line_initial_guess(params):
    """
    Generate a straight line initial guess for trajectory and control.

    Parameters:
    params (dict): Dictionary containing parameters.

    Returns:
    dict: Updated params with initial guesses for trajectory and control.
    """
    # Initialization trajectory
    params['dt_init']   = (params['T_init'] / (params['N'] - 1)) * np.ones(params['N'] - 1)
    params['Ts_init']   = params['T_init'] / params['nondim']['nt']
    params['dts_init']  = params['dt_init'] / params['nondim']['nt']
    ts_init             = np.cumsum(np.concatenate(([0], params['dts_init'])))

    # Initial state
    zs_init             = np.array([np.linspace(params['zi'][i], params['zf'][i], params['N']) for i in range(params['n'])]).T

    # Initial control
    us_init             = np.zeros((params['N'],params['m']))

    # Create initial state and control vector
    params['ts_init']   = ts_init
    params['zs_init']   = zs_init
    params['us_init']   = us_init

    return params


def waypoint_initial_guess(params):
    """
    Generate an initial guess for trajectory and control using waypoints.

    Parameters:
    params (dict): Dictionary containing parameters.

    Returns:
    dict: Updated params with initial guesses for trajectory and control.
    """
    # Initialization trajectory
    params['dt_init']   = (params['T_init'] / (params['N'] - 1)) * np.ones(params['N'] - 1)
    params['Ts_init']   = params['T_init'] / params['nondim']['nt']
    params['dts_init']  = params['dt_init'] / params['nondim']['nt']
    ts_init             = np.cumsum(np.concatenate(([0], params['dts_init'])))

    # Waypoint
    if 'z_waypt' not in params:
        params['z_waypt'] = np.zeros(params['n'])
        
        # Loop through initial conditions
        for i_zi in range(params['n_init']):
            i_state = params['zi_idx'][i_zi]

            # Loop through terminal conditions
            for i_zf in range(params['n_term']):
                if params['zi_idx'][i_zi] == params['zf_idx'][i_zf]:
                    if params['zi'][i_zi] != params['zf'][i_zf]:
                        params['z_waypt'][i_state] = (params['zf'][i_zf] - params['zi'][i_zi]) / 2
                    else:
                        params['z_waypt'][i_state] = params['zi'][i_zi]

    N1 = params['N'] // 2
    idx1 = np.arange(1, N1 + 1)

    N2 = params['N'] - N1
    idx2 = np.arange(N1, params['N'])

    # Initialize
    zs_init = np.zeros((params['N'],params['n']))

    # Initial state
    for i_state in range(min(params['n'], len(params['z_waypt']))):
        if i_state in params['zi_idx'] and i_state in params['zf_idx']:
            i_init = np.where(params['zi_idx'] == i_state)[0][0]
            i_term = np.where(params['zf_idx'] == i_state)[0][0]

            zs_init[idx1-1, i_state]    = np.linspace(params['zi'][i_init], params['z_waypt'][i_state], N1)
            zs_init[idx2, i_state]      = np.linspace(params['z_waypt'][i_state], params['zf'][i_term], N2)

    # Initial control
    us_init             = np.zeros((params['N'], params['m']))

    # Create initial state and control vector
    params['ts_init']   = ts_init
    params['zs_init']   = zs_init
    params['us_init']   = us_init

    return params


def nonlinear_initial_guess(us_range, params):
    """
    Generate a nonlinear initial guess for trajectory and control.

    Parameters:
        us_range (ndarray): 2×1 or 2×2 array giving control input range (min/max per axis)
        params (dict): Parameter dictionary with fields 'N', 'T_init', 'nondim', 'z0s', etc.

    Returns:
        dict: Updated params with initial guesses (zs_init: N×n, us_init: N×m)
    """

    # ---- Time grid initialization ----
    params['dt_init'] = (params['T_init'] / (params['N'] - 1)) * np.ones(params['N'] - 1)
    params['Ts_init']  = params['T_init'] / params['nondim']['nt']
    params['dts_init'] = params['dt_init'] / params['nondim']['nt']
    ts_init = np.cumsum(np.concatenate(([0], params['dts_init'])))

    # ---- Control initialization ----
    # us_range is expected to have shape (m, 2): [[u1_min, u1_max], [u2_min, u2_max], ...]
    m = us_range.shape[1]
    N = params['N']

    # Create N×m matrix: each column is a linspace for that control dimension
    us_init = np.zeros((N, m))
    for i in range(m):
        umin, umax = us_range[0,i], us_range[1,i]
        us_init[:, i] = np.linspace(umin, umax, N)

    # ---- Propagate initial trajectory ----
    odesettings = {'atol': 1e-12, 'rtol': 1e-12}
    sol = solve_ivp(
        get_system_dynamics(params),
        [ts_init[0], ts_init[-1]],
        params['z0s'],
        args=(us_init, params, ts_init),
        t_eval=ts_init,
        **odesettings
    )

    # sol.y is (n, N) → transpose to (N, n)
    zs_init = sol.y.T

    # ---- Store results ----
    params['ts_init'] = ts_init
    params['zs_init'] = zs_init
    params['us_init'] = us_init

    return params


def ctcs_initial_guess(params):
    """
    Initialize the guess for the constrained trajectory control system (CTCS).

    Parameters:
    params (dict): Dictionary containing parameters.

    Returns:
    dict: Updated params with initial guesses for the state vector.
    """
    # Extend zs_init with zeros for the inequality constraints
    params['zs_init'] = np.hstack([params['zs_init'], np.zeros((params['zs_init'].shape[0], params['n_ineq']))])


    return params

# Example usage
if __name__ == "__main__":
    # Define dummy data for testing
    params = {
        'T_init': 10,
        'N': 50,
        'n': 4,
        'm': 2,
        'zi': np.array([0, 0, 0, 0]),
        'zf': np.array([1, 1, 1, 1]),
        'nondim': {'nt': 1}
    }
    
    updated_params = straight_line_initial_guess(params)
    print(updated_params)


    # Define dummy data for testing
    params = {
        'T_init': 10,
        'N': 50,
        'n': 4,
        'm': 2,
        'zi': np.array([0, 0, 0, 0]),
        'zf': np.array([1, 1, 1, 1]),
        'zi_idx': np.array([0, 1, 2, 3]),
        'zf_idx': np.array([0, 1, 2, 3]),
        'n_init': 4,
        'n_term': 4,
        'nondim': {'nt': 1}
    }
    
    updated_params = waypoint_initial_guess(params)
    print(updated_params)


    # Define dummy data for testing
    us_range = np.array([[0, 1], [1, 2]])
    params = {
        'T_init': 10,
        'N': 50,
        'nondim': {'nt': 1},
        'z0s': np.zeros(4)
    }
    
    updated_params = nonlinear_initial_guess(us_range, params)
    print(updated_params)


    # Define dummy data for testing
    params = {
        'zs_init': np.array([[1, 2, 3], [4, 5, 6]]),
        'n_ineq': 2,
        'N': 3
    }
    
    updated_params = ctcs_initial_guess(params)
    print(updated_params)