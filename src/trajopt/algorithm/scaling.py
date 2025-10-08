import numpy as np
import cvxpy as cp

def wrap_to_pi(angle):
    """
    Wrap angle to [-pi, pi]
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def nondim_vars(t, x, u, params):
    """
    Non-Dimensionalize state and control variables.
    
    Parameters:
    t (numpy.ndarray): Time array
    x (numpy.ndarray): State variables array
    u (numpy.ndarray): Control variables array
    params (dict): Parameters dictionary with scaling constants
    
    Returns:
    tuple: Tuple containing rescaled time, state variables, and control variables
    """
    # Extract scaling constants
    nx_ind          = np.arange(x.shape[0])
    nu_ind          = np.arange(u.shape[0])
    nu_rad_ind      = params['nondim']['nu_rad_ind']
    M_state_d2nd    = params['nondim']['M']['state']['d2nd'][np.ix_(nx_ind, nx_ind)]
    M_ctrl_d2nd     = params['nondim']['M']['ctrl']['d2nd'][np.ix_(nu_ind, nu_ind)]
    
    # Rescale time to physical units of seconds
    ts              = params['nondim']['nt'] * t
    
    # Return dimensionalized x and u
    xs              = M_state_d2nd @ x
    us              = M_ctrl_d2nd @ u
    
    return ts, xs, us

def dim_vars(ts, xs, us, params):
    """
    Dimensionalize state and control variables to physical quantities.
    
    Parameters:
    ts (numpy.ndarray): Non-dimensionalized time array
    xs (numpy.ndarray): Non-dimensionalized state variables array
    us (numpy.ndarray): Non-dimensionalized control variables array
    params (dict): Parameters dictionary with scaling constants
    
    Returns:
    tuple: Tuple containing dimensionalized time, state variables, and control variables
    """
    # Extract scaling constants
    nx_ind          = np.arange(params['n'])
    nu_ind          = np.arange(params['m'])
    nu_rad_ind      = [idx for idx in params['nondim']['nu_rad_ind'] if idx <= nu_ind[-1]]
    M_state_nd2d    = np.linalg.pinv(params['nondim']['M']['state']['d2nd'][np.ix_(nx_ind, nx_ind)])
    M_ctrl_nd2d     = np.linalg.pinv(params['nondim']['M']['ctrl']['d2nd'][np.ix_(nu_ind, nu_ind)])
    
    # Rescale time to physical units of seconds
    t               = params['nondim']['nt'] * ts
    
    # Return dimensionalized x and u
    x                   = M_state_nd2d @ xs[nx_ind, :]
    u                   = M_ctrl_nd2d @ us[nu_ind, :]
    u[nu_rad_ind, :]    = wrap_to_pi(u[nu_rad_ind, :])
    
    return t, x, u

def subprob_variable_scaling(problem, local_vars):

    # Extract input struct
    I               = local_vars['I']

    # Extract params
    n               = problem['params']['nz']
    m               = problem['params']['m']
    N               = problem['params']['N']
    bool_dev_var    = problem['params']['bools']['dev_var']
    var_scl_flag    = problem['params']['bools']['var_scl_flag']

    zs_ref          = I['zs_ref']
    us_ref          = I['us_ref']
    z_max           = problem['params']['z_max']
    z_min           = problem['params']['z_min']
    u_max           = problem['params']['u_max']
    u_min           = problem['params']['u_min']

    # DEVIATION VARIABLES
    if bool_dev_var:
        dzhat = cp.Variable((N, n))
        duhat = cp.Variable((N, m))

        M_x = np.zeros((N, n, n))
        b_x = np.zeros((N, n))
        M_u = np.zeros((N, m, m))
        b_u = np.zeros((N, m))

        if var_scl_flag == 1:  # affine scaling
            for k in range(N):
                dz_max = z_max - zs_ref[k]
                dz_min = zs_ref[k] - z_min

                du_max = u_max - us_ref[k]
                du_min = us_ref[k] - u_min

                M_x[k] = np.diag(dz_max - dz_min)
                b_x[k] = dz_min

                M_u[k] = np.diag(du_max - du_min)
                b_u[k] = du_min

        elif var_scl_flag == 2:  # linear scaling
            for k in range(N):
                dz_max = z_max - zs_ref[k]
                dz_min = zs_ref[k] - z_min

                du_max = u_max - us_ref[k]
                du_min = us_ref[k] - u_min

                M_x[k] = np.diag(dz_max - dz_min)
                b_x[k] = np.zeros(n)

                M_u[k] = np.diag(du_max - du_min)
                b_u[k] = np.zeros(m)

        elif var_scl_flag == 0:  # no scaling
            for k in range(N):
                M_x[k] = np.eye(n)
                b_x[k] = np.zeros(n)

                M_u[k] = np.eye(m)
                b_u[k] = np.zeros(m)

        else:
            raise ValueError('Undefined var_scl_flag!')

        dz = cp.vstack(cp.reshape(M_x[k] @ dzhat[k] + b_x[k], (1, n), order='C') for k in range(N))
        
        du = cp.vstack(cp.reshape(M_u[k] @ duhat[k] + b_u[k], (1, m), order='C') for k in range(N))

    # FULL STATE VARIABLES
    else:
        xhat = cp.Variable((N, n))
        uhat = cp.Variable((N, m))

        M_x = np.zeros((N, n, n))
        b_x = np.zeros((N, n))
        M_u = np.zeros((N, m, m))
        b_u = np.zeros((N, m))

        if var_scl_flag == 1:  # affine scaling
            for k in range(N):
                M_x[k] = np.diag(z_max - z_min)
                b_x[k] = z_min

                M_u[k] = np.diag(u_max - u_min)
                b_u[k] = u_min

        elif var_scl_flag == 2:  # linear scaling
            for k in range(N):
                M_x[k] = np.diag(z_max - z_min)
                b_x[k] = np.zeros(n)

                M_u[k] = np.diag(u_max - u_min)
                b_u[k] = np.zeros(m)

        elif var_scl_flag == 0:
            for k in range(N):
                M_x[k] = np.eye(n)
                b_x[k] = np.zeros(n)

                M_u[k] = np.eye(m)
                b_u[k] = np.zeros(m)

        else:
            raise ValueError('Undefined var_scl_flag!')

        x = np.zeros((N, n))
        u = np.zeros((N, m))
        for k in range(N):
            x[k] = M_x[k] @ xhat[k].value + b_x[k]
            u[k] = M_u[k] @ uhat[k].value + b_u[k]

        dz = x - zs_ref
        du = u - us_ref

    return dz, du

# Example usage
if __name__ == "__main__":

    # Define a dummy params dictionary for testing
    params = {
        'nondim': {
            'nu_rad_ind': [0, 1, 2],
            'M_state_d2nd': np.eye(3),
            'M_ctrl_d2nd': np.eye(3),
            'nt': 0.1
        }
    }
    
    t = np.array([0, 1, 2, 3])
    x = np.random.rand(3, 4)
    u = np.random.rand(3, 4)
    
    ts, xs, us = nondim_vars(t, x, u, params)
    print(f"ts: {ts}")
    print(f"xs: {xs}")
    print(f"us: {us}")

    # Define a dummy params dictionary for testing
    params = {
        'n': 3,
        'm': 2,
        'nondim': {
            'nu_rad_ind': [0, 1],
            'M_state_d2nd': np.eye(3),
            'M_ctrl_d2nd': np.eye(2),
            'nt': 0.1
        }
    }
    
    ts = np.array([0, 1, 2, 3])
    xs = np.random.rand(3, 4)
    us = np.random.rand(2, 4)
    
    t, x, u = dim_vars(ts, xs, us, params)
    print(f"t: {t}")
    print(f"x: {x}")
    print(f"u: {u}")

    # Define a dummy problem for testing
    problem = {
        'I': [{'zs_ref': np.random.rand(3, 4), 'us_ref': np.random.rand(2, 4)}],
        'params': {
            'nz': 3,
            'm': 2,
            'N': 4,
            'bools': {'dev_var': True, 'var_scl_flag': 1},
            'z_max': np.array([1, 1, 1]),
            'z_min': np.array([0, 0, 0]),
            'u_max': np.array([1, 1]),
            'u_min': np.array([0, 0])
        }
    }
    dz, du = subproblem_variable_scaling(problem)
    print(f"dz: {dz}")
    print(f"du: {du}")