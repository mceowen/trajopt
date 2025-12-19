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
    nu_rad_ind      = params["nondim"]["nu_rad_ind"]
    M_state_d2nd    = params["nondim"]["M"]["state"]["d2nd"][np.ix_(nx_ind, nx_ind)]
    M_ctrl_d2nd     = params["nondim"]["M"]["ctrl"]["d2nd"][np.ix_(nu_ind, nu_ind)]
    
    # Rescale time to physical units of seconds
    ts              = params["nondim"]["nt"] * t
    
    # Return dimensionalized x and u
    zs             = M_state_d2nd @ x
    nu            = M_ctrl_d2nd @ u
    
    return t, z, nu

def dim_vars(t, xs, nu, params):
    """
    Dimensionalize state and control variables to physical quantities.
    
    Parameters:
    ts (numpy.ndarray): Non-dimensionalized time array
    zs(numpy.ndarray): Non-dimensionalized state variables array
    us (numpy.ndarray): Non-dimensionalized control variables array
    params (dict): Parameters dictionary with scaling constants
    
    Returns:
    tuple: Tuple containing dimensionalized time, state variables, and control variables
    """
    # Extract scaling constants
    nx_ind          = np.arange(params["model"]["n"])
    nu_ind          = np.arange(params["model"]["m"])
    nu_rad_ind      = [idx for idx in params["nondim"]["nu_rad_ind"] if idx <= nu_ind[-1]]
    M_state_nd2d    = np.linalg.pinv(params["nondim"]["M"]["state"]["d2nd"][np.ix_(nx_ind, nx_ind)])
    M_ctrl_nd2d     = np.linalg.pinv(params["nondim"]["M"]["ctrl"]["d2nd"][np.ix_(nu_ind, nu_ind)])
    
    # Rescale time to physical units of seconds
    t               = params["nondim"]["nt"] * ts
    
    # Return dimensionalized x and u
    x                   = M_state_nd2d @ xs[nx_ind, :]
    u                   = M_ctrl_nd2d @ nu[nu_ind, :]
    u[nu_rad_ind, :]    = wrap_to_pi(u[nu_rad_ind, :])
    
    return t, x, u

def subprob_variable_scaling(trajopt_obj, local_vars):

    mission = trajopt_obj.mission
    model = trajopt_obj.model
    method = trajopt_obj.method

    # Extract input struct
    I               = local_vars["I"]

    # Extract params
    n               = model.nz
    m               = model.m
    N               = method.N
    bool_dev_var    = method.flags["dev_var"]
    var_scl_flag    = method.flags["var_scl_flag"]

    z_ref          = I["z_ref"]
    nu_ref          = I["nu_ref"]
    z_max           = mission.z_max
    z_min           = mission.z_min
    u_max           = mission.u_max
    u_min           = mission.u_min

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
                dz_max = z_max - z_ref[k]
                dz_min = z_ref[k] - z_min

                du_max = u_max - nu_ref[k]
                du_min = nu_ref[k] - u_min

                M_x[k] = np.diag(dz_max - dz_min)
                b_x[k] = dz_min

                M_u[k] = np.diag(du_max - du_min)
                b_u[k] = du_min

        elif var_scl_flag == 2:  # linear scaling
            for k in range(N):
                dz_max = z_max - z_ref[k]
                dz_min = z_ref[k] - z_min

                du_max = u_max - nu_ref[k]
                du_min = nu_ref[k] - u_min

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
            raise ValueError("Undefined var_scl_flag!")

        dz = cp.vstack(cp.reshape(M_x[k] @ dzhat[k] + b_x[k], (1, n), order="C") for k in range(N))
        
        du = cp.vstack(cp.reshape(M_u[k] @ duhat[k] + b_u[k], (1, m), order="C") for k in range(N))

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
            raise ValueError("Undefined var_scl_flag!")

        x = np.zeros((N, n))
        u = np.zeros((N, m))
        for k in range(N):
            x[k] = M_x[k] @ xhat[k].value + b_x[k]
            u[k] = M_u[k] @ uhat[k].value + b_u[k]

        dz = x - z_ref
        du = u - nu_ref

    return dz, du

# Example usage
if __name__ == "__main__":

    # Define a dummy params dictionary for testing
    params = {
        "nondim": {
            "nu_rad_ind": [0, 1, 2],
            "M_state_d2nd": np.eye(3),
            "M_ctrl_d2nd": np.eye(3),
            "nt": 0.1
        }
    }
    
    t = np.array([0, 1, 2, 3])
    x = np.random.rand(3, 4)
    u = np.random.rand(3, 4)
    
    t, xs, nu = nondim_vars(t, x, u, params)
    print(f"ts: {ts}")
    print(f"xs: {xs}")
    print(f"us: {us}")

    # Define a dummy params dictionary for testing
    params = {
        "n": 3,
        "m": 2,
        "nondim": {
            "nu_rad_ind": [0, 1],
            "M_state_d2nd": np.eye(3),
            "M_ctrl_d2nd": np.eye(2),
            "nt": 0.1
        }
    }
    
    t = np.array([0, 1, 2, 3])
    zs= np.random.rand(3, 4)
    nu = np.random.rand(2, 4)
    
    t, x, u = dim_vars(t, xs, nu, params)
    print(f"t: {t}")
    print(f"x: {x}")
    print(f"u: {u}")

    # Define a dummy trajopt_obj for testing
    trajopt_obj = {
        "I": [{"z_ref": np.random.rand(3, 4), "nu_ref": np.random.rand(2, 4)}],
        "params": {
            "nz": 3,
            "m": 2,
            "N": 4,
            'flags': {"dev_var": True, "var_scl_flag": 1},
            "z_max": np.array([1, 1, 1]),
            "z_min": np.array([0, 0, 0]),
            "u_max": np.array([1, 1]),
            "u_min": np.array([0, 0])
        }
    }
    dz, du = scp_variable_scaling(trajopt_obj)
    print(f"dz: {dz}")
    print(f"du: {du}")