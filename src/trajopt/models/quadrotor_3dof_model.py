import numpy as np
import importlib

import trajopt.utils.set_defaults           as defaults
import trajopt.utils.tools                  as tools
import trajopt.algorithm.initial_guess      as guess
import trajopt.algorithm.convergence        as convergence
import trajopt.algorithm.convexification    as convexify
import trajopt.utils.nondim                 as nondim

def system_dynamics(ts, zs, us, model, t_vec=None):
    """
    x1, x2: r (position)
    u1, u2: v (velocity)
    """
    # extracts params if "problem" parent struct is passed in
    problem = model.problem
    params = problem['params']

    # extract constant param values
    m       = int( params['m'] )
    n       = int( params['n'] )
    mass    = params['mass'] / params['nondim']['nm']
    ge      = params['ge'] / params['nondim']['na']

    # extract states
    r = zs[0:3]
    v = zs[3:6]

    # extract controls 
    if t_vec is None:
        us2 = us
    else:
        us2 = np.array([np.interp(ts, t_vec, us[:, i]) for i in range(m)])
            
    # extract control
    T = us2

    # compute velocity and acceleration
    xDot        = np.empty(6) # initialize
    xDot[0:3]   = v
    xDot[3:6]   = T/mass + ge

    if np.issubdtype(r.dtype, np.number):
        if r[2] <= -1: # set xDot = 0 if the vehicle hits the ground
            xDot = np.zeros(n)
    elif np.issubdtype(r.dtype, np.nan) or any(np.isinf(r)):
        breakpoint()
        
    return xDot

def analytical_linsys(ts, zs, us, model):

    problem = model.problem 
    params = problem['params']
    
    # Extract parameters

    n       = params['n']
    m       = params['m']
    mass    = params['mass'] / params['nondim']['nm']

    # Sanity check for vector shapes
    zs = np.asarray(zs).flatten()
    us = np.asarray(us).flatten()

    assert len(zs) == n, f"Expected state vector of length {n}, got {len(zs)}"
    assert len(us) == m, f"Expected control vector of length {m}, got {len(us)}"

    # Compute A matrix (Jacobian w.r.t. state)
    n2 = n // 2
    Ac = np.block([
        [np.zeros((n2, n2)), np.eye(n2)],
        [np.zeros((n2, n))]
    ])

    # Compute B matrix (Jacobian w.r.t. control)
    Bc = np.vstack([
        np.zeros((n2, m)),
        np.eye(m)
    ]) * (1.0 / mass)

    # Evaluate nonlinear dynamics
    fc = system_dynamics(ts, zs, us, model)

    # Return in dictionary format
    linsys = {
        "dfcn_dz": Ac,
        "dfcn_du": Bc,
        "fcn":     fc
    }

    return linsys

def nonlinear_inequality_constraints(ts, zs, us, model):

    problem = model.problem
    params  = problem['params']

    N       = tools.num_timesteps(zs)
    n_nfz   = params.get("n_nfz", 0)
    n_path  = params.get("n_path", 0)  # placeholder

    # Handle state unpacking
    if zs.ndim == 2:
        rx = zs[:, 0]
        ry = zs[:, 1]
    elif zs.ndim == 1:
        rx = np.full(N, zs[0])
        ry = np.full(N, zs[1])
    else:
        raise ValueError(f"Unhandled zs shape {zs.shape}")

    # === NFZ constraints ===
    if n_nfz > 0:
        xc = params["obs"]["posc"][0]
        yc = params["obs"]["posc"][1]
        rc = params["obs"]["rc"]

        P_nfz = np.stack([
            rc[i]**2 - (rx - xc[i])**2 - (ry - yc[i])**2
            for i in range(n_nfz)
        ], axis=1)  # shape: (N, n_nfz)
    else:
        P_nfz = np.empty((N, 0))

    # === Path constraints (placeholder) ===
    P_path = np.empty((N, 0))  # currently unused

    # === Stack all inequality constraints ===
    P = np.hstack([P_path, P_nfz]) if P_path.size or P_nfz.size else np.empty((N, 0))

    if zs.ndim == 1:
        P = P.flatten()

    return P

def analytical_inequality_constraints(ts, zs, us, model):

    problem = model.problem
    params = problem['params']

    N         = tools.num_timesteps(zs)
    n         = params["n"]
    m         = params["m"]
    n_path    = params["n_path"]
    n_nfz     = params["n_nfz"]
    path_idx  = params["path_idx"]

    # Scale path limits using nondimensional constraint weights
    scale = params["nondim"]["np_ineq"][:n_path]
    path_lim_scaled = np.linalg.solve(np.diag(scale), params["path_lim"])

    # Obstacle info (broadcasted for speed)
    if n_nfz > 0:
        xc = params["obs"]["posc"][0]
        yc = params["obs"]["posc"][1]
        rc = params["obs"]["rc"]

    # === Preallocate flattened output arrays ===
    fcn_all   = np.zeros((N, n_path + n_nfz))
    dPdz_all  = np.zeros((N, n_path + n_nfz, n))
    dPdu_all  = np.zeros((N, n_path + n_nfz, m))

    # Also collect detailed path and NFZ constraint data if needed
    path_data = {"P": [], "Praw": [], "dPdz": [], "dPdu": []}
    nfz_data  = {"P": [], "dPdz": [], "dPdu": []}

    if zs.ndim == 1:
        ts = np.array([ts])
        zs = zs.reshape((1, -1))
        us = us.reshape((1, -1))

    for k in range(N):
        tk = ts[k]
        zk = zs[k]
        uk = us[k]
        rx_k, ry_k = zk[0], zk[1]

        # Evaluate all inequality constraints
        P_full = nonlinear_inequality_constraints(tk, zk, uk, model)

        # === Path constraints ===
        if n_path > 0:
            P_path = P_full[path_idx]
            fcn_all[k, :n_path] = P_path - path_lim_scaled
            dPdz_all[k, :n_path, :] = np.zeros((n_path, n))  # Placeholder
            dPdu_all[k, :n_path, :] = np.zeros((n_path, m))

            # Append to raw data
            path_data["P"].append(P_path - path_lim_scaled)
            path_data["Praw"].append(P_path)
            path_data["dPdz"].append(np.zeros((n_path, n)))
            path_data["dPdu"].append(np.zeros((n_path, m)))

        # === No-fly zone constraints ===
        if n_nfz > 0:
            P_nfz = P_full[n_path:n_path + n_nfz]
            fcn_all[k, n_path:n_path + n_nfz] = P_nfz

            dPdz_nfz = np.zeros((n_nfz, n))
            dPdz_nfz[:, 0] = - 2 * (rx_k - xc)
            dPdz_nfz[:, 1] = - 2 * (ry_k - yc)

            dPdu_nfz = np.zeros((n_nfz, m))

            dPdz_all[k, n_path:n_path + n_nfz, :] = dPdz_nfz
            dPdu_all[k, n_path:n_path + n_nfz, :] = dPdu_nfz

            # Store full data
            nfz_data["P"].append(P_nfz)
            nfz_data["dPdz"].append(dPdz_nfz)
            nfz_data["dPdu"].append(dPdu_nfz)

    return {
        "fcn": fcn_all,
        "dfcn_dz": dPdz_all,
        "dfcn_du": dPdu_all,
        "data": {
            "path": path_data,
            "nfz": nfz_data,
        }
    }