import numpy as np
import jax 
import jax.numpy as jnp
import trajopt.utils.tools as tools
jax.config.update("jax_enable_x64", True)
import trajopt.core.modules.models.constraints     as constraints

def system_dynamics_jax(ts, zs, us, problem, t_vec=None):

    mission = problem.mission
    model = problem.model
    method = problem.method

    ctrl_type = model.bools['ctrl_type']

    # Extract constant param values from struct
    Om = mission.planet["omega"] / (method.nondim["nang"] / method.nondim["nt"])
    Kg = mission.planet["mu"] / (method.nondim["na"] * method.nondim["nd"] ** 2)

    # Extract states
    rs, theta, phi, vs, gamma, psi = zs

    # Determine lift and drag coefficients from velocity
    aero = mission.nonlinear_aero(ts, zs, us)
    L    = aero["L"]
    D    = aero["D"]

    # extract controls 
    if t_vec is None:
        us2 = us
    else:
        us2 = np.array([jnp.interp(ts, t_vec, us[:, i]) for i in range(model.m)])

    # Extract bank angle
    if ctrl_type == 'bank_aoa':
        alpha   = us2[1]
    
    sigma   = us2[0]

    # Extract sines and cosines of various values
    cp  = jnp.cos(phi)
    sp  = jnp.sin(phi)
    tp  = jnp.tan(phi)
    cg  = jnp.cos(gamma)
    sg  = jnp.sin(gamma)
    tg  = jnp.tan(gamma)
    cps = jnp.cos(psi)
    sps = jnp.sin(psi)

    cs  = jnp.cos(sigma)
    ss  = jnp.sin(sigma)
    
    # state derivative function
    xDot = jnp.array([
        vs * sg,
        vs * cg * sps / (rs * cp),
        vs * cg * cps / rs, 
        - D - Kg * sg / rs**2 + Om**2 * rs * cp * (sg * cp - cg * sp * cps),
        (1 / vs) * ( L * cs + (vs**2 - Kg / rs) * cg / rs ) + 2 * Om * cp * sps + Om**2 * rs * (1 / vs) * cp * (cg * cp + sg * cps * sp),
        (1 / vs) * ( L * ss / cg + vs**2 * cg * sps * tp / rs ) - 2 * Om * (tg * cps * cp - sp) + Om**2 * rs * (1 / (vs * cg)) * sps * sp * cp
    ])

    return xDot

def nonlinear_inequality_constraints(ts, zs, us, problem):

    mission = problem.mission
    model = problem.model
    method = problem.method

    N       = tools.num_timesteps(zs)
    n_nfz   = mission.n_nfz
    n_path  = mission.n_path  # placeholder

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
        xc = mission.obs["posc"][0]
        yc = mission.obs["posc"][1]
        rc = mission.obs["rc"]

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

def analytical_inequality_constraints(ts, zs, us, problem):
    mission = problem.mission
    model = problem.model
    method = problem.method

    N         = tools.num_timesteps(zs)
    n         = model.n
    m         = model.m
    n_path    = mission.n_path
    n_nfz     = mission.n_nfz
    path_idx  = mission.path_idx

    # Scale path limits using nondimensional constraint weights
    scale = method.nondim["np_ineq"][:n_path]
    path_lim_scaled = np.linalg.solve(np.diag(scale), mission.path_lim)

    # Obstacle info (broadcasted for speed)
    if n_nfz > 0:
        xc = mission.obs["posc"][0]
        yc = mission.obs["posc"][1]
        rc = mission.obs["rc"]

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
        P_full = nonlinear_inequality_constraints(tk, zk, uk, problem)

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