# Skye Mceowen and Aman Tiwary
# Feb. 14, 2024
# functions that set default problem parameters

import numpy as np
from trajopt.algorithm.discretization import set_ltv_indices

def set_params_constraint_default(problem):
    """
    Set default constraint dimensions, duals, weights, and convergence criteria
    for SCvx-style trajectory optimization.
    """

    mission = problem.mission
    model   = problem.model
    method  = problem.method

    # --- CTCS-specific adjustment ---
    if method.bools.get("ctcs") and method.bools.get("buff_dyn") == "term":
        method.bools["buff_dyn"] = "l1"

    # --- Constraint bookkeeping ---
    mission.n_init       = len(mission.zi_idx)
    mission.n_init_ineq  = len(mission.zi_min_idx) + len(mission.zi_max_idx)
    mission.n_term       = len(mission.zf_idx)
    mission.n_term_ineq  = len(mission.zf_min_idx) + len(mission.zf_max_idx)
    mission.n_ctrl       = len(mission.u_min_idx) + len(mission.u_max_idx)
    mission.n_state      = len(mission.z_min_idx) + len(mission.z_max_idx)
    mission.n_udot       = len(mission.udot_max)
    mission.n_path       = len(mission.path_idx)
    mission.n_nfz        = len(mission.nfz_idx)
    mission.n_aux        = len(mission.aux_idx)
    mission.n_ineq       = mission.n_path + mission.n_nfz + mission.n_aux

    # --- State vector size (ctcs mode) ---
    if method.bools.get("ctcs", False):
        model.nz = model.n + mission.n_ineq
    else:
        model.nz = model.n

    mission.n_dyn = model.nz

    buff_dyn = str(method.bools.get("buff_dyn", "term"))

    # --- Dynamics buffering ---
    if buff_dyn in {"term", "l1", "l2"}:
        method.n_plus = 0
        method.n_minus = 0
        method.Npm = 0
    elif buff_dyn == "quad-1":
        method.n_plus = 1
        method.n_minus = 1
        method.Npm = 1
    elif buff_dyn == "quad-2":
        method.n_plus = 1
        method.n_minus = 1
        method.Npm = method.N - 1
    elif buff_dyn == "quad-3":
        method.n_plus = model.nz
        method.n_minus = model.nz
        method.Npm = 1
    else:
        raise ValueError("Invalid buff_dyn flag.")

    # --- Terminal conditions nondimensionalization ---
    # Get the diagonal of the source matrix
    M_diag = np.diag(method.nondim["M"]["state"]["d2nd"])
    # Stack selected diagonals
    selected = np.concatenate([
        M_diag[mission.zf_idx],
        M_diag[mission.zf_min_idx],
        M_diag[mission.zf_max_idx]
    ])
    # Create the new diagonal matrix
    method.nondim["M"]["term"]["d2nd"] = np.diag(selected)

    # --- Default weights ---
    weights = method.weights
    weights["w_fac_N"]      = method.N
    weights["w_fac_Nm1"]    = method.N - 1
    weights["w_cost"]       = 1

    weights["dual_path"]    = np.zeros((method.N, mission.n_path))
    weights["dual_nfz"]     = np.zeros((method.N, mission.n_nfz))
    weights["dual_aux"]     = np.zeros((method.N, mission.n_aux))
    weights["dual_term"]    = np.zeros(mission.n_term + mission.n_term_ineq)
    weights["dual_dyn"]     = np.zeros((method.N - 1, mission.n_dyn))
    weights["dual_plus"]    = np.zeros((method.N - 1, mission.n_dyn))
    weights["dual_minus"]   = np.zeros((method.N - 1, mission.n_dyn))

    weights["W_path"]       = np.zeros((method.N, mission.n_path))
    weights["W_nfz"]        = np.zeros((method.N, mission.n_nfz))
    weights["W_aux"]        = np.zeros((method.N, mission.n_aux))
    weights["W_term"]       = np.zeros(mission.n_term + mission.n_term_ineq)
    weights["W_dyn"]        = np.zeros((method.N - 1, mission.n_dyn))
    weights["W_plus"]       = np.zeros((method.Npm, method.n_plus))
    weights["W_minus"]      = np.zeros((method.Npm, method.n_minus))

    # --- Convergence tolerances ---
    conv = method.conv
    conv["eps_cost"]    = 0.
    conv["eps_state"]   = 0.
    conv["eps_path"]    = 0.
    conv["eps_nfz"]     = 0.
    conv["eps_aux"]     = 0.
    conv["eps_term"]    = 0.
    conv["eps_dyn"]     = 0.
    conv["ctcs_mult_state"] = 1.0
    conv["ctcs_mult_cnst"] = 1.0
    conv["epcs_ctcs"] = 1e-5

    # --- Terminal nondimensionalization matrix ---
    M_state_vec = np.diag(method.nondim["M"]["state"]["d2nd"])
    zf_idx      = mission.zf_idx
    zf_min_idx  = mission.zf_min_idx
    zf_max_idx  = mission.zf_max_idx
    M_term_diag = np.concatenate([M_state_vec[zf_idx],
                                  M_state_vec[zf_min_idx],
                                  M_state_vec[zf_max_idx]])
    method.nondim["M"]["term"]["d2nd"] = np.diag(M_term_diag)

    # --- LTV indexing ---
    set_ltv_indices(problem)

    # --- Initialize virtual buffers ---
    conv_data = method.conv_data
    conv_data["vb_path"] = np.zeros((method.N,   mission.n_path))
    conv_data["vb_nfz"]  = np.zeros((method.N,   mission.n_nfz))
    conv_data["vb_aux"]  = np.zeros((method.N,   mission.n_aux))
    conv_data["vb_dyn"]  = np.zeros((method.N-1, model.nz))
    conv_data["vb_term"] = np.zeros(model.nz)





