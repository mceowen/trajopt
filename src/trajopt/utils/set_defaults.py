# Skye Mceowen and Aman Tiwary
# Feb. 14, 2024
# functions that set default problem parameters

import numpy as np
from trajopt.algorithm.discretization import set_ltv_indices

# def set_params_default(config=None):
#     # --- Initialize empty params dict ---
#     params = {}

#     # --- Default booleans ---
#     params["bools"] = {
#         "auto_jac": 1,
#         "auto_jac_aero": 1,
#         "auto_jac_cnst": 1,
#         "dev_var": 1,
#         "nondim": 0,
#         "var_scl_flag": 0,
#         "free_final_time": 0,
#         "equal_dt": 0,
#         "flag_nfz": 0,
#         "flag_autotune": 3,
#         "flag_Wauto_memory": 0,
#         "weight_zero": 0,
#         "stepsize_auto_primal": 0,
#         "stepsize_auto_dual": 0,
#         "flag_conv": 0,
#         "buff_dyn": "l2",
#         "buff_dyn_dual": "l1",
#         "ctcs": 0,
#         "ode_fixed_dt": 0,
#         "aoa_vb": 0,
#         "earth_rot": 0,
#         "init_ctrl": 1,
#         "jax_dyn": 0
#     }

#     # --- Basic structure ---
#     params["model"]["n"]             = 1
#     params["model"]["m"]             = 1

#     # --- Constraint structure ---
#     params["mission"]["path_lim"]      = np.array([])
#     params["mission"]["path_idx"]      = np.array([], dtype=np.int64)
#     params["n_path"]        = 0
#     params["mission"]["nfz_idx"]       = np.array([], dtype=np.int64)
#     params["mission"]["n_nfz"]         = 0
#     params["mission"]["aux_idx"]       = np.array([], dtype=np.int64)
#     params["mission"]["n_aux"]         = 0
#     params["mission"]["n_ineq"]        = 0
#     params["mission"]["n_eq"]          = 0

#     # --- Cost label ---
#     params["mission"]["cost_name"]     = "Cost"

#     # --- Initial boundary conditions ---
#     params["mission"]["zi"]            = np.array([])
#     params["mission"]["zi_idx"]        = np.array([], dtype=np.int64)
#     params["mission"]["zi_min"]        = np.array([])
#     params["mission"]["zi_min_idx"]    = np.array([], dtype=np.int64)
#     params["mission"]["zi_max"]        = np.array([])
#     params["mission"]["zi_max_idx"]    = np.array([], dtype=np.int64)
#     params["mission"]["n_init"]        = 0
#     params["mission"]["n_init_ineq"]   = 0

#     # --- Terminal boundary conditions ---
#     params["mission"]["zf"]            = np.array([])
#     params["mission"]["zf_idx"]        = np.array([], dtype=np.int64)
#     params["mission"]["zf_min"]        = np.array([])
#     params["mission"]["zf_min_idx"]    = np.array([], dtype=np.int64)
#     params["mission"]["zf_max"]        = np.array([])
#     params["mission"]["zf_max_idx"]    = np.array([], dtype=np.int64)
#     params["mission"]["n_term"]        = 0
#     params["mission"]["n_term_ineq"]   = 0

#     # --- State constraints ---
#     params["mission"]["z_min"]         = np.array([])
#     params["mission"]["z_min_idx"]     = np.array([], dtype=np.int64)
#     params["mission"]["z_max"]         = np.array([])
#     params["mission"]["z_max_idx"]     = np.array([], dtype=np.int64)
#     params["mission"]["n_state"]       = 0

#     # --- Control constraints ---
#     params["mission"]["u_min"]         = np.array([])
#     params["mission"]["u_min_idx"]     = np.array([], dtype=np.int64)
#     params["mission"]["u_max"]         = np.array([])
#     params["mission"]["u_max_idx"]     = np.array([], dtype=np.int64)
#     params["mission"]["udot_max"]      = np.array([])
#     params["mission"]["udot_max_idx"]  = np.array([], dtype=np.int64)
#     params["mission"]["n_ctrl"]        = 0
#     params["mission"]["n_udot"]        = 0

#     # --- Weight structure ---
#     params["method"]["weights"]       = {}
#     params["mission"]["n_dyn"]         = 0

#     # --- Convergence settings ---
#     params["method"]["conv"] = {
#         "eps_feas_conv": 1e-1,
#         "eps_conv": 1e-1,
#         "eps_cost_conv": 1e-1
#     }

#     # --- Overwrite with config ---
#     if config is not None:
#         params["mission"] = config.get("mission", None)
#         if "params" in config:
#             for key, value in config["params"].items():
#                 if key == "bools":
#                     params["bools"].update(value)
#                 else:
#                     params[key] = value

#     # --- CTCS-specific adjustment ---
#     if params["method"]["bools"].get("ctcs") and params["method"]["bools"].get("buff_dyn") == "term":
#         params["method"]["bools"]["buff_dyn"] = "l1"

#     return params


def set_params_constraint_default(params):
    """
    Set default constraint dimensions, duals, weights, and convergence criteria
    for SCvx-style trajectory optimization.
    """

    # --- Constraint bookkeeping ---
    params["mission"]["n_init"]       = len(params["mission"].get("zi_idx",       np.array([], dtype=np.int64)))
    params["mission"]["n_init_ineq"]  = len(params["mission"].get("zi_min_idx",   np.array([], dtype=np.int64))) + len(params["mission"].get("zi_max_idx", np.array([], dtype=np.int64)))
    params["mission"]["n_term"]       = len(params["mission"].get("zf_idx",       np.array([], dtype=np.int64)))
    params["mission"]["n_term_ineq"]  = len(params["mission"].get("zf_min_idx",   np.array([], dtype=np.int64))) + len(params["mission"].get("zf_max_idx", np.array([], dtype=np.int64)))
    params["mission"]["n_ctrl"]       = len(params["mission"].get("u_min_idx",    np.array([], dtype=np.int64))) + len(params["mission"].get("u_max_idx", np.array([], dtype=np.int64)))
    params["mission"]["n_state"]      = len(params["mission"].get("z_min_idx",    np.array([], dtype=np.int64))) + len(params["mission"].get("z_max_idx", np.array([], dtype=np.int64)))
    params["mission"]["n_udot"]       = len(params["mission"].get("udot_max",     np.array([], dtype=np.int64)))
    params["mission"]["n_path"]       = len(params["mission"].get("path_idx",     np.array([], dtype=np.int64)))
    params["mission"]["n_nfz"]        = len(params["mission"].get("nfz_idx",      np.array([], dtype=np.int64)))
    params["mission"]["n_aux"]        = len(params["mission"].get("aux_idx",      np.array([], dtype=np.int64)))
    params["mission"]["n_ineq"]       = params["mission"]["n_path"] + params["mission"]["n_nfz"] + params["mission"]["n_aux"]

    # --- State vector size (ctcs mode) ---
    if params["method"]["bools"].get("ctcs", False):
        params["model"]["nz"] = params["model"]["n"] + params["mission"]["n_ineq"]
    else:
        params["model"]["nz"] = params["model"]["n"]

    params["mission"]["n_dyn"] = params["model"]["nz"]

    buff_dyn = str(params["method"]["bools"].get("buff_dyn", "term"))


    # --- Dynamics buffering ---
    if buff_dyn in {"term", "l1", "l2"}:
        params["method"]["n_plus"] = 0
        params["method"]["n_minus"] = 0
        params["method"]["Npm"] = 0
    elif buff_dyn == "quad-1":
        params["method"]["n_plus"] = 1
        params["method"]["n_minus"] = 1
        params["method"]["Npm"] = 1
    elif buff_dyn == "quad-2":
        params["method"]["n_plus"] = 1
        params["method"]["n_minus"] = 1
        params["method"]["Npm"] = params["method"]["N"] - 1
    elif buff_dyn == "quad-3":
        params["method"]["n_plus"] = params["model"]["nz"]
        params["method"]["n_minus"] = params["model"]["nz"]
        params["method"]["Npm"] = 1
    else:
        raise ValueError("Invalid buff_dyn flag.")

    # --- Terminal conditions nondimensionalization ---
    # Get the diagonal of the source matrix
    M_diag = np.diag(params["method"]["nondim"]["M"]["state"]["d2nd"])
    # Stack selected diagonals
    selected = np.concatenate([
        M_diag[params["mission"]["zf_idx"]],
        M_diag[params["mission"]["zf_min_idx"]],
        M_diag[params["mission"]["zf_max_idx"]]
    ])
    # Create the new diagonal matrix
    params["method"]["nondim"]["M"]["term"]["d2nd"] = np.diag(selected)

    # --- Default weights ---
    weights = params["method"].setdefault("weights", {})
    weights["w_fac_N"]      = params["method"]["N"]
    weights["w_fac_Nm1"]    = params["method"]["N"] - 1
    weights["w_cost"]       = 1.0

    weights["dual_path"]    = np.zeros((params["method"]["N"], params["mission"]["n_path"]))
    weights["dual_nfz"]     = np.zeros((params["method"]["N"], params["mission"]["n_nfz"]))
    weights["dual_aux"]     = np.zeros((params["method"]["N"], params["mission"]["n_aux"]))
    weights["dual_term"]    = np.zeros(params["mission"]["n_term"] + params["mission"]["n_term_ineq"])
    weights["dual_dyn"]     = np.zeros((params["method"]["N"] - 1, params["mission"]["n_dyn"]))
    weights["dual_plus"]    = np.zeros((params["method"]["N"] - 1, params["mission"]["n_dyn"]))
    weights["dual_minus"]   = np.zeros((params["method"]["N"] - 1, params["mission"]["n_dyn"]))

    weights["W_path"]       = np.zeros((params["method"]["N"], params["mission"]["n_path"]))
    weights["W_nfz"]        = np.zeros((params["method"]["N"], params["mission"]["n_nfz"]))
    weights["W_aux"]        = np.zeros((params["method"]["N"], params["mission"]["n_aux"]))
    weights["W_term"]       = np.zeros(params["mission"]["n_term"] + params["mission"]["n_term_ineq"])
    weights["W_dyn"]        = np.zeros((params["method"]["N"] - 1, params["mission"]["n_dyn"]))
    weights["W_plus"]       = np.zeros((params["method"]["Npm"], params["method"]["n_plus"]))
    weights["W_minus"]      = np.zeros((params["method"]["Npm"], params["method"]["n_minus"]))

    # --- Convergence tolerances ---
    conv = params["method"].setdefault("conv", {})
    conv["eps_cost"]    = 0.
    conv["eps_state"]   = 0.
    conv["eps_path"]    = 0.
    conv["eps_nfz"]     = 0.
    conv["eps_aux"]     = 0.
    conv["eps_term"]    = 0.
    conv["eps_dyn"]     = 0.

    setup = conv.setdefault("setup", {})
    for key in ["eps_cost", "eps_state", "eps_path", "eps_nfz", "eps_aux", "eps_term", "eps_dyn"]:
        setup[key] = np.array([])

    setup["ctcs_mult_state"] = 1.0
    setup["ctcs_mult_cnst"] = 1.0

    params["eps_ctcs"] = 1e-5

    # --- Terminal nondimensionalization matrix ---
    M_state_vec = np.diag(params["method"]["nondim"]["M"]["state"]["d2nd"])
    zf_idx      = params["mission"].get("zf_idx", np.array([], dtype=np.int64))
    zf_min_idx  = params["mission"].get("zf_min_idx", np.array([], dtype=np.int64))
    zf_max_idx  = params["mission"].get("zf_max_idx", np.array([], dtype=np.int64))
    M_term_diag = np.concatenate([M_state_vec[zf_idx],
                                  M_state_vec[zf_min_idx],
                                  M_state_vec[zf_max_idx]])
    params["method"]["nondim"]["M"]["term"]["d2nd"] = np.diag(M_term_diag)

    # --- LTV indexing ---
    params = set_ltv_indices(params)

    # --- Initialize virtual buffers ---
    conv_data = params["method"].setdefault("conv_data", {})
    conv_data["vb_path"] = np.zeros((params["method"]["N"],   params["mission"]["n_path"]))
    conv_data["vb_nfz"]  = np.zeros((params["method"]["N"],   params["mission"]["n_nfz"]))
    conv_data["vb_aux"]  = np.zeros((params["method"]["N"],   params["mission"]["n_aux"]))
    conv_data["vb_dyn"]  = np.zeros((params["method"]["N"]-1, params["model"]["nz"]))
    conv_data["vb_term"] = np.zeros(params["model"]["nz"])

    return params

def set_problem_default(problem):
    """
    Populate default state/control bounds and boundary conditions
    from problem.params into top-level problem fields.

    Parameters:
        problem : dict with nested "params" dict

    Returns:
        problem : updated dict
    """
    params = problem["params"]

    #
    # BOUNDARY CONDITIONS
    #
    # Initial conditions
    problem["zi"]           = params["mission"]["zi"]
    problem["zi_idx"]       = params["mission"]["zi_idx"]
    problem["zi_min"]       = params["mission"]["zi_min"]
    problem["zi_min_idx"]   = params["mission"]["zi_min_idx"]
    problem["zi_max"]       = params["mission"]["zi_max"]
    problem["zi_max_idx"]   = params["mission"]["zi_max_idx"]
    problem["n_init"]       = params["mission"]["n_init"]
    problem["n_init_ineq"]  = params["mission"]["n_init_ineq"]

    # Terminal conditions
    problem["zf"]           = params["mission"]["zf"]
    problem["zf_idx"]       = params["mission"]["zf_idx"]
    problem["zf_min"]       = params["mission"]["zf_min"]
    problem["zf_min_idx"]   = params["mission"]["zf_min_idx"]
    problem["zf_max"]       = params["mission"]["zf_max"]
    problem["zf_max_idx"]   = params["mission"]["zf_max_idx"]
    problem["n_term"]       = params["mission"]["n_term"]
    problem["n_term_ineq"]  = params["mission"]["n_term_ineq"]

    #
    # STATE CONSTRAINTS
    #
    problem["z_min"]        = params["mission"]["z_min"]
    problem["z_min_idx"]    = params["mission"]["z_min_idx"]
    problem["z_max"]        = params["mission"]["z_max"]
    problem["z_max_idx"]    = params["mission"]["z_max_idx"]

    #
    # CONTROL CONSTRAINTS
    #
    problem["u_min"]        = params["mission"]["u_min"]
    problem["u_min_idx"]    = params["mission"]["u_min_idx"]
    problem["u_max"]        = params["mission"]["u_max"]
    problem["u_max_idx"]    = params["mission"]["u_max_idx"]
    problem["n_ctrl"]       = params["mission"]["n_ctrl"]
    problem["udot_max"]     = params["mission"]["udot_max"]
    problem["udot_max_idx"] = params["mission"]["udot_max_idx"]
    problem["n_udot"]       = params["mission"]["n_udot"]

    return problem


# testing functions
if __name__ == "__main__":
    print("..:: Testing set_params_default ::..")

    # call without config
    print("no config argument passed")
    params = set_params_default()
    tf = "N" in params
    print("N in params?: ", tf)
    print("params['bools']['flag_nfz'] = ", params["mission"]["bools"]["flag_nfz"])

    # test config params overwriting
    print("calling again with config argument")
    # make dummy config w/ data to overwrite params defaults
    config = {
        "params": { # config["params"]
            "N": 40,
            "T_init": 10,
            "bools": { # config["params"]["bools"]
                "flag_nfz": 2,
                "flag_autotune": 0,
                "free_final_time": 1,
                "buff_dyn": 0,
                "ctcs": 0
            },
        },
    }
    params = set_params_default(config)
    print("params['mission']['bools']['flag_nfz'] = ", params["mission"]["bools"]["flag_nfz"])
    print("params['method']['N'] = ", params["method"]["N"])

    # need to set nondim params before passing into set_params_constraint_defualt
    from trajopt.problem_models.quadrotor_3dof import set_nondim_params
    params = set_nondim_params(params)
    tf = "nondim" in params # true/false
    print("nondim in params?: ", tf)
    # print("params["nondim"] = ", params["nondim"])

    # check if conv_data exists (it shouldn"t yet)
    tf = "conv_data" in params # true/false
    print("conv_data in params?: ", tf)

    # Now call set_params_constraint_default with params
    print("..:: Calling set_params_constraint_default ::..")
    params = set_params_constraint_default(params)
    
    # check if conv_data exists again (it should)
    tf = "conv_data" in params # true/false
    print("conv_data in params?: ", tf)





