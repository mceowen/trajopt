import numpy as np
import cvxpy as cp

def configure_penalty_weights(problem):
    """
    Configure all scalar and matrix weights for the optimization method,
    using YAML-defined parameters under method.weights.
    """

    method  = problem.method
    mission = problem.mission

    # --- Default weights ---
    n_ineq = mission.n_path + mission.n_nfz + mission.n_custom
    method.weights["W_ineq"] = np.zeros((method.N, n_ineq))

    method.weights["W_term"]     = np.zeros(mission.n_term + mission.n_term_ineq)
    method.weights["W_dyn"]      = np.zeros((method.N - 1, mission.n_dyn))
    method.weights["W_plus"]     = np.zeros((method.Npm, method.n_plus))
    method.weights["W_minus"]    = np.zeros((method.Npm, method.n_minus))

    method.weights["dual_ineq"]  = np.zeros((method.N, n_ineq))
    method.weights["dual_term"]  = np.zeros(mission.n_term + mission.n_term_ineq)
    method.weights["dual_dyn"]   = np.zeros((method.N - 1, mission.n_dyn))
    method.weights["dual_plus"]  = np.zeros((method.N - 1, mission.n_dyn))
    method.weights["dual_minus"] = np.zeros((method.N - 1, mission.n_dyn))

    # local block arrays
    W_path      = np.zeros((method.N, mission.n_path))
    W_nfz       = np.zeros((method.N, mission.n_nfz))
    W_custom       = np.zeros((method.N, mission.n_custom))
    dual_path   = np.zeros((method.N, mission.n_path))
    dual_nfz    = np.zeros((method.N, mission.n_nfz))
    dual_custom    = np.zeros((method.N, mission.n_custom))

    # PTR penalty weights
        # Wtr: weight for trust region cost                        
        # w_term: weight for terminal constraint buffer cost
        # w_path: weight for path constraint buffer cost
        # w_nfz:  weight for nfz constraint buffer cost

    method.weights["wtr_z"] = 1 / (2 * method.weights["alpha_z"])
    method.weights["wtr_u"] = 0 if np.isinf(method.weights["alpha_u"]) else 1 / (2 * method.weights["alpha_u"])

    method.weights["w_fac_N"]      = method.N
    method.weights["w_fac_Nm1"]    = method.N - 1

    # === Autotune modes (flag_autotune ∈ {0,2,3,al-scvx}) ===
    if str(method.flags["flag_autotune"]) in {"0", "2", "3", "al-scvx"}:

        # --- Buffer weights ---
        if str(method.flags["flag_autotune"]) in {"0", "al-scvx"}:
            if str(method.flags["flag_autotune"]) == "0":

                w_path = method.weights["w_path_scale"] * method.weights["wbuff"] / method.weights["w_fac_N"]
                w_nfz  = method.weights["w_nfz_scale"]  * method.weights["wbuff"] / method.weights["w_fac_N"]
                w_custom  = method.weights["w_custom_scale"]  * method.weights["wbuff"] / method.weights["w_fac_N"]
                w_dyn  = method.weights["w_dyn_scale"]  * method.weights["wbuff"] / method.weights["w_fac_Nm1"]
                w_term = method.weights["w_term_scale"] * method.weights["wbuff"]

            else:
                w_path = method.weights["wbuff"] / method.weights["w_fac_N"]
                w_nfz  = method.weights["wbuff"] / method.weights["w_fac_N"]
                w_custom  = method.weights["wbuff"] / method.weights["w_fac_N"]
                w_dyn  = method.weights["wbuff"] / method.weights["w_fac_Nm1"]
                w_term = method.weights["wbuff"]
        else:
            method.weights["wbuff"] = 1
            w_path = method.weights["wbuff"] / method.weights["w_fac_N"]
            w_nfz  = method.weights["wbuff"] / method.weights["w_fac_N"]
            w_custom  = method.weights["wbuff"] / method.weights["w_fac_N"]
            w_dyn  = method.weights["wbuff"] / method.weights["w_fac_Nm1"]
            w_term = method.weights["wbuff"]

        W_path += w_path
        W_nfz  += w_nfz
        W_custom  += w_custom

        # Stack into the master W_ineq
        method.weights["W_ineq"] = np.hstack([W_path, W_nfz, W_custom])

        if method.flags["dynamics_nonconvex"] or method.flags["ctcs"] != "none":
            buff_dyn = str(method.flags.get("buff_dyn", ""))
            if buff_dyn in {"l1", "l2"}:
                method.weights["W_dyn"] += w_dyn
            elif buff_dyn in {"quad-1", "quad-2", "quad-3"}:
                method.weights["W_plus"] += w_dyn
                method.weights["W_minus"] += w_dyn
            else:
                method.weights["W_term"] += w_term

    # === Autotune mode: {1,3,al-scvx} ===
    if str(method.flags["flag_autotune"]) in {"1", "3", "al-scvx"}:

        dual_path += method.weights["eps_nonzero1"]
        dual_nfz  += method.weights["eps_nonzero1"]
        dual_custom  += method.weights["eps_nonzero1"]

        method.weights["dual_ineq"] = np.hstack([dual_path, dual_nfz, dual_custom])

        if method.flags["dynamics_nonconvex"] or method.flags["ctcs"] != "none":
            buff_dyn = str(method.flags.get("buff_dyn", ""))
            if buff_dyn == "term":
                method.weights["dual_term"] += method.weights["eps_nonzero1"]
            else:
                method.weights["dual_dyn"] += method.weights["eps_nonzero1"]

                if str(method.flags.get("buff_dyn_dual", "")) == "l1":
                    method.weights["dual_plus"] += method.weights["eps_nonzero1"]
                    method.weights["dual_minus"] += method.weights["eps_nonzero1"]

    ### ctcs convergence adjustments ###
    method.weights["w_ctcs"] = 10.0




# -------------- PENALTIES ----------------------------------------------------------------------------------------

def build_virtual_buffer_cost(subprob) -> cp.Expression:
    """
    Build the virtual buffer penalty term VB (stacked inequality form).
    """
    problem = subprob.problem
    idx = problem.indices
    N   = subprob.N

    VB = 0.0

    # --- Terminal term ---
    if subprob.vb_term is not None and subprob.n_term > 0:
        VB += cp.sum_squares(cp.diag(subprob.W_term_sqrt) @ subprob.vb_term)

    # --- Stacked inequality constraints ---
    if subprob.vb_ineq is not None and subprob.n_ineq > 0:
        for k in range(N):
            VB += cp.sum_squares(cp.diag(subprob.W_ineq_sqrt[k, :]) @ subprob.vb_ineq[k, :])

    # --- Dynamics buffers ---
    diff = subprob.vb_dyn_p - subprob.vb_dyn_m
    if subprob.buff_dyn == "l1":
        for k in range(N - 1):
            VB += subprob.w_dyn_row[k] * cp.norm1(diff[k])
    elif subprob.buff_dyn == "l2":
        for k in range(N - 1):
            VB += cp.sum_squares(cp.diag(subprob.W_dyn_sqrt[k, :]) @ diff[k])
    elif subprob.buff_dyn in {"quad-1", "quad-2"}:
        for k in range(max(subprob.Npm, 1)):
            if subprob.vb_plus is not None and subprob.n_plus > 0:
                VB += cp.sum_squares(cp.diag(subprob.W_plus_sqrt[k, :]) @ subprob.vb_plus[k])
            if subprob.vb_minus is not None and subprob.n_minus > 0:
                VB += cp.sum_squares(cp.diag(subprob.W_minus_sqrt[k, :]) @ subprob.vb_minus[k])

    return 0.5 * subprob.flag_vb * VB




def build_dual_buffer_cost(subprob) -> cp.Expression:
    """
    Build dual penalty term DUAL for stacked inequality constraints (unified _ineq form).
    """
    DUAL = 0.0

    # --- Unified inequality duals ---
    if subprob.vb_ineq is not None and subprob.n_ineq > 0:
        DUAL += cp.sum(cp.multiply(subprob.vb_ineq, subprob.dual_ineq))

    # --- Dynamic duals ---
    diff = subprob.vb_dyn_p - subprob.vb_dyn_m
    DUAL += cp.sum(cp.multiply(diff, subprob.dual_dyn))

    # --- Aggregate duals ---
    if subprob.vb_plus is not None and subprob.n_plus > 0:
        DUAL += cp.sum(cp.multiply(subprob.vb_plus, subprob.dual_plus))
    if subprob.vb_minus is not None and subprob.n_minus > 0:
        DUAL += cp.sum(cp.multiply(subprob.vb_minus, subprob.dual_minus))
    if subprob.vb_term is not None and subprob.n_term > 0:
        DUAL += subprob.dual_term @ subprob.vb_term

    return DUAL




# -------------- AUTOTUNING SCHEMES ----------------------------------------------------------------------------------------

def autotune1(problem, iter_record):
    """
    Unified version of autotune1 using stacked inequality form (_ineq only).
    """
    method = problem.method

    # Access iteration number
    iter_num = iter_record["iter_num"]

    # Extract variables from local_vars
    vb_ineq = np.array(iter_record["conv_data"]["vb_ineq"])
    vb_term = np.array(iter_record["conv_data"]["vb_term"])
    vb_dyn  = np.array(iter_record["conv_data"]["vb_dyn"])  # from O since not in sol_vars

    dual_ineq = iter_record["weights"]["dual_ineq"]
    dual_dyn  = iter_record["weights"]["dual_dyn"]
    dual_term = iter_record["weights"]["dual_term"]

    # Hyperparameters
    if method.flags["stepsize_auto_dual"]:
        beta = gamma = 1 / iter_num
    else:
        beta = method.weights["beta"]
        gamma = method.weights["gamma"]

    # === Updates ===
    dual_ineq_plus = np.maximum(0, gamma * vb_ineq + dual_ineq)
    dual_dyn_plus  = beta * vb_dyn + dual_dyn
    dual_term_plus = beta * vb_term + dual_term

    # Constraint feasibility thresholds
    conv = method.conv
    eps_ineq = conv.get("eps_ineq", 1e-6)
    eps_term = conv["eps_term"]
    eps_dyn  = conv["eps_dyn"]

    # Apply saturation logic
    mask_ineq = vb_ineq <= eps_ineq
    dual_ineq_plus[mask_ineq] = dual_ineq[mask_ineq]

    dual_dyn_plus[np.abs(vb_dyn) <= eps_dyn] = dual_dyn[np.abs(vb_dyn) <= eps_dyn]
    dual_term_plus[np.abs(vb_term) <= eps_term] = dual_term[np.abs(vb_term) <= eps_term]

    # Update output dictionary
    weights = iter_record["weights"]
    weights.update({
        "dual_ineq": dual_ineq_plus,
        "dual_dyn": dual_dyn_plus,
        "dual_term": dual_term_plus,
        "data": {
            "dmu_ineq": dual_ineq_plus - dual_ineq,
            "dmu_eq": dual_term_plus - dual_term
        }
    })

    return iter_record


def autotune2(problem, iter_record):
    """
    Unified stacked-inequality version of autotune2.
    """
    method = problem.method
    
    # Extract variables from local_vars
    N = method.N
    vb_ineq = np.array(iter_record["conv_data"]["vb_ineq"])
    vb_dyn  = np.array(iter_record["conv_data"]["vb_dyn"])  # Assuming vb_dyn_plus is in sol_vars
    vb_term = np.array(iter_record["conv_data"]["vb_term"])

    W_ineq = np.array(iter_record["weights"]["W_ineq"])
    W_dyn  = np.array(iter_record["weights"]["W_dyn"])
    W_term = np.array(iter_record["weights"]["W_term"])

    # Extract parameters for autotuning
    eps_feas_ineq = method.conv.get("eps_ineq", 1e-6)
    eps_feas_term = method.conv["eps_term"]
    eps_feas_dyn  = method.conv["eps_dyn"]

    eps_nonzero2 = method.weights["eps_nonzero2"]

    buff_dyn = method.flags["buff_dyn"]

    dual_ineq_buff = []
    dual_dyn_buff  = []

    Wh_ineq = np.zeros((N, method.problem.mission.n_ineq))
    Wh_dyn  = np.zeros((N, method.problem.mission.n_dyn))
    Wh_term = np.zeros(method.problem.mission.n_term + method.problem.mission.n_term_ineq)

    Wh_plus  = np.zeros((N, method.n_plus))
    Wh_minus = np.zeros((N, method.n_minus))

    # Autotune matrices via dual variables and feasibility tolerance
    for k in range(N):
        dual_ineq_buff.append(np.diag(W_ineq[k, :]) @ vb_ineq[k, :].flatten())

        if method.problem.mission.n_ineq > 0:
            Wh_ineq[k, :] = np.abs(dual_ineq_buff[-1] / eps_feas_ineq)
        else:
            Wh_ineq[k, :] = np.abs(dual_ineq_buff[-1])

        if k < N - 1:
            dual_dyn_buff.append(np.diag(W_dyn[k, :]) @ vb_dyn[k, :])
            if buff_dyn:
                Wh_dyn[k, :] = np.sum(np.abs(dual_dyn_buff[-1]) / eps_feas_dyn)
            else:
                Wh_dyn[k, :] = np.sum(np.abs(dual_dyn_buff[-1]))

    if (method.problem.mission.n_term + method.problem.mission.n_term_ineq) > 0:
        dual_term_buff = np.diag(W_term) @ vb_term
        Wh_term = np.abs(dual_term_buff / eps_feas_term).flatten()

    # Extract field names and create buffer nametags
    W_fn = [key for key in iter_record["weights"].keys() if key.startswith("W")]
    nametags = [key.split("_")[1] for key in W_fn if key.startswith("W")]

    for i_field in nametags:
        if i_field not in {"plus", "minus"}:
            W_field = f"W_{i_field}"
            Wh_field = f"Wh_{i_field}"
            vb_field = f"vb_{i_field}"
            eps_feas = f"eps_feas_{i_field}"
            Wconv_field = f"Wconv_{i_field}"

            if np.sum(method.weights[W_field]) == 0:
                iter_record["weights"][W_field] = eval(W_field)
            else:
                exec(f"{Wh_field}[{Wh_field} <= eps_nonzero2] = eps_nonzero2")

                # Create updated weight
                iter_record["weights"][W_field] = eval(Wh_field)

    # --- Store diagnostics ---
    iter_record["weights"]["data"] = {}
    iter_record["weights"]["data"]["eps_feas"] = eps_feas_ineq

    iter_record["weights"]["data"]["term"] = {
        "Wxq": np.diag(W_term) @ vb_term,
        "dual": dual_term_buff
    }

    for k in range(N):
        iter_record["weights"]["data"]["ineq"] = {
            "Wxq": np.diag(W_ineq[k, :]) @ vb_ineq[k, :],
            "dual": dual_ineq_buff[k]
        }

    iter_record["weights"]["data"]["term"]["delta"] = (
        iter_record["weights"]["data"]["term"]["Wxq"]
        - iter_record["weights"]["data"]["term"]["dual"]
    )
    iter_record["weights"]["data"]["ineq"]["delta"] = (
        iter_record["weights"]["data"]["ineq"]["Wxq"]
        - iter_record["weights"]["data"]["ineq"]["dual"]
    )

    return iter_record


def autotune3(problem, iter_record):
    iter_record = autotune1(problem, iter_record)
    iter_record = autotune2(problem, iter_record)

    return iter_record