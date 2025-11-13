import numpy as np

def configure_penalty_weights(problem):
    """
    Configure all scalar and matrix weights for the optimization method,
    using YAML-defined parameters under method.weights.
    """

    method  = problem.method
    mission = problem.mission

    # --- Default weights ---
    method.weights["dual_path"]    = np.zeros((method.N, mission.n_path))
    method.weights["dual_nfz"]     = np.zeros((method.N, mission.n_nfz))
    method.weights["dual_aux"]     = np.zeros((method.N, mission.n_aux))
    method.weights["dual_term"]    = np.zeros(mission.n_term + mission.n_term_ineq)
    method.weights["dual_dyn"]     = np.zeros((method.N - 1, mission.n_dyn))
    method.weights["dual_plus"]    = np.zeros((method.N - 1, mission.n_dyn))
    method.weights["dual_minus"]   = np.zeros((method.N - 1, mission.n_dyn))

    method.weights["W_path"]       = np.zeros((method.N, mission.n_path))
    method.weights["W_nfz"]        = np.zeros((method.N, mission.n_nfz))
    method.weights["W_aux"]        = np.zeros((method.N, mission.n_aux))
    method.weights["W_term"]       = np.zeros(mission.n_term + mission.n_term_ineq)
    method.weights["W_dyn"]        = np.zeros((method.N - 1, mission.n_dyn))
    method.weights["W_plus"]       = np.zeros((method.Npm, method.n_plus))
    method.weights["W_minus"]      = np.zeros((method.Npm, method.n_minus))

    # PTR penalty weights
        # Wtr: weight for trust region cost                        
        # w_term: weight for terminal constraint buffer cost
        # w_path: weight for path constraint buffer cost
        # w_nfz: weight for nfz constraint buffer cost


    # TODO (carlos): this is a temporary fix to keep quadrotor converging the same, will remove soon!
    if method.flags["match_dim_nondim_weights"]:
        M_state  = method.nondim["M"]["state"]["nd2d"]
        avg_state_nd_sq = np.mean(np.diag(M_state)**2)
    else:
        avg_state_nd_sq = 1

    method.weights["wtr_z"] = avg_state_nd_sq  * 1 / (2 * method.weights["alpha_z"])
    method.weights["wtr_u"] = 0 if np.isinf(method.weights["alpha_u"]) else 1 / (2 * method.weights["alpha_u"])

    method.weights["w_fac_N"]      = method.N
    method.weights["w_fac_Nm1"]    = method.N - 1

    # === Autotune modes (flag_autotune ∈ {0,2,3,al-scvx}) ===
    if str(method.flags["flag_autotune"]) in {"0", "2", "3", "al-scvx"}:

        # --- Buffer weights ---
        if str(method.flags["flag_autotune"]) in {"0", "al-scvx"}:
            if str(method.flags["flag_autotune"]) == "0":

                w_nfz_dim  = method.weights["w_nfz_scale"] * method.weights["wbuff"] / method.weights["w_fac_N"]
                w_dyn_dim  = method.weights["w_dyn_scale"] * method.weights["wbuff"] / method.weights["w_fac_Nm1"]
                w_term_dim = method.weights["w_term_scale"] * method.weights["wbuff"]

                # TODO (carlos): this is a temporary fix to keep quadrotor converging the same
                # will remove soon!
                if method.flags["match_dim_nondim_weights"]:
                    # scaled nondim weights to approximately preserve relative scaling between cost terms
                    M_nfz  = method.nondim["M"]["nfz"]["nd2d"]
                    M_dyn  = method.nondim["M"]["dyn"]["nd2d"]
                    M_term = method.nondim["M"]["term"]["nd2d"]

                    avg_nfz_nd_sq  = np.mean(np.diag(M_nfz)**2)
                    avg_dyn_nd_sq  = np.mean(np.diag(M_dyn)**2)
                    avg_term_nd_sq = np.mean(np.diag(M_term)**2)
                else:
                    avg_nfz_nd_sq  = 1
                    avg_dyn_nd_sq  = 1
                    avg_term_nd_sq = 1

                w_nfz   = avg_nfz_nd_sq  * w_nfz_dim
                w_dyn   = avg_dyn_nd_sq  * w_dyn_dim
                w_term  = avg_term_nd_sq * w_term_dim
            else:
                w_nfz = method.weights["wbuff"] / method.weights["w_fac_N"]
                w_dyn = method.weights["wbuff"] / method.weights["w_fac_Nm1"]
                w_term = method.weights["wbuff"]

        method.weights["W_nfz"] += w_nfz

        if method.flags["dynamics_nonconvex"] or method.flags["ctcs"]:
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

        method.weights["dual_nfz"] += method.weights["eps_nonzero1"]

        if method.flags["dynamics_nonconvex"] or method.flags["ctcs"]:
            buff_dyn = str(method.flags.get("buff_dyn", ""))
            if buff_dyn == "term":
                method.weights["dual_term"] += method.weights["eps_nonzero1"]
            else:
                method.weights["dual_dyn"] += method.weights["eps_nonzero1"]

                if str(method.flags.get("buff_dyn_dual", "")) == "l1":
                    method.weights["dual_plus"] += method.weights["eps_nonzero1"]
                    method.weights["dual_minus"] += method.weights["eps_nonzero1"]

    ### ctcs convergence adjustments ###
    # TODO: will probably need to change this weight later (shouldn't be tied to nondim["nd"])
    method.weights["w_ctcs"] = method.nondim["nd"]**2


# -------------- PENALTIES ----------------------------------------------------------------------------------------

def build_virtual_buffer_cost(subprob) -> cp.Expression:
    """
    Build the virtual buffer penalty term VB for a given Subproblem instance.
    Encapsulates all flag/buff_dyn logic outside of scp.py for clarity.
    """
    problem = subprob.problem
    mission, model, method = problem.mission, problem.model, problem.method

    VB = 0.0
    N = subprob.N

    # ----- Terminal term -----
    if subprob.vb_term is not None and subprob.n_term > 0:
        VB += cp.sum_squares(cp.diag(subprob.W_term_sqrt) @ subprob.vb_term)

    # ----- Path / NFZ / AUX buffers -----
    if subprob.vb_ineq is not None:
        if subprob.n_path > 0:
            for k in range(N):
                VB += cp.sum_squares(cp.diag(subprob.W_path_sqrt[k, :])
                                     @ subprob.vb_ineq[k, mission.path_idx])
        if subprob.n_nfz > 0:
            for k in range(N):
                VB += cp.sum_squares(cp.diag(subprob.W_nfz_sqrt[k, :])
                                     @ subprob.vb_ineq[k, mission.nfz_idx])
        if subprob.n_aux > 0:
            for k in range(N):
                VB += cp.sum_squares(cp.diag(subprob.W_aux_sqrt[k, :])
                                     @ subprob.vb_ineq[k, mission.aux_idx])

    # ----- Dynamics buffers -----
    diff = subprob.vb_dyn_p - subprob.vb_dyn_m
    if subprob.buff_dyn == "l1":
        for k in range(N - 1):
            VB += subprob.w_dyn_row[k] * cp.norm1(diff[k])
    elif subprob.buff_dyn == "l2":
        for k in range(N - 1):
            VB += cp.sum_squares(cp.diag(subprob.W_dyn_sqrt[k, :]) @ diff[k])
    elif subprob.buff_dyn in {"quad-1", "quad-2"}:
        if subprob.vb_plus is not None and subprob.n_plus > 0:
            for k in range(max(subprob.Npm, 1)):
                VB += cp.sum_squares(cp.diag(subprob.W_plus_sqrt[k, :])
                                     @ subprob.vb_plus[k])
        if subprob.vb_minus is not None and subprob.n_minus > 0:
            for k in range(max(subprob.Npm, 1)):
                VB += cp.sum_squares(cp.diag(subprob.W_minus_sqrt[k, :])
                                     @ subprob.vb_minus[k])

    return 0.5 * subprob.flag_vb * VB




def build_dual_buffer_cost(subprob) -> cp.Expression:
    """
    Build dual penalty term DUAL for a given Subproblem instance.
    """
    problem = subprob.problem
    mission = problem.mission

    DUAL = 0.0

    if subprob.vb_ineq is not None and subprob.n_path > 0:
        DUAL += cp.sum(cp.multiply(subprob.vb_ineq[:, mission.path_idx],
                                   subprob.dual_path))
    if subprob.vb_ineq is not None and subprob.n_nfz > 0:
        DUAL += cp.sum(cp.multiply(subprob.vb_ineq[:, mission.nfz_idx],
                                   subprob.dual_nfz))
    if subprob.vb_ineq is not None and subprob.n_aux > 0:
        DUAL += cp.sum(cp.multiply(subprob.vb_ineq[:, mission.aux_idx],
                                   subprob.dual_aux))

    diff = subprob.vb_dyn_p - subprob.vb_dyn_m
    DUAL += cp.sum(cp.multiply(diff, subprob.dual_dyn))

    if subprob.vb_plus is not None and subprob.n_plus > 0:
        DUAL += cp.sum(cp.multiply(subprob.vb_plus, subprob.dual_plus))
    if subprob.vb_minus is not None and subprob.n_minus > 0:
        DUAL += cp.sum(cp.multiply(subprob.vb_minus, subprob.dual_minus))
    if subprob.vb_term is not None and subprob.n_term > 0:
        DUAL += subprob.dual_term @ subprob.vb_term

    return DUAL


# -------------- AUTOTUNING SCHEMES ----------------------------------------------------------------------------------------

def autotune1(problem, local_vars, O):
    
    mission = problem.mission
    method = problem.method
    
    # Access iter_num from local_vars
    iter_num = local_vars["iter_num"]

    # Extract variables from local_vars dict
    sol_vars = local_vars["sol_vars"]
    vb_path = np.array(sol_vars["vb_path"])
    vb_nfz = np.array(sol_vars["vb_nfz"])
    vb_aux = np.array(sol_vars["vb_aux"])
    vb_term = np.array(sol_vars["vb_term"])
    vb_dyn = np.array(O["conv_data"]["vb_dyn"])  # From O since not in sol_vars

    dual_path = local_vars["dual_path"]
    dual_nfz = local_vars["dual_nfz"]
    dual_aux = local_vars["dual_aux"]
    dual_dyn = local_vars["dual_dyn"]
    dual_term = local_vars["dual_term"]

    # Hyperparameters
    if method.flags["stepsize_auto_dual"]:
        beta = gamma = 1 / iter_num
    else:
        beta = method.weights["beta"]
        gamma = method.weights["gamma"]

    # Inequality updates
    dual_path_plus = np.maximum(0, gamma * vb_path + dual_path)
    dual_nfz_plus = np.maximum(0, gamma * vb_nfz + dual_nfz)
    dual_aux_plus = np.maximum(0, gamma * vb_aux + dual_aux)
    
    # Equality updates
    dual_dyn_plus = beta * vb_dyn + dual_dyn
    dual_term_plus = beta * vb_term + dual_term

    # Constraint feasibility thresholds
    conv = method.conv
    eps_path = conv["eps_path"]
    eps_nfz = conv["eps_nfz"]
    eps_aux = conv["eps_aux"]
    eps_term = conv["eps_term"]
    eps_dyn = conv["eps_dyn"]

    # Apply saturation logic
    for var, eps in zip([vb_path, vb_nfz, vb_aux], [eps_path, eps_nfz, eps_aux]):
        mask = var <= eps
        if var is vb_path: dual_path_plus[mask] = dual_path[mask]
        if var is vb_nfz: dual_nfz_plus[mask] = dual_nfz[mask] 
        if var is vb_aux: dual_aux_plus[mask] = dual_aux[mask]

    dual_dyn_plus[np.abs(vb_dyn) <= eps_dyn] = dual_dyn[np.abs(vb_dyn) <= eps_dyn]
    dual_term_plus[np.abs(vb_term) <= eps_term] = dual_term[np.abs(vb_term) <= eps_term]

    # Update output dictionary
    weights = O["method"]["weights"]
    weights.update({
        "dual_path": dual_path_plus,
        "dual_nfz": dual_nfz_plus,
        "dual_aux": dual_aux_plus,
        "dual_dyn": dual_dyn_plus,
        "dual_term": dual_term_plus,
        "data": {
            "dmu_ineq": np.concatenate([dual_path_plus, dual_nfz_plus, dual_aux_plus]) - 
                       np.concatenate([dual_path, dual_nfz, dual_aux]),
            "dmu_eq": dual_term_plus - dual_term
        }
    })

    return O


def autotune2(problem, local_vars, O):

    mission = problem.mission
    method = problem.method
    
    # Extract variables from local_vars
    N = local_vars["N"]
    sol_vars = local_vars["sol_vars"]
    vb_path = np.array(sol_vars["vb_path"])
    vb_nfz = np.array(sol_vars["vb_nfz"])
    vb_aux = np.array(sol_vars["vb_aux"])
    vb_dyn = np.array(sol_vars["vb_dyn_plus"])  # Assuming vb_dyn_plus is in sol_vars
    vb_term = np.array(sol_vars["vb_term"])

    W_path = local_vars["W_path"]
    W_nfz = local_vars["W_nfz"]
    W_aux = local_vars["W_aux"]
    W_dyn = local_vars["W_dyn"]
    W_term = local_vars["W_term"]

    # Extract parameters for autotuning
    eps_feas_path = method.conv["eps_path"]
    eps_feas_nfz = method.conv["eps_nfz"]
    eps_feas_aux = method.conv["eps_aux"]
    eps_feas_term = method.conv["eps_term"]
    eps_feas_dyn = method.conv["eps_dyn"]

    eps_nonzero2 = method.weights["eps_nonzero2"]
    flag_Wmemory = method['flags']["flag_Wauto_memory"]

    buff_dyn = method['flags']["buff_dyn"]
    
    dual_ineq = []
    dual_path_buff = []
    dual_nfz_buff = []
    dual_aux_buff = []
    dual_dyn_buff = []
    
    Wh_path = []
    Wh_nfz = []
    Wh_aux = []
    Wh_dyn = []
    
    Wh_term = []

    # Autotune matrices via dual variables and feasibility tolerance
    for k in range(N):
        dual_path_buff.append(np.diag(W_path[:, k]) @ vb_path[:, k].flatten())
        dual_nfz_buff.append(np.diag(W_nfz[:, k]) @ vb_nfz[:, k].flatten())
        dual_aux_buff.append(np.diag(W_aux[:, k]) @ vb_aux[:, k].flatten())

        if mission.n_ineq > 0:
            if mission.n_path > 0:
                Wh_path.append(np.abs(dual_path_buff[-1] / eps_feas_path))
            if mission.n_nfz > 0:
                Wh_nfz.append(np.abs(dual_nfz_buff[-1] / eps_feas_nfz))
            if mission.n_aux > 0:
                Wh_aux.append(np.abs(dual_aux_buff[-1] / eps_feas_aux))
        else:
            Wh_path.append(np.abs(dual_path_buff[-1]))
            Wh_nfz.append(np.abs(dual_nfz_buff[-1]))
            Wh_aux.append(np.abs(dual_aux_buff[-1]))

        if k < N - 1:
            dual_dyn_buff.append(np.diag(W_dyn[:, k].flatten()) @ vb_dyn[:, k])
            if buff_dyn:
                Wh_dyn.append(np.sum(np.abs(dual_dyn_buff[-1]) / eps_feas_dyn))
            else:
                Wh_dyn.append(np.sum(np.abs(dual_dyn_buff[-1])))

    if (mission.n_term + mission.n_term_ineq) > 0:
        dual_term_buff = np.diag(W_term.flatten()) @ vb_term
        Wh_term = np.abs(dual_term_buff / eps_feas_term)

    # Extract field names and create buffer nametags
    W_fn = [key for key in method.weights.keys() if key.startswith("W")]
    nametags = [key.split("_")[1] for key in W_fn if key.startswith("W")]

    for i_field in nametags:
        W_field = f"W_{i_field}"
        Wh_field = f"Wh_{i_field}"
        vb_field = f"vb_{i_field}"
        eps_feas = f"eps_feas_{i_field}"
        Wconv_field = f"Wconv_{i_field}"

        if np.sum(method.weights[W_field]) == 0:
            O["method"]["weights"][W_field] = eval(W_field)
        else:
            if flag_Wmemory == 0:
                # Remove nonzero elements from new candidate weight (derived from dual)
                exec(f"{Wh_field}[{Wh_field} <= eps_nonzero2] = eps_nonzero2")
            elif flag_Wmemory == 1:
                # Stop updating weight after desired threshold
                exec(f"eps_feas = {eps_feas}")
                exec(f"Wconv_field = problem['params']['conv'][{Wconv_field}]")
                exec(f"idx_feas_thresh = (Wconv_field @ value({vb_field}) <= eps_feas)")
                exec(f"{Wh_field}[idx_feas_thresh] = {W_field}[idx_feas_thresh]")
            elif flag_Wmemory == 2:
                exec(f"eps_feas = {eps_feas}")
                exec(f"Wconv_field = problem['params']['conv'][{Wconv_field}]")
                exec(f"idx_feas_thresh = (Wconv_field @ value({vb_field}) <= eps_feas)")
                exec(f"{Wh_field}[idx_feas_thresh] = np.minimum(eps_nonzero2, {W_field}[idx_feas_thresh])")

            # Create updated weight
            O["method"]["weights"][W_field] = eval(Wh_field)

    # TODO - clean me
    O["method"]["weights"]["data"]["eps_feas"] = eps_feas_path

    # CHECKS
    O["method"]["weights"]["data"] = {}

    O["method"]["weights"]["data"]["term"] = {
        "Wxq": np.diag(W_term.flatten()) @ vb_term,
        "dual": dual_term_buff
    }

    for k in range(N):
        O["method"]["weights"]["data"]["path"] = {
            "Wxq": np.diag(W_path[:, k].flatten()) @ vb_path[:, k],
            "dual": dual_path_buff[k]
        }
        O["method"]["weights"]["data"]["nfz"] = {
            "Wxq": np.diag(W_nfz[:, k].flatten()) @ vb_nfz[:, k],
            "dual": dual_nfz_buff[k]
        }
        O["method"]["weights"]["data"]["aux"] = {
            "Wxq": np.diag(W_aux[:, k].flatten()) @ vb_aux[:, k],
            "dual": dual_aux_buff[k]
        }
        # if k < N - 1:
        #     O["method"]["weights"]["data"]["dyn"] = {
        #         "Wxq": np.diag(W_dyn[:, k].flatten()) @ vb_dyn[:, k],
        #         "dual": dual_dyn_buff[k]
        #     }

    O["method"]["weights"]["data"]["term"]["delta"] = O["method"]["weights"]["data"]["term"]["Wxq"] - O["method"]["weights"]["data"]["term"]["dual"]
    O["method"]["weights"]["data"]["path"]["delta"] = O["method"]["weights"]["data"]["path"]["Wxq"] - O["method"]["weights"]["data"]["path"]["dual"]
    O["method"]["weights"]["data"]["nfz"]["delta"] = O["method"]["weights"]["data"]["nfz"]["Wxq"] - O["method"]["weights"]["data"]["nfz"]["dual"]
    O["method"]["weights"]["data"]["aux"]["delta"] = O["method"]["weights"]["data"]["aux"]["Wxq"] - O["method"]["weights"]["data"]["aux"]["dual"]
    # O["method"]["weights"]["data"]["dyn"]["delta"] = O["method"]["weights"]["data"]["dyn"]["Wxq"] - O["method"]["weights"]["data"]["dyn"]["dual"]

    return O


def autotune3(problem, local_vars, O):
    O = autotune1(problem, local_vars, O)
    O = autotune2(problem, local_vars, O)

    return O