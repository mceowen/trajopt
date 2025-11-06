import numpy as np

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

    path_idx = mission.path_idx
    nfz_idx = mission.nfz_idx
    aux_idx = mission.aux_idx
    
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


# Example usage
if __name__ == "__main__":
    # Define dummy data for testing
    O = {
        "weights": {
            "dual_path": None,
            "dual_nfz": None,
            "dual_aux": None,
            "dual_dyn": None,
            "dual_term": None,
            "data": {
                "dmu_ineq": None,
                "dmu_eq": None
            }
        }
    }

    problem = {
        "params": {
            'flags': {
                "stepsize_auto_dual": True
            },
            "weights": {
                "beta": 0.1,
                "gamma": 0.1
            },
            "conv": {
                "eps_path": 0.01,
                "eps_nfz": 0.01,
                "eps_aux": 0.01,
                "eps_term": 0.01,
                "eps_dyn": 0.01
            }
        }
    }

    iter_num = 10
    vb_path = np.array([0.02, 0.03])
    vb_nfz = np.array([0.02])
    vb_aux = np.array([0.02])
    vb_dyn = np.array([0.02])
    vb_term = np.array([0.02])

    dual_path = np.array([0.01, 0.01])
    dual_nfz = np.array([0.01])
    dual_aux = np.array([0.01])
    dual_dyn = np.array([0.01])
    dual_term = np.array([0.01])

    updated_O = autotune1(O, problem, iter_num, vb_path, vb_nfz, vb_aux, vb_dyn, vb_term, dual_path, dual_nfz, dual_aux, dual_dyn, dual_term)
    print(updated_O)


    # Define dummy data for testing
    O = {
        "weights": {
            "dual_path": None,
            "dual_nfz": None,
            "dual_aux": None,
            "dual_dyn": None,
            "dual_term": None,
            "data": {
                "dmu_ineq": None,
                "dmu_eq": None
            }
        }
    }

    problem = {
        "params": {
            'flags': {
                "flag_Wauto_memory": 0,
                "buff_dyn": True
            },
            "weights": {
                "eps_nonzero2": 1e-6,
                "W_path": np.zeros((2, 3)),
                "W_nfz": np.zeros((1, 3)),
                "W_aux": np.zeros((1, 3)),
                "W_dyn": np.zeros((1, 2)),
                "W_term": np.zeros((1,))
            },
            "conv": {
                "eps_path": 0.01,
                "eps_nfz": 0.01,
                "eps_aux": 0.01,
                "eps_term": 0.01,
                "eps_dyn": 0.01
            },
            "path_idx": [0, 1],
            "nfz_idx": [0],
            "aux_idx": [0]
        }
    }

    N = 3
    vb_path = np.random.rand(2, 3)
    vb_nfz = np.random.rand(1, 3)
    vb_aux = np.random.rand(1, 3)
    vb_dyn = np.random.rand(1, 2)
    vb_term = np.random.rand(1,)

    W_path = np.random.rand(2, 3)
    W_nfz = np.random.rand(1, 3)
    W_aux = np.random.rand(1, 3)
    W_dyn = np.random.rand(1, 2)
    W_term = np.random.rand(1,)

    updated_O = autotune2(O, problem, N, vb_path, vb_nfz, vb_aux, vb_dyn, vb_term, W_path, W_nfz, W_aux, W_dyn, W_term)
    print(updated_O)

    updated_O = autotune3(O, problem, N, vb_path, vb_nfz, vb_aux, vb_dyn, vb_term, W_path, W_nfz, W_aux, W_dyn, W_term)
    print(updated_O)