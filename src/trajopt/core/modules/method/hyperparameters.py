import numpy as np
import cvxpy as cp

def configure_penalty_weights(problem):
    """
    Configure all scalar and matrix weights for the optimization method,
    using YAML-defined parameters under method.weights.
    """

    mission, model, method  = problem.mission, problem.model, problem.method
    indices = problem.indices

    # --- Default weights ---
    n_ineq = mission.n_path + mission.n_nfz + mission.n_custom
    method.weights["W_ineq"] = np.zeros((method.N, n_ineq))

    method.weights["W_term"]     = np.zeros(mission.n_term + mission.n_term_ineq + mission.n_term_ctcs)
    method.weights["W_dyn"]      = np.zeros((method.N - 1, model.n_dyn))
    method.weights["W_plus_real"]     = np.zeros((method.Npm_real, method.n_plus_real))
    method.weights["W_minus_real"]    = np.zeros((method.Npm_real, method.n_minus_real))
    method.weights["W_plus_ctcs"]     = np.zeros((method.Npm_ctcs, method.n_plus_ctcs))
    method.weights["W_minus_ctcs"]    = np.zeros((method.Npm_ctcs, method.n_minus_ctcs))

    method.weights["dual_ineq"]  = np.zeros((method.N, n_ineq))
    method.weights["dual_term"]  = np.zeros(mission.n_term + mission.n_term_ineq + mission.n_term_ctcs)
    method.weights["dual_dyn"]   = np.zeros((method.N - 1, model.n_dyn))

    method.weights["dual_plus_real"]  = np.zeros((method.Npm_real, method.n_plus_real))
    method.weights["dual_minus_real"] = np.zeros((method.Npm_real, method.n_minus_real))
    method.weights["dual_plus_ctcs"]  = np.zeros((method.Npm_ctcs, method.n_plus_ctcs))
    method.weights["dual_minus_ctcs"] = np.zeros((method.Npm_ctcs, method.n_minus_ctcs))

    # local block arrays
    W_path          = np.zeros((method.N, mission.n_path))
    W_nfz           = np.zeros((method.N, mission.n_nfz))
    W_custom        = np.zeros((method.N, mission.n_custom))
    dual_path       = np.zeros((method.N, mission.n_path))
    dual_nfz        = np.zeros((method.N, mission.n_nfz))
    dual_custom     = np.zeros((method.N, mission.n_custom))

    # PTR penalty weights
        # Wtr: weight for trust region cost                        
        # w_term: weight for terminal constraint buffer cost
        # w_path: weight for path constraint buffer cost
        # w_nfz:  weight for nfz constraint buffer cost

    method.weights["w_fac_N"]      = method.N
    method.weights["w_fac_Nm1"]    = method.N - 1

    method.weights["wtr_z"] = 1 / (2 * method.weights["alpha_z"])
    method.weights["wtr_u"] = 0 if np.isinf(method.weights["alpha_u"]) else 1 / (2 * method.weights["alpha_u"])

    # === Autotune modes (flag_autotune ∈ {0,2,3,al-scvx}) ===
    if str(method.flags["flag_autotune"]) in {"0", "2", "3", "al-scvx"}:

        # --- Buffer weights ---
        if str(method.flags["flag_autotune"]) in {"0", "al-scvx"}:
            if str(method.flags["flag_autotune"]) == "0":

                w_path      = method.weights["w_path_scale"] * method.weights["wbuff"] / method.weights["w_fac_N"]
                w_nfz       = method.weights["w_nfz_scale"]  * method.weights["wbuff"] / method.weights["w_fac_N"]
                w_custom    = method.weights["w_custom_scale"]  * method.weights["wbuff"] / method.weights["w_fac_N"]
                w_dyn       = method.weights["w_dyn_scale"]  * method.weights["wbuff"] / method.weights["w_fac_Nm1"]
                w_term      = method.weights["w_term_scale"] * method.weights["wbuff"]

            else:
                w_path      = method.weights["wbuff"] / method.weights["w_fac_N"]
                w_nfz       = method.weights["wbuff"] / method.weights["w_fac_N"]
                w_custom    = method.weights["wbuff"] / method.weights["w_fac_N"]
                w_dyn       = method.weights["wbuff"] / method.weights["w_fac_Nm1"]
                w_term      = method.weights["wbuff"]
        else:
            method.weights["wbuff"] = 1
            w_path          = method.weights["wbuff"] / method.weights["w_fac_N"]
            w_nfz           = method.weights["wbuff"] / method.weights["w_fac_N"]
            w_custom        = method.weights["wbuff"] / method.weights["w_fac_N"]
            w_dyn           = method.weights["wbuff"] / method.weights["w_fac_Nm1"]
            w_term          = method.weights["wbuff"]

        W_path += w_path
        W_nfz  += w_nfz
        W_custom  += w_custom

        # Stack into the master W_ineq
        method.weights["W_ineq"] = np.hstack([W_path, W_nfz, W_custom])

        if method.flags["dynamics_nonconvex"] or method.flags["ctcs"] != "none":

            z_state_idx = indices.z["state"]
            z_ctcs_idx = indices.z["ctcs"]
            term_idx  = indices.constraints.terminal 
            
            # real dynamics portion weights
            if method.flags["buff_dyn"] in {"l1", "l2"}:
                method.weights["W_dyn"][:, z_state_idx] += w_dyn

            elif method.flags["buff_dyn"] in {"quad-1", "quad-2", "quad-3"}:
                method.weights["W_plus_real"] += w_dyn
                method.weights["W_minus_real"] += w_dyn

            else:
                if len(term_idx["eq"]) > 0:
                    method.weights["W_term"][term_idx["eq"]] += w_term

                if len(term_idx["ineq"]) > 0:
                    method.weights["W_term"][term_idx["ineq"]] += w_term

            # ctcs portion weights
            if method.flags["ctcs"] in {"l1", "l2"}:
                method.weights["W_dyn"][:, z_ctcs_idx] += w_dyn
    
            elif method.flags["ctcs"] in {"quad-1", "quad-2", "quad-3"}:
                method.weights["W_plus_ctcs"]+= w_dyn
                method.weights["W_minus_ctcs"] += w_dyn
            
            else:
                method.weights["W_term"][term_idx["ctcs"]] += w_term


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
    method.weights["w_ctcs"] = 1.5


# -------------- PENALTIES ----------------------------------------------------------------------------------------

def build_virtual_buffer_cost(subprob) -> cp.Expression:
    """
    Virtual-buffer cost VB with SEPARATE buffering logic for:
        • real dynamics  (buff_dyn flag)
        • CTCS dynamics  (ctcs flag)
    """
    problem = subprob.problem
    method  = problem.method
    model   = problem.model

    N      = subprob.N
    n      = model.n
    n_ctcs = model.n_ctcs

    VB = 0.0

    # ------------------------------------------------------------
    # TERMINAL TERM (REAL + CTCS)
    # ------------------------------------------------------------
    if subprob.vb_term is not None and subprob.vb_term.size > 0:
        VB += cp.sum_squares(cp.diag(subprob.W_term_sqrt) @ subprob.vb_term)

    # ------------------------------------------------------------
    # STACKED NONLINEAR INEQUALITY BUFFERS
    # ------------------------------------------------------------
    if subprob.vb_ineq is not None and subprob.n_ineq > 0:
        for k in range(N):
            VB += cp.sum_squares(
                cp.diag(subprob.W_ineq_sqrt[k, :]) @ subprob.vb_ineq[k, :]
            )

    # ============================================================
    # REAL DYNAMICS BUFFERING   (first n states)
    # ============================================================
    mode_real = method.flags["buff_dyn"]   # {"none","term","l1","l2","quad-1","quad-2"}

    if subprob.vb_dyn_real_p is not None and n > 0:
        diff_real = subprob.vb_dyn_p[:, :n] - subprob.vb_dyn_m[:, :n]

        # --------------------------------------------------------
        # L1 penalty
        # --------------------------------------------------------
        if mode_real == "l1":
            for k in range(N - 1):
                VB += subprob.w_dyn_row[k] * cp.norm1(diff_real[k, :])

        # --------------------------------------------------------
        # L2 penalty
        # --------------------------------------------------------
        elif mode_real == "l2":
            for k in range(N - 1):
                VB += cp.sum_squares(
                    cp.diag(subprob.W_dyn_sqrt[k, :n]) @ diff_real[k, :] # TODO(Skye/Carlos): use indices here
                )

        # --------------------------------------------------------
        # QUAD-1 or QUAD-3: k = 1 only
        # --------------------------------------------------------
        elif mode_real == "quad-1" or mode_real == "quad-3":
            if subprob.vb_plus_real is not None:
                VB += cp.sum_squares(
                    cp.diag(subprob.W_plus_real_sqrt[0, :]) @ subprob.vb_plus_real[0, :]
                )
            if subprob.vb_minus_real is not None:
                VB += cp.sum_squares(
                    cp.diag(subprob.W_minus_real_sqrt[0, :]) @ subprob.vb_minus_real[0, :]
                )

        # --------------------------------------------------------
        # QUAD-2 : Per-time-step quadratic penalties
        # --------------------------------------------------------
        elif mode_real == "quad-2":
            if subprob.vb_plus_real is not None:
                for k in range(subprob.Npm_real):
                    VB += cp.sum_squares(
                        cp.diag(subprob.W_plus_real_sqrt[k, :]) @ subprob.vb_plus_real[k, :]
                    )
            if subprob.vb_minus_real is not None:
                for k in range(subprob.Npm_real):
                    VB += cp.sum_squares(
                        cp.diag(subprob.W_minus_real_sqrt[k, :]) @ subprob.vb_minus_real[k, :]
                    )

    # ============================================================
    # CTCS DYNAMICS BUFFERING   (last n_ctcs states)
    # ============================================================
    mode_ctcs = method.flags["ctcs"]       # {"none","term","l1","l2","quad-1","quad-2"}

    if subprob.vb_dyn_p is not None and n_ctcs > 0:
        diff_ctcs = subprob.vb_dyn_p[:, n:] - subprob.vb_dyn_m[:, n:]

        # --------------------------------------------------------
        # L1 penalty
        # --------------------------------------------------------
        if mode_ctcs == "l1":
            for k in range(N - 1):
                VB += subprob.w_dyn_row[k] * cp.norm1(diff_ctcs[k, :])

        # --------------------------------------------------------
        # L2 penalty
        # --------------------------------------------------------
        elif mode_ctcs == "l2":
            for k in range(N - 1):
                VB += cp.sum_squares(
                    cp.diag(subprob.W_dyn_sqrt[k, n:n+n_ctcs]) @ diff_ctcs[k, :] # TODO(Skye/Carlos): use indices here
                )

        # --------------------------------------------------------
        # QUAD-1 or QUAD-3: k = 1 only
        # --------------------------------------------------------
        elif mode_ctcs == "quad-1" or mode_ctcs == "quad-3":
            if subprob.vb_plus_ctcs is not None:
                VB += cp.sum_squares(
                    cp.diag(subprob.W_plus_ctcs_sqrt[0, :]) @ subprob.vb_plus_ctcs[0, :]
                )
            if subprob.vb_minus_ctcs is not None:
                VB += cp.sum_squares(
                    cp.diag(subprob.W_minus_ctcs_sqrt[0, :]) @ subprob.vb_minus_ctcs[0, :]
                )

        # --------------------------------------------------------
        # QUAD-2 : Per-time-step quadratic penalties
        # --------------------------------------------------------
        elif mode_ctcs == "quad-2":
            if subprob.vb_plus_ctcs is not None:
                for k in range(subprob.Npm_ctcs):
                    VB += cp.sum_squares(
                        cp.diag(subprob.W_plus_ctcs_sqrt[k, :]) @ subprob.vb_plus_ctcs[k, :]
                    )
            if subprob.vb_minus_ctcs is not None:
                for k in range(subprob.Npm_ctcs):
                    VB += cp.sum_squares(
                        cp.diag(subprob.W_minus_ctcs_sqrt[k, :]) @ subprob.vb_minus_ctcs[k, :]
                    )

    # ------------------------------------------------------------
    # Final scaling (flag multiplies entire VB block)
    # ------------------------------------------------------------
    return 0.5 * subprob.flag_vb * VB

def build_dual_buffer_cost(subprob) -> cp.Expression:
    """
    Dual penalty term DUAL for inequality constraints, dynamic buffers,
    and aggregate quadratic-buffer (quad-1, quad-2) modes.
    """
    method = subprob.problem.method
    mode_real     = method.flags["buff_dyn"]        # {'none','term','l1','l2','quad-1','quad-2','quad-3'}
    mode_real_dual = method.flags["buff_dyn_dual"]  # MATLAB: buff_dyn_dual
    mode_ctcs     = method.flags["ctcs"]            # {'none','term','l1','l2','quad-1','quad-2','quad-3'}
    mode_ctcs_dual = method.flags["ctcs_dual"]

    DUAL = 0.0

    # ============================================================
    # Unified INEQ DUAL COST
    # ============================================================
    if subprob.vb_ineq is not None and subprob.n_ineq > 0:
        DUAL += cp.sum(cp.multiply(subprob.vb_ineq, subprob.dual_ineq))

    # ============================================================
    # Dynamic dual: dual_dyn .* (vb_dyn_plus - vb_dyn_minus)
    # ============================================================
    diff = subprob.vb_dyn_p - subprob.vb_dyn_m
    DUAL += cp.sum(cp.multiply(diff, subprob.dual_dyn))

    # ============================================================
    # QUAD-1, QUAD-2, QUAD-3 dual components
    # ============================================================
    # These exist ONLY if buff_dyn_dual == 'l1'
    if mode_real_dual == "l1":

        # -----------------------
        # QUAD-1 or QUAD-3: k == 1 only
        # -----------------------
        if mode_real == "quad-1" or mode_real == "quad-3":
            if subprob.vb_plus_real is not None and subprob.dual_plus_real is not None:
                DUAL += subprob.dual_plus_real[0, :] @ subprob.vb_plus_real[0, :]
            if subprob.vb_minus_real is not None and subprob.dual_minus_real is not None:
                DUAL += subprob.dual_minus_real[0, :] @ subprob.vb_minus_real[0, :]

        # -----------------------
        # QUAD-2  (per time index)
        # -----------------------
        elif mode_real == "quad-2":
            if subprob.vb_plus_real is not None and subprob.dual_plus_real is not None:
                for k in range(subprob.Npm):
                    DUAL += subprob.dual_plus_real[k, :] @ subprob.vb_plus_real[k, :]
            if subprob.vb_minus_real is not None and subprob.dual_minus_real is not None:
                for k in range(subprob.Npm_real):
                    DUAL += subprob.dual_minus_real[k, :] @ subprob.vb_minus_real[k, :]

    # ============================================================
    # CTCS DUAL COST
    # ============================================================
    # These exist ONLY if ctcs_dual == 'l1'
    if mode_ctcs_dual == "l1":

        # -----------------------
        # QUAD-1 or QUAD-3: k == 1 only
        # -----------------------
        if mode_ctcs == "quad-1" or mode_ctcs == "quad-3":
            if subprob.vb_plus_ctcs is not None and subprob.dual_plus_ctcs is not None:
                DUAL += subprob.dual_plus_ctcs[0, :] @ subprob.vb_plus_ctcs[0, :]
            if subprob.vb_minus_ctcs is not None and subprob.dual_minus_ctcs is not None:
                DUAL += subprob.dual_minus_ctcs[0, :] @ subprob.vb_minus_ctcs[0, :]

        # -----------------------
        # QUAD-2  (per time index)
        # -----------------------
        elif mode_ctcs == "quad-2":
            if subprob.vb_plus_ctcs is not None and subprob.dual_plus_ctcs is not None:
                for k in range(subprob.Npm_ctcs):
                    DUAL += subprob.dual_plus_ctcs[k, :] @ subprob.vb_plus_ctcs[k, :]
            if subprob.vb_minus_ctcs is not None and subprob.dual_minus_ctcs is not None:
                for k in range(subprob.Npm_ctcs):
                    DUAL += subprob.dual_minus_ctcs[k, :] @ subprob.vb_minus_ctcs[k, :]

    # ============================================================
    # Terminal dual cost
    # ============================================================
    if subprob.vb_term is not None and subprob.n_term > 0:
        DUAL += subprob.dual_term @ subprob.vb_term

    return DUAL


# -------------- AUTOTUNING SCHEMES ----------------------------------------------------------------------------------------

def autotune1(problem, iter_record):
    """
    Unified version of autotune1 including dual_plus and dual_minus.
    Works with stacked inequality and dynamic buffer system.
    """
    mission, model, method  = problem.mission, problem.model, problem.method

    # Iteration number
    iter_num = iter_record["iter_num"]

    # Extract primal buffers
    vb_ineq  = np.array(iter_record["conv_data"]["vb_ineq"])
    vb_term  = np.array(iter_record["conv_data"]["vb_term"])
    vb_dyn   = np.array(iter_record["conv_data"]["vb_dyn"])
    
    vb_plus_real  = np.array(iter_record["conv_data"]["vb_plus_real"])
    vb_minus_real = np.array(iter_record["conv_data"]["vb_minus_real"])
    
    vb_plus_ctcs  = np.array(iter_record["conv_data"]["vb_plus_ctcs"])
    vb_minus_ctcs = np.array(iter_record["conv_data"]["vb_minus_ctcs"])

    # Extract duals
    dual_ineq  = iter_record["weights"]["dual_ineq"]
    dual_dyn   = iter_record["weights"]["dual_dyn"]
    dual_term  = iter_record["weights"]["dual_term"]
    
    dual_plus_real  = iter_record["weights"]["dual_plus_real"]
    dual_minus_real = iter_record["weights"]["dual_minus_real"]
    
    dual_plus_ctcs  = iter_record["weights"]["dual_plus_ctcs"]
    dual_minus_ctcs = iter_record["weights"]["dual_minus_ctcs"]

    # Hyperparameters
    if method.flags["stepsize_auto_dual"]:
        beta = gamma = 1 / iter_num
    else:
        beta  = method.weights["beta"]
        gamma = method.weights["gamma"]

    # ==========================================
    # Dual updates
    # ==========================================

    # inequality
    dual_ineq_plus = np.maximum(0, gamma * vb_ineq + dual_ineq)

    # dynamics
    dual_dyn_plus = beta * vb_dyn + dual_dyn

    # terminal
    dual_term_plus = beta * vb_term + dual_term

    # plus/minus (quadratic 1-norm decomposition)
    dual_plus_plus_real  = beta * vb_plus_real  + dual_plus_real
    dual_minus_plus_real = beta * vb_minus_real + dual_minus_real

    dual_plus_plus_ctcs  = beta * vb_plus_ctcs  + dual_plus_ctcs
    dual_minus_plus_ctcs = beta * vb_minus_ctcs + dual_minus_ctcs

    # ==========================================
    # Saturation thresholds
    # ==========================================
    conv = method.conv
    eps_ineq = conv.get("eps_ineq", 1e-6)
    eps_term = conv["eps_term"]
    eps_dyn  = conv["eps_dyn"]
    eps_quad = conv.get("eps_quad", eps_dyn)   # for vb_plus/vb_minus

    # inequality saturation
    mask_ineq = vb_ineq <= eps_ineq
    dual_ineq_plus[mask_ineq] = dual_ineq[mask_ineq]

    # dynamics
    mask_dyn = np.abs(vb_dyn) <= eps_dyn
    dual_dyn_plus[mask_dyn] = dual_dyn[mask_dyn]

    # terminal
    mask_term = np.abs(vb_term) <= eps_term
    dual_term_plus[mask_term] = dual_term[mask_term]


    # real plus/minus
    # plus
    mask_plus_real = np.abs(vb_plus_real) <= eps_quad
    dual_plus_plus_real[mask_plus_real] = dual_plus_real[mask_plus_real]

    # minus
    mask_minus_real = np.abs(vb_minus_real) <= eps_quad
    dual_minus_plus_real[mask_minus_real] = dual_minus_real[mask_minus_real]

    # ctcs plus/minus
    # plus
    mask_plus_ctcs = np.abs(vb_plus_ctcs) <= eps_quad
    dual_plus_plus_ctcs[mask_plus_ctcs] = dual_plus_ctcs[mask_plus_ctcs]

    # minus
    mask_minus_ctcs = np.abs(vb_minus_ctcs) <= eps_quad
    dual_minus_plus_ctcs[mask_minus_ctcs] = dual_minus_ctcs[mask_minus_ctcs]

    # ==========================================
    # Update weights
    # ==========================================
    weights = iter_record["weights"]
    weights.update({
        "dual_ineq": dual_ineq_plus,
        "dual_dyn":  dual_dyn_plus,
        "dual_term": dual_term_plus,
        "dual_plus_real": dual_plus_plus_real,
        "dual_minus_real": dual_minus_plus_real,
        "dual_plus_ctcs": dual_plus_plus_ctcs,
        "dual_minus_ctcs": dual_minus_plus_ctcs,
        "data": {
            "dmu_ineq": dual_ineq_plus - dual_ineq,
            "dmu_eq":   dual_term_plus - dual_term,
            "dmu_plus_real": dual_plus_plus_real  - dual_plus_real,
            "dmu_minus_real": dual_minus_plus_real - dual_minus_real,
            "dmu_plus_ctcs": dual_plus_plus_ctcs  - dual_plus_ctcs,
            "dmu_minus_ctcs": dual_minus_plus_ctcs - dual_minus_ctcs,
        }
    })

    return iter_record


def autotune2(problem, iter_record):
    """
    Unified stacked-inequality version of autotune2.
    """
    mission, model, method  = problem.mission, problem.model, problem.method
    
    # Extract variables from local_vars
    N = method.N
    vb_ineq = np.array(iter_record["conv_data"]["vb_ineq"])
    vb_dyn  = np.array(iter_record["conv_data"]["vb_dyn"])  # Assuming vb_dyn_plus is in sol_vars
    vb_term = np.array(iter_record["conv_data"]["vb_term"])

    vb_plus_real = np.array(iter_record["conv_data"]["vb_plus_real"])
    vb_minus_real = np.array(iter_record["conv_data"]["vb_minus_real"])
    vb_plus_ctcs = np.array(iter_record["conv_data"]["vb_plus_ctcs"])
    vb_minus_ctcs = np.array(iter_record["conv_data"]["vb_minus_ctcs"])

    W_ineq = np.array(iter_record["weights"]["W_ineq"])
    W_dyn  = np.array(iter_record["weights"]["W_dyn"])
    W_term = np.array(iter_record["weights"]["W_term"])

    W_plus_real  = np.array(iter_record["weights"]["W_plus_real"])
    W_minus_real = np.array(iter_record["weights"]["W_minus_real"])
    W_plus_ctcs  = np.array(iter_record["weights"]["W_plus_ctcs"])
    W_minus_ctcs = np.array(iter_record["weights"]["W_minus_ctcs"])

    # Extract parameters for autotuning
    eps_feas_ineq = method.conv.get("eps_ineq", 1e-6)
    eps_feas_term = method.conv["eps_term"]
    eps_feas_dyn  = method.conv["eps_dyn"]

    eps_nonzero2 = method.weights["eps_nonzero2"]

    buff_dyn = method.flags["buff_dyn"]
    ctcs = method.flags["ctcs"]

    Wh_ineq = np.zeros((N, mission.n_ineq))
    Wh_dyn  = np.zeros((N, model.n_dyn))
    Wh_term = np.zeros(mission.n_term + mission.n_term_ineq + mission.n_term_ctcs)

    Wh_plus_real  = np.zeros((method.Npm_real, method.n_plus_real))
    Wh_minus_real = np.zeros((method.Npm_real, method.n_minus_real))
    Wh_plus_ctcs  = np.zeros((method.Npm_ctcs, method.n_plus_ctcs))
    Wh_minus_ctcs = np.zeros((method.Npm_ctcs, method.n_minus_ctcs))

    # ==========================================
    # COMPUTE AUTOTUNE UPDATES
    # ==========================================

    if buff_dyn == "quad-1":
        Wh_plus_real = np.abs(W_plus_real @ vb_plus_real) / eps_feas_dyn
        Wh_minus_real = np.abs(W_minus_real @ vb_minus_real) / eps_feas_dyn

    if ctcs == "quad-1":
        Wh_plus_ctcs = np.abs(W_plus_ctcs @ vb_plus_ctcs) / eps_feas_dyn
        Wh_minus_ctcs = np.abs(W_minus_ctcs @ vb_minus_ctcs) / eps_feas_dyn

    # TODO: add quad-3 case

    if buff_dyn == "quad-3":
        for j in range(model.n):
            Wh_plus_real[:, j] = np.sum(np.abs(np.diag(W_plus_real[:, j]) @ vb_plus_real[:, j] / eps_feas_dyn))
            Wh_minus_real[:, j] = np.sum(np.abs(np.diag(W_minus_real[:, j]) @ vb_minus_real[:, j] / eps_feas_dyn))
    if ctcs == "quad-3":
        for j in range(model.n_ctcs):
            Wh_plus_ctcs[:, j] = np.sum(np.abs(np.diag(W_plus_ctcs[:, j]) @ vb_plus_ctcs[:, j] / eps_feas_dyn))
            Wh_minus_ctcs[:, j] = np.sum(np.abs(np.diag(W_minus_ctcs[:, j]) @ vb_minus_ctcs[:, j] / eps_feas_dyn))

    for k in range(N):
        dual_ineq_buff = np.diag(W_ineq[k, :]) @ vb_ineq[k, :].flatten()

        if mission.n_ineq > 0:
            Wh_ineq[k, :] = np.abs(dual_ineq_buff / eps_feas_ineq)
        else:
            Wh_ineq[k, :] = np.abs(dual_ineq_buff)

        if k < N - 1:
            dual_dyn_buff = np.diag(W_dyn[k, :]) @ vb_dyn[k, :]
            
            if buff_dyn:
                Wh_dyn[k, :] = np.sum(np.abs(dual_dyn_buff) / eps_feas_dyn)
            else:
                Wh_dyn[k, :] = np.sum(np.abs(dual_dyn_buff))

            if buff_dyn == "quad-2":
                Wh_plus_real[k]  = np.sum(np.abs(np.diag(W_plus_real[k, :]) @ vb_plus_real[k, :] / eps_feas_dyn))
                Wh_minus_real[k] = np.sum(np.abs(np.diag(W_minus_real[k, :]) @ vb_minus_real[k, :] / eps_feas_dyn))
            
            if ctcs == "quad-2":
                Wh_plus_ctcs[k]  = np.sum(np.abs(np.diag(W_plus_ctcs[k, :]) @ vb_plus_ctcs[k, :] / eps_feas_dyn))
                Wh_minus_ctcs[k] = np.sum(np.abs(np.diag(W_minus_ctcs[k, :]) @ vb_minus_ctcs[k, :] / eps_feas_dyn))

    if (mission.n_term + mission.n_term_ineq + mission.n_term_ctcs) > 0:
        dual_term_buff = np.diag(W_term) @ vb_term
        Wh_term = np.abs(dual_term_buff / eps_feas_term).flatten()

    # ==========================================
    # UPDATE WEIGHTS WITH COMPUTED AUTOTUNE UPDATES
    # ==========================================

    if np.sum(method.weights["W_plus_real"]) > 0: Wh_plus_real[Wh_plus_real <= eps_nonzero2] = eps_nonzero2 
    if np.sum(method.weights["W_minus_real"]) > 0: Wh_minus_real[Wh_minus_real <= eps_nonzero2] = eps_nonzero2
    if np.sum(method.weights["W_plus_ctcs"]) > 0: Wh_plus_ctcs[Wh_plus_ctcs <= eps_nonzero2] = eps_nonzero2
    if np.sum(method.weights["W_minus_ctcs"]) > 0: Wh_minus_ctcs[Wh_minus_ctcs <= eps_nonzero2] = eps_nonzero2

    if np.sum(method.weights["W_ineq"]) > 0: Wh_ineq[Wh_ineq <= eps_nonzero2] = eps_nonzero2  
    if np.sum(method.weights["W_dyn"]) > 0: Wh_dyn[Wh_dyn <= eps_nonzero2] = eps_nonzero2
    if np.sum(method.weights["W_term"]) > 0: Wh_term[Wh_term <= eps_nonzero2] = eps_nonzero2

    iter_record["weights"]["W_plus_real"] = Wh_plus_real 
    iter_record["weights"]["W_minus_real"] = Wh_minus_real
    iter_record["weights"]["W_plus_ctcs"] = Wh_plus_ctcs
    iter_record["weights"]["W_minus_ctcs"] = Wh_minus_ctcs

    iter_record["weights"]["W_ineq"] = Wh_ineq
    iter_record["weights"]["W_dyn"] = Wh_dyn
    iter_record["weights"]["W_term"] = Wh_term

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
            "dual": 0.0 #dual_ineq_buff[k] # TODO (CARLOS): add this back
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