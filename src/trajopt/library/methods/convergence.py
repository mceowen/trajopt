import numpy as np

import trajopt.utils.tools as tools

def set_convergence_tolerance(problem, method):
    """
    Compute convergence tolerances for all trajopt_obj components.
    This refactored version stacks path/NFZ/AUX tolerances into unified eps_ineq/Wconv_ineq.
    """
    
    # =======================
    # STATE CONVERGENCE (augmented with ct when "ct" is present)
    # =======================
    n_z = problem.index_map.n['z']

    ctcs_mult_state = 1.0 # method.conv.ctcs_mult_state
    ctcs_mult_cnst  = getattr(method.conv, "ctcs_fac_cnst", 1.0) * method.guess.T_init / (method.index_map.N['N']-1)

    if len(method.conv.eps_state) == 1:
        eps_state    = method.conv.eps_state * np.ones(n)
        M_state_d2nd = np.eye(n)
    else:
        eps_state    = method.conv.eps_state
        M_state_d2nd = method.nondim.M["state"]["d2nd"]

    if problem.constraints.has(ct=1):
        eps_list = np.concatenate([constraint.eps for constraint in problem.constraints.get(ct=1)])
        ctcs_eps_list = ctcs_mult_cnst * (method.penalty.w_ctcs * eps_list)**2

        eps_state = np.concatenate([eps_state, ctcs_eps_list])
        M_state_d2nd = np.diag(np.concatenate([np.diag(M_state_d2nd), np.diag(method.nondim.M["ineq_ct"]["d2nd"])**2 / method.nondim.nt]))

    eps_state_nd  = M_state_d2nd @ eps_state
    eps_min_state = float(np.min(eps_state_nd))
    Wconv_state   = np.diag(eps_min_state / eps_state_nd)

    method.conv.eps_state_nd    = eps_state_nd
    method.conv.eps_state       = eps_min_state 
    method.conv.Wconv_state     = Wconv_state
    method.conv.Wconv_state_vec = np.diag(Wconv_state).copy()

    # =======================
    # COST CONVERGENCE
    # =======================
    eps_cost    = method.conv.eps_cost
    nd_cost = method.nondim.nd_cost
    method.conv.eps_cost = eps_cost / nd_cost

    # =======================
    # stacked inequality
    # =======================

    nodal_ncvx_constraints = problem.constraints.get(ct=0, type='nonconvex_inequality')
    M_ineq_d2nd = method.nondim.M["ineq_nodal"]["d2nd"]

    # Compute dimensional tolerances

    if len(nodal_ncvx_constraints) > 0:
        eps_ineq = np.concatenate([constraint.eps for constraint in nodal_ncvx_constraints])
        eps_ineq_nd = M_ineq_d2nd @ eps_ineq
        eps_min_ineq = float(np.min(eps_ineq_nd)) if eps_ineq_nd.size > 0 else 0.
        Wconv_ineq = np.diag(eps_min_ineq / eps_ineq_nd) if eps_ineq_nd.size > 0 else np.zeros((1,1))
    else:
        eps_ineq = np.array([])
        eps_ineq_nd = np.array([])
        eps_min_ineq = 0.0
        Wconv_ineq = np.zeros((0, 0)) 

    method.conv.eps_ineq_nd    = eps_ineq_nd
    method.conv.eps_ineq       = eps_min_ineq 
    method.conv.Wconv_ineq     = Wconv_ineq
    method.conv.Wconv_ineq_vec = np.diag(Wconv_ineq).copy()

    # =======================
    # TERMINAL CONSTRAINTS
    # =======================
    n_term = problem.index_map.n['term_total']
    if n_term > 0:
        if len(method.conv.eps_term) == 1 and n_term != 1:
            eps_term    = method.conv.eps_term * np.ones(n_term)
            M_term_d2nd = np.eye(n_term)
        else:
            eps_term    = method.conv.eps_term
            M_term_d2nd = method.nondim.M["term_total"]["d2nd"]
        eps_term_nd  = M_term_d2nd @ eps_term
        eps_min_term = float(np.min(eps_term_nd))
    else:
        eps_term_nd = np.array([0.])
        eps_min_term = 0.

    Wconv_term = np.diag(eps_min_term / eps_term_nd) if np.any(eps_term_nd) else np.zeros((1,1))
    method.conv.eps_term_nd    = eps_term_nd
    method.conv.eps_term       = eps_min_term 
    method.conv.Wconv_term     = Wconv_term
    method.conv.Wconv_term_vec = np.diag(Wconv_term).copy()

    # =======================
    # MULTIPLE SHOOTING DEFECT
    # =======================
    if len(method.conv.eps_defect) == 1:
        eps_defect    = method.conv.eps_defect * np.ones(n_z)
        M_defect_d2nd = np.eye(n_z)
    else:
        eps_defect    = method.conv.eps_defect
        M_defect_d2nd = method.nondim.M["state"]["d2nd"]

    eps_defect_nd  = M_defect_d2nd @ eps_defect
    eps_min_defect = float(np.min(eps_defect_nd))
    Wconv_defect   = np.diag(eps_min_defect / eps_defect_nd)

    method.conv.eps_defect_nd    = eps_defect_nd
    method.conv.eps_defect       = eps_min_defect 
    method.conv.Wconv_defect     = Wconv_defect
    method.conv.Wconv_defect_vec = np.diag(Wconv_defect).copy()

    # =======================
    # DYNAMICS CONVERGENCE
    # =======================
    n_dyn = problem.index_map.n['z']
    if n_dyn > 0:
        eps_dyn    = method.conv.eps_dyn
        M_dyn_d2nd = method.nondim.M["state"]["d2nd"]
    else:
        eps_dyn    = np.zeros((problem.index_map.n['z'],))
        M_dyn_d2nd = np.zeros((1, problem.index_map.n['z']))

    if problem.constraints.has(ct=1):
        eps_ct_list = np.concatenate([constraint.eps for constraint in problem.constraints.get(ct=1)])
    else: 
        eps_ct_list = np.array([])

    if problem.constraints.has(ct=1):
        eps_dyn = np.concatenate([
            ctcs_mult_state * eps_dyn,
            ctcs_mult_cnst  * (method.penalty.w_ctcs * eps_ct_list)**2
        ])
        M_dyn_d2nd = np.diag(np.concatenate([
            np.diag(M_dyn_d2nd),
            np.diag(method.nondim.M["ineq_ct"]["d2nd"])**2 / method.nondim.nt
        ]))

    eps_dyn_nd  = M_dyn_d2nd @ eps_dyn
    eps_min_dyn = float(np.min(eps_dyn_nd)) if np.any(eps_dyn_nd) else 0.

    if np.all(eps_dyn_nd == 0):
        Wconv_dyn = np.zeros((len(eps_dyn_nd), len(eps_dyn_nd)))
    else:
        Wconv_dyn = np.diag(eps_min_dyn / eps_dyn_nd)

    method.conv.eps_dyn_nd    = eps_dyn_nd
    method.conv.eps_dyn       = eps_min_dyn 
    method.conv.Wconv_dyn     = Wconv_dyn
    method.conv.Wconv_dyn_vec = np.diag(Wconv_dyn).copy()


# ----------------------------------------------------------------------------------------------

def check_convergence_tolerance(problem, method, iter_record):
    """Check convergence using unified stacked inequality (_ineq) structure."""

    # --- Load convergence data
    conv_data = iter_record.conv_data

    # --- Extract dimensions from Subproblem
    n_z = problem.index_map.n.z
    N   = method.index_map.N.N

    # --- Extract optimization variables
    dz      = iter_record.dz_s
    dcost   = iter_record.cost - conv_data.cost_ref
    defect  = conv_data.defect
    vb_dyn  = conv_data.vb_dyn
    vb_ineq = conv_data.vb_ineq      # unified inequality VB
    vb_term = conv_data.vb_term

    # --- Extract convergence criteria
    eps_state  = method.conv.eps_state
    eps_cost   = method.conv.eps_cost
    eps_ineq   = getattr(method.conv, "eps_ineq", 1e-6)
    eps_term   = method.conv.eps_term
    eps_defect = method.conv.eps_defect
    eps_dyn    = method.conv.eps_dyn

    W_state  = method.conv.Wconv_state
    W_ineq   = method.conv.Wconv_ineq
    W_term   = method.conv.Wconv_term
    W_dyn    = method.conv.Wconv_dyn
    W_defect = method.conv.Wconv_defect

    # --- Extract linearized inequality constraint residuals
    cnst_all  = iter_record.cnst_path  # full stacked nonlinear constraints

    conv_ineq_nl = np.maximum(0.0, cnst_all) if cnst_all is not None else np.zeros((N, 0))

    # === Optimality ===
    dz_array = tools.safe_val(dz, rows=N, cols=n_z)
    chk_dz   = np.max([np.max(W_state @ np.abs(dz_k)) for dz_k in dz_array])
    chk_cost = np.abs(dcost)

    # === Feasibility: Virtual Buffer violations ===
    chk_vb_ineq = (np.max([np.max(W_ineq @ vb_ineq[k].reshape(-1, 1)) for k in range(N)]) if vb_ineq.size else 0.0)
    chk_vb_term = np.max(W_term @ np.abs(vb_term))
    chk_vb_dyn  = (np.max([np.max(W_dyn @ np.abs(vb_dyn[k]).reshape(-1, 1)) for k in range(N - 1)]) if vb_dyn.size else 0.0)
    chk_defect = (np.max([np.max(W_defect @ np.abs(defect[k]).reshape(-1, 1)) for k in range(N)]) if defect.size else 0.0)

    # === Feasibility: Linearized constraint residuals ===
    chk_ineq_2 = (np.max([np.max(W_ineq @ conv_ineq_nl[k].reshape(-1, 1)) for k in range(N)]) if conv_ineq_nl.size else 0.0)

    if method.flags["ctcs"] != "none":
        chk_feas_1 = np.array([chk_vb_term, chk_vb_dyn])
        chk_feas_2 = np.array([chk_vb_term, chk_defect])
        eps_feas_1 = np.array([eps_term, eps_dyn])
        eps_feas_2 = np.array([eps_term, eps_defect])
    else:
        chk_feas_1 = np.array([chk_vb_ineq, chk_vb_term, chk_vb_dyn])
        chk_feas_2 = np.array([chk_ineq_2, chk_vb_term])
        eps_feas_1 = np.array([eps_ineq, eps_term, eps_dyn])
        eps_feas_2 = np.array([eps_ineq, eps_term])

    if method.flags["flag_conv"] == 0:
        chk_opt = np.array([chk_dz, chk_cost])
        eps_opt = np.array([eps_state, eps_cost])
    elif method.flags["flag_conv"] == 1:
        chk_opt = np.array([chk_dz, np.nan])
        eps_opt = np.array([eps_state, np.nan])
    elif method.flags["flag_conv"] == 2:
        chk_opt = np.array([np.nan, chk_cost])
        eps_opt = np.array([np.nan, eps_cost])

    # === Convergence check
    bool_conv = (
        (np.all(chk_feas_1 <= eps_feas_1) and np.all(chk_opt[0] <= eps_opt[0])) or
        (np.all(chk_feas_2 <= eps_feas_2) and np.all(chk_opt[1] <= eps_opt[1]))
    )

    # === Populate convergence summary
    conv_data.bool_conv = bool_conv
    conv_data.chk_dz = chk_opt[0]
    conv_data.chk_opt = np.nanmax(chk_opt)
    conv_data.chk_feas_term = chk_vb_term
    conv_data.chk_feas_ineq = chk_vb_ineq
    conv_data.chk_feas_dyn = chk_vb_dyn
    conv_data.chk_feas = max(np.max(chk_feas_1), np.max(chk_feas_2))
    conv_data.status = iter_record.cp_subprob.status

    iter_record.converged = bool_conv
    iter_record.conv_data = conv_data

    return iter_record


