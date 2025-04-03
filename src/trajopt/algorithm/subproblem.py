# recall: pip install -e /Users/skye/ACL/entry/python/hypersonic_entry_opt/trajopt

# basic imports
import cvxpy as cp
import numpy as np
import time

# trajopt imports
import trajopt.algorithm.hyperparameters as hp
import trajopt.algorithm.scaling as scaling
import trajopt.algorithm.discretization as discretization
import trajopt.algorithm.convergence as convergence
import trajopt.utils.tools as tools


def solve_subproblem(problem):
    inputs, model_data = baseline_subprob_inputs(problem)
    problem.custom_inputs(problem, inputs)

    solution_vars = baseline_subprob_variables(inputs)

    constraints = []
    constraints += baseline_subprob_constraints(inputs, solution_vars)
    constraints += problem.custom_constraints(problem, inputs, solution_vars)

    PTR_COST = baseline_subprob_cost(inputs, solution_vars)

    objective = cp.Minimize(PTR_COST)
    prob = cp.Problem(objective, constraints)
    solve_stats = prob.solve(**inputs["cvxpy_opts"])

    soln_stats = {
        "solntime": prob.solver_stats.solve_time,
        "parse_time": prob.solver_stats.setup_time,
        "soln_object": prob
    }

    O = baseline_subprob_outputs(
        soln_obj=prob,
        inputs=inputs,
        solution_vars=solution_vars,
        model_data=model_data
    )

    problem.custom_outputs(problem, inputs, solution_vars, O)
    O = convergence.check_convergence_tolerance(O, problem)
    baseline_autotune(problem, O)
    display_baseline_subprob_status(O, problem, nt=inputs["params"]["nt"], ncost=inputs["params"]["ncost"])

    return O


def baseline_subprob_inputs(problem):
    I = problem['I'][-1][0]
    iter_num = I['iter_num']
    case_flag = problem['params']['case_flag']

    # Dynamics and cost
    start = time.time()
    Ak, Bk, Bkp, Sk, zs_minus = discretization.compute_linsys_discrete(I['zs_ref'], I['us_ref'], I['dts_ref'], problem)
    prop_time = time.time() - start

    dcostdz, dcostdu, cost = problem.compute_cost(I['ts_ref'], I['zs_ref'], I['us_ref'], problem)
    dgdz, dgdu, g = problem.problem.compute_path_constraints(I['ts_ref'], I['zs_ref'], I['us_ref'], problem)

    # Reference trajectories
    ts_ref = I['ts_ref']
    dts_ref = I['dts_ref']
    zs_ref = I['zs_ref']
    us_ref = I['us_ref']

    vb_path_ref = I['conv_data']['vb_path']
    vb_nfz_ref = I['conv_data']['vb_nfz']
    vb_aux_ref = I['conv_data']['vb_aux']
    vb_dyn_ref = I['conv_data']['vb_dyn']
    vb_term_ref = I['conv_data']['vb_term']

    # Dimensions and problem parameters
    params = problem['params']
    N = params['N']
    n = params['n']
    m = params['m']
    nz = params['nz']

    nt = params['nondim']['nt']
    nd = params['nondim']['nd']
    nv = params['nondim']['nv']
    na = params['nondim']['na']
    ncost = params['nondim']['ncost']

    # Boundary conditions
    z1 = problem['zi']
    z1_idx = problem['zi_idx']
    z1_min = problem['zi_min']
    z1_min_idx = problem['zi_min_idx']
    z1_max = problem['zi_max']
    z1_max_idx = problem['zi_max_idx']
    n_init = problem['n_init']
    n_init_ineq = problem['n_init_ineq']

    zN = problem['zf']
    zN_idx = problem['zf_idx']
    zN_min = problem['zf_min']
    zN_min_idx = problem['zf_min_idx']
    zN_max = problem['zf_max']
    zN_max_idx = problem['zf_max_idx']
    n_term = problem['n_term']
    n_term_ineq = problem['n_term_ineq']

    vb_N_idx = list(range(n_term))
    vb_N_ineq_idx = list(range(n_term, n_term + n_term_ineq))

    # State and control constraints
    z_min = problem['z_min']
    z_min_idx = problem['z_min_idx']
    z_max = problem['z_max']
    z_max_idx = problem['z_max_idx']
    n_state = params['n_state']

    u_min = problem['u_min']
    u_min_idx = problem['u_min_idx']
    u_max = problem['u_max']
    u_max_idx = problem['u_max_idx']
    n_ctrl = problem['n_ctrl']
    udot_max = problem['udot_max']
    udot_max_idx = problem['udot_max_idx']
    n_udot = problem['n_udot']
    bool_init_ctrl = params['bools']['init_ctrl']

    # Time settings
    bools = params['bools']
    free_final_time = bools['free_final_time']
    equal_dt_bool = bools['equal_dt']

    dts_min = dts_max = ddts_max = None
    if free_final_time:
        dts_min = params['dts_min']
        dts_max = params['dts_max']
        ddts_max = params['ddts_max']

    # Contact constraints
    ctcs = bools['ctcs']
    eps_ctcs = params['eps_ctcs']

    # Constraint structure
    n_path = params['n_path']
    n_nfz = params['n_nfz']
    n_aux = params['n_aux']
    n_ineq = n_path + n_nfz + n_aux
    n_dyn = params['n_dyn']
    n_eq = n_dyn

    # Weighting and duals
    flag_autotune = bools['flag_autotune']
    buff_dyn = bools['buff_dyn']
    buff_dyn_dual = bools.get('buff_dyn_dual', None)

    weights = I['weights']
    W_path = weights['W_path']
    W_nfz = weights['W_nfz']
    W_aux = weights['W_aux']
    W_dyn = weights['W_dyn']
    W_term = weights['W_term']
    W_plus = weights['W_plus']
    W_minus = weights['W_minus']

    w_cost = weights['w_cost']
    wtr_z = weights['wtr_z']
    wtr_u = weights['wtr_u']

    dual_path = weights['dual_path']
    dual_nfz = weights['dual_nfz']
    dual_aux = weights['dual_aux']
    dual_dyn = weights['dual_dyn']
    dual_plus = weights['dual_plus']
    dual_minus = weights['dual_minus']
    dual_term = weights['dual_term']

    opts = params['yalmip_opts']

    return locals()


def baseline_subprob_variables(problem):
    params = problem['params']
    bools = params['bools']
    N = params['N']

    # Variation in state and control
    dz, du = scaling.subprob_variable_scaling(problem)

    # Time-step or time-horizon dilation
    if bools['free_final_time']:
        if not bools['equal_dt']:
            dt = cp.Variable(N - 1)  # Variable timestep
        else:
            dT = cp.Variable()       # Time horizon scalar
            dt = (1 / (N - 1)) * dT * np.ones(N - 1)
    else:
        dt = np.zeros(N - 1)         # Fixed timestep (not a variable)

    # Virtual buffer and virtual control variables
    vb_path, vb_nfz, vb_aux, vb_dyn, vb_term = subprob_virtual_variables(problem)

    return {
        'dz': dz,
        'du': du,
        'dt': dt,
        'vb_path': vb_path,
        'vb_nfz': vb_nfz,
        'vb_aux': vb_aux,
        'vb_dyn': vb_dyn,
        'vb_term': vb_term,
        # 't_l1': t_l1,
    }

def subprob_virtual_variables(problem):
    """
    Determines which virtual variables (virtual buffer and virtual control) to
    create for subproblem constraints. If the weight associated with each
    term is zero, then the variables are set to the zero vector.
    """

    # Extract autotune flag and dynamics buffer toggle
    flag_autotune = problem['params']['bools']['flag_autotune']
    buff_dyn = problem['params']['bools']['buff_dyn']

    # Extract dimensions
    n = {
        'path':  problem['params']['n_path'],
        'nfz':   problem['params']['n_nfz'],
        'aux':   problem['params']['n_aux'],
        'dyn':   problem['params']['nz'],
        'term':  problem['params']['n_term'] + problem['params']['n_term_ineq'],
        'plus':  problem['params']['n_plus'],
        'minus': problem['params']['n_minus']
    }

    N = {
        'path':  problem['params']['N'],
        'nfz':   problem['params']['N'],
        'aux':   problem['params']['N'],
        'dyn':   problem['params']['N'] - 1,
        'term':  1,
        'plus':  problem['params']['Npm'],
        'minus': problem['params']['Npm']
    }

    # Extract weight and dual variable fields
    weights = problem['I'][-1]['weights']
    wght_keys = weights.keys()
    W_keys = [k for k in wght_keys if 'W' in k]
    dual_keys = [k for k in wght_keys if 'dual' in k]

    vb_vals = {}

    if str(flag_autotune) in {'0','2','3','al-scvx'}:
        vb_tags = [k.split('_')[-1] for k in W_keys]

        for tag in vb_tags:
            W = weights.get(f'W_{tag}', None)
            if W is not None and np.sum(W) != 0:
                vb_vals[tag] = cp.Variable((n[tag], N[tag]))
            else:
                vb_vals[tag] = np.zeros((n[tag], N[tag]))

    elif str(flag_autotune) == '1':
        vb_tags = [k.split('_')[-1] for k in dual_keys if 'eq' not in k]

        for tag in vb_tags:
            dual = weights.get(f'dual_{tag}', None)
            is_dyn_excluded = (tag == 'dyn') and (buff_dyn == 'term')
            if dual is not None and np.sum(dual) != 0 and not is_dyn_excluded:
                vb_vals[tag] = cp.Variable((n[tag], N[tag]))
            else:
                vb_vals[tag] = np.zeros((n[tag], N[tag]))

    # Dynamics plus/minus buffers depend on buff_dyn
    if buff_dyn == 'term':
        vb_dyn_plus = np.zeros((n['dyn'], N['dyn']))
        vb_dyn_minus = np.zeros((n['dyn'], N['dyn']))
    else:
        vb_dyn_plus = cp.Variable((n['dyn'], N['dyn']))
        vb_dyn_minus = cp.Variable((n['dyn'], N['dyn']))

    # Optional plus/minus aggregate buffers (e.g., for quad-1, quad-2)
    vb_plus = cp.Variable((n['plus'], N['plus']))
    vb_minus = cp.Variable((n['minus'], N['minus']))

    return (
        vb_vals.get('path', np.zeros((n['path'], N['path']))),
        vb_vals.get('nfz', np.zeros((n['nfz'], N['nfz']))),
        vb_vals.get('aux', np.zeros((n['aux'], N['aux']))),
        vb_vals.get('term', np.zeros((n['term'], N['term']))),
        vb_dyn_plus,
        vb_dyn_minus,
        vb_plus,
        vb_minus
    )


def baseline_subprob_constraints(inputs, solution_vars):
    dz = solution_vars["dz"]
    du = solution_vars["du"]
    dt = solution_vars["dt"]
    vb_path = solution_vars["vb_path"]
    vb_nfz = solution_vars["vb_nfz"]
    vb_aux = solution_vars["vb_aux"]
    vb_term = solution_vars["vb_term"]
    vb_dyn_plus = solution_vars["vb_dyn_plus"]
    vb_dyn_minus = solution_vars["vb_dyn_minus"]
    vb_plus = solution_vars["vb_plus"]
    vb_minus = solution_vars["vb_minus"]

    I = inputs["I"]
    params = inputs["params"]
    Ak = inputs["Ak"]
    Bk = inputs["Bk"]
    Bkp = inputs["Bkp"]
    Sk = inputs["Sk"]
    zs_minus = inputs["zs_minus"]
    dgdz = inputs["dgdz"]
    dgdu = inputs["dgdu"]
    g = inputs["g"]

    CNST = []
    bools = params['bools']
    flag_autotune = bools['flag_autotune']
    buff_dyn = bools['buff_dyn']

    # Extract dimensions and references
    N = params['N']
    n = params['n']
    m = params['m']
    nz = params['nz']
    n_state = params['n_state']
    n_ctrl = params['n_ctrl']
    n_udot = params['n_udot']
    n_ineq = params['n_path'] + params['n_nfz'] + params['n_aux']

    z1, z1_idx = params['z1'], params['z1_idx']
    z1_min, z1_min_idx = params['z1_min'], params['z1_min_idx']
    z1_max, z1_max_idx = params['z1_max'], params['z1_max_idx']
    zN, zN_idx = params['zN'], params['zN_idx']
    zN_min, zN_min_idx = params['zN_min'], params['zN_min_idx']
    zN_max, zN_max_idx = params['zN_max'], params['zN_max_idx']

    vb_N_idx = list(range(params['n_term']))
    vb_N_ineq_idx = list(range(params['n_term'], params['n_term'] + params['n_term_ineq']))

    z_min, z_min_idx = params['z_min'], params['z_min_idx']
    z_max, z_max_idx = params['z_max'], params['z_max_idx']
    u_min, u_min_idx = params['u_min'], params['u_min_idx']
    u_max, u_max_idx = params['u_max'], params['u_max_idx']

    udot_max, udot_max_idx = params['udot_max'], params['udot_max_idx']
    dts_ref = I['dts_ref']
    zs_ref = I['zs_ref']
    us_ref = I['us_ref']

    eps_ctcs = params['eps_ctcs']
    dts_min, dts_max, ddts_max = params['dts_min'], params['dts_max'], params['ddts_max']

    # Initial control
    if bools['init_ctrl']:
        CNST.append(du[:, 0] == 0)

    # Initial state
    if params['n_init'] > 0:
        CNST.append(dz[z1_idx, 0] + zs_ref[z1_idx, 0] == z1)

    if params['n_init_ineq'] > 0:
        M_sel = tools.constraint_index_selector(z1_min_idx, z1_max_idx, n)
        CNST.append(M_sel @ (dz[:n, 0] + zs_ref[:n, 0]) <= np.concatenate([-z1_min, z1_max]))

    # Terminal state
    if params['n_term'] > 0:
        CNST.append(dz[zN_idx, -1] + zs_ref[zN_idx, -1] - vb_term[vb_N_idx, 0] == zN)

    if params['n_term_ineq'] > 0:
        M_sel = tools.constraint_index_selector(zN_min_idx, zN_max_idx, n)
        CNST.append(M_sel @ (dz[:n, -1] + zs_ref[:n, -1]) - vb_term[vb_N_ineq_idx, 0] <= np.concatenate([-zN_min, zN_max]))

    if buff_dyn == 'quad-1':
        CNST.append(cp.sum(cp.vec(vb_dyn_plus)) == vb_plus)
        CNST.append(cp.sum(cp.vec(vb_dyn_minus)) == vb_minus)

    elif buff_dyn == 'quad-3':
        for j in range(nz):
            CNST.append(cp.sum(vb_dyn_plus[j, :]) == vb_plus[j, :])
            CNST.append(cp.sum(vb_dyn_minus[j, :]) == vb_minus[j, :])

    for k in range(N):
        if k < N - 1:
            # Dynamics
            lhs = dz[:, k + 1] + zs_ref[:, k + 1] - zs_minus[:, k + 1]
            rhs = (Ak[:, :, k] @ dz[:, k] + Bk[:, :, k] @ du[:, k] +
                   Bkp[:, :, k] @ du[:, k + 1] + Sk[:, k] * dt[k] +
                   vb_dyn_plus[:, k] - vb_dyn_minus[:, k])
            CNST.append(lhs == rhs)

            if buff_dyn != 'term':
                CNST.append(vb_dyn_plus[:, k] >= 0)
                CNST.append(vb_dyn_minus[:, k] >= 0)

            if buff_dyn == 'quad-2':
                CNST.append(cp.sum(vb_dyn_plus[:, k]) == vb_plus[:, k])
                CNST.append(cp.sum(vb_dyn_minus[:, k]) == vb_minus[:, k])

            if bools['ctcs']:
                CNST.append(zs_ref[n:nz, k + 1] + dz[n:nz, k + 1] - (zs_ref[n:nz, k] + dz[n:nz, k]) <= eps_ctcs)

            if bools['free_final_time']:
                CNST.append(dts_ref[k] + dt[k] <= dts_max)
                CNST.append(dts_ref[k] + dt[k] >= dts_min)
                CNST.append(cp.abs(dt[k]) <= ddts_max)

            if n_udot > 0 and k < N - 2:
                M_sel = tools.constraint_index_selector(udot_max_idx, udot_max_idx, m)
                CNST.append(M_sel @ (us_ref[:, k + 1] + du[:, k + 1] - (us_ref[:, k] + du[:, k])) <= (dts_ref[k] + dt[k]) * np.concatenate([udot_max, udot_max]))

        # State constraints
        if n_state > 0:
            M_sel = tools.constraint_index_selector(z_min_idx, z_max_idx, n)
            CNST.append(M_sel @ (zs_ref[:n, k] + dz[:n, k]) <= np.concatenate([-z_min, z_max]))

        # Control constraints
        if n_ctrl > 0:
            M_sel = tools.constraint_index_selector(u_min_idx, u_max_idx, m)
            CNST.append(M_sel @ (us_ref[:, k] + du[:, k]) <= np.concatenate([-u_min, u_max]))

        # Linearized inequality constraints
        if n_ineq > 0 and not bools['ctcs']:
            vb_combined = cp.vstack([vb_path[:, k], vb_nfz[:, k], vb_aux[:, k]])
            CNST.append(dgdz[:, :, k] @ dz[:n, k] + dgdu[:, :, k] @ du[:, k] + g[:, k] - vb_combined <= 0)

            if str(flag_autotune) in {'1', '3', 'al-scvx'}:
                CNST.append(vb_combined >= 0)

    return CNST



def baseline_subprob_cost(inputs, solution_vars):
    dz = solution_vars["dz"]
    du = solution_vars["du"]
    dt = solution_vars["dt"]
    vb_path = solution_vars["vb_path"]
    vb_nfz = solution_vars["vb_nfz"]
    vb_aux = solution_vars["vb_aux"]
    vb_term = solution_vars["vb_term"]
    vb_dyn_plus = solution_vars["vb_dyn_plus"]
    vb_dyn_minus = solution_vars["vb_dyn_minus"]

    dual_vars = inputs["dual_vars"]
    cost_terms = inputs["cost_terms"]
    weights = inputs["weights"]
    params = inputs["params"]


    N = params["N"]
    n = params["n"]
    solver_type = params.get("solver_type", "osqp")
    flag_autotune = params["flag_autotune"]
    buff_dyn = params["buff_dyn"]
    bools = params.get("bools", {})
    
    # --- TRUE COST ---
    TRUE_COST = 0
    for k in range(N):
        TRUE_COST += weights["w_cost"] * (
            cost_terms["dcostdz"][:, :, k] @ dz[:n, k] +
            cost_terms["dcostdu"][:, :, k] @ du[:, k] +
            cost_terms["cost"][:, k]
        )

    # --- TRUST REGION COST ---
    if solver_type == 'osqp':
        TR_COST = (
            weights["wtr_z"] * cp.sum_squares(cp.vec(dz)) +
            weights["wtr_u"] * cp.sum_squares(cp.vec(du))
        )
    else:
        dz_slacks = cp.Variable((1, N))
        du_slacks = cp.Variable((1, N))
        slack_constraints = []
        for k in range(N):
            slack_constraints.append(cp.norm(dz[:, k], 2) ** 2 <= dz_slacks[0, k])
            slack_constraints.append(cp.norm(du[:, k], 2) ** 2 <= du_slacks[0, k])
        TR_COST = weights["wtr_z"] * cp.norm(dz_slacks, 1) + weights["wtr_u"] * cp.norm(du_slacks, 1)

    # --- VIRTUAL BUFFER COST ---
    VIRTUAL_COST = 0
    if flag_autotune in {'0', '2', '3', 'al-scvx'}:
        VIRTUAL_COST += cp.quad_form(vb_term, np.diag(weights["W_term"]))

        for k in range(N):
            if buff_dyn == 'l1':
                if params["n_path"] > 0:
                    VIRTUAL_COST += np.max(weights["W_path"][:, k]) * cp.norm(vb_path[:, k], 1)
                if params["n_nfz"] > 0:
                    VIRTUAL_COST += np.max(weights["W_nfz"][:, k]) * cp.norm(vb_nfz[:, k], 1)
                if params["n_aux"] > 0:
                    VIRTUAL_COST += np.max(weights["W_aux"][:, k]) * cp.norm(vb_aux[:, k], 1)
                if k < N - 1:
                    VIRTUAL_COST += np.max(weights["W_dyn"][:, k]) * cp.norm(vb_dyn_plus[:, k] - vb_dyn_minus[:, k], 1)

            else:  # Quadratic buffer penalties
                VIRTUAL_COST += (
                    cp.quad_form(vb_path[:, k], np.diag(weights["W_path"][:, k])) +
                    cp.quad_form(vb_nfz[:, k], np.diag(weights["W_nfz"][:, k])) +
                    cp.quad_form(vb_aux[:, k], np.diag(weights["W_aux"][:, k]))
                )
                if k < N - 1:
                    if buff_dyn == 'l2':
                        VIRTUAL_COST += cp.quad_form(
                            vb_dyn_plus[:, k] - vb_dyn_minus[:, k],
                            np.diag(weights["W_dyn"][:, k])
                        )
                    elif buff_dyn == 'quad-1' and k == 0:
                        VIRTUAL_COST += (
                            cp.quad_form(vb_dyn_plus[:, 0], np.diag(weights["W_plus"])) +
                            cp.quad_form(vb_dyn_minus[:, 0], np.diag(weights["W_minus"]))
                        )
                    elif buff_dyn == 'quad-2':
                        VIRTUAL_COST += (
                            cp.quad_form(vb_dyn_plus[:, k], np.diag(weights["W_plus"][:, k])) +
                            cp.quad_form(vb_dyn_minus[:, k], np.diag(weights["W_minus"][:, k]))
                        )

    # --- DUAL COST ---
    DUAL_COST = 0
    if flag_autotune in {'1', '3', 'al-scvx'}:
        for k in range(N):
            if params["n_ineq"] > 0:
                DUAL_COST += cp.sum(cp.multiply(
                    cp.hstack([vb_path[:, k], vb_nfz[:, k], vb_aux[:, k]]),
                    cp.hstack([dual_vars["dual_path"][:, k],
                               dual_vars["dual_nfz"][:, k],
                               dual_vars["dual_aux"][:, k]])
                ))

            if params["n_eq"] > 0 and k < N - 1:
                DUAL_COST += cp.sum(cp.multiply(
                    vb_dyn_plus[:, k] - vb_dyn_minus[:, k],
                    dual_vars["dual_dyn"][:, k]
                ))
                DUAL_COST += (
                    dual_vars["dual_plus"][:, k] @ vb_dyn_plus[:, k] +
                    dual_vars["dual_minus"][:, k] @ vb_dyn_minus[:, k]
                )

        if params["n_term"] > 0:
            DUAL_COST += dual_vars["dual_term"].T @ vb_term

    # --- TOTAL COST ---
    PTR_COST = TRUE_COST + 0.5 * VIRTUAL_COST + DUAL_COST + TR_COST

    return PTR_COST

def baseline_autotune(flag_autotune):
    match flag_autotune:
        case 1:
            hp.autotune1()
        case 2:
            hp.autotune2()
        case 3:
            hp.autotune3()


def baseline_subprob_outputs(solution, inputs, solution_vars, model_data):
    dz = solution_vars['dz']
    du = solution_vars['du']
    dt = solution_vars['dt']
    vb_path = solution_vars['vb_path']
    vb_nfz = solution_vars['vb_nfz']
    vb_aux = solution_vars['vb_aux']
    vb_dyn_plus = solution_vars['vb_dyn_plus']
    vb_dyn_minus = solution_vars['vb_dyn_minus']
    vb_term = solution_vars['vb_term']
    vb_plus = solution_vars['vb_plus']
    vb_minus = solution_vars['vb_minus']

    zs_ref = inputs['reference_data']['zs_ref']
    us_ref = inputs['reference_data']['us_ref']
    dts_ref = inputs['reference_data']['dts_ref']
    ts_ref = inputs['reference_data']['ts_ref']

    wtr_z = model_data['wtr_z']
    wtr_u = model_data['wtr_u']
    zs_minus = model_data['zs_minus']

    # Extract primal variables
    dz_val = dz.value
    du_val = du.value
    dt_val = dt.value

    O = {}

    # Primal recovered solution
    O["dz_s"] = dz_val
    O["du_s"] = du_val
    O["zs"] = dz_val + zs_ref
    O["us"] = du_val + us_ref
    O["dts"] = dt_val + dts_ref
    O["ts"] = np.concatenate(([0], np.cumsum(O["dts"])))
    O["Ts"] = np.sum(O["dts"])

    # Reference data
    O["zs_ref"] = zs_ref
    O["us_ref"] = us_ref
    O["dts_ref"] = dts_ref
    O["ts_ref"] = ts_ref

    # Hyperparameter weights
    O["weights"] = model_data["weights"]
    O["weights_ref"] = model_data["weights"]

    # Convergence data
    conv = {}
    conv["soln"] = soln_stats["soln_object"]

    conv["vb_path"] = vb_path.value
    conv["vb_nfz"] = vb_nfz.value
    conv["vb_aux"] = vb_aux.value
    conv["vb_term"] = vb_term.value
    conv["vb_dyn"] = vb_dyn_plus.value - vb_dyn_minus.value

    conv["defect"] = dz_val + zs_ref - zs_minus

    conv["Jtr"] = (wtr_z * np.sum(dz_val**2) + wtr_u * np.sum(du_val**2))

    O["conv_data"] = conv

    # Solve timing
    O["solve_time"] = soln_stats["solntime"] * 1000
    O["parse_time"] = soln_stats["parse_time"] * 1000
    O["prop_time"] = model_data["prop_time"] * 1000

    # Discretization model
    O["zs_minus"] = zs_minus
    O["Ak"] = model_data["Ak"]
    O["Bk"] = model_data["Bk"]
    O["Bkp"] = model_data["Bkp"]
    O["Sk"] = model_data["Sk"]

    # Path constraint residuals
    _, _, O["cnst_path"] = problem.compute_path_constraints_fn(O["ts"], O["zs"], O["us"])

    # Total cost via user-defined cost function (if available)
    if hasattr(problem, 'cost_fn'):
        O["cost"] = problem.cost_fn(O["ts"], O["zs"], O["us"])
        O["conv_data"]["cost_ref"] = problem.cost_fn(ts_ref, zs_ref, us_ref)

    return O

def display_baseline_subprob_status(O, problem, nt, ncost):
    """
    Print formatted diagnostic summary for a subproblem solve.

    Args:
        O: dict containing subproblem output data
        problem: an object containing at least I (iteration info)
        nt: multiplier for time of flight (e.g., number of time steps)
        ncost: scaling factor for total cost
    """

    conv = O["conv_data"]

    # Determine inequality feasibility measure
    chk_feas_path = conv.get("chk_feas_path", 0.0)
    chk_feas_nfz = conv.get("chk_feas_nfz", 0.0)
    ineq_vb = chk_feas_path + chk_feas_nfz if chk_feas_nfz != 0 else chk_feas_path

    # Extract values with safe fallback
    chk_dz = conv.get("chk_dz", 1e-12)
    chk_feas_term = conv.get("chk_feas_term", 1e-12)
    chk_feas_dyn = conv.get("chk_feas_dyn", 1e-12)

    log_dz = np.log10(max(chk_dz, 1e-12))
    log_vb_ineq = np.log10(max(ineq_vb, 1e-12))
    log_vb_term = np.log10(max(chk_feas_term, 1e-12))
    log_vb_dyn = np.log10(max(chk_feas_dyn, 1e-12))

    solve_stat = conv.get("status", "UNKNOWN")
    iter_num = len(getattr(problem, "I", []))

    Ts = O.get("Ts", 0.0)
    cost = O.get("cost", 0.0)

    print(
        "     {:02d}     |    {:07.1f}   |   {:06.1f}  |   {:06.1f}   |   {:+04.1f}    |      {:+05.1f}      |    {:+05.1f}    |     {:+05.1f}   |    {:s}    |   {:4.2f}   |  {:4.1f}".format(
            iter_num,
            O["prop_time"],
            O["solve_time"],
            O["parse_time"],
            log_dz,
            log_vb_ineq,
            log_vb_term,
            log_vb_dyn,
            solve_stat,
            Ts * nt,
            cost * ncost
        )
    )



### UNIT TEST

def main():

    # Define minimal dummy problem input structure compatible with updated solve_subproblem
    N, n, m = 5, 3, 2
    dummy_problem = {
        "I": [{
            "iter_num": 0,
            "zs_ref": np.zeros((n, N)),
            "us_ref": np.zeros((m, N)),
            "dts_ref": np.ones(N - 1),
            "ts_ref": np.linspace(0, 1, N),
            "conv_data": {
                "vb_path": np.zeros((1, N)),
                "vb_nfz": np.zeros((1, N)),
                "vb_aux": np.zeros((1, N)),
                "vb_dyn": np.zeros((n, N - 1)),
                "vb_term": np.zeros((2, 1)),
            },
            "weights": {
                "W_path": np.ones((1, N)),
                "W_nfz": np.ones((1, N)),
                "W_aux": np.ones((1, N)),
                "W_dyn": np.ones((n, N - 1)),
                "W_term": np.ones((2, 1)),
                "W_plus": np.ones((n,)),
                "W_minus": np.ones((n,)),
                "w_cost": 1.0,
                "wtr_z": 1.0,
                "wtr_u": 1.0,
                "dual_path": np.ones((1, N)),
                "dual_nfz": np.ones((1, N)),
                "dual_aux": np.ones((1, N)),
                "dual_dyn": np.ones((n, N - 1)),
                "dual_plus": np.ones((n, N - 1)),
                "dual_minus": np.ones((n, N - 1)),
                "dual_term": np.ones((2, 1)),
            }
        }],
        "params": {
            "N": N, "n": n, "m": m, "nz": n,
            "nondim": {"nt": 1, "nd": 1, "nv": 1, "na": 1, "ncost": 1},
            "n_state": n, "n_ctrl": m, "n_udot": 1,
            "n_path": 1, "n_nfz": 1, "n_aux": 1, "n_dyn": n,
            "n_term": 1, "n_term_ineq": 1,
            "z_min": np.zeros(n), "z_max": np.ones(n),
            "u_min": np.zeros(m), "u_max": np.ones(m),
            "udot_max": np.ones(m), "udot_max_idx": [0, 1],
            "z_min_idx": list(range(n)), "z_max_idx": list(range(n)),
            "u_min_idx": list(range(m)), "u_max_idx": list(range(m)),
            "z1": np.zeros(n), "z1_idx": list(range(n)),
            "z1_min": np.zeros(n), "z1_min_idx": list(range(n)),
            "z1_max": np.ones(n), "z1_max_idx": list(range(n)),
            "zN": np.zeros(n), "zN_idx": list(range(n)),
            "zN_min": np.zeros(n), "zN_min_idx": list(range(n)),
            "zN_max": np.ones(n), "zN_max_idx": list(range(n)),
            "n_init": 1, "n_init_ineq": 1,
            "bools": {
                "init_ctrl": True,
                "free_final_time": False,
                "equal_dt": True,
                "ctcs": False,
                "flag_autotune": "0",
                "buff_dyn": "term"
            },
            "eps_ctcs": 1e-3,
            "yalmip_opts": {"solver": "OSQP"},
        },
        "zi": np.zeros(n),
        "zi_idx": list(range(n)),
        "zi_min": np.zeros(n),
        "zi_min_idx": list(range(n)),
        "zi_max": np.ones(n),
        "zi_max_idx": list(range(n)),
        "zf": np.zeros(n),
        "zf_idx": list(range(n)),
        "zf_min": np.zeros(n),
        "zf_min_idx": list(range(n)),
        "zf_max": np.ones(n),
        "zf_max_idx": list(range(n)),
    }

    try:
        output = solve_subproblem(dummy_problem)
        print("Solve completed.")
        print("Final cost:", output.get("cost", "N/A"))
        print("Final state (zs):", output.get("zs", "N/A")[:, -1])
        print("Final control (us):", output.get("us", "N/A")[:, -1])
        print("Ts:", output.get("Ts", "N/A"))
    except Exception as e:
        print("Solve failed:", e)

if __name__ == "__main__":
    main()
    
    "Main function updated successfully."


