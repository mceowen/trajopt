# recall: pip install -e /Users/skye/ACL/entry/python/hypersonic_entry_opt/trajopt

# basic imports
import cvxpy as cp
import numpy as np
import time

# trajopt imports
import trajopt.problem_models.quadrotor_3dof    as quad3dof
import trajopt.algorithm.hyperparameters        as hp
import trajopt.algorithm.scaling                as scaling
import trajopt.algorithm.convexification        as convexify
import trajopt.algorithm.discretization         as discretize
import trajopt.algorithm.convergence            as convergence
import trajopt.utils.tools                      as tools


def solve_subproblem(problem):

    local_vars = baseline_subprob_inputs(problem)
    problem['custom_inputs'](problem, local_vars)

    local_vars = baseline_subprob_variables(problem,local_vars)
    problem['custom_variables'](problem, local_vars)

    constraints = baseline_subprob_constraints(problem, local_vars)
    constraints = problem['custom_constraints'](constraints, local_vars)

    PTR_COST    = 0.0
    PTR_COST    = baseline_subprob_cost(problem, local_vars)
    PTR_COST    = problem['custom_cost'](PTR_COST, local_vars)

    # TODO(Skye): vectorize cost computation for speedup
    objective   = cp.Minimize(PTR_COST)
    subprob     = cp.Problem(objective, constraints)
    subprob.solve(solver=problem['params']['solver_opts']['solver'])

    O = baseline_subprob_outputs(
        problem,
        local_vars,
        subprob,
    )

    # TODO: Add custom outputs
    # problem.custom_outputs(problem, local_vars, O)

    O = convergence.check_convergence_tolerance(problem, local_vars, O)

    O = baseline_autotune(problem, local_vars, O)
    
    display_baseline_subprob_status(problem, local_vars, O)

    return O

def baseline_subprob_inputs(problem):

    I               = problem['I'][-1]
    iter_num        = I['iter_num']
    case_flag       = problem['params']['case_flag']

    # Dynamics and cost
    start                   = time.time()
    Ak, Bk, Bkp, Sk, zs_minus = discretize.compute_linsys_discrete(I['zs_ref'], I['us_ref'], I['dts_ref'], problem)
    prop_time               = time.time() - start

    dcostdz, dcostdu, cost  = convexify.compute_cost(I['ts_ref'], I['zs_ref'], I['us_ref'], problem)
    dgdz, dgdu, g           = convexify.compute_path_constraints(I['ts_ref'], I['zs_ref'], I['us_ref'], problem)

    # Reference trajectories
    ts_ref          = I['ts_ref']
    dts_ref         = I['dts_ref']
    zs_ref          = I['zs_ref']
    us_ref          = I['us_ref']

    vb_path_ref     = I['conv_data']['vb_path']
    vb_nfz_ref      = I['conv_data']['vb_nfz']
    vb_aux_ref      = I['conv_data']['vb_aux']
    vb_dyn_ref      = I['conv_data']['vb_dyn']
    vb_term_ref     = I['conv_data']['vb_term']

    # Dimensions and problem parameters
    params          = problem['params']
    N               = params['N']
    n               = params['n']
    m               = params['m']
    nz              = params['nz']

    nt              = params['nondim']['nt']
    nd              = params['nondim']['nd']
    nv              = params['nondim']['nv']
    na              = params['nondim']['na']
    ncost           = params['nondim']['ncost']

    # Boundary conditions
    z1              = problem['zi']
    z1_idx          = problem['zi_idx']
    z1_min          = problem['zi_min']
    z1_min_idx      = problem['zi_min_idx']
    z1_max          = problem['zi_max']
    z1_max_idx      = problem['zi_max_idx']
    n_init          = problem['n_init']
    n_init_ineq     = problem['n_init_ineq']

    zN              = problem['zf']
    zN_idx          = problem['zf_idx']
    zN_min          = problem['zf_min']
    zN_min_idx      = problem['zf_min_idx']
    zN_max          = problem['zf_max']
    zN_max_idx      = problem['zf_max_idx']
    n_term          = problem['n_term']
    n_term_ineq     = problem['n_term_ineq']

    vb_N_idx = list(range(n_term))
    vb_N_ineq_idx   = list(range(n_term, n_term + n_term_ineq))

    # State and control constraints
    z_min           = problem['z_min']
    z_min_idx       = problem['z_min_idx']
    z_max           = problem['z_max']
    z_max_idx       = problem['z_max_idx']
    n_state         = params['n_state']

    u_min           = problem['u_min']
    u_min_idx       = problem['u_min_idx']
    u_max           = problem['u_max']
    u_max_idx       = problem['u_max_idx']
    n_ctrl          = problem['n_ctrl']
    udot_max        = problem['udot_max']
    udot_max_idx    = problem['udot_max_idx']
    n_udot          = problem['n_udot']
    bool_init_ctrl  = params['bools']['init_ctrl']

    # Time settings
    bools           = params['bools']
    free_final_time = bools['free_final_time']
    equal_dt_bool   = bools['equal_dt']

    dts_min = dts_max = ddts_max = None
    if free_final_time:
        dts_min     = params['dts_min']
        dts_max     = params['dts_max']
        ddts_max    = params['ddts_max']

    # Contact constraints
    ctcs            = bools['ctcs']
    eps_ctcs        = params['eps_ctcs']

    # Constraint structure
    n_path          = params['n_path']
    n_nfz           = params['n_nfz']
    n_aux           = params['n_aux']
    n_ineq          = n_path + n_nfz + n_aux
    n_dyn           = params['n_dyn']
    n_eq            = n_dyn

    # Weighting and duals
    flag_autotune   = bools['flag_autotune']
    buff_dyn        = bools['buff_dyn']
    buff_dyn_dual   = bools.get('buff_dyn_dual', None)

    weights         = I['weights']
    W_path          = weights['W_path']
    W_nfz           = weights['W_nfz']
    W_aux           = weights['W_aux']
    W_dyn           = weights['W_dyn']
    W_term          = weights['W_term']
    W_plus          = weights['W_plus']
    W_minus         = weights['W_minus']

    w_cost          = weights['w_cost']
    wtr_z           = weights['wtr_z']
    wtr_u           = weights['wtr_u']

    dual_path       = weights['dual_path']
    dual_nfz        = weights['dual_nfz']
    dual_aux        = weights['dual_aux']
    dual_dyn        = weights['dual_dyn']
    dual_plus       = weights['dual_plus']
    dual_minus      = weights['dual_minus']
    dual_term       = weights['dual_term']

    opts            = params['solver_opts']

    local_vars = dict(locals())

    return local_vars


def baseline_subprob_variables(problem, local_vars):
    bools   = local_vars['bools']
    N       = local_vars['N']

    # Variation in state and control
    dz, du = scaling.subprob_variable_scaling(problem, local_vars)

    # Time-step or time-horizon dilation
    if bools['free_final_time']:
        if not bools['equal_dt']:
            dt = cp.Variable((N-1,1))  # Variable timestep
        else:
            dT = cp.Variable()       # Time horizon scalar
            dt = (1 / (N - 1)) * dT * np.ones((N - 1,1))
    else:
        dt = np.zeros((N - 1,1))         # Fixed timestep (not a variable)

    # Virtual buffer and virtual control variables
    vb_path,vb_nfz,vb_aux,vb_term,vb_dyn_plus,vb_dyn_minus,vb_plus,vb_minus = subprob_virtual_variables(problem, local_vars)

    local_vars['sol_vars'] = {
        'dz': dz,
        'du': du,
        'dt': dt,
        'vb_path': vb_path,
        'vb_nfz': vb_nfz,
        'vb_aux': vb_aux,
        'vb_term': vb_term,
        'vb_dyn_plus': vb_dyn_plus,
        'vb_dyn_minus': vb_dyn_minus,
        'vb_plus': vb_plus,
        'vb_minus': vb_minus,
    }

    return local_vars

def subprob_virtual_variables(problem, local_vars):
    """
    Determines which virtual variables (virtual buffer and virtual control) to
    create for subproblem constraints. If the weight associated with each
    term is zero, then the variables are set to the zero vector.
    """

    # Extract autotune flag and dynamics buffer toggle
    flag_autotune   = problem['params']['bools']['flag_autotune']
    buff_dyn        = problem['params']['bools']['buff_dyn']

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
    weights     = problem['I'][-1]['weights']
    wght_keys   = weights.keys()
    W_keys      = [k for k in wght_keys if 'W' in k]
    dual_keys   = [k for k in wght_keys if 'dual' in k]

    vb_vals     = {}

    if str(flag_autotune) in {'0','2','3','al-scvx'}:
        vb_tags = [k.split('_')[-1] for k in W_keys]

        for tag in vb_tags:
            W = weights.get(f'W_{tag}', None)
            if W is not None and np.sum(W) != 0:
                vb_vals[tag] = cp.Variable((N[tag], n[tag]))
            else:
                vb_vals[tag] = np.zeros((N[tag], n[tag]))

    elif str(flag_autotune) == '1':
        vb_tags = [k.split('_')[-1] for k in dual_keys if 'eq' not in k]

        for tag in vb_tags:
            dual            = weights.get(f'dual_{tag}', None)
            is_dyn_excluded = (tag == 'dyn') and (buff_dyn == 'term')
            if dual is not None and np.sum(dual) != 0 and not is_dyn_excluded:
                vb_vals[tag] = cp.Variable((N[tag], n[tag]))
            else:
                vb_vals[tag] = np.zeros((N[tag], n[tag]))

    # Dynamics plus/minus buffers depend on buff_dyn
    if buff_dyn == 'term':
        vb_dyn_plus     = np.zeros((N['dyn'],n['dyn']))
        vb_dyn_minus    = np.zeros((N['dyn'],n['dyn']))
    else:
        vb_dyn_plus     = cp.Variable((N['dyn'],n['dyn']))
        vb_dyn_minus    = cp.Variable((N['dyn'],n['dyn']))

    # Optional plus/minus aggregate buffers (e.g., for quad-1, quad-2)
    if n['plus'] != 0:
        vb_plus         = cp.Variable((N['plus'], n['plus']))
        vb_minus        = cp.Variable((N['minus'], n['minus']))
    else:
        vb_plus         = np.zeros((N['plus'], n['plus']))
        vb_minus        = np.zeros((N['minus'], n['minus']))

    return (
        vb_vals.get('path', np.zeros((N['path'], n['path']))),
        vb_vals.get('nfz',  np.zeros(( N['nfz'], n['nfz']))),
        vb_vals.get('aux',  np.zeros((N['aux'], n['aux']))),
        vb_vals.get('term', np.zeros((N['term'], n['term']))).flatten(order='C'),
        vb_dyn_plus,
        vb_dyn_minus,
        vb_plus,
        vb_minus
    )


def baseline_subprob_constraints(problem,local_vars):
    dz                  = local_vars['sol_vars']["dz"]
    du                  = local_vars['sol_vars']["du"]
    dt                  = local_vars['sol_vars']["dt"]
    vb_path             = local_vars['sol_vars']["vb_path"]
    vb_nfz              = local_vars['sol_vars']["vb_nfz"]
    vb_aux              = local_vars['sol_vars']["vb_aux"]
    vb_term             = local_vars['sol_vars']["vb_term"]
    vb_dyn_plus         = local_vars['sol_vars']["vb_dyn_plus"]
    vb_dyn_minus        = local_vars['sol_vars']["vb_dyn_minus"]
    vb_plus             = local_vars['sol_vars']["vb_plus"]
    vb_minus            = local_vars['sol_vars']["vb_minus"]

    I                   = local_vars["I"]
    params              = local_vars["params"]
    Ak                  = local_vars["Ak"]
    Bk                  = local_vars["Bk"]
    Bkp                 = local_vars["Bkp"]
    Sk                  = local_vars["Sk"]
    zs_minus            = local_vars["zs_minus"]
    dgdz                = local_vars["dgdz"]
    dgdu                = local_vars["dgdu"]
    g                   = local_vars["g"]

    CNST                = []
    bools               = params['bools']
    flag_autotune       = bools['flag_autotune']
    buff_dyn            = bools['buff_dyn']

    # Extract dimensions and references
    N                   = local_vars['N']
    n                   = local_vars['n']
    m                   = local_vars['m']
    nz                  = local_vars['nz']
    n_state             = local_vars['n_state']
    n_ctrl              = local_vars['n_ctrl']
    n_udot              = local_vars['n_udot']
    n_ineq              = local_vars['n_path'] + local_vars['n_nfz'] + local_vars['n_aux']

    z1, z1_idx          = local_vars['z1'],     local_vars['z1_idx']
    z1_min, z1_min_idx  = local_vars['z1_min'], local_vars['z1_min_idx']
    z1_max, z1_max_idx  = local_vars['z1_max'], local_vars['z1_max_idx']
    zN, zN_idx          = local_vars['zN'],     local_vars['zN_idx']
    zN_min, zN_min_idx  = local_vars['zN_min'], local_vars['zN_min_idx']
    zN_max, zN_max_idx  = local_vars['zN_max'], local_vars['zN_max_idx']

    vb_N_idx            = list(range(params['n_term']))
    vb_N_ineq_idx       = list(range(params['n_term'], params['n_term'] + params['n_term_ineq']))

    z_min, z_min_idx    = local_vars['z_min'], local_vars['z_min_idx']
    z_max, z_max_idx    = local_vars['z_max'], local_vars['z_max_idx']
    u_min, u_min_idx    = local_vars['u_min'], local_vars['u_min_idx']
    u_max, u_max_idx    = local_vars['u_max'], local_vars['u_max_idx']

    udot_max, udot_max_idx = local_vars['udot_max'], local_vars['udot_max_idx']
    dts_ref             = I['dts_ref']
    zs_ref              = I['zs_ref']
    us_ref              = I['us_ref']

    eps_ctcs            = local_vars['eps_ctcs']
    dts_min, dts_max, ddts_max = local_vars['dts_min'], local_vars['dts_max'], local_vars['ddts_max']

    # Initial control
    if bools['init_ctrl']:
        CNST.append(du[:, 0] == 0)

    # Initial state 
    if params['n_init'] > 0:
        CNST.append(dz[0,z1_idx] + zs_ref[0,z1_idx] == z1)

    if params['n_init_ineq'] > 0:
        M_sel = tools.constraint_index_selector(z1_min_idx, z1_max_idx, n)
        CNST.append(M_sel @ (dz[0,:n] + zs_ref[0,:n]) <= np.concatenate([-z1_min, z1_max]))

    # Terminal state
    if params['n_term'] > 0:
        CNST.append(dz[-1,zN_idx] + zs_ref[-1,zN_idx] - vb_term[vb_N_idx] == zN[zN_idx])

    if params['n_term_ineq'] > 0:
        M_sel = tools.constraint_index_selector(zN_min_idx, zN_max_idx, n)
        CNST.append(M_sel @ (dz[-1, :n] + zs_ref[-1, :n]) - vb_term[vb_N_ineq_idx] <= np.concatenate([-zN_min, zN_max]))

    if buff_dyn == 'quad-1':
        CNST.append(cp.sum(cp.vec(vb_dyn_plus, order='F')) == vb_plus)
        CNST.append(cp.sum(cp.vec(vb_dyn_minus, order='F')) == vb_minus)

    elif buff_dyn == 'quad-3':
        for j in range(nz):
            CNST.append(cp.sum(vb_dyn_plus[:,j]) == vb_plus[:,j])
            CNST.append(cp.sum(vb_dyn_minus[:,j]) == vb_minus[:,j])

    for k in range(N):
        if k < N - 1:
            # Dynamics
            lhs = dz[k+1] + zs_ref[k + 1] - zs_minus[k + 1]
            rhs = (Ak[k] @ dz[k] + Bk[k] @ du[k] +
                   Bkp[k] @ du[k+1] + Sk[k] * dt[k] +
                   vb_dyn_plus[k] - vb_dyn_minus[k])
            CNST.append(lhs == rhs)

            if buff_dyn != 'term':
                CNST.append(vb_dyn_plus[k] >= 0)
                CNST.append(vb_dyn_minus[k] >= 0)

            if buff_dyn == 'quad-2':
                CNST.append(cp.sum(vb_dyn_plus[k])  == vb_plus[k])
                CNST.append(cp.sum(vb_dyn_minus[k]) == vb_minus[k])

            if bools['ctcs'] and not n == nz:
                CNST.append(zs_ref[k + 1, n:nz] + dz[k + 1, n:nz] - (zs_ref[k,n:nz] + dz[k,n:nz]) <= eps_ctcs)

            if bools['free_final_time']:
                CNST.append(dts_ref[k] + dt[k] <= dts_max)
                CNST.append(dts_ref[k] + dt[k] >= dts_min)
                CNST.append(cp.abs(dt[k]) <= ddts_max)

            if n_udot > 0 and k < N - 2:
                M_sel = tools.constraint_index_selector(udot_max_idx, udot_max_idx, m)
                CNST.append(M_sel @ (us_ref[k + 1] + du[k + 1] - (us_ref[k] + du[k])) <= (dts_ref[k] + dt[k]) * np.concatenate([udot_max, udot_max]))

        # State constraints
        if n_state > 0:
            M_sel = tools.constraint_index_selector(z_min_idx, z_max_idx, n)
            CNST.append(M_sel @ (zs_ref[k,:n] + dz[k,:n]) <= np.concatenate([-z_min, z_max]))

        # Control constraints
        if n_ctrl > 0:
            M_sel = tools.constraint_index_selector(u_min_idx, u_max_idx, m)
            CNST.append(M_sel @ (us_ref[k] + du[k]) <= np.concatenate([-u_min, u_max]))

        # Linearized inequality constraints
        if n_ineq > 0 and not bools['ctcs']:
            vb_combined = cp.vstack([
                    x for x in [vb_path[k], vb_nfz[k], vb_aux[k]]
                    if getattr(x, "shape", (0,))[0] > 0
                ])
            CNST.append(dgdz[k] @ dz[k] + dgdu[k] @ du[k] + g[k] - vb_combined <= 0)

            if str(flag_autotune) in {'1', '3', 'al-scvx'}:
                CNST.append(vb_combined >= 0)
    
    return CNST


def baseline_subprob_cost(problem, local_vars):
    dz          = local_vars['sol_vars']["dz"]
    du          = local_vars['sol_vars']["du"]
    dt          = local_vars['sol_vars']["dt"]
    vb_path     = local_vars['sol_vars']["vb_path"]
    vb_nfz      = local_vars['sol_vars']["vb_nfz"]
    vb_aux      = local_vars['sol_vars']["vb_aux"]
    vb_term     = local_vars['sol_vars']["vb_term"]
    vb_dyn_plus = local_vars['sol_vars']["vb_dyn_plus"]
    vb_dyn_minus= local_vars['sol_vars']["vb_dyn_minus"]
    vb_plus     = local_vars['sol_vars']["vb_plus"]
    vb_minus    = local_vars['sol_vars']["vb_minus"]

    params        = local_vars["params"]
    N             = params["N"]
    n             = params['n']
    solver_type   = params.get("solver_type", "osqp")
    flag_autotune = local_vars['bools']["flag_autotune"]
    buff_dyn      = local_vars['bools']["buff_dyn"]

    dcostdz = local_vars["dcostdz"][:,0,:]    # (N,n)
    dcostdu = local_vars["dcostdu"][:,0,:]    # (N,m)
    cost    = local_vars["cost"][:,0,0]       # (N,)
    w_cost  = local_vars["w_cost"]

    # ---- TRUE COST (vectorized) ----
    # Stack along first dimension then use cp.sum directly.
    TRUE_COST = w_cost * (
        cp.sum(cp.multiply(dcostdz, dz[:, :n])) +
        cp.sum(cp.multiply(dcostdu, du)) +
        cp.sum(cost)
    )

    # ---- TRUST REGION COST (vectorized) ----
    TR_COST = (
        local_vars["wtr_z"] * cp.sum(cp.sum_squares(dz)) +
        local_vars["wtr_u"] * cp.sum(cp.sum_squares(du))
    )

    # ---- VIRTUAL COST ----
    VIRTUAL_COST = 0.0
    if flag_autotune in {'0', '2', '3', 'al-scvx'}:
        W_term = np.diag(local_vars["W_term"])
        VIRTUAL_COST += cp.quad_form(vb_term, W_term)

        if buff_dyn == 'l1':
            for vb_mat, W in zip([vb_path, vb_nfz, vb_aux],
                                 [local_vars["W_path"], local_vars["W_nfz"], local_vars["W_aux"]]):
                if vb_mat.shape[1] > 0:
                    weights = np.max(W, axis=1)
                    VIRTUAL_COST += cp.sum(cp.multiply(weights, cp.norm(vb_mat, 1, axis=1)))

            if vb_dyn_plus.shape[1] > 0:
                w_dyn = np.max(local_vars["W_dyn"], axis=1)
                diff  = vb_dyn_plus - vb_dyn_minus
                VIRTUAL_COST += cp.sum(cp.multiply(w_dyn, cp.norm(diff, 1, axis=1)))

        else:
            for vb_mat, W in zip([vb_path, vb_nfz, vb_aux],
                                 [local_vars["W_path"], local_vars["W_nfz"], local_vars["W_aux"]]):
                if vb_mat.shape[1] > 0:
                    W_diag = np.array([np.diag(Wk.flatten(order='C')) for Wk in W])
                    # Vectorized quad_form equivalent: sum_i (x_i.T * D_i * x_i)
                    VIRTUAL_COST += cp.sum(cp.sum(cp.multiply(vb_mat**2, np.array([np.diag(Wi) for Wi in W_diag]))))

            if buff_dyn == 'l2' and vb_dyn_plus.shape[1] > 0:
                diff = vb_dyn_plus - vb_dyn_minus
                W_dyn = np.array([np.diag(Wk.flatten(order='C')) for Wk in local_vars["W_dyn"]])
                VIRTUAL_COST += cp.sum(cp.sum(cp.multiply(diff**2, np.array([np.diag(Wi) for Wi in W_dyn]))))
            elif buff_dyn == 'quad-1':
                if vb_plus.shape[1] > 0:
                    VIRTUAL_COST += cp.quad_form(vb_plus[0], np.diag(local_vars["W_plus"].flatten(order='C')))
                if vb_minus.shape[1] > 0:
                    VIRTUAL_COST += cp.quad_form(vb_minus[0], np.diag(local_vars["W_minus"].flatten(order='C')))
            elif buff_dyn == 'quad-2':
                if vb_plus.shape[1] > 0:
                    VIRTUAL_COST += cp.sum([
                        cp.quad_form(vb_plus[k], np.diag(local_vars["W_plus"][k]))
                        for k in range(vb_plus.shape[1])
                    ])
                if vb_minus.shape[1] > 0:
                    VIRTUAL_COST += cp.sum([
                        cp.quad_form(vb_minus[k], np.diag(local_vars["W_minus"][k]))
                        for k in range(vb_minus.shape[1])
                    ])

    # ---- DUAL COST ----
    DUAL_COST = 0.0
    if flag_autotune in {'1', '3', 'al-scvx'}:
        n_ineq = params["n_ineq"]
        n_eq   = params["n_eq"]
        n_term = params["n_term"]

        if n_ineq > 0:
            for k in range(N):
                stack_v = cp.hstack([v for v in [vb_path[k], vb_nfz[k], vb_aux[k]] if v.size > 0])
                if stack_v.size > 0:
                    duals = cp.hstack([d for v, d in zip(
                        [vb_path[k], vb_nfz[k], vb_aux[k]],
                        [local_vars["dual_path"][k], local_vars["dual_nfz"][k], local_vars["dual_aux"][k]]
                    ) if v.size > 0])
                    DUAL_COST += cp.sum(cp.multiply(stack_v, duals))

        if n_eq > 0:
            diff = vb_dyn_plus - vb_dyn_minus
            DUAL_COST += cp.sum(cp.multiply(diff, local_vars["dual_dyn"][:N-1]))
            DUAL_COST += cp.sum(cp.multiply(vb_plus[:N-1], local_vars["dual_plus"][:N-1]))
            DUAL_COST += cp.sum(cp.multiply(vb_minus[:N-1], local_vars["dual_minus"][:N-1]))

        if n_term > 0:
            DUAL_COST += local_vars["dual_term"].T @ vb_term

    # ---- TOTAL COST ----
    PTR_COST = TRUE_COST + 0.5 * VIRTUAL_COST + DUAL_COST + TR_COST
    return PTR_COST


def baseline_autotune(problem, local_vars, O):
    match problem['params']['bools']['flag_autotune']:
        case 1:
            O = hp.autotune1(problem, local_vars, O)
        case 2:
            O = hp.autotune2(problem, local_vars, O)
        case 3:
            O = hp.autotune3(problem, local_vars, O)
    return O


def baseline_subprob_outputs(problem, local_vars, subprob):

    N = problem['params']['N']
    n = problem['params']['n']
    m = problem['params']['m']

    dz          = local_vars['sol_vars']['dz']
    du          = local_vars['sol_vars']['du']
    dt          = local_vars['sol_vars']['dt']
    vb_path     = local_vars['sol_vars']['vb_path']
    vb_nfz      = local_vars['sol_vars']['vb_nfz']
    vb_aux      = local_vars['sol_vars']['vb_aux']
    vb_dyn_plus = local_vars['sol_vars']['vb_dyn_plus']
    vb_dyn_minus= local_vars['sol_vars']['vb_dyn_minus']
    vb_term     = local_vars['sol_vars']['vb_term']
    vb_plus     = local_vars['sol_vars']['vb_plus']
    vb_minus    = local_vars['sol_vars']['vb_minus']

    zs_ref      = local_vars['zs_ref']
    us_ref      = local_vars['us_ref']
    dts_ref     = local_vars['dts_ref']
    ts_ref      = local_vars['ts_ref']

    wtr_z       = local_vars['wtr_z']
    wtr_u       = local_vars['wtr_u']
    zs_minus    = local_vars['zs_minus']

    # dimensions
    n_path = local_vars["n_path"]
    n_nfz  = local_vars["n_nfz"]
    n_aux  = local_vars["n_aux"]
    n_term = local_vars["n_term"]
    n_dyn  = local_vars["n_dyn"]

    # Extract primal variables
    dz_val      = dz.value
    du_val      = du.value
    dt_val      = dt.value if local_vars['bools']['free_final_time'] else dt

    if subprob.solver_stats is not None:
        soln_stats = {
            "solntime":     subprob.solver_stats.solve_time,
            "parse_time":   subprob.solver_stats.setup_time,
            "soln_object":  subprob
        }
    else:
        soln_stats = {
            "solntime":     None,
            "parse_time":   None,
            "soln_object":  subprob
        }

    O = {}

    # solver status
    O["subprob"]         = subprob

    # Primal recovered solution
    O["dz_s"]           = dz_val
    O["du_s"]           = du_val
    O['dt_s']           = dt_val
    O["zs"]             = tools.safe_val(dz_val, rows=N, cols=n) + zs_ref
    O["us"]             = tools.safe_val(du_val, rows=N, cols=m) + us_ref
    O["dts"]            = np.squeeze( tools.safe_val(dt_val) )  + dts_ref
    O["ts"]             = np.concatenate(([0], np.cumsum(O["dts"])))
    O["Ts"]             = np.sum(O["dts"])

    # Reference data
    O["zs_ref"]         = zs_ref
    O["us_ref"]         = us_ref
    O["dts_ref"]        = dts_ref
    O["ts_ref"]         = ts_ref

    # Hyperparameter weights
    O["weights"]        = problem['params']["weights"]
    O["weights_ref"]    = problem['params']["weights"]

    # Convergence data
    conv                = {}
    conv["soln"]        = soln_stats["soln_object"]

    conv["vb_path"]     = tools.get_val(vb_path,    rows=N, cols=n_path) 
    conv["vb_nfz"]      = tools.get_val(vb_nfz,     rows=N, cols=n_nfz) 
    conv["vb_aux"]      = tools.get_val(vb_aux,     rows=N, cols=n_aux) 
    conv["vb_term"]     = tools.get_val(vb_term,    rows=1, cols=n_term) 
    conv["vb_dyn"]      = tools.get_val(vb_dyn_plus, rows=N, cols=n_dyn)  - tools.get_val(vb_dyn_minus, cols=n_dyn, rows=N) 

    conv["defect"]      = tools.safe_val(dz, cols=n, rows=N) + zs_ref - zs_minus

    conv["Jtr"]         = (wtr_z * np.sum(tools.safe_val(dz, cols=n, rows=N)**2) + wtr_u * np.sum(tools.safe_val(du, rows=N, cols=m)**2))

    O["conv_data"]      = conv

    # Solve timing
    O["solve_time"]     = tools.safe_val(soln_stats["solntime"]) * 1000
    O["parse_time"]     = tools.safe_val(soln_stats["parse_time"]) * 1000
    O["prop_time"]      = local_vars["prop_time"] * 1000

    # Discretization model
    O["zs_minus"]       = zs_minus
    O["Ak"]             = local_vars["Ak"]
    O["Bk"]             = local_vars["Bk"]
    O["Bkp"]            = local_vars["Bkp"]
    O["Sk"]             = local_vars["Sk"]

    # Path constraint residuals
    _, _, O["cnst_path"]                = convexify.compute_path_constraints(O["ts"], O["zs"], O["us"], problem)

    # Total cost via user-defined cost function (if available)
    O["cost"]                           = convexify.compute_cost(O["ts"], O["zs"], O["us"], problem)[2].sum().item()
    O["conv_data"]["cost_ref"]          = convexify.compute_cost(ts_ref, zs_ref, us_ref, problem)[2].sum().item()

    return O



def display_baseline_subprob_status(problem, local_vars, O):

    conv            = O["conv_data"]

    # Determine inequality feasibility measure
    chk_feas_path   = conv.get("chk_feas_path", 0.0)
    chk_feas_nfz    = conv.get("chk_feas_nfz", 0.0)
    ineq_vb         = chk_feas_path + chk_feas_nfz if chk_feas_nfz != 0 else chk_feas_path

    # Extract values with safe fallback
    chk_dz          = conv.get("chk_dz", 1e-12)
    chk_feas_term   = conv.get("chk_feas_term", 1e-12)
    chk_feas_dyn    = conv.get("chk_feas_dyn", 1e-12)

    log_dz          = np.log10(max(chk_dz, 1e-12))
    log_vb_ineq     = np.log10(max(ineq_vb, 1e-12))
    log_vb_term     = np.log10(max(chk_feas_term, 1e-12))
    log_vb_dyn      = np.log10(max(chk_feas_dyn, 1e-12))

    solve_stat      = conv.get("status", "UNKNOWN")
    iter_num        = problem['I'][-1]['iter_num']
    nt              = local_vars['nt']
    ncost           = local_vars['ncost']

    Ts              = O.get("Ts", 0.0)
    cost            = O.get("cost", 0.0)

    print(
        "     {:02d}     |    {:07.1f}   |   {:06.1f}  |   {:06.1f}   |   {:+04.1f}    |      {:+05.1f}      |    {:+05.1f}    |     {:+05.1f}   |    {:s}    |   {:4.2f}   |  {:4.1f}".format(
            int(iter_num),
            float(O["prop_time"]),
            float(O["solve_time"]),
            float(O["parse_time"]),
            float(log_dz),
            float(log_vb_ineq),
            float(log_vb_term),
            float(log_vb_dyn),
            str(solve_stat),
            float(Ts * nt),
            float(cost * ncost)
        )
    )

### UNIT TEST

if __name__ == "__main__":
    pass


