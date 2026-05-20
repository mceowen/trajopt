import numpy as np
import trajopt.utils.tools as tools

def create_eps_stack(method):

    problem = method.problem
    nondim  = problem.nondim
    n       = problem.index_map.n

    eps_stack = tools.AttrDict()

    # state deviation convergence epsilon
    eps_state = tools.expand_to_array_if_scalar(method.conv_config.eps_state, n.state)
    eps_stack.state = nondim.M.state.d2nd @ eps_state

    # cost change convergence epsilon (optimality)
    eps_stack.cost = np.atleast_1d(method.conv_config.eps_cost)

    # time deviation convergence epsilon
    eps_time = method.conv_config.eps_time
    eps_stack.time = np.asarray(eps_time).reshape(-1) / nondim.time_scale

    # nonlinear dynamics defect convergence epsilon
    eps_defect = tools.expand_to_array_if_scalar(method.conv_config.eps_defect, n.state)
    eps_stack.defect = nondim.M.state.d2nd @ eps_defect

    # set nodal nonconvex inequality constraint tolerances
    if problem.constraints.has(ct=0, type='nonconvex_inequality'):
        eps_stack.nonconvex_inequality = np.concatenate([c.eps for c in problem.constraints.get(ct=0, type='nonconvex_inequality')])

    if problem.constraints.has(type="initial_state"):
        for c in problem.constraints.get(type="initial_state"):
            eps_stack.initial_state = np.concatenate([method.conv_config.eps_term, np.atleast_1d(c.eps)])

    if problem.constraints.has(type="final_state"):
        for c in problem.constraints.get(type="final_state"):
            eps_stack.final_state = np.concatenate([method.conv_config.eps_term, np.atleast_1d(c.eps)])

    eps_stack = augment_convergence_tolerance(method, eps_stack)

    return eps_stack


def augment_convergence_tolerance(method, eps_stack):
    """
    Augment convergence tolerances to the full z space, adding time (z.time
    indices) and CTCS (z.ctcs indices) components to eps_dyn and eps_term.
    """
    idx = method.index_map.indices
    n   = method.index_map.n
    problem = method.problem

    eps_dyn              = np.empty(n.z)
    eps_dyn[idx.z.state] = eps_stack.state
    eps_dyn[idx.z.time]  = eps_stack.time

    if problem.constraints.has(ct=1):
        eps_ctcs = np.concatenate([c.eps for c in problem.constraints.get(ct=1)])
        eps_dyn[idx.z.ctcs]  = (eps_ctcs) * 0.1 * 0.25
        if "initial_state" in eps_stack:
            eps_stack.initial_state = np.concatenate([eps_stack.initial_state, eps_ctcs])
        if "final_state" in eps_stack:
            eps_stack.final_state = np.concatenate([eps_stack.final_state, eps_ctcs])

    eps_stack.dynamics = eps_dyn

    return eps_stack


def check_convergence_tolerance(method):

    # --- Load convergence data
    prev_iter_data = method.iter_data_list[-1]
    current_iter_data = method.current_iter_data

    index_map = method.index_map

    # dz and dcost to measure optimality
    dstate  = current_iter_data.dz[:, index_map.indices.z.state]
    dcost   = current_iter_data.cost - prev_iter_data.cost
    
    # linearized constraint violations
    vb_dyn  = current_iter_data.vb.get("dynamics", 0)
    vb_ineq = current_iter_data.vb.get("nonconvex_inequality", 0)
    vb_term = current_iter_data.vb.get("final_state", 0)

    # nonlinear constraint violations (default to 0 when constraint type absent)
    defect    = current_iter_data.get("defect", 0)
    ncvx_ineq = current_iter_data.get("g_nonconvex_inequality", 0)

    # convergence epsilons (default to inf so absent checks pass trivially)
    eps_z      = method.eps_stack.state
    eps_dcost  = method.eps_stack.cost
    eps_ineq   = method.eps_stack.get("nonconvex_inequality", np.inf)
    eps_term   = method.eps_stack.get("final_state", np.inf)
    eps_dyn    = method.eps_stack.dynamics

    # absolute values of dz and dcost to measure optimality
    abs_dz = np.abs(dstate)
    abs_dcost = np.abs(dcost)

    # absolute values of linearized constraint violations
    abs_vb_dyn = np.abs(vb_dyn)

    abs_vb_ineq = np.abs(vb_ineq)
    abs_vb_term = np.abs(vb_term)

    # absolute values nonconvex constraint violations
    abs_ncvx_dyn = np.abs(defect)
    abs_ncvx_ineq = np.abs(ncvx_ineq)

    eps_term_trimmed = np.atleast_1d(eps_term)[:np.asarray(abs_vb_term).size] if np.ndim(eps_term) > 0 else eps_term
    bool_term = np.all(abs_vb_term <= 1.0*eps_term_trimmed)
    bool_vb_ineq = np.all(abs_vb_ineq <= 1.0*eps_ineq)
    bool_dz = np.all(abs_dz <= 1.0*eps_z)
    bool_dcost = np.all(abs_dcost <= 1.0*eps_dcost)
    bool_vb_dyn = np.all(abs_vb_dyn <= 1.0*eps_dyn)
    bool_ncvx_dyn_state = np.all(abs_ncvx_dyn <= 1.0*eps_dyn)
    bool_ncvx_ineq = np.all(abs_ncvx_ineq <= 1.0*eps_ineq)

    # convergence criteria option 1: 
    # check optimality with dz and feasibility with linearized constraint violations
    bool_opt1  = bool_dz
    bool_feas1 = bool_term and bool_vb_ineq and bool_vb_dyn

    # convergence criteria option 2: 
    # check optimality with dcost and feasibility with nonlinear constraint violations
    bool_opt2  = bool_dcost
    bool_feas2 = bool_term and bool_ncvx_ineq and bool_ncvx_dyn_state

    if method.flags.flag_conv == 0:
        bool_conv = (bool_opt1 and bool_feas1) or (bool_opt2 and bool_feas2) 
    elif method.flags.flag_conv == 1:
        bool_conv = (bool_opt1 and bool_feas1) 
    elif method.flags.flag_conv == 2:
        bool_conv = (bool_opt2 and bool_feas2) 

    # === Populate convergence summary
    current_iter_data.bool_conv = bool_conv
    current_iter_data.chk = tools.AttrDict(
        dz                   = np.max(abs_dz / eps_z),
        dcost                = np.max(abs_dcost / eps_dcost),
        final_state          = np.max(abs_vb_term / eps_term_trimmed),
        nonconvex_inequality = np.max(abs_vb_ineq / eps_ineq),
        dynamics             = np.max(abs_vb_dyn / eps_dyn),
    )
    current_iter_data.status    = method.cp_subproblem.status
    current_iter_data.converged = bool_conv