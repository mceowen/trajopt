import numpy as np
import trajopt.utils.tools as tools

def set_convergence_tolerance(problem, method):
    """
    Compute convergence tolerances for all trajopt_obj components.
    """

    # nondim state deviation convergence epsilon
    nondim = method.nondim
    method.conv.eps_state = nondim.M.state.d2nd @ method.conv.eps_state

    # nondim multiple shooting state defect convergence epsilon
    eps_defect = tools.expand_to_array_if_scalar(method.conv.eps_defect, method.index_map.n.state)
    method.conv.eps_defect = nondim.M.state.d2nd @ eps_defect

    # nondim dynamics convergence epsilon

    # method.conv.eps_dyn is still in dimensional units here
    eps_dyn = tools.expand_to_array_if_scalar(method.conv.eps_dyn, method.index_map.n.z)
    eps_dyn_real = nondim.M.state.d2nd @ eps_dyn
    
    # augment epsilon with ctcs contributions
    if problem.constraints.has(ct=1):
        # constraint epsilons have already been nondimensionalized with "nondim_constraints()"
        eps_dyn_ctcs = np.concatenate([c.eps for c in problem.constraints.get(ct=1)])
        
        # approximation of constraint violation integral
        eps_dyn_ctcs = (1* eps_dyn_ctcs)**2 * method.dt_min * 0.25
        eps_dyn = np.concatenate([eps_dyn_real, eps_dyn_ctcs])
    else:
        eps_dyn = eps_dyn_real

    method.conv.eps_dyn = eps_dyn

    # set nodal nonconvex inequality constraint tolerances
    if problem.constraints.has(ct=0, type='nonconvex_inequality'):
        ncvx_ineq_constraints = problem.constraints.get(ct=0, type='nonconvex_inequality')
        method.conv.eps_ineq = np.concatenate([constraint.eps for constraint in ncvx_ineq_constraints])
    else:
        method.conv.eps_ineq = np.array([])
    
    method.conv.eps_term = np.array([])
    if problem.constraints.has(ct=0, type="equality_bc", boundary="final", set="state"):
        # terminal and nodal nonconvex inequality constraints
        term_constraints     = problem.constraints.get(ct=0, type='equality_bc', boundary="final", set="state")
        method.conv.eps_term = np.concatenate([constraint.eps for constraint in term_constraints])

    # stack epsilons for terminal inequality constraints and augmented ctcs cosntraints
    if problem.constraints.has(ct=0, type='inequality_bc', boundary="final", set="state"):
        eps_term_ineq = [c.eps for c in problem.constraints.get(ct=0, type='inequality_bc', boundary="final", set="state")]
        method.conv.eps_term = np.concatenate([method.conv.eps_term, eps_term_ineq])

    if problem.constraints.has(ct=1):
        eps_term_ctcs = np.concatenate([c.eps for c in problem.constraints.get(ct=1)])
        method.conv.eps_term = np.concatenate([method.conv.eps_term, eps_term_ctcs])

def check_convergence_tolerance(problem, method, iter_record):
    """Check convergence using unified stacked inequality (_ineq) structure."""

    # --- Load convergence data
    conv_data = iter_record.conv_data

    # --- Extract dimensions from Subproblem
    n_state = problem.index_map.n.state
    N   = method.index_map.N.N

    # dz and dcost to measure optimality
    dstate  = iter_record.dz_s[:, :n_state]
    dcost   = iter_record.cost - conv_data.cost_ref
    
    # linearized constraint violations
    vb_dyn  = conv_data.vb_dyn
    vb_ineq = conv_data.vb_ineq
    vb_term = conv_data.vb_terminal

    # nonlinear constraint violations
    defect    = conv_data.defect
    ncvx_ineq = conv_data.ncvx_ineq

    # convergence epsilons
    eps_state  = method.conv.eps_state
    eps_dcost  = method.conv.eps_cost
    eps_ineq   = method.conv.eps_ineq   if method.conv.eps_ineq.size   > 0 else np.array([1.0])
    eps_term   = method.conv.eps_term   if method.conv.eps_term.size   > 0 else np.array([1.0])
    eps_defect = method.conv.eps_defect
    eps_dyn    = method.conv.eps_dyn

    # absolute values of dz and dcost to measure optimality
    abs_dz = np.abs(dstate)
    abs_dcost = np.abs(dcost)

    # absolute values of linearized constraint violations
    abs_vb_dyn = np.abs(vb_dyn)

    abs_vb_ineq = np.abs(vb_ineq) if problem.constraints.has(type="nonconvex_inequality", ct=0) else np.zeros((1, ))
    abs_vb_term = np.abs(vb_term) if np.asarray(vb_term).size > 0 else np.zeros(1)

    # absolute values nonconvex constraint violations
    abs_ncvx_dyn = np.abs(defect)
    abs_ncvx_ineq = np.abs(ncvx_ineq)

    bool_term = np.all(abs_vb_term <= 1.0*eps_term)
    bool_vb_ineq = np.all(abs_vb_ineq <= 1.0*eps_ineq)
    bool_dz = np.all(abs_dz <= 1.0*eps_state)
    bool_dcost = np.all(abs_dcost <= 1.0*eps_dcost)
    bool_vb_dyn = np.all(abs_vb_dyn <= 1.0*eps_dyn)
    bool_ncvx_dyn = np.all(abs_ncvx_dyn <= 1.0*eps_defect)
    bool_ncvx_ineq = np.all(abs_ncvx_ineq <= 1.0*eps_ineq)

    # convergence criteria option 1: 
    # check optimality with dz and feasibility with linearized constraint violations
    bool_opt1  = bool_dz
    bool_feas1 = bool_term and bool_vb_ineq and bool_vb_dyn

    # convergence criteria option 2: 
    # check optimality with dcost and feasibility with nonlinear constraint violations
    bool_opt2  = bool_dcost
    bool_feas2 = bool_term and bool_ncvx_ineq and bool_ncvx_dyn

    if method.flags.flag_conv == 0:
        bool_conv = (bool_opt1 and bool_feas1) or (bool_opt2 and bool_feas2) 
    elif method.flags.flag_conv == 1:
        bool_conv = (bool_opt1 and bool_feas1) 
    elif method.flags.flag_conv == 2:
        bool_conv = (bool_opt2 and bool_feas2) 

    # === Populate convergence summary
    conv_data.bool_conv   = bool_conv
    conv_data.chk_dz      = np.max(abs_dz / eps_state)
    conv_data.chk_dcost   = np.max(abs_dcost / eps_dcost)
    conv_data.chk_vb_term = np.max(abs_vb_term / eps_term)
    conv_data.chk_vb_ineq = np.max(abs_vb_ineq / eps_ineq)
    conv_data.chk_vb_dyn  = np.max(abs_vb_dyn / eps_dyn)
    conv_data.status      = iter_record.cp_subprob.status

    iter_record.converged = bool_conv
    iter_record.conv_data = conv_data

    return iter_record