import numpy as np
import trajopt.utils.tools as tools

def set_convergence_tolerance(problem, method):
    """
    Set convergence tolerances in physical (x, t) space and nominal constraints,
    then augment to the full z space.
    """
    nondim = method.nondim
    n      = problem.index_map.n

    # --- State deviation (optimality) ---
    eps_state = tools.expand_to_array_if_scalar(method.conv.eps_state, n.state)
    method.conv.eps_state = nondim.M.state.d2nd @ eps_state

    # --- Time deviation (optimality; default inf = not checked) ---
    eps_t_cfg = getattr(method.conv, 'eps_t', None)
    if eps_t_cfg is None:
        method.conv.eps_t = np.full(n.time, np.inf)
    else:
        method.conv.eps_t = np.asarray(eps_t_cfg).reshape(-1) / nondim.time_scale

    # --- Multiple-shooting state defect (monitoring only) ---
    eps_defect = tools.expand_to_array_if_scalar(method.conv.eps_defect, n.state)
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
        method.conv.eps_ineq = np.concatenate([c.eps for c in problem.constraints.get(ct=0, type='nonconvex_inequality')])
    else:
        method.conv.eps_ineq = np.array([])

    # --- Terminal feasibility (physical: eq + ineq, no CTCS yet) ---
    method.conv.eps_term = np.array([])
    if problem.constraints.has(ct=0, type="equality_bc", boundary="final", set="state"):
        term_constraints     = problem.constraints.get(ct=0, type='equality_bc', boundary="final", set="state")
        method.conv.eps_term = np.concatenate([c.eps for c in term_constraints])

    if problem.constraints.has(ct=0, type='inequality_bc', boundary="final", set="state"):
        eps_term_ineq = [c.eps for c in problem.constraints.get(ct=0, type='inequality_bc', boundary="final", set="state")]
        method.conv.eps_term = np.concatenate([method.conv.eps_term, *eps_term_ineq])

    augment_convergence_tolerance(problem, method)


def augment_convergence_tolerance(problem, method):
    """
    Augment convergence tolerances to the full z space, adding time (z.time
    indices) and CTCS (z.ctcs indices) components to eps_dyn and eps_term.
    """
    idx = problem.index_map.indices
    n   = problem.index_map.n

    # Build eps_dyn indexed over the full z vector
    eps_dyn              = np.empty(n.z)
    eps_dyn[idx.z.state] = method.conv.eps_state
    eps_dyn[idx.z.time]  = method.conv.eps_t

    if problem.constraints.has(ct=1):
        eps_ctcs = np.concatenate([c.eps for c in problem.constraints.get(ct=1)])
        # Approximation of constraint violation integral
        eps_dyn[idx.z.ctcs]  = eps_ctcs * method.dt_min * 0.25
        method.conv.eps_term = np.concatenate([method.conv.eps_term, eps_ctcs])

    method.conv.eps_dyn = eps_dyn


def check_convergence_tolerance(problem, method, iter_record):
    """Check convergence using unified stacked inequality (_ineq) structure."""

    # --- Load convergence data
    conv_data = iter_record.conv_data

    idx = problem.index_map.indices

    # --- Extract optimization variables
    dstate  = iter_record.dz_s[:, idx.z.state]
    dt_sol  = iter_record.dz_s[:, idx.z.time]
    dcost   = iter_record.cost - conv_data.cost_ref
    vb_dyn  = conv_data.vb_dyn
    vb_ineq = conv_data.vb_ineq
    vb_term = conv_data.vb_terminal

    # --- Extract convergence criteria
    eps_state  = method.conv.eps_state
    eps_t      = method.conv.eps_t
    eps_cost   = method.conv.eps_cost
    eps_ineq   = method.conv.eps_ineq
    eps_term   = method.conv.eps_term
    eps_defect = method.conv.eps_defect
    eps_dyn    = method.conv.eps_dyn

    abs_dz = np.abs(dstate)
    abs_dt = np.abs(dt_sol)
    abs_opt = np.abs(dcost)
    abs_vb_dyn = np.abs(vb_dyn)
    abs_vb_ineq = np.abs(vb_ineq)
    abs_vb_term = np.abs(vb_term)

    bool_term  = np.all(abs_vb_term <= 1.0*eps_term)
    bool_ineq  = np.all(abs_vb_ineq <= 1.0*eps_ineq)
    bool_state = np.all(abs_dz <= 1.0*eps_state)
    bool_time  = np.all(abs_dt <= 1.0*eps_t)

    bool_conv = bool_term and bool_ineq and bool_state and bool_time

    # === Populate convergence summary
    conv_data.bool_conv = bool_conv
    conv_data.chk_dz = np.max(abs_dz)
    conv_data.chk_opt = np.max(abs_opt)
    conv_data.chk_feas_term = np.max(abs_vb_term)
    if eps_ineq.size > 0:
        conv_data.chk_feas_ineq = np.max(abs_vb_ineq)
    else:
        conv_data.chk_feas_ineq = 0.0
    conv_data.chk_feas_dyn = np.max(abs_vb_dyn)
    conv_data.status = iter_record.cp_subprob.status

    iter_record.converged = bool_conv
    iter_record.conv_data = conv_data

    return iter_record