import numpy as np
from trajopt.utils import tools


def check_convergence_tolerance(method_segment) -> None:
    prev_iter_data    = method_segment.iter_data_list[-1]
    current_iter_data = method_segment.current_iter_data

    index_map = method_segment.index_map
    constraints = list(method_segment.constraints.values())

    dstate    = current_iter_data.dz[:, index_map.indices.z.state]
    dcost     = current_iter_data.cost - prev_iter_data.cost
    abs_dz    = np.abs(dstate)
    abs_dcost = np.abs(dcost)
    eps_z     = method_segment.eps_state
    eps_dcost = method_segment.eps_cost

    bool_dz    = np.all(abs_dz <= eps_z)
    bool_dcost = np.all(abs_dcost <= eps_dcost)

    bool_vb_dyn  = all(cnstr.is_feasible for cnstr in constraints if cnstr.type == "dynamics")
    bool_vb_ineq = all(cnstr.is_feasible for cnstr in constraints if "nonconvex_inequality" in cnstr.type)
    bool_vb_eq   = all(cnstr.is_feasible for cnstr in constraints if "nonconvex_equality" in cnstr.type)
    bool_term    = all(cnstr.is_feasible for cnstr in constraints if cnstr.type == "final_state")
    bool_cont    = all(cnstr.is_feasible for cnstr in constraints if "continuity" in cnstr.type)

    defect              = current_iter_data.get("defect", 0)
    bool_ncvx_dyn_state = np.all(np.abs(defect) <= method_segment.eps_dyn)
    bool_ncvx_ineq      = all(
        np.all(np.abs(h.g_nl) <= np.atleast_1d(h.eps))
        for h in constraints
        if "nonconvex_inequality" in h.type and getattr(h, "g_nl", None) is not None
    )
    bool_ncvx_eq        = all(
        np.all(np.abs(h.g_nl) <= np.atleast_1d(h.eps))
        for h in constraints
        if "nonconvex_equality" in h.type and getattr(h, "g_nl", None) is not None
    )

    bool_opt1  = bool_dz
    bool_feas1 = bool_term and bool_vb_ineq and bool_vb_eq and bool_vb_dyn and bool_cont

    bool_opt2  = bool_dcost
    bool_feas2 = bool_term and bool_ncvx_ineq and bool_ncvx_eq and bool_ncvx_dyn_state and bool_cont

    flag_conv = method_segment.flags.flag_conv
    if flag_conv == 0:
        bool_conv = (bool_opt1 and bool_feas1) or (bool_opt2 and bool_feas2)
    elif flag_conv == 1:
        bool_conv = (bool_opt1 and bool_feas1)
    elif flag_conv == 2:
        bool_conv = (bool_opt2 and bool_feas2)
    else:
        bool_conv = False

    current_iter_data.bool_conv = bool_conv
    current_iter_data.chk = tools.AttrDict(
        dz                   = np.max(abs_dz / eps_z),
        dcost                = np.max(abs_dcost / eps_dcost),
        final_state          = max((cnstr.vb_ratio for cnstr in constraints if cnstr.type == "final_state"), default=0.0),
        nonconvex_inequality = max((cnstr.vb_ratio for cnstr in constraints if "nonconvex_inequality" in cnstr.type), default=0.0),
        nonconvex_equality   = max((cnstr.vb_ratio for cnstr in constraints if "nonconvex_equality" in cnstr.type), default=0.0),
        dynamics             = max((cnstr.vb_ratio for cnstr in constraints if cnstr.type == "dynamics"), default=0.0),
    )
    current_iter_data.status    = method_segment.cp_subproblem_status
    current_iter_data.converged = bool_conv
