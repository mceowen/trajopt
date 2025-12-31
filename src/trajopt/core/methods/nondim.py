import numpy as np

# example usage: set_nondim_params(["d", "d", "d", "v", "v", "v"], ["f", "f"], [("d", 10), ("v", 10), ("m", 1)], params)

def set_nondim_params(problem, method, base_unit_labels=["m", "s", "kg"]):
    """
    Initializes all nondimensional parameters
    """

    n = problem.n
    m = problem.m

    # this solves the following linear system to backout base scales for
    # distance, time, and mass:
    # A @ ln([d, t, m]^T) = ln([anchor0, anchor1, anchor2]^T)
    # then ([d, t, m]^T) = exp(log([d, t, m]^T))

    exponents = {
        "d": np.array([1,  0,  0]),
        "t": np.array([0,  1,  0]),
        "m": np.array([0,  0,  1]),
        "v": np.array([1, -1,  0]),
        "a": np.array([1, -2,  0]),
        "f": np.array([1, -2,  1]),
    }

    A = np.vstack([exponents[key] for key in problem.params['model']['nondim']['anchor_types']])
    b = np.log(np.array([val for val in problem.params['model']['nondim']['anchor_scales']]))

    log_base_scales = np.linalg.solve(A, b)
    base_scales = np.exp(log_base_scales)

    # retrieve remaining scales from base scales
    if method.flags["nondim"]:
        nd = base_scales[0]
        nt = base_scales[1]
        nm = base_scales[2]
    else:
        nd = 1.0
        nt = 1.0
        nm = 1.0

    scales = {
        "d"    : nd,
        "t"    : nt,
        "m"    : nm,
        "v"    : nd / nt,
        "a"    : nd / (nt**2),
        "f"    : nm * nd / (nt**2),
        "ang"  : 180 / np.pi,
        "angv" : (180 / np.pi) / nt,
        "none": 1.0 
    }

    d_lbl = base_unit_labels[0]
    t_lbl = base_unit_labels[1]
    m_lbl = base_unit_labels[2]

    scale_labels = {
        "d"    : d_lbl,
        "t"    : t_lbl,
        "m"    : m_lbl,
        "v"    : f"{d_lbl} / {t_lbl}" ,
        "a"    : f"{d_lbl} / ({t_lbl}^2)",
        "f"    : f"{m_lbl} * {d_lbl} / ({t_lbl}^2)",
        "ang"  : "deg",
        "angv" : f"deg / {t_lbl}",
        "none": ""
    }

    print("scales: ")
    print(", ".join(f"{k}: {v:.4f}" for k, v in scales.items()))

    nd_state = np.array([scales[problem.params['model']['nondim']['z_types'][i]] for i in range(n)])
    nd_ctrl  = np.array([scales[problem.params['model']['nondim']['u_types'][i]] for i in range(m)])

    method.nondim               = {}
    method.nondim["M"]          = {}
    method.nondim["M"]["state"] = {}
    method.nondim["M"]["ctrl"]  = {}
    method.nondim["M"]["ineq_nodal"] = {}
    method.nondim["M"]["ineq_ct"] = {}
    method.nondim["M"]["term_total"] = {}

    method.nondim["labels"]     = {}

    method.nondim["M"]["state"]["d2nd"] = np.diag(1 / nd_state).copy()
    method.nondim["M"]["state"]["nd2d"] = np.diag(nd_state).copy()
    method.nondim["M"]["ctrl"]["d2nd"] = np.diag(1 / nd_ctrl).copy()
    method.nondim["M"]["ctrl"]["nd2d"] = np.diag(nd_ctrl).copy()

    # add scalar nondim variables to nondim substruct
    method.nondim["scales"] = scales
    method.nondim["nd"] = scales["d"]
    method.nondim["na"] = scales["a"]
    method.nondim["nt"] = scales["t"]
    method.nondim["nt_inv"] = 1 / scales["t"]
    method.nondim["nv"] = scales["v"]
    method.nondim["nm"] = scales["m"]
    method.nondim["nm_dot"] = scales["m"] / scales["t"]
    method.nondim["nf"] = scales["f"]
    method.nondim["nang"] = scales["ang"]
    method.nondim["nangv"] = scales["ang"] / scales["t"]
    method.nondim["labels"]["state"] = [scale_labels[problem.params['model']['nondim']['z_types'][i]] for i in range(n)]
    method.nondim["labels"]["ctrl"]  = [scale_labels[problem.params['model']['nondim']['u_types'][i]] for i in range(m)]

    # set nondim for cost and constraints
    
    # TODO(carlos): setting to one rn, need to think about how to properly specifiy this
    # when there are multiple summed cost functions

    nd_cost = 1.0
    method.nondim["nd_cost"] = nd_cost

    # nodal inequality nondim
    nd_ineq = np.ones(problem.n_ineq)
    
    idx = 0
    # TODO: change 'nonconvex_inequality' to 'inequality' once we add general buffering
    for constraint in problem.constraints.get('nodal', 'nonconvex_inequality'):
        if getattr(constraint, 'units', None) is not None:
            scale_list = [scales[key]**exponent for key, exponent in constraint.units.items()]
            scale = np.prod(scale_list)
            nd_ineq[idx:idx + constraint.dimension] = scale
        idx += constraint.dimension

    method.nondim["nd_ineq"] = nd_ineq
    method.nondim["M"]["ineq_nodal"]["d2nd"] = np.diag(1 / nd_ineq).copy()
    method.nondim["M"]["ineq_nodal"]["nd2d"] = np.diag(nd_ineq).copy()

    # ct inequality nondim
    nd_ctcs = np.ones(problem.n_ctcs)
    idx = 0
    # TODO: change 'nonconvex_inequality' to 'inequality' once we add general buffering
    for constraint in problem.constraints.get('ct', 'nonconvex_inequality'):
        if getattr(constraint, 'units', None) is not None:
            scale_list = [scales[key]**exponent for key, exponent in constraint.units.items()]
            scale = np.prod(scale_list)
            nd_ctcs[idx:idx + constraint.dimension] = scale
        idx += constraint.dimension

    method.nondim["nd_ctcs"] = nd_ctcs
    method.nondim["M"]["ineq_ct"]["d2nd"] = np.diag(1 / nd_ctcs).copy()
    method.nondim["M"]["ineq_ct"]["nd2d"] = np.diag(nd_ctcs).copy()


    # TODO (CARLOS): fix this pls, set to all ones so i can run it
    # the problem is vb_term mixes different constraint types 
    # together so indexing into is a bit tricky

    nd_term_total = np.hstack([np.ones(problem.n_term), np.ones(problem.n_term_ineq), np.ones(problem.n_ctcs)])
    method.nondim["nd_term_total"] = nd_term_total
    method.nondim["M"]["term_total"]["d2nd"] = np.diag(1 / nd_term_total).copy()
    method.nondim["M"]["term_total"]["nd2d"] = np.diag(nd_term_total).copy()