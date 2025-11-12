import numpy as np

# example usage: set_nondim_params(["d", "d", "d", "v", "v", "v"], ["f", "f"], [("d", 10), ("v", 10), ("m", 1)], params)

def set_nondim_params(problem, base_unit_labels=["m", "s", "kg"]):
    """
    Initializes all nondimensional parameters
    """

    mission = problem.mission
    model = problem.model
    method = problem.method

    n = model.n
    m = model.m

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

    A = np.vstack([exponents[key] for key in model.anchor_types])
    b = np.log(np.array([val for val in model.anchor_scales]))

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
        "angv" : (180 / np.pi) / nt 
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
        "angv" : f"deg / {t_lbl}"
    }

    print("scales: ")
    print(", ".join(f"{k}: {v:.4f}" for k, v in scales.items()))

    nd_state = np.array([scales[model.z_types[i]] for i in range(n)])
    nd_ctrl  = np.array([scales[model.u_types[i]] for i in range(m)])

    method.nondim               = {}
    method.nondim["M"]          = {}
    method.nondim["M"]["state"] = {}
    method.nondim["M"]["ctrl"]  = {}
    method.nondim["M"]["cnst"]  = {}
    method.nondim["M"]["nfz"]   = {}
    method.nondim["M"]["dyn"]   = {}
    method.nondim["M"]["path"]  = {}
    method.nondim["M"]["aux"]   = {}
    method.nondim["M"]["cost"]  = {}
    method.nondim["M"]["term"]  = {}

    method.nondim["labels"]     = {}

    method.nondim["M"]["state"]["d2nd"] = np.diag(1 / nd_state).copy()
    method.nondim["M"]["state"]["nd2d"] = np.diag(nd_state).copy()
    method.nondim["M"]["ctrl"]["d2nd"] = np.diag(1 / nd_ctrl).copy()
    method.nondim["M"]["ctrl"]["nd2d"] = np.diag(nd_ctrl).copy()

    nd_dyn = nd_state / scales["t"]
    method.nondim["M"]["dyn"]["d2nd"] = np.diag(1 / nd_dyn).copy()
    method.nondim["M"]["dyn"]["nd2d"] = np.diag(nd_dyn).copy()
    method.nondim["M"]["term"]["d2nd"] = np.diag(1 / nd_state).copy()
    method.nondim["M"]["term"]["nd2d"] = np.diag(nd_state).copy()
    method.nondim["nu_rad_ind"] = []

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
    method.nondim["labels"]["state"] = [scale_labels[model.z_types[i]] for i in range(n)]
    method.nondim["labels"]["ctrl"]  = [scale_labels[model.u_types[i]] for i in range(m)]

    # set nondim for cost and constraints
    ncost, np_ineq = mission.get_cost_cnstr_nondim()

    method.nondim["ncost"] = ncost
    method.nondim["M"]["cost"]["d2nd"] = 1 / ncost

    method.nondim["np_ineq"] = np_ineq
    method.nondim["M"]["cnst"]["d2nd"] = np.diag(np_ineq ** -1).copy()
    method.nondim["M"]["cnst"]["nd2d"] = np.diag(np_ineq).copy()

    method.nondim["M"]["nfz"]["d2nd"] = np.diag(1 / np_ineq[mission.nfz_idx]).copy()
    method.nondim["M"]["nfz"]["nd2d"] = np.diag(np_ineq[mission.nfz_idx]).copy()

    method.nondim["M"]["path"]["d2nd"] = np.diag(1 / np_ineq[mission.path_idx]).copy()
    method.nondim["M"]["path"]["nd2d"] = np.diag(np_ineq[mission.path_idx]).copy()

    method.nondim["M"]["aux"]["d2nd"] = np.diag(1 / np_ineq[mission.aux_idx]).copy()
    method.nondim["M"]["aux"]["nd2d"] = np.diag(np_ineq[mission.aux_idx]).copy()
    