import numpy as np

# example usage: set_nondim_params(["d", "d", "d", "v", "v", "v"], ["f", "f"], [("d", 10), ("v", 10), ("m", 1)], params)

def set_nondim_params(z_types, u_types, anchor_types, anchor_scales, params, base_unit_labels=["m", "s", "kg"]):
    """
    Initializes all nondimensional parameters
    """
    # Extract dimension constants
    # path_lim = params["mission"]["path_lim"]
    # n_path = params["n_path"]
    # n_nfz = params["mission"]["n_nfz"]
    n = params["model"]["n"]
    m = params["model"]["m"]

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

    A = np.vstack([exponents[key] for key in anchor_types])
    b = np.log(np.array([val for val in anchor_scales]))

    log_base_scales = np.linalg.solve(A, b)
    base_scales = np.exp(log_base_scales)

    # retrieve remaining scales from base scales
    if params["method"]["bools"]["nondim"]:
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
        "ang"  : 1.0,
        "angv" : 1.0 / nt 
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
        "ang"  : "rad",
        "angv" : f"rad / {t_lbl}"
    }

    print("scales: ")
    print(", ".join(f"{k}: {v:.2f}" for k, v in scales.items()))

    nd_state = np.array([scales[z_types[i]] for i in range(n)])
    nd_ctrl  = np.array([scales[u_types[i]] for i in range(m)])

    if "nondim" not in params["method"]: # initialize if it doesn"t already exist
       params["method"]["nondim"] = {}
       params["method"]["nondim"]["M"] = {}
       params["method"]["nondim"]["M"]["state"] = {}
       params["method"]["nondim"]["M"]["ctrl"] = {}
       params["method"]["nondim"]["M"]["cnst"] = {}
       params["method"]["nondim"]["M"]["nfz"] = {}
       params["method"]["nondim"]["M"]["dyn"] = {}
       params["method"]["nondim"]["M"]["cost"] = {}
       params["method"]["nondim"]["M"]["term"] = {}

       params["method"]["nondim"]["labels"]  = {}

    params["method"]["nondim"]["M"]["state"]["d2nd"] = np.diag(1 / nd_state).copy()
    params["method"]["nondim"]["M"]["state"]["nd2d"] = np.diag(nd_state).copy()
    
    params["method"]["nondim"]["M"]["ctrl"]["d2nd"] = np.diag(1 / nd_ctrl).copy()
    params["method"]["nondim"]["M"]["ctrl"]["nd2d"] = np.diag(nd_ctrl).copy()

    nd_dyn = nd_state / scales["t"]
    params["method"]["nondim"]["M"]["dyn"]["d2nd"] = np.diag(1 / nd_dyn).copy()
    params["method"]["nondim"]["M"]["dyn"]["nd2d"] = np.diag(nd_dyn).copy()

    params["method"]["nondim"]["M"]["term"]["d2nd"] = np.diag(1 / nd_state).copy()
    params["method"]["nondim"]["M"]["term"]["nd2d"] = np.diag(nd_state).copy()

    params["method"]["nondim"]["nu_rad_ind"] = []

    # add scalar nondim variables to nondim substruct
    params["method"]["nondim"]["scales"] = scales
    params["method"]["nondim"]["nd"] = scales["d"]
    params["method"]["nondim"]["na"] = scales["a"]
    params["method"]["nondim"]["nt"] = scales["t"]
    params["method"]["nondim"]["nt_inv"] = 1 / scales["t"]
    params["method"]["nondim"]["nv"] = scales["v"]
    params["method"]["nondim"]["nm"] = scales["m"]
    params["method"]["nondim"]["nm_dot"] = scales["m"] / scales["t"]
    params["method"]["nondim"]["nf"] = scales["f"]

    params["method"]["nondim"]["labels"]["state"] = [scale_labels[z_types[i]] for i in range(n)]
    params["method"]["nondim"]["labels"]["ctrl"]  = [scale_labels[u_types[i]] for i in range(m)]

    return params

def set_cost_cnst_nondim_params(np_ineq, ncost, params):

    params["method"]["nondim"]["M"]["cnst"]["d2nd"] = np.diag(np_ineq ** -1).copy()
    params["method"]["nondim"]["M"]["cnst"]["nd2d"] = np.diag(np_ineq).copy()

    params["method"]["nondim"]["M"]["nfz"]["d2nd"] = np.diag(1 / np_ineq[params["mission"]["nfz_idx"]]).copy()
    params["method"]["nondim"]["M"]["nfz"]["nd2d"] = np.diag(np_ineq[params["mission"]["nfz_idx"]]).copy()
    params["method"]["nondim"]["M"]["cost"]["d2nd"] = 1 / ncost
    params["method"]["nondim"]["np_ineq"] = np_ineq
    params["method"]["nondim"]["ncost"] = ncost

    return params