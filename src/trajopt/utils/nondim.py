import numpy as np

# example usage: set_nondim_params(['d', 'd', 'd', 'v', 'v', 'v'], ['f', 'f'], [('d', 10), ('v', 10), ('m', 1)], params)

def set_nondim_params(z_types, u_types, anchor_scales, params, base_unit_labels=['m', 's', 'kg']):
    """
    Initializes all nondimensional parameters
    """
    # Extract dimension constants
    path_lim = params['path_lim']
    n_path = params['n_path']
    n_nfz = params['n_nfz']
    n = params['n']
    m = params['m']

    # this solves the following linear system to backout base scales for
    # distance, time, and mass:
    # A @ ln([d, t, m]^T) = ln([anchor0, anchor1, anchor2]^T)
    # then ([d, t, m]^T) = exp(log([d, t, m]^T))

    exponents = {
        'd': np.array([1,  0,  0]),
        't': np.array([0,  1,  0]),
        'm': np.array([0,  0,  1]),
        'v': np.array([1, -1,  0]),
        'a': np.array([1, -2,  0]),
        'f': np.array([1, -2,  1]),
    }

    A = np.vstack([exponents[key] for key, _ in anchor_scales])
    b = np.log(np.array([val for _, val in anchor_scales]))

    log_base_scales = np.linalg.solve(A, b)
    base_scales = np.exp(log_base_scales)

    # retrieve remaining scales from base scales
    if params['bools']['nondim']:
        nd = base_scales[0]
        nt = base_scales[1]
        nm = base_scales[2]
    else:
        nd = 1.0
        nt = 1.0
        nm = 1.0

    scales = {
        'd'    : nd,
        't'    : nt,
        'm'    : nm,
        'v'    : nd / nt,
        'a'    : nd / (nt**2),
        'f'    : nm * nd / (nt**2),
        'ang'  : 1.0,
        'angv' : 1.0 / nt 
    }

    d_lbl = base_unit_labels[0]
    t_lbl = base_unit_labels[1]
    m_lbl = base_unit_labels[2]

    scale_labels = {
        'd'    : d_lbl,
        't'    : t_lbl,
        'm'    : m_lbl,
        'v'    : f"{d_lbl} / {t_lbl}" ,
        'a'    : f"{d_lbl} / ({t_lbl}^2)",
        'f'    : f"{m_lbl} * {d_lbl} / ({t_lbl}^2)",
        'ang'  : "rad",
        'angv' : f"rad / {t_lbl}"
    }

    print("scales: ")
    print(", ".join(f"{k}: {v:.2f}" for k, v in scales.items()))

    nd_state = np.array([scales[z_types[i]] for i in range(n)])
    nd_ctrl  = np.array([scales[u_types[i]] for i in range(m)])

    if 'nondim' not in params: # initialize if it doesn't already exist
       params['nondim'] = {}
       params['nondim']['M'] = {}
       params['nondim']['M']['state'] = {}
       params['nondim']['M']['ctrl'] = {}
       params['nondim']['M']['cnst'] = {}
       params['nondim']['M']['nfz'] = {}
       params['nondim']['M']['dyn'] = {}
       params['nondim']['M']['cost'] = {}
       params['nondim']['M']['term'] = {}

       params['nondim']['labels']  = {}

    params['nondim']['M']['state']['d2nd'] = np.diag(1 / nd_state).copy()
    params['nondim']['M']['state']['nd2d'] = np.diag(nd_state).copy()
    
    params['nondim']['M']['ctrl']['d2nd'] = np.diag(1 / nd_ctrl).copy()
    params['nondim']['M']['ctrl']['nd2d'] = np.diag(nd_ctrl).copy()

    nd_dyn = nd_state / scales['t']
    params['nondim']['M']['dyn']['d2nd'] = np.diag(1 / nd_dyn).copy()
    params['nondim']['M']['dyn']['nd2d'] = np.diag(nd_dyn).copy()

    params['nondim']['M']['term']['d2nd'] = np.diag(1 / nd_state).copy()
    params['nondim']['M']['term']['nd2d'] = np.diag(nd_state).copy()

    params['nondim']['nu_rad_ind'] = []

    # add scalar nondim variables to nondim substruct
    params['nondim']['scales'] = scales
    params['nondim']['nd'] = scales['d']
    params['nondim']['na'] = scales['a']
    params['nondim']['nt'] = scales['t']
    params['nondim']['nt_inv'] = 1 / scales['t']
    params['nondim']['nv'] = scales['v']
    params['nondim']['nm'] = scales['m']
    params['nondim']['nm_dot'] = scales['m'] / scales['t']
    params['nondim']['nf'] = scales['f']

    params['nondim']['labels']['state'] = [scale_labels[z_types[i]] for i in range(n)]
    params['nondim']['labels']['ctrl']  = [scale_labels[u_types[i]] for i in range(m)]

    return params

def set_cost_cnst_nondim_params(np_ineq, ncost, params):

    params['nondim']['M']['cnst']['d2nd'] = np.diag(np_ineq ** -1).copy()
    params['nondim']['M']['cnst']['nd2d'] = np.diag(np_ineq).copy()

    params['nondim']['M']['nfz']['d2nd'] = np.diag(1 / np_ineq[params['nfz_idx']]).copy()
    params['nondim']['M']['nfz']['nd2d'] = np.diag(np_ineq[params['nfz_idx']]).copy()
    params['nondim']['M']['cost']['d2nd'] = 1 / ncost
    params['nondim']['np_ineq'] = np_ineq
    params['nondim']['ncost'] = ncost

    return params