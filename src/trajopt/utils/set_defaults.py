# Skye Mceowen and Aman Tiwary
# Feb. 14, 2024
# functions that set default problem parameters

import numpy as np
from trajopt.algorithm.discretization import set_ltv_indices

def set_params_default(config=None):
    # --- Initialize empty params dict ---
    params = {}

    # --- Default booleans ---
    params['bools'] = {
        'auto_jac': 1,
        'auto_jac_aero': 1,
        'auto_jac_cnst': 1,
        'dev_var': 1,
        'nondim': 0,
        'var_scl_flag': 0,
        'free_final_time': 0,
        'equal_dt': 0,
        'flag_nfz': 0,
        'flag_autotune': 3,
        'flag_Wauto_memory': 0,
        'weight_zero': 0,
        'stepsize_auto_primal': 0,
        'stepsize_auto_dual': 0,
        'flag_conv': 0,
        'buff_dyn': 'l2',
        'buff_dyn_dual': 'l1',
        'ctcs': 0,
        'ode_fixed_dt': 0,
        'aoa_vb': 0,
        'earth_rot': 0,
        'init_ctrl': 1,
        'jax_dyn': 0
    }

    # --- Basic structure ---
    params['n']             = 1
    params['m']             = 1

    # --- Constraint structure ---
    params['path_lim']      = np.array([])
    params['path_idx']      = np.array([], dtype=np.int64)
    params['n_path']        = 0
    params['nfz_idx']       = np.array([], dtype=np.int64)
    params['n_nfz']         = 0
    params['aux_idx']       = np.array([], dtype=np.int64)
    params['n_aux']         = 0
    params['n_ineq']        = 0
    params['n_eq']          = 0

    # --- Cost label ---
    params['cost_name']     = 'Cost'

    # --- Initial boundary conditions ---
    params['zi']            = np.array([])
    params['zi_idx']        = np.array([], dtype=np.int64)
    params['zi_min']        = np.array([])
    params['zi_min_idx']    = np.array([], dtype=np.int64)
    params['zi_max']        = np.array([])
    params['zi_max_idx']    = np.array([], dtype=np.int64)
    params['n_init']        = 0
    params['n_init_ineq']   = 0

    # --- Terminal boundary conditions ---
    params['zf']            = np.array([])
    params['zf_idx']        = np.array([], dtype=np.int64)
    params['zf_min']        = np.array([])
    params['zf_min_idx']    = np.array([], dtype=np.int64)
    params['zf_max']        = np.array([])
    params['zf_max_idx']    = np.array([], dtype=np.int64)
    params['n_term']        = 0
    params['n_term_ineq']   = 0

    # --- State constraints ---
    params['z_min']         = np.array([])
    params['z_min_idx']     = np.array([], dtype=np.int64)
    params['z_max']         = np.array([])
    params['z_max_idx']     = np.array([], dtype=np.int64)
    params['n_state']       = 0

    # --- Control constraints ---
    params['u_min']         = np.array([])
    params['u_min_idx']     = np.array([], dtype=np.int64)
    params['u_max']         = np.array([])
    params['u_max_idx']     = np.array([], dtype=np.int64)
    params['udot_max']      = np.array([])
    params['udot_max_idx']  = np.array([], dtype=np.int64)
    params['n_ctrl']        = 0
    params['n_udot']        = 0

    # --- Weight structure ---
    params['weights']       = {}
    params['n_dyn']         = 0

    # --- Convergence settings ---
    params['conv'] = {
        'eps_feas_conv': 1e-1,
        'eps_conv': 1e-1,
        'eps_cost_conv': 1e-1
    }

    # --- Overwrite with config ---
    if config is not None:
        params['mission'] = config.get('mission', None)
        if 'params' in config:
            for key, value in config['params'].items():
                if key == 'bools':
                    params['bools'].update(value)
                else:
                    params[key] = value

    # --- CTCS-specific adjustment ---
    if params['bools'].get('ctcs') and params['bools'].get('buff_dyn') == 'term':
        params['bools']['buff_dyn'] = 'l1'

    return params


def set_params_constraint_default(params):
    """
    Set default constraint dimensions, duals, weights, and convergence criteria
    for SCvx-style trajectory optimization.
    """

    # --- Constraint bookkeeping ---
    params['n_init']       = len(params.get('zi_idx',       np.array([], dtype=np.int64)))
    params['n_init_ineq']  = len(params.get('zi_min_idx',   np.array([], dtype=np.int64))) + len(params.get('zi_max_idx', np.array([], dtype=np.int64)))
    params['n_term']       = len(params.get('zf_idx',       np.array([], dtype=np.int64)))
    params['n_term_ineq']  = len(params.get('zf_min_idx',   np.array([], dtype=np.int64))) + len(params.get('zf_max_idx', np.array([], dtype=np.int64)))
    params['n_ctrl']       = len(params.get('u_min_idx',    np.array([], dtype=np.int64))) + len(params.get('u_max_idx', np.array([], dtype=np.int64)))
    params['n_state']      = len(params.get('z_min_idx',    np.array([], dtype=np.int64))) + len(params.get('z_max_idx', np.array([], dtype=np.int64)))
    params['n_udot']       = len(params.get('udot_max',     np.array([], dtype=np.int64)))

    params['n_path']       = len(params.get('path_idx',     np.array([], dtype=np.int64)))
    params['n_nfz']        = len(params.get('nfz_idx',      np.array([], dtype=np.int64)))
    params['n_aux']        = len(params.get('aux_idx',      np.array([], dtype=np.int64)))
    params['n_ineq']       = params['n_path'] + params['n_nfz'] + params['n_aux']

    # --- State vector size (ctcs mode) ---
    if params['bools'].get('ctcs', False):
        params['nz'] = params['n'] + params['n_ineq']
    else:
        params['nz'] = params['n']

    params['n_dyn'] = params['nz']

    buff_dyn = str(params['bools'].get('buff_dyn', 'term'))


    # --- Dynamics buffering ---
    if buff_dyn in {'term', 'l1', 'l2'}:
        params['n_plus'] = 0
        params['n_minus'] = 0
        params['Npm'] = 0
    elif buff_dyn == 'quad-1':
        params['n_plus'] = 1
        params['n_minus'] = 1
        params['Npm'] = 1
    elif buff_dyn == 'quad-2':
        params['n_plus'] = 1
        params['n_minus'] = 1
        params['Npm'] = params['N'] - 1
    elif buff_dyn == 'quad-3':
        params['n_plus'] = params['nz']
        params['n_minus'] = params['nz']
        params['Npm'] = 1
    else:
        raise ValueError("Invalid buff_dyn flag.")

    # --- Terminal conditions nondimensionalization ---
    # Get the diagonal of the source matrix
    M_diag = np.diag(params['nondim']['M']['state']['d2nd'])
    # Stack selected diagonals
    selected = np.concatenate([
        M_diag[params['zf_idx']],
        M_diag[params['zf_min_idx']],
        M_diag[params['zf_max_idx']]
    ])
    # Create the new diagonal matrix
    params['nondim']['M']['term']['d2nd'] = np.diag(selected)

    # --- Default weights ---
    weights = params.setdefault('weights', {})
    weights['w_fac_N']      = params['N']
    weights['w_fac_Nm1']    = params['N'] - 1
    weights['w_cost']       = 1.0

    weights['dual_path']    = np.zeros((params['N'], params['n_path']))
    weights['dual_nfz']     = np.zeros((params['N'], params['n_nfz']))
    weights['dual_aux']     = np.zeros((params['N'], params['n_aux']))
    weights['dual_term']    = np.zeros(params['n_term'] + params['n_term_ineq'])
    weights['dual_dyn']     = np.zeros((params['N'] - 1, params['n_dyn']))
    weights['dual_plus']    = np.zeros((params['N'] - 1, params['n_dyn']))
    weights['dual_minus']   = np.zeros((params['N'] - 1, params['n_dyn']))

    weights['W_path']       = np.zeros((params['N'], params['n_path']))
    weights['W_nfz']        = np.zeros((params['N'], params['n_nfz']))
    weights['W_aux']        = np.zeros((params['N'], params['n_aux']))
    weights['W_term']       = np.zeros(params['n_term'] + params['n_term_ineq'])
    weights['W_dyn']        = np.zeros((params['N'] - 1, params['n_dyn']))
    weights['W_plus']       = np.zeros((params['Npm'], params['n_plus']))
    weights['W_minus']      = np.zeros((params['Npm'], params['n_minus']))

    # --- Convergence tolerances ---
    conv = params.setdefault('conv', {})
    conv['eps_cost']    = 0.
    conv['eps_state']   = 0.
    conv['eps_path']    = 0.
    conv['eps_nfz']     = 0.
    conv['eps_aux']     = 0.
    conv['eps_term']    = 0.
    conv['eps_dyn']     = 0.

    setup = conv.setdefault('setup', {})
    for key in ['eps_cost', 'eps_state', 'eps_path', 'eps_nfz', 'eps_aux', 'eps_term', 'eps_dyn']:
        setup[key] = np.array([])

    setup['ctcs_mult_state'] = 1.0
    setup['ctcs_mult_cnst'] = 1.0

    params['eps_ctcs'] = 1e-5

    # --- Terminal nondimensionalization matrix ---
    M_state_vec = np.diag(params['nondim']['M']['state']['d2nd'])
    zf_idx      = params.get('zf_idx', np.array([], dtype=np.int64))
    zf_min_idx  = params.get('zf_min_idx', np.array([], dtype=np.int64))
    zf_max_idx  = params.get('zf_max_idx', np.array([], dtype=np.int64))
    M_term_diag = np.concatenate([M_state_vec[zf_idx],
                                  M_state_vec[zf_min_idx],
                                  M_state_vec[zf_max_idx]])
    params['nondim']['M']['term']['d2nd'] = np.diag(M_term_diag)

    # --- LTV indexing ---
    params = set_ltv_indices(params)

    # --- Initialize virtual buffers ---
    conv_data = params.setdefault('conv_data', {})
    conv_data['vb_path'] = np.zeros((params['N'],   params['n_path']))
    conv_data['vb_nfz']  = np.zeros((params['N'],   params['n_nfz']))
    conv_data['vb_aux']  = np.zeros((params['N'],   params['n_aux']))
    conv_data['vb_dyn']  = np.zeros((params['N']-1, params['nz']))
    conv_data['vb_term'] = np.zeros(params['nz'])

    return params


def set_problem_default(problem):
    """
    Populate default state/control bounds and boundary conditions
    from problem.params into top-level problem fields.

    Parameters:
        problem : dict with nested 'params' dict

    Returns:
        problem : updated dict
    """
    params = problem["params"]

    #
    # BOUNDARY CONDITIONS
    #
    # Initial conditions
    problem["zi"]           = params["zi"]
    problem["zi_idx"]       = params["zi_idx"]
    problem["zi_min"]       = params["zi_min"]
    problem["zi_min_idx"]   = params["zi_min_idx"]
    problem["zi_max"]       = params["zi_max"]
    problem["zi_max_idx"]   = params["zi_max_idx"]
    problem["n_init"]       = params["n_init"]
    problem["n_init_ineq"]  = params["n_init_ineq"]

    # Terminal conditions
    problem["zf"]           = params["zf"]
    problem["zf_idx"]       = params["zf_idx"]
    problem["zf_min"]       = params["zf_min"]
    problem["zf_min_idx"]   = params["zf_min_idx"]
    problem["zf_max"]       = params["zf_max"]
    problem["zf_max_idx"]   = params["zf_max_idx"]
    problem["n_term"]       = params["n_term"]
    problem["n_term_ineq"]  = params["n_term_ineq"]

    #
    # STATE CONSTRAINTS
    #
    problem["z_min"]        = params["z_min"]
    problem["z_min_idx"]    = params["z_min_idx"]
    problem["z_max"]        = params["z_max"]
    problem["z_max_idx"]    = params["z_max_idx"]

    #
    # CONTROL CONSTRAINTS
    #
    problem["u_min"]        = params["u_min"]
    problem["u_min_idx"]    = params["u_min_idx"]
    problem["u_max"]        = params["u_max"]
    problem["u_max_idx"]    = params["u_max_idx"]
    problem["n_ctrl"]       = params["n_ctrl"]
    problem["udot_max"]     = params["udot_max"]
    problem["udot_max_idx"] = params["udot_max_idx"]
    problem["n_udot"]       = params["n_udot"]

    return problem


# testing functions
if __name__ == "__main__":
    print('..:: Testing set_params_default ::..')

    # call without config
    print('no config argument passed')
    params = set_params_default()
    tf = 'N' in params
    print('N in params?: ', tf)
    print("params['bools']['flag_nfz'] = ", params['bools']['flag_nfz'])

    # test config params overwriting
    print('calling again with config argument')
    # make dummy config w/ data to overwrite params defaults
    config = {
        'params': { # config['params']
            'N': 40,
            'T_init': 10,
            'bools': { # config['params']['bools']
                'flag_nfz': 2,
                'flag_autotune': 0,
                'free_final_time': 1,
                'buff_dyn': 0,
                'ctcs': 0
            },
        },
    }
    params = set_params_default(config)
    print("params['bools']['flag_nfz'] = ", params['bools']['flag_nfz'])
    print("params['N'] = ", params['N'])

    # need to set nondim params before passing into set_params_constraint_defualt
    from trajopt.problem_models.quadrotor_3dof import set_nondim_params
    params = set_nondim_params(params)
    tf = 'nondim' in params # true/false
    print('nondim in params?: ', tf)
    # print("params['nondim'] = ", params['nondim'])

    # check if conv_data exists (it shouldn't yet)
    tf = 'conv_data' in params # true/false
    print('conv_data in params?: ', tf)

    # Now call set_params_constraint_default with params
    print('..:: Calling set_params_constraint_default ::..')
    params = set_params_constraint_default(params)
    
    # check if conv_data exists again (it should)
    tf = 'conv_data' in params # true/false
    print('conv_data in params?: ', tf)





