# Skye Mceowen and Aman Tiwary
# Feb. 14, 2024
# functions that set default problem parameters

import numpy as np
from trajopt.algorithm.discretization import set_ltv_indices

def set_params_default(config=None): # TODO: Edit docstring
    """
    Set default problem parameters

    Parameters:
    config (dict): problem configuration data

    Returns:
    params (dict): problem parameters
    """
    params = {}

    if config is None:
        config = {}

    # booleans
    params['bools'] = {
        'auto_jac': 1,               # (1=symbolic jacobians for dynamics, 0=analytical)
        'auto_jac_aero': 1,          # (1=symbolic jacobians for aerodynamics, 0=analytical)
        'auto_jac_cnst': 1,          # (1=symbolic jacobians for constraints, 0=analytical)
        'dev_var': 1,                # (1=deviation variables, 0=full-state)
        'nondim': 0,                 # (1=nondimensionalization, 0=dimensional)
        'var_scl_flag': 0,           # (2=linear scaling, 1=affine, 0=no scaling)
        'free_final_time': 0,        # final time of trajectory is a variable
        'equal_dt': 0,               # all time intervals are equal
        'flag_nfz': 0,               # no fly zone constraints 
        'flag_autotune': 3,
        'flag_Wauto_memory': 0,
        'weight_zero': 0,
        'stepsize_auto_primal': 0,
        'stepsize_auto_dual': 0,
        'flag_conv': 0,              # 0-both, 1-dx/violation, 2-cost/violation
        'buff_dyn': 0,
        'ctcs': 0,
        'ode_fixed_dt': 0
    }

    # state and control
    params['n'] = 1
    params['m'] = 1

    # constraints
    params['path_lim'] = np.array([])
    params['path_idx'] = np.array([], dtype=int)
    params['n_path'] = params['path_idx'].size
    params['nfz_idx'] = np.array([], dtype=int)
    params['n_nfz'] = params['nfz_idx'].size
    params['aux_idx'] = np.array([], dtype=int)
    params['n_aux'] = params['aux_idx'].size
    params['n_ineq'] = params['n_path'] + params['n_nfz'] + params['n_aux']
    params['n_eq'] = 0

    # initial boundary conditions
    params['zi'] = np.array([])
    params['zi_idx'] = np.array([], dtype=int)
    params['zi_min'] = np.array([])
    params['zi_min_idx'] = np.array([], dtype=int)
    params['zi_max'] = np.array([])
    params['zi_max_idx'] = np.array([], dtype=int)
    params['n_init'] = params['zi_idx'].size
    params['n_init_ineq'] = params['zi_min_idx'].size + params['zi_max_idx'].size

    # terminal boundary conditions
    params['zf'] = np.array([])
    params['zf_idx'] = np.array([], dtype=int)
    params['zf_min'] = np.array([])
    params['zf_min_idx'] = np.array([], dtype=int)
    params['zf_max'] = np.array([])
    params['zf_max_idx'] = np.array([], dtype=int)
    params['n_term'] = params['zf_idx'].size
    params['n_term_ineq'] = params['zf_min_idx'].size + params['zf_max_idx'].size

    # state constraints
    params['z_min'] = np.array([])
    params['z_min_idx'] = np.array([], dtype=int)
    params['z_max'] = np.array([])
    params['z_max_idx'] = np.array([], dtype=int)
    params['n_state'] = params['z_min_idx'].size + params['z_max_idx'].size

    # control constraints
    params['bools']['init_ctrl'] = 1
    params['u_min'] = np.array([])
    params['u_min_idx'] = np.array([], dtype=int)
    params['u_max'] = np.array([])
    params['u_max_idx'] = np.array([], dtype=int)
    params['n_ctrl'] = params['u_min_idx'].size + params['u_max_idx'].size
    params['udot_max'] = np.array([])
    params['udot_max_idx'] = np.array([], dtype=int)
    params['n_udot'] = params['udot_max_idx'].size

    # weights
    params['weights'] = {}
    params['n_dyn'] = 0

    # convergence
    params['conv'] = {
        'eps_feas_conv': 1e-1,
        'eps_conv': 1e-1,
        'eps_cost_conv': 1e-1
    }

    # overwrite fields with config
    if 'params' in config:
        for key, value in config['params'].items():
            if key == 'bools':
                for subkey, subvalue in value.items():
                    params['bools'][subkey] = subvalue
            else:
                params[key] = value

    if params['bools']['ctcs']:
        params['bools']['buff_dyn'] = 1

    return params


def set_params_constraint_default(params): # TODO: Test
    # Constraint bookkeeping
    params['n_init'] = len(params['zi_idx'])
    params['n_init_ineq'] = len(params['zi_min_idx'] + params['zi_max_idx'])
    params['n_term'] = len(params['zf_idx'])
    params['n_term_ineq'] = len(params['zf_min_idx'] + params['zf_max_idx'])
    params['n_ctrl'] = len(params['u_min_idx'] + params['u_max_idx'])
    params['n_state'] = len(params['z_min_idx'] + params['z_max_idx'])
    params['n_udot'] = len(params['udot_max'])

    params['n_path'] = len(params['path_idx'])  # Number of path constraints
    params['n_nfz'] = len(params['nfz_idx'])  # Number of NFZs
    params['n_aux'] = len(params['aux_idx'])  # Number of auxiliary constraints
    params['n_ineq'] = params['n_nfz'] + params['n_path'] + params['n_aux']

    # CTCS
    if params['bools']['ctcs']:
        params['nz'] = params['n'] + params['n_ineq']
    else:
        params['nz'] = params['n']

    # Buffering dynamics
    params['n_dyn'] = params['bools']['buff_dyn'] * params['nz']

    params['weights']['w_fac_N'] = params['N']
    params['weights']['w_fac_Nm1'] = params['N'] - 1

    # All autotuning schemes - true cost and dual initialization
    params['weights']['w_cost'] = 1e0
    params['weights']['dual_path'] = np.zeros((params['n_path'], params['N']))
    params['weights']['dual_nfz'] = np.zeros((params['n_nfz'], params['N']))
    params['weights']['dual_aux'] = np.zeros((params['n_aux'], params['N']))
    params['weights']['dual_dyn'] = np.zeros((params['nz'], params['N'] - 1))
    params['weights']['dual_term'] = np.zeros((params['n_term'] + params['n_term_ineq'], 1))

    # Buffer variables penalty weights initialization
    params['weights']['W_path'] = np.zeros((params['n_path'], params['N']))
    params['weights']['W_nfz'] = np.zeros((params['n_nfz'], params['N']))
    params['weights']['W_aux'] = np.zeros((params['n_aux'], params['N']))
    params['weights']['W_dyn'] = np.zeros((1, params['N'] - 1))
    params['weights']['W_term'] = np.zeros((params['n_term'] + params['n_term_ineq'], 1))

    # Convergence criteria
    params['conv']['eps_cost'] = 0
    params['conv']['eps_state'] = 0
    params['conv']['eps_path'] = 0
    params['conv']['eps_nfz'] = 0
    params['conv']['eps_aux'] = 0
    params['conv']['eps_term'] = 0
    params['conv']['eps_dyn'] = 0
    params['conv'].setdefault('setup', {})['eps_cost'] = []
    params['conv']['setup']['eps_state'] = []
    params['conv']['setup']['eps_path'] = []
    params['conv']['setup']['eps_nfz'] = []
    params['conv']['setup']['eps_aux'] = []
    params['conv']['setup']['eps_term'] = []
    params['conv']['setup']['eps_dyn'] = []
    params['conv']['setup']['ctcs_mult_state'] = 1
    params['conv']['setup']['ctcs_mult_cnst'] = 1
    params['eps_ctcs'] = 1e-5

    # Terminal state nondimensionalization
    M_state_d2nd_vec = params['nondim']['M_state_d2nd']
    params['nondim']['M_term_d2nd'] = np.diag(np.concatenate((M_state_d2nd_vec[params['zf_idx']], 
                                                             M_state_d2nd_vec[params['zf_min_idx']], 
                                                             M_state_d2nd_vec[params['zf_max_idx']])))

    # Discrete LTV matrix data
    params = set_ltv_indices(params)

    # Initial virtual buffers
    params.setdefault('conv_data', {})['vb_path'] = np.zeros((params['n_path'], params['N']))
    params['conv_data']['vb_nfz'] = np.zeros((params['n_nfz'], params['N']))
    params['conv_data']['vb_aux'] = np.zeros((params['n_aux'], params['N']))
    params['conv_data']['vb_dyn'] = np.zeros((params['nz'], params['N'] - 1))
    params['conv_data']['vb_term'] = np.zeros((params['nz'], 1))

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





