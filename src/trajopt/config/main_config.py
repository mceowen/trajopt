import trajopt.utils.set_defaults           as defaults
import trajopt.utils.tools                  as tools
import trajopt.algorithm.initial_guess      as guess
import trajopt.algorithm.convergence        as convergence
import trajopt.algorithm.convexification    as convexify
import trajopt.utils.nondim                 as nondim

# TODO consolidate imports 
from scipy.interpolate  import interp1d
from scipy.integrate    import solve_ivp
import numpy as np
import cvxpy as cp
import os
from pathlib import Path

def config_main():

    config = {}

    # --- Paths ---
    config['paths'] = {
        'home': os.path.expanduser('~/ACL/sandbox/scp_sandbox/'),
        'shim': ''
    }

    # NOTE: cd and addpath are MATLAB-specific and not used in Python.
    # Instead, assume working directory is already set or handled externally.

    # --- Set base defaults ---
    config['model_type']    = 'quadrotor_3dof'
    config['mission']       = config['model_type']
    config['case_flag']     = 1  # 1: double integrator

    config['bools'] = {
        'opt': 1,
        'plot': 1,
        'multiplot': 0,
        'it_plots': 1,
        'save': 0,
        'save_fig': 0,
        'setfig_paper': 1,
        'dock_fig': 1
    }

    config['dataset_id'] = ''

    # --- Plot settings (placeholder) --- TODO
    if config['bools']['setfig_paper']:
        # In Python, you would use matplotlib.rcParams or seaborn.set_context()
        pass

    # --- User problem setup ---
    config['params'] = {}
    config['params']['N'] = 40  # timesteps; 40 for no ctcs, 10 for ctcs

    config['params']['bools'] = {
        'flag_nfz': 2,           # 0, 1, 2
        'free_final_time': 1,    # 0, 1
        'equal_dt': 1,           # 0, 1
        'flag_autotune': '0',    # '0', '1', '2', '3', 'al-scvx'
        'buff_dyn': 'term',      # 'term', 'l1', 'l2', 'quad-1', 'quad-2'
        'buff_dyn_dual': 'none', # 'l1', 'none'
        'ctcs': 0,               # 0, 1
        'ode_fixed_dt': 0,       # 0, 1 ,
        'nondim': 1,             # 0, 1
    }

    # todo: clean this
    config['params']['model_type'] = config['model_type']

    # --- Solver options - TODO:expand ---
    config['params']['solver_opts'] = {
        'solver': 'qoco',
    }

    # --- Paths for problem-specific model and data ---
    model_path = Path(f"test_problems/{config['model_type']}/")
    config['paths']['model_path'] = str(model_path)

    if config['case_flag'] == 1:
        case_path = 'case1'
    elif config['case_flag'] == 2:
        case_path = 'case2'
    else:
        raise ValueError("Undefined case_flag!")

    config['paths']['problem_path'] = str(model_path / case_path)
    config['paths']['data_path'] = f"data/{config['model_type']}/"

    if config['bools']['multiplot']:
        dataset_path = Path(config['paths']['data_path']) / f"dataset{config['dataset_id']}/iterations/"
    else:
        dataset_path = Path(config['paths']['data_path']) / f"dataset{config['dataset_id']}/standalone/"

    config['paths']['dataset_path'] = str(dataset_path)
    config['paths']['dataset_file'] = str(dataset_path / f"case{config['case_flag']}_standalone.mat")

    return config

def problem_params(config=None): # replacing init_params_struct TODO: Test
    """
    Configures parameters dictionary for quadrotor_3dof example problem
    """
    # Initialize
    params = defaults.set_params_default(config)

    params['system']                    = 'quad3dof'
    params['case_flag']                 = 1    # case1
    params['bools']['auto_jac']         = 0    # (1=symbolic jacobians for dynamics, 0=analytical)
    params['bools']['auto_jac_aero']    = 0    # (1=symbolic jacobians for aerodynamics, 0=analytical)
    params['bools']['auto_jac_cnst']    = 0    # (1=symbolic jacobians for constraints, 0=analytical)
    params['bools']['init_ctrl']        = 0

    # Problem params
    params['n'] = 6
    params['m'] = 3

    # set nondim scaling
    z_types = ['d', 'd', 'd', 'v', 'v', 'v']
    u_types = ['f', 'f', 'f']

    if params['bools']['nondim'] == 1:
        anchor_scales = [('d', 10), ('v', 10), ('m', 1)]
    else:
        anchor_scales = [('d', 1 ), ('v', 1 ), ('m', 1)]
    
    base_unit_labels = ['m', 's', 'kg']
    params = nondim.set_nondim_params(z_types, u_types, anchor_scales, params, base_unit_labels=base_unit_labels)

    params['ge']   = np.array([0, 0, -9.81])

    params['T_init']    = 7.
    
    ### Vehicle Parameters ###
    params['mass']          = 0.35;                 # [kg], quadrotor mass
    params['theta_max']     = np.deg2rad(100.);     # [rad], maximum tilt angle

    return params

def method_params(params):

    #============================================
    # Optimization parameters and hyperparameters
    #============================================
    # PTR penalty weights
        # Wtr: weight for trust region cost                        
        # w_term: weight for terminal constraint buffer cost
        # w_path: weight for path constraint buffer cost
        # w_nfz: weight for path constraint buffer cost

    # === Baseline cost + trust region weights ===
    params['weights']['w_cost']         = 0.
    params['weights']['eps_nonzero1']   = 2e-1
    params['weights']['eps_nonzero2']   = 1e-10

    M_state  = params['nondim']['M']['state']['nd2d']
    avg_state_nd_sq  = np.mean(np.diag(M_state)**2)

    # === Trust region weights ===
    params['weights'].setdefault('alpha_z', 0.5)
    params['weights'].setdefault('alpha_u', np.inf)

    params['weights']['wtr_z']          = avg_state_nd_sq  * 1 / (2 * params['weights']['alpha_z'])
    params['weights']['wtr_u']          = 0 if np.isinf(params['weights']['alpha_u']) else 1 / (2 * params['weights']['alpha_u'])

    # === Autotune modes (flag_autotune ∈ {0,2,3,al-scvx}) ===
    if str(params['bools']['flag_autotune']) in {'0', '2', '3', 'al-scvx'}:

        params['weights'].setdefault('beta', 1)
        params['weights'].setdefault('gamma', 1e-1)

        # --- Buffer weights ---
        if str(params['bools']['flag_autotune']) in {'0', 'al-scvx'}:
            if 'wbuff' not in params['weights']:
                wbuff = 1e2
                if str(params['bools']['flag_autotune']) == '0':

                    w_nfz_dim  = wbuff / params['weights']['w_fac_N']
                    w_dyn_dim  = 1e5 * wbuff / params['weights']['w_fac_Nm1']
                    w_term_dim = 1e2 * wbuff

                    # scaled nondim weights to approximately preserve relative scaling between cost terms
                    M_nfz  = params['nondim']['M']['nfz']['nd2d']
                    M_dyn  = params['nondim']['M']['dyn']['nd2d']
                    M_term = params['nondim']['M']['term']['nd2d']

                    avg_nfz_nd_sq  = np.mean(np.diag(M_nfz)**2)
                    avg_dyn_nd_sq  = np.mean(np.diag(M_dyn)**2)
                    avg_term_nd_sq = np.mean(np.diag(M_term)**2)

                    w_nfz   = avg_nfz_nd_sq  * w_nfz_dim
                    w_dyn   = avg_dyn_nd_sq  * w_dyn_dim
                    w_term  = avg_term_nd_sq * w_term_dim
            else:
                wbuff = params['weights']['wbuff']
                w_nfz = wbuff / params['weights']['w_fac_N']
                w_dyn = wbuff / params['weights']['w_fac_Nm1']
                w_term = wbuff
        else:
            wbuff = 1
            w_nfz = wbuff / params['weights']['w_fac_N']
            w_dyn = wbuff / params['weights']['w_fac_Nm1']
            w_term = wbuff

        params['weights']['W_nfz'] += w_nfz

        if params['bools']['free_final_time'] or params['bools']['ctcs']:
            buff_dyn = str(params['bools'].get('buff_dyn', ''))
            if buff_dyn in {'l1', 'l2'}:
                params['weights']['W_dyn'] += w_dyn
            elif buff_dyn in {'quad-1', 'quad-2', 'quad-3'}:
                params['weights']['W_plus'] += w_dyn
                params['weights']['W_minus'] += w_dyn
            else:
                params['weights']['W_term'] += w_term

    # === Autotune mode: {1,3,al-scvx} ===
    if str(params['bools']['flag_autotune']) in {'1', '3', 'al-scvx'}:

        params['weights'].setdefault('beta', 1)
        params['weights'].setdefault('gamma', 1e-1)

        params['weights']['dual_nfz'] += params['weights']['eps_nonzero1']

        if params['bools']['free_final_time']:
            buff_dyn = str(params['bools'].get('buff_dyn', ''))
            if buff_dyn == 'term':
                params['weights']['dual_term'] += params['weights']['eps_nonzero1']
            else:
                params['weights']['dual_dyn'] += params['weights']['eps_nonzero1']

                if str(params['bools'].get('buff_dyn_dual', '')) == 'l1':
                    params['weights']['dual_plus'] += params['weights']['eps_nonzero1']
                    params['weights']['dual_minus'] += params['weights']['eps_nonzero1']

    ### ctcs convergence adjustments ###
    ctcs_mult_state         = 1e0
    ctcs_mult_cnst          = 1e0 
    eps_ctcs                = 1e-4

    params['conv']['setup']['ctcs_mult_state']                  = ctcs_mult_state
    params['conv']['setup']['ctcs_mult_cnst']                   = ctcs_mult_cnst

    params['eps_ctcs']                                          = eps_ctcs
    params['weights']['w_ctcs']                                 = params['nondim']['nd']**2

    ### State convergence ###
    eps_d_state             = 1e-1  # [m]
    eps_v_state             = 1e-1   # [m/s]
    params['conv']['setup']['eps_state']                        = np.concatenate((eps_d_state * np.ones(params['n'] // 2), 
                                                                    eps_v_state * np.ones(params['n'] // 2)))

    params['conv']['setup'].setdefault('state', {})['eps_d']    = eps_d_state
    params['conv']['setup']['state']['eps_v']                   = eps_v_state

    ### Cost convergence ###
    eps_F_cost              = 1e0 # N

    # Assign to cost eps and store data
    params['conv']['setup']['eps_cost'] = eps_F_cost
    params['conv']['setup'].setdefault('cost', {})['eps_v']     = eps_F_cost

    ### NFZ convergence values ###
    eps_nfz_dim             = 1e-1 # [m]
    eps_nfz_cnst            = 2 * params['obs']['rc'] * eps_nfz_dim - eps_nfz_dim**2
    params['conv']['setup']['eps_nfz']                          = eps_nfz_cnst * np.ones(params['n_nfz'])
    params['conv']['setup'].setdefault('cnst', {})['eps_nfz']   = eps_nfz_cnst

    ### Terminal constraint values ###
    eps_d_term              = 1e-1 # [m]
    eps_v_term              = 1e-2 # [m/s]

    # Create eps_vector for full terminal state equality, min, max constraints
    eps_term                = np.array([eps_d_term, eps_d_term, eps_d_term, eps_v_term, eps_v_term, eps_v_term])
    eps_term_min            = eps_term.copy()
    eps_term_max            = eps_term.copy()

    # Extract only those terminal constraints used
    params['conv']['setup']['eps_term']                         = np.concatenate((eps_term[params['zf_idx']], 
                                                                    eps_term_min[params['zf_min_idx']], 
                                                                    eps_term_max[params['zf_max_idx']]))

    # Store data
    params['conv']['setup'].setdefault('term', {})['eps_d']     = eps_d_term

    #### Configure multiple shooting dynamics defect convergence values ###
    params['conv']['setup']['eps_defect']                       = np.array([1e-2])

    ### Dynamics convergence ###
    eps_d_dyn               = 1e-3  # [m]
    eps_v_dyn               = 1e-3   # [m/s]
    params['conv']['setup']['eps_dyn']                          = np.concatenate((eps_d_dyn * np.ones(params['n'] // 2), 
                                                                    eps_v_dyn * np.ones(params['n'] // 2)))

    # Store data
    params['conv']['setup'].setdefault('dyn', {})['eps_d']      = eps_d_dyn
    params['conv']['setup']['dyn']['eps_v']                     = eps_v_dyn

    ### Configure generic convergence criterion and max iterations ###
    params = convergence.set_convergence_tolerance(params)

    # Iterations
    params['conv']['iter_max']  = 20

    # Save variable names
    params['save_var_names']    = ['ts_opt', 'zs_opt', 'us_opt', 'params', 'O']

    return params
