from trajopt.defaults.set_defaults import set_params_default, set_params_constraint_default
from trajopt.algorithm.initial_guess import nonlinear_initial_guess, ctcs_initial_guess, waypoint_initial_guess
from trajopt.algorithm.convergence import set_convergence_tolerance

def test_func():
    return "Hello, World!"

# Skye Mceowen
# Feb. 17th, 2024
# 2D circular NFZ
# Single integrator linear dynamics

# TODO consolidate imports 
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import numpy as np


def system_dynamics(ts,zs,us,params,t_vec=None):
    """
    x1, x2: r (position)
    u1, u2: v (velocity)
    """
    # extracts params if "problem" parent struct is passed in
    if hasattr(params, 'params'):
        params = params.params

    # extract constant param values
    m       = int( params['m'] )
    n       = int( params['n'] )
    mass    = params['mass']
    ge      = params['ge']

    # extract states
    r = zs[0:3]
    v = zs[3:6]

    # extract controls 
    if t_vec is None:
        us2 = us
    else:
        us2 = np.empty(m)
        for i in range(m):
            interp = interp1d(t_vec, us[i,:]) # does this work?
            us2[i] = interp(ts)
            
    # extract control
    T = us2

    # compute velocity and acceleration
    xDot = np.empty(6) # initialize
    xDot[0:3] = v
    xDot[3:6] = T/mass + ge

    if np.issubdtype(r.dtype, np.number):
        if r[2] <= -1: # set xDot = 0 if the vehicle hits the ground
            xDot = np.zeros(n)
    elif np.issubdtype(r.dtype, np.nan) or any(np.isinf(r)):
        breakpoint()
        
    return xDot


def set_nondim_params(params): # TODO: Test
    """
    Initializes all nondimensional parameters
    """
    # Extract dimension constants
    path_lim = params['path_lim']
    n_path = params['n_path']
    n_nfz = params['n_nfz']
    n = params['n']
    m = params['m']

    if params['bools']['nondim']:
        # set nondim params
        nd = 10
        nv = 10
        nt = nd / nv
        nt_inv = 1 / nt
        na = nv / nt
        nm = 1
        nm_dot = 1
        nf = 1
        np_ineq = np.ones(n_nfz) * nd
        ncost = nv
    else:
        # set dim params
        nt = 1
        nt_inv = 1
        nd = 1
        nv = 1
        na = 1
        nm = 1
        nm_dot = 1
        nf = 1
        np_ineq = np.ones(n_path + n_nfz)
        ncost = 1

    nd_state = np.array([1/nd, 1/nd, 1/nd, 1/nv, 1/nv, 1/nv])

    if 'nondim' not in params: # initialize if it doesn't already exist
       params['nondim'] = {}

    params['nondim']['M_state_d2nd'] = np.diag(nd_state).copy()
    params['nondim']['M_ctrl_d2nd'] = np.diag(np.ones(m) / na).copy()

    params['nondim']['M_term_d2nd'] = np.diag(np.concatenate([
        nd_state[params['zf_idx']],
        nd_state[params['zf_min_idx']],
        nd_state[params['zf_max_idx']]
    ])).copy()
    params['nondim']['M_cnst_d2nd'] = np.diag(np_ineq ** -1).copy()
    params['nondim']['M_nfz_d2nd'] = np.diag(np_ineq[params['nfz_idx']] ** -1).copy()

    nd_dyn = np.array([1/nv, 1/nv, 1/nv, 1/na, 1/na, 1/na])
    params['nondim']['M_dyn_d2nd'] = np.diag(nd_dyn).copy()

    params['nondim']['M_cost_d2nd'] = 1 / ncost

    params['nondim']['nu_rad_ind'] = []

    # add scalar nondim variables to nondim substruct
    params['nondim']['nd'] = nd
    params['nondim']['na'] = na
    params['nondim']['nt'] = nt
    params['nondim']['nt_inv'] = nt_inv
    params['nondim']['nv'] = nv
    params['nondim']['nm'] = nm
    params['nondim']['nm_dot'] = nm_dot
    params['nondim']['nf'] = nf
    params['nondim']['np_ineq'] = np_ineq
    params['nondim']['ncost'] = ncost

    return params


def config_params(config=None): # replacing init_params_struct TODO: Test
    """
    Configures parameters dictionary for quadrotor_3dof example problem
    """
    # Initialize
    params = set_params_default(config)
    params['case_flag']             = 1    # case1: bank angle only
    params['bools']['auto_jac']        = 0    # (1=symbolic jacobians for dynamics, 0=analytical)
    params['bools']['auto_jac_aero']   = 0    # (1=symbolic jacobians for aerodynamics, 0=analytical)
    params['bools']['auto_jac_cnst']   = 0    # (1=symbolic jacobians for constraints, 0=analytical)
    params['bools']['init_ctrl']       = 0

    # Physical constants
    params['ge'] = np.array([0, 0, -9.81]) # [m/s^2], grav accel at sea lvl

    # Problem params
    params['n'] = 6
    params['m'] = 3

    # Time of flight
    params['T_init'] = 10

    # Define Cost Function
    params['cost'] = lambda t, z, u: np.sum(u**2) #equivalent to dot product...TODO this will be a method at some point


    #======================
    # Path /NFZ constraints
    #======================
    # no fly zones, specified by position and radius [rad]
    if params['bools']['flag_nfz'] == 0:
        xc = np.array([])
        yc = np.array([])
        rc = np.array([])
    elif params['bools']['flag_nfz'] == 1:
        xc = 5
        yc = 4
        rc = 2
    elif params['bools']['flag_nfz'] == 2:
        xc = np.array([2.5, 5,  2.5, 5.5,  8,  5.5])  # 5
        yc = np.array([2,   2.5,  5, 5.25, 5.5, 8])   # 4
        rc = np.ones(xc.size)  # 2, 1
    else:
        xc = np.array([])
        yc = np.array([])
        rc = np.array([])

    params.setdefault('obs', {})['posc'] = np.array([xc, yc]) # xc and yc may be vectors
    params['obs']['rc'] = rc
    
    params['nfz_idx'] = np.arange(0, xc.size)
    params['n_nfz'] = len(params['nfz_idx'])


    ### Vehicle Parameters ###
    params['mass'] = 0.35;                  # [kg], quadrotor mass
    params['theta_max'] = np.deg2rad(100);  # [rad], maximum tilt angle

    ### Set dim/nondim params based on flag ###
    # scaling values for nondim
    params = set_nondim_params(params)

    #====================
    # Boundary Conditions
    #====================
    # initial conditions
    params['z0s'] = np.array([0,0,0.5,0,0,0])

    # equality initial conditions
    params['zi'] = params['z0s']
    params['zi_idx'] = np.arange(0, params['n'])

    # inequality initial conditions
    # none

    # equality terminal conditions
    params['zs'] = params['nondim']['M_state_d2nd']*np.array([10,10,0.5,0,0,0])
    params['zf_idx'] = np.arange(0,params['n'])


    #==============================
    # Control and state constraints
    #==============================
    # no state constraints
    params['z_min'] = np.array([0, 0, 0.25])
    params['z_min_idx'] = np.arange(0,3)
    params['z_max'] = np.array([12, 12, 0.75])
    params['z_max_idx'] = np.arange(0,3)

    params['u_norm_min'] = 0.21 # [N]
    params['u_norm_max'] = 8.12 # [N]

    params['udot_max'] = 5*np.ones(3) # [N/s]
    params['udot_max_idx'] = np.arange(0,3)


    ### Time of flight constraints ###
    Ts_min = 1 / params['nondim']['nt']  # 50
    Ts_max = 10 / params['nondim']['nt']
    params['ddts_max'] = 5 / ((params['N'] - 1) * params['nondim']['nt'])  # 0.025
    params['dts_min'] = Ts_min / (params['N'] - 1)
    params['dts_max'] = Ts_max / (params['N'] - 1)

    ### Set default constraint data ###
    params = set_params_constraint_default(params)


    #======================================
    # Initialize trajectory (initial guess)
    #======================================
    if params['bools']['free_final_time'] and not params['bools']['buff_dyn']:
        us_range = ( -params['ge'].reshape(-1,1) * params['mass'] ) @ np.ones((1, 2)) + np.array([0.08, 0.08, 0]).reshape(-1,1)
        # need to manually set the left-hand side vector to a column vector for multiplacation to work
        params = nonlinear_initial_guess(us_range, params)
    else:
        params = waypoint_initial_guess(params)  # waypoint_initial_guess(params) # straight_line_initial_guess
        params['us_init'] = ( -params['ge'].reshape(-1,1) * params['mass'] ) @ np.ones((1, params['N']))

    if params['bools']['ctcs']:
        params = ctcs_initial_guess(params)

    #============================================
    # Optimization parameters and hyperparameters
    #============================================
    # PTR penalty weights
        # Wtr: weight for trust region cost                        
        # w_term: weight for terminal constraint buffer cost
        # w_path: weight for path constraint buffer cost
        # w_nfz: weight for path constraint buffer cost

    params['weights']['w_cost'] = 0
    params['weights']['eps_nonzero1'] = 2e-1
    params['weights']['eps_nonzero2'] = 1e-10

    # trust region
    if 'alpha_z' not in params['weights']:
        params['weights']['alpha_z'] = 5e-1  # 1e0, 1e-1, 5e-1
        params['weights']['alpha_u'] = np.inf  # 1e1, 1e0, 1e1

    params['weights']['wtr_z'] = 1 / (2 * params['weights']['alpha_z'])

    # Handle division by infinity for wtr_u
    if params['weights']['alpha_u'] == np.inf:
        params['weights']['wtr_u'] = 0
    else:
        params['weights']['wtr_u'] = 1 / (2 * params['weights']['alpha_u'])


    # no autotuning, Skye1/autotune2, Skye-Behcet3/autotune3
    if params['bools']['flag_autotune'] in [0, 2, 3]:
        
        # Autotuning meta-tuning dual variable trust region weights
            # beta-dual eq, gamma-dual ineq
        if 'beta' not in params['weights']:
            params['weights']['beta'] = 1
        if 'gamma' not in params['weights']:
            params['weights']['gamma'] = 1e-1  # 5e-3, 1e-2, 1e-1

        # Buffer variables penalty weights
        if 'wbuff' not in params['weights']:
            if params['bools']['flag_autotune'] == 0:
                wbuff = 1e2
            else:
                wbuff = 1
            w_nfz = wbuff / params['weights']['w_fac_N']  # NFZ
            w_dyn = 1e5 * wbuff  # DYNAMICS (CONTROL)
            w_term = 1e2 * wbuff  # TERMINAL
            if 'W_nfz' in params['weights']:
                params['weights']['W_nfz'] += w_nfz
            else:
                params['weights']['W_nfz'] = w_nfz
        else:
            wbuff = 1
            w_nfz = wbuff / params['weights']['w_fac_N']  # NFZ
            w_dyn = 1e6 * wbuff  # DYNAMICS (CONTROL)
            w_term = 1e2 * wbuff  # TERMINAL

        if params['bools']['free_final_time']:
            if params['bools']['buff_dyn']:
                if 'W_dyn' in params['weights']:
                    params['weights']['W_dyn'] += w_dyn
                else:
                    params['weights']['W_dyn'] = w_dyn
            else:
                if 'W_term' in params['weights']:
                    params['weights']['W_term'] += w_term
                else:
                    params['weights']['W_term'] = w_term

    if params['bools']['flag_autotune'] in [1,3]:

        # autotuning meta-tuning dual variable trust region weights
            # beta-dual eq, gamma-dual ineq
        if 'beta' not in params['weights']:
            params['weights']['beta'] = 1
        if 'gamma' not in params['weights']:
            params['weights']['gamma'] = 1e-1 # 5e-3, 1e-2, 1e-1

        # Update dual_nfz
        if 'dual_nfz' in params['weights']:
            if 'eps_nonzero2' in params['weights']:
                params['weights']['dual_nfz'] += params['weights']['eps_nonzero2']
            else:
                raise ValueError("eps_nonzero2 is not defined.")
        else:
            if 'eps_nonzero2' in params['weights']:
                params['weights']['dual_nfz'] = params['weights']['eps_nonzero2']
            else:
                raise ValueError("Both dual_nfz and eps_nonzero2 are not defined.")

        if params['bools']['free_final_time']: 
            # Update dual_dyn or dual_term depending on buff_dyn bool
            if params['bools']['buff_dyn']:
                if 'dual_dyn' in params['weights'] and 'eps_nonzero2' in params['weights']:
                    params['weights']['dual_dyn'] += params['weights']['eps_nonzero2']
                else:
                    raise ValueError("Either dual_dyn or eps_nonzero2 is not defined.")
            else:
                if 'dual_term' in params['weights'] and 'eps_nonzero2' in params['weights']:
                    params['weights']['dual_term'] += params['weights']['eps_nonzero2']
                else:
                    raise ValueError("Either dual_term or eps_nonzero2 is not defined.")

    ### ctcs convergence adjustments ###
    ctcs_mult_state = 5e-1
    ctcs_mult_cnst = 1e0
    eps_ctcs = 1e-5

    params['conv']['setup']['ctcs_mult_state'] = ctcs_mult_state
    params['conv']['setup']['ctcs_mult_cnst'] = ctcs_mult_cnst

    params['eps_ctcs'] = eps_ctcs

    ### State convergence ###
    eps_d_state = 1e-1  # [m]
    eps_v_state = 1e0   # [m/s]
    params['conv']['setup']['eps_state'] = np.concatenate((eps_d_state * np.ones(params['n'] // 2), 
                                                        eps_v_state * np.ones(params['n'] // 2)))

    params['conv']['setup'].setdefault('state', {})['eps_d'] = eps_d_state
    params['conv']['setup']['state']['eps_v'] = eps_v_state


    
    ### Cost convergence ###
    eps_F_cost = 1 # N

    # Assign to cost eps and store data
    params['conv']['setup']['eps_cost'] = eps_F_cost
    params['conv']['setup'].setdefault('cost', {})['eps_v'] = eps_F_cost

    ### NFZ convergence values ###
    eps_nfz_cnst = 1e-1
    params['conv']['setup']['eps_nfz'] = eps_nfz_cnst * np.ones(params['n'])
    params['conv']['setup'].setdefault('cnst', {})['eps_nfz'] = eps_nfz_cnst

    ### Terminal constraint values ###
    eps_d_term = 1e-1
    eps_v_term = 1e-2

    # Create eps_vector for full terminal state equality, min, max constraints
    eps_term = np.array([eps_d_term, eps_d_term, eps_d_term, eps_v_term, eps_v_term, eps_v_term])
    eps_term_min = eps_term.copy()
    eps_term_max = eps_term.copy()

    # Extract only those terminal constraints used
    params['conv']['setup']['eps_term'] = np.concatenate((eps_term[params['zf_idx']], 
                                                        eps_term_min[params['zf_min_idx']], 
                                                        eps_term_max[params['zf_max_idx']]))

    # Store data
    params['conv']['setup'].setdefault('term', {})['eps_d'] = eps_d_term

    #### Configure multiple shooting dynamics defect convergence values ###
    params['conv']['setup']['eps_defect'] = np.array([1e-2])

    ### Dynamics convergence ###
    eps_d_dyn = 1e-1  # [m]
    eps_v_dyn = 1e0   # [m/s]
    params['conv']['setup']['eps_dyn'] = np.concatenate((eps_d_dyn * np.ones(params['n'] // 2), 
                                                        eps_v_dyn * np.ones(params['n'] // 2)))

    # Store data
    params['conv']['setup'].setdefault('dyn', {})['eps_d'] = eps_d_dyn
    params['conv']['setup']['dyn']['eps_v'] = eps_v_dyn

    ### Configure generic convergence criterion and max iterations ###
    params = set_convergence_tolerance(params)

    # Iterations
    params['conv']['iter_max'] = 20  # 14, 30
    # params['conv']['num_buffers'] = 4

    # Save variable names
    params['save_var_names'] = ['ts_opt', 'zs_opt', 'us_opt', 'params', 'O']

    return params


# # testing set_nondim_params()
# if __name__ == "__main__":
#     print('..:: Testing set_nondim_params() ::..')
#     params = {
#         'path_lim': None,
#         'n_path': 0,
#         'n_nfz': 6,
#         'nfz_idx': [0,1,2,3,4,5],
#         'zf_idx': None,
#         'zf_min_idx': None,
#         'zf_max_idx': None,
#         'n': 6,
#         'm': 3,
#         'bools' : {
#             'nondim': 1,
#         },
#     }
#     params = set_nondim_params(params)
#     print("params['nondim'] = ", params['nondim'])

# TESTING CONFIG_PARAMS
if __name__ == "__main__":
    print('..:: Testing config_params() ::..')
    # make dummy config
    config = {
        'params': { # config['params']
            'N': 40,
            'T_init': 10,
            'bools': { # config['params']['bools']
                'flag_nfz': 0,
                'flag_autotune': 0,
                'free_final_time': 1,
                'buff_dyn': 0,
                'ctcs': 0
            },
        },
    }
    params = config_params(config)
    print(f"function call successful... \n\tparams['save_var_names'] = {params['save_var_names']}")