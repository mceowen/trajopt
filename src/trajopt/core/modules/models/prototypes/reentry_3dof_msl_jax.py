import trajopt.utils.set_defaults           as defaults
import trajopt.utils.tools                  as tools
import trajopt.core.modules.methods.initial_guess      as guess
import trajopt.core.modules.methods.convergence        as convergence
import trajopt.core.modules.methods.convexify    as convexify
import trajopt.core.modules.methods.discretize     as discretize
import trajopt.utils.nondim                 as nondim

# TODO consolidate imports 
from scipy.interpolate  import interp1d
from scipy.integrate    import solve_ivp
import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import os
import sys
sys.path.append("/Users/carlosm/Documents/guidance/hypersonics/prototypes/local")
import marsgram_dens_lut as dens
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
    config['model_type']    = 'reentry_3dof_msl_jax'
    config['mission']       = config['model_type']
    config['case_flag']     = 1  # 1: double integrator

    config['flags'] = {
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
    if config['flags']['setfig_paper']:
        # In Python, you would use matplotlib.rcParams or seaborn.set_context()
        pass

    # --- User problem setup ---
    config['params'] = {}

    config['params']['flags'] = {
        'flag_nfz': 0,          # 0, 1, 2
        'free_final_time': 1,   # 0, 1
        'equal_dt': 1,          # 0, 1
        'flag_autotune': '0',   # '0', '1', '2', '3', 'al-scvx'
        'buff_dyn': 'term',       # 'term', 'l1', 'l2', 'quad-1', 'quad-2'
        'buff_dyn_dual': 'none',# 'l1', 'none'
        'ctcs': 0,              # 0, 1
        'ode_fixed_dt': 0,      # 0, 1 ,
        'nondim': 1,            # 0, 1
        'jax_dyn': 1
    }

    # todo: clean this
    config['params']['model_type'] = config['model_type']

    # --- Solver options - TODO:expand ---
    config['params']['solver_opts'] = {
        'solver': 'CLARABEL',
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

    if config['flags']['multiplot']:
        dataset_path = Path(config['paths']['data_path']) / f"dataset{config['dataset_id']}/iterations/"
    else:
        dataset_path = Path(config['paths']['data_path']) / f"dataset{config['dataset_id']}/standalone/"

    config['paths']['dataset_path'] = str(dataset_path)
    config['paths']['dataset_file'] = str(dataset_path / f"case{config['case_flag']}_standalone.mat")

    return config

def ocp(config):
    """
    Define the optimal control problem (OCP):
    - cost
    - constraints
    - dynamics

    Parameters:
        config : configuration dictionary or object

    Returns:
        problem : dictionary describing the OCP
    """
    problem = {}

    problem["name"] = "3DoF Quadrotor"

    # Ingest parameters
    problem["config"]   = config
    params              = config_params(config)
    problem["params"]   = params
    problem["ts_init"]  = params["ts_init"]
    problem["zs_init"]  = params["zs_init"]
    problem["us_init"]  = params["us_init"]

    # Default state/control bounds
    problem             = defaults.set_problem_default(problem)

    # Cost function
    problem["cost"]     = params["cost"]
    problem["cost_init"] = problem["cost"](params["ts_init"], params["zs_init"], params["us_init"])

    if params["method"]['flags']["auto_jac"]:
        problem["lin_cost"] = convexify.generate_jacobians(
            lambda ts, zs, us: problem["cost"](ts, zs, us, problem),
            problem
        )
    else:
        problem["lin_cost"] = lambda ts, zs, us: analytical_cost(ts, zs, us, problem)

    # Dynamics
    # problem["xdot"] = lambda ts, zs, us, t_vec: system_dynamics(ts, zs, us, problem, t_vec)

    # if params["method"]['flags']["auto_jac"]:
    #     problem["lin_dyn"] = convexify.generate_jacobians(
    #         lambda ts, zs, us: system_dynamics(ts, zs, us, problem),
    #         problem
    #     )
    # else:
    #     problem["lin_dyn"] = lambda ts, zs, us: analytical_linsys(ts, zs, us, problem)

    problem["lin_dyn"] = convexify.generate_lin_sys_jax(system_dynamics, params)

    # # Nonconvex inequality constraints
    problem["mission"]["path_lim"] = params["mission"]["path_lim"]
    problem["P"] = lambda ts, zs, us, t_vec: nonlinear_inequality_constraints(ts, zs, us, problem)

    if params["method"]['flags']["auto_jac_cnst"]:
        problem["lin_constr"] = convexify.generate_jacobians(
            lambda ts, zs, us: nonlinear_inequality_constraints(ts, zs, us, problem),
            problem
        )
    else:
        problem["lin_constr"] = lambda ts, zs, us: analytical_inequality_constraints(ts, zs, us, problem)

    # precompile discretize functions for jax
    if problem['params']['flags']['jax_dyn'] == 1:
        problem = discretize.jit_jax_discretize(problem)

    # Algorithm - custom formulation
    problem["custom_inputs"]        = lambda problem,   local_vars:     custom_inputs(problem, local_vars)
    problem["custom_variables"]     = lambda problem,   local_vars:     custom_subprob_variables(problem, local_vars)
    problem["custom_constraints"]   = lambda CNST,      local_vars:     custom_subprob_constraints(CNST, local_vars)
    problem["custom_cost"]          = lambda PTR_COST,  local_vars:     custom_subprob_cost(PTR_COST, local_vars)

    # Plotting
    # TODO
    # problem["plots"] = lambda prob: init_plot_struct(prob)

    return problem


def config_params(config=None): # replacing init_params_struct TODO: Test
    """
    Configures parameters dictionary for quadrotor_3dof example problem
    """
    # Initialize
    params = defaults.set_params_default(config)
    params['system']                    = 'msl_reentry_jax'
    params['case_flag']                 = 1    # case1
    params['flags']['auto_jac']         = 0    # (1=symbolic jacobians for dynamics, 0=analytical)
    params['flags']['auto_jac_aero']    = 0    # (1=symbolic jacobians for aerodynamics, 0=analytical)
    params['flags']['auto_jac_cnst']    = 0    # (1=symbolic jacobians for constraints, 0=analytical)
    params['flags']['init_ctrl']        = 0

    # === Case setup ===
    params['nondim_on'] = False
    params['case_flag'] = 1
    params['N']         = 40
    params['n']         = 6
    params['m']         = 2
    params['T_init']    = 200.0

    # === Physical constants ===
    params['ge']    = 3.7132                         # [m/s^2]
    params['re']    = 3396190                        # Mars radius [m]
    params['rhoe']  = 0.020                         # Surface atmospheric density [kg/m^3]
    params['H']     = 11.1e3                        # Scale height [m]
    params['beta']  = 1.0 / params['H']             # Inverse scale height
    params['mue']   = params['ge'] * params['re']**2  # Gravitational parameter for Mars
    # No rotation for now (can adjust later)
    params['omega'] = 0.0

    # === Vehicle mass & reference geometry ===
    params['mass']  = 2900           # kg (MSL landed mass ~900, entry mass ~2900)
    params['sref']  = 15.9           # m^2 (MSL aeroshell reference area)
    params['LD']    = 0.24           # Ballistic coefficient L/D
    params['bc']    = 120            # Ballistic coefficient β = m / (Cd * S)

    # Define Cost Function
    params['cost']      = lambda t, z, u: z[-1, 3]

    #======================
    # Path /NFZ constraints
    #======================
    # no fly zones, specified by position and radius [rad]
    if params['flags']['flag_nfz'] == 1:
        xc_dim = np.array([5]) 
        yc_dim = np.array([4])
        rc_dim = np.array([2])
    elif params['flags']['flag_nfz'] == 2:
        xc_dim = np.array([2.5, 5,  2.5, 5.5,  8,  5.5]) # 5
        yc_dim = np.array([2,   2.5,  5, 5.25, 5.5, 8]) # 4
        rc_dim = np.ones(xc_dim.size)# 2, 1
    else:
        xc_dim = np.array([])
        yc_dim = np.array([])
        rc_dim = np.array([])

    params['nfz_idx']       = np.arange(0, xc_dim.size)
    params['n_nfz']         = len(params['nfz_idx'])

    # set nondim scaling

    z_types = ['d', 'ang', 'ang', 'v', 'ang', 'ang']
    u_types = ['ang', 'ang']

    nt = np.sqrt(params['re'] / params['ge'])
    nv = np.sqrt(params['re'] * params['ge'])
    nm = params['mass']

    anchor_scales = [('t', nt), ('v', nv), ('m', nm)]
    base_unit_labels = ['m', 's', 'kg']

    params = nondim.set_nondim_params(z_types, u_types, anchor_scales, params, base_unit_labels=base_unit_labels)

    # set nondim for cost and constraints
    np_ineq = np.ones(params['n_nfz']) * params['nondim']['nd']**2
    ncost = params['nondim']['nv']

    params = nondim.set_cost_cnst_nondim_params(np_ineq, ncost, params)

    xc = xc_dim / params['nondim']['nd']
    yc = yc_dim / params['nondim']['nd']
    rc = rc_dim / params['nondim']['nd']

    params['kg']      = params['mue'] / (params['nondim']['na'] * params['nondim']['nd']**2)
    params['omega_s'] = params['omega'] * params['nondim']['nt'] 

    params.setdefault('obs', {})['posc'] = np.array([xc, yc]) # xc and yc may be vectors
    params['obs']['rc']     = rc

    #====================
    # Boundary Conditions
    #====================
    # initial conditions

    # === Initial state
    h0      = 126e3                  # Entry altitude [m]
    theta0  = np.deg2rad(0)
    phi0    = np.deg2rad(0)
    v0      = 5845                   # Entry velocity [m/s]
    gamma0  = np.deg2rad(-15.47)     # Entry flight path angle [rad]
    psi0    = 0.0  

    params['z0_dim'] = np.array([(params['re'] + h0) , theta0 , phi0 , v0 , gamma0 , psi0])

    # equality initial conditions
    params['zi']            = params['nondim']['M']['state']['d2nd'] @ params['z0_dim']
    params['zi_idx']        = np.arange(0, params['n'])

    # inequality initial conditions
    # none

    # equality terminal conditions
    params['zf_dim']        = np.array([(params['re'] + 10e3), 0, np.deg2rad(10.5768), 406, np.deg2rad(-10), 0])
    params['zf']            = params['nondim']['M']['state']['d2nd'] @ params['zf_dim']
    params['zf_idx']        = np.arange(0,params['n'] // 2)

    # # control boundary conditions
    # params['ui']            = -params['ge']*params['mass'] / params['nondim']['nf']
    # params['uf']            = -params['ge']*params['mass'] / params['nondim']['nf']

    #==============================
    # Control and state constraints
    #==============================
    # no state constraints
    params['z_min']         = np.array([(params['re'] ) / params['nondim']['nd'], 100 / params['nondim']['nv']]) 
    params['z_min_idx']     = np.array([0, 3])

    params['u_min']         = np.array([np.deg2rad(-170), np.deg2rad(-5)]) 
    params['u_min_idx']     = np.array([0, 1])

    params['u_max']         = np.array([np.deg2rad(170), np.deg2rad(5)]) 
    params['u_max_idx']     = np.array([0, 1])


    ### Time of flight constraints ###
    Ts_min                  = 180. / params['nondim']['nt']  # 50
    Ts_max                  = 400. / params['nondim']['nt']
    params['ddts_max']      = 20. / ((params['N'] - 1) * params['nondim']['nt'])  # 0.025
    params['dts_min']       = Ts_min / (params['N'] - 1)
    params['dts_max']       = Ts_max / (params['N'] - 1)

    ### Set default constraint data ###
    params                  = defaults.set_params_constraint_default(params)


    #======================================
    # Initialize trajectory (initial guess)
    #======================================
    if params['flags']['free_final_time'] and (params['flags'].get('buff_dyn')=='term'):

        us_range = np.tile(np.array([np.deg2rad(5), 0]).reshape(-1, 1), (1, 2))
        
        # need to manually set the left-hand side vector to a column vector for multiplacation to work
        params              = guess.nonlinear_initial_guess(us_range, params)

    else:
        params              = guess.straight_line_initial_guess(params) 
        params['us_init']   =  np.tile([np.deg2rad(5), 0], (params['N'], 1))

    if params['flags']['ctcs']:
        params              = guess.ctcs_initial_guess(params)


    #============================================
    # Optimization parameters and hyperparameters
    #============================================
    # PTR penalty weights
        # Wtr: weight for trust region cost                        
        # w_term: weight for terminal constraint buffer cost
        # w_path: weight for path constraint buffer cost
        # w_nfz: weight for path constraint buffer cost

    # === Baseline cost + trust region weights ===
    params['weights']['w_cost']         = 1e3
    params['weights']['eps_nonzero1']   = 2e-1
    params['weights']['eps_nonzero2']   = 1e-10

    # === Trust region weights ===
    params['weights'].setdefault('alpha_z', 0.5)
    params['weights'].setdefault('alpha_u', 0.5)

    params['weights']['wtr_z']          = 1 / (2 * params['weights']['alpha_z'])
    params['weights']['wtr_u']          = 0 if np.isinf(params['weights']['alpha_u']) else 1 / (2 * params['weights']['alpha_u'])

    # === Autotune modes (flag_autotune ∈ {0,2,3,al-scvx}) ===
    if str(params['flags']['flag_autotune']) in {'0', '2', '3', 'al-scvx'}:

        params['weights'].setdefault('beta', 1)
        params['weights'].setdefault('gamma', 1e-1)

        # --- Buffer weights ---
        if str(params['flags']['flag_autotune']) in {'0', 'al-scvx'}:
            if 'wbuff' not in params['weights']:
                wbuff = 1e2
                if str(params['flags']['flag_autotune']) == '0':
                    w_nfz   = wbuff / params['weights']['w_fac_N']
                    w_dyn   = 1e6 * wbuff / params['weights']['w_fac_Nm1']
                    w_term  = 1e2 * wbuff
                else:
                    w_nfz   = wbuff / params['weights']['w_fac_N']
                    w_dyn   = wbuff / params['weights']['w_fac_Nm1']
                    w_term  = wbuff
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

        if params['flags']['free_final_time'] or params['flags']['ctcs']:
            buff_dyn = str(params['flags'].get('buff_dyn', ''))
            if buff_dyn in {'l1', 'l2'}:
                params['weights']['W_dyn'] += w_dyn
            elif buff_dyn in {'quad-1', 'quad-2', 'quad-3'}:
                params['weights']['W_plus'] += w_dyn
                params['weights']['W_minus'] += w_dyn
            else:
                params['weights']['W_term'] += w_term

    # === Autotune mode: {1,3,al-scvx} ===
    if str(params['flags']['flag_autotune']) in {'1', '3', 'al-scvx'}:

        params['weights'].setdefault('beta', 1)
        params['weights'].setdefault('gamma', 1e-1)

        params['weights']['dual_nfz'] += params['weights']['eps_nonzero1']

        if params['flags']['free_final_time']:
            buff_dyn = str(params['flags'].get('buff_dyn', ''))
            if buff_dyn == 'term':
                params['weights']['dual_term'] += params['weights']['eps_nonzero1']
            else:
                params['weights']['dual_dyn'] += params['weights']['eps_nonzero1']

                if str(params['flags'].get('buff_dyn_dual', '')) == 'l1':
                    params['weights']['dual_plus'] += params['weights']['eps_nonzero1']
                    params['weights']['dual_minus'] += params['weights']['eps_nonzero1']

    ### ctcs convergence adjustments ###
    ctcs_mult_state         = 5e-1
    ctcs_mult_cnst          = 1e0
    eps_ctcs                = 1e-4

    params['conv']['setup']['ctcs_mult_state']                  = ctcs_mult_state
    params['conv']['setup']['ctcs_mult_cnst']                   = ctcs_mult_cnst

    params['eps_ctcs']                                          = eps_ctcs

    ### State convergence ###
    eps_d_state             = 10  # [m]
    eps_v_state             = 10   # [m/s]
    eps_ang_state           = np.deg2rad(1) # [rad]

    params['conv']['setup']['eps_state']                        = np.array([eps_d_state, eps_ang_state, eps_ang_state, eps_d_state, eps_ang_state, eps_ang_state])

    params['conv']['setup'].setdefault('state', {})['eps_d']    = eps_d_state
    params['conv']['setup']['state']['eps_v']                   = eps_v_state

    ### Cost convergence ###
    eps_F_cost              = 1e0 # [m/s]

    # Assign to cost eps and store data
    params['conv']['setup']['eps_cost'] = eps_F_cost
    params['conv']['setup'].setdefault('cost', {})['eps_v']     = eps_F_cost

    ### NFZ convergence values ###
    eps_nfz_cnst            = 1e-1
    params['conv']['setup']['eps_nfz']                          = eps_nfz_cnst * np.ones(params['n_nfz'])
    params['conv']['setup'].setdefault('cnst', {})['eps_nfz']   = eps_nfz_cnst

    ### Terminal constraint values ###
    eps_d_term              = 10
    eps_v_term              = 10
    eps_ang_term            = np.deg2rad(1) # [rad]

    # Create eps_vector for full terminal state equality, min, max constraints
    eps_term                = np.array([eps_d_term, eps_ang_term, eps_ang_term, eps_v_term, eps_ang_term, eps_ang_term])
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
    eps_d_dyn               = 10  # [m/s]
    eps_v_dyn               = 10   # [m/s^2]
    eps_ang_dyn             = np.deg2rad(1) # [rad/s]
    params['conv']['setup']['eps_dyn']                          = np.array([eps_d_dyn, eps_ang_dyn, eps_ang_dyn, eps_v_dyn, eps_ang_dyn, eps_ang_dyn])

    # Store data
    params['conv']['setup'].setdefault('dyn', {})['eps_d']      = eps_d_dyn
    params['conv']['setup']['dyn']['eps_v']                     = eps_v_dyn

    ### Configure generic convergence criterion and max iterations ###
    params = convergence.set_convergence_tolerance(params)

    # Iterations
    params['conv']['iter_max']  = 10

    # Save variable names
    params['save_var_names']    = ['ts_opt', 'zs_opt', 'us_opt', 'params', 'O']

    return params

# ============================================================
# Dynamics defintions
# ============================================================

def nonlinear_aero(ts, zs, us, params):
    # Extract states and controls
    rs, _, _, vs, _, _ = zs

    # Compute altitude
    rdim = rs*params['nondim']['nd']
    hdim = rdim - params['re']
    
    rho = jnp.interp(hdim/1e3, dens.h_grid, dens.rho_vals)

    rho_s = rho / (params['nondim']['nm'] / params['nondim']['nd']**3)
    sref_s = params['sref'] / params['nondim']['nd']**2
    bc_s = params['bc'] / (params['nondim']['nm'] / (params['nondim']['nd']**2))

    D    = 0.5 * (1 / bc_s) * rho_s * vs**2
    L    = D * params['LD'] 

    alpha = 0

    return {'L': L, 'D': D, 'alpha': alpha, 'rho': rho}

def system_dynamics(ts, zs, us, params, t_vec=None):

    # Extract constant param values from struct
    Om = params['omega_s']
    Kg = params['kg']

    # Extract states
    rs, theta, phi, vs, gamma, psi = zs

    # Determine lift and drag coefficients from velocity
    aero = nonlinear_aero(ts, zs, us, params)
    L    = aero['L']
    D    = aero['D']

    # extract controls 
    if t_vec is None:
        us2 = us
    else:
        us2 = np.array([jnp.interp(ts, t_vec, us[:, i]) for i in range(params['m'])])

    # Extract bank angle
    sigma   = us2[0]
    alpha   = us2[1]

    # Extract sines and cosines of various values
    cp  = jnp.cos(phi)
    sp  = jnp.sin(phi)
    tp  = jnp.tan(phi)
    cg  = jnp.cos(gamma)
    sg  = jnp.sin(gamma)
    tg  = jnp.tan(gamma)
    cps = jnp.cos(psi)
    sps = jnp.sin(psi)

    cs  = jnp.cos(sigma)
    ss  = jnp.sin(sigma)
    
    # state derivative function
    xDot = jnp.array([
        vs * sg,
        vs * cg * sps / (rs * cp),
        vs * cg * cps / rs, 
        - D - Kg * sg / rs**2 + Om**2 * rs * cp * (sg * cp - cg * sp * cps),
        (1 / vs) * ( L * cs + (vs**2 - Kg / rs) * cg / rs ) + 2 * Om * cp * sps + Om**2 * rs * (1 / vs) * cp * (cg * cp + sg * cps * sp),
        (1 / vs) * ( L * ss / cg + vs**2 * cg * sps * tp / rs ) - 2 * Om * (tg * cps * cp - sp) + Om**2 * rs * (1 / (vs * cg)) * sps * sp * cp
    ])

    return xDot


def nonlinear_inequality_constraints(ts, zs, us, params):

    # Extract nested params if needed
    if "params" in params:
        params = params["params"]

    N       = tools.num_timesteps(zs)
    n_nfz   = params.get("n_nfz", 0)
    n_path  = params.get("n_path", 0)  # placeholder

    # Handle state unpacking
    if zs.ndim == 2:
        rx = zs[:, 0]
        ry = zs[:, 1]
    elif zs.ndim == 1:
        rx = np.full(N, zs[0])
        ry = np.full(N, zs[1])
    else:
        raise ValueError(f"Unhandled zs shape {zs.shape}")

    # === NFZ constraints ===
    if n_nfz > 0:
        xc = params["obs"]["posc"][0]
        yc = params["obs"]["posc"][1]
        rc = params["obs"]["rc"]

        P_nfz = np.stack([
            rc[i]**2 - (rx - xc[i])**2 - (ry - yc[i])**2
            for i in range(n_nfz)
        ], axis=1)  # shape: (N, n_nfz)
    else:
        P_nfz = np.empty((N, 0))

    # === Path constraints (placeholder) ===
    P_path = np.empty((N, 0))  # currently unused

    # === Stack all inequality constraints ===
    P = np.hstack([P_path, P_nfz]) if P_path.size or P_nfz.size else np.empty((N, 0))

    if zs.ndim == 1:
        P = P.flatten()

    return P

def analytical_inequality_constraints(ts, zs, us, problem):
    params    = problem.get("params", problem)
    N         = tools.num_timesteps(zs)
    n         = params["model"]["n"]
    m         = params["model"]["m"]
    n_path    = params["n_path"]
    n_nfz     = params["mission"]["n_nfz"]
    path_idx  = params["mission"]["path_idx"]

    # Scale path limits using nondimensional constraint weights
    scale = params["nondim"]["np_ineq"][:n_path]
    path_lim_scaled = np.linalg.solve(np.diag(scale), params["mission"]["path_lim"])

    # Obstacle info (broadcasted for speed)
    if n_nfz > 0:
        xc = params["obs"]["posc"][0]
        yc = params["obs"]["posc"][1]
        rc = params["obs"]["rc"]

    # === Preallocate flattened output arrays ===
    fcn_all   = np.zeros((N, n_path + n_nfz))
    dPdz_all  = np.zeros((N, n_path + n_nfz, n))
    dPdu_all  = np.zeros((N, n_path + n_nfz, m))

    # Also collect detailed path and NFZ constraint data if needed
    path_data = {"P": [], "Praw": [], "dPdz": [], "dPdu": []}
    nfz_data  = {"P": [], "dPdz": [], "dPdu": []}

    if zs.ndim == 1:
        ts = np.array([ts])
        zs = zs.reshape((1, -1))
        us = us.reshape((1, -1))

    for k in range(N):
        tk = ts[k]
        zk = zs[k]
        uk = us[k]
        rx_k, ry_k = zk[0], zk[1]

        # Evaluate all inequality constraints
        P_full = nonlinear_inequality_constraints(tk, zk, uk, params)

        # === Path constraints ===
        if n_path > 0:
            P_path = P_full[path_idx]
            fcn_all[k, :n_path] = P_path - path_lim_scaled
            dPdz_all[k, :n_path, :] = np.zeros((n_path, n))  # Placeholder
            dPdu_all[k, :n_path, :] = np.zeros((n_path, m))

            # Append to raw data
            path_data["P"].append(P_path - path_lim_scaled)
            path_data["Praw"].append(P_path)
            path_data["dPdz"].append(np.zeros((n_path, n)))
            path_data["dPdu"].append(np.zeros((n_path, m)))

        # === No-fly zone constraints ===
        if n_nfz > 0:
            P_nfz = P_full[n_path:n_path + n_nfz]
            fcn_all[k, n_path:n_path + n_nfz] = P_nfz

            dPdz_nfz = np.zeros((n_nfz, n))
            dPdz_nfz[:, 0] = - 2 * (rx_k - xc)
            dPdz_nfz[:, 1] = - 2 * (ry_k - yc)

            dPdu_nfz = np.zeros((n_nfz, m))

            dPdz_all[k, n_path:n_path + n_nfz, :] = dPdz_nfz
            dPdu_all[k, n_path:n_path + n_nfz, :] = dPdu_nfz

            # Store full data
            nfz_data["P"].append(P_nfz)
            nfz_data["dPdz"].append(dPdz_nfz)
            nfz_data["dPdu"].append(dPdu_nfz)

    return {
        "fcn": fcn_all,
        "dfcn_dz": dPdz_all,
        "dfcn_du": dPdu_all,
        "data": {
            "path": path_data,
            "nfz": nfz_data,
        }
    }


def analytical_cost(ts, zs, us, problem):

    # Extract params
    params = problem.get("params", problem)
    n               = params["model"]["n"]
    m               = params["model"]["m"]
    N               = params["N"]

    ts              = np.asarray(ts).flatten()
    zs              = np.asarray(zs)
    us              = np.asarray(us)
    dt              = np.diff(ts)

    # Preallocate outputs
    dcostdz         = np.zeros((N, 1, n))
    dcostdu         = np.zeros((N, 1, m))
    cost            = np.zeros((N, 1, 1))

    # Last step (N) (minimize terminal velocity)
    dcostdz[-1, 0, :] = np.array([0, 0, 0, 1, 0, 0])
    dcostdu[-1]  = 0
    cost[-1]     = zs[-1, 3]

    # Package into output dict
    lincost = {
        "dfcn_dz": dcostdz,
        "dfcn_du": dcostdu,
        "fcn":     cost     
    }

    return lincost


####### ALGORITHM 

def custom_inputs(problem,local_vars):

    return local_vars 

def custom_subprob_variables(problem,local_vars): 

    return local_vars 

def custom_subprob_constraints(CNST,local_vars):

    return CNST

def custom_subprob_cost(PTR_COST,local_vars):

    return PTR_COST


# TESTING CONFIG_PARAMS
if __name__ == "__main__":
    print('..:: Testing config_params() ::..')
    # make dummy config
    config = config_main()
    params = config_params(config)
    print(f"function call successful... \n\tparams['save_var_names'] = {params['save_var_names']}")