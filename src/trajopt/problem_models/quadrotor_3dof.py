import trajopt.utils.set_defaults           as defaults
import trajopt.utils.tools                  as tools
import trajopt.algorithm.initial_guess      as guess
import trajopt.algorithm.convergence        as convergence
import trajopt.algorithm.convexification    as convexify
import trajopt.algorithm.discretization     as discretize

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
    config['params']['N'] = 40

    config['params']['bools'] = {
        'flag_nfz': 2,          # 0, 1, 2
        'free_final_time': 1,   # 0, 1
        'equal_dt': 1,          # 0, 1
        'flag_autotune': '0',   # '0', '1', '2', '3', 'al-scvx'
        'buff_dyn': 'l1',       # 'term', 'l1', 'l2', 'quad-1', 'quad-2'
        'buff_dyn_dual': 'none',# 'l1', 'none'
        'ctcs': 0,              # 0, 1
        'ode_fixed_dt': 0,      # 0, 1 ,
        'nondim': 1,            # 0, 1
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

    if params["bools"]["auto_jac"]:
        problem["lin_cost"] = convexify.generate_jacobians(
            lambda ts, zs, us: problem["cost"](ts, zs, us, problem),
            problem
        )
    else:
        problem["lin_cost"] = lambda ts, zs, us: analytical_cost(ts, zs, us, problem)

    # Dynamics
    problem["xdot"] = lambda ts, zs, us, t_vec: system_dynamics(ts, zs, us, problem, t_vec)

    if params["bools"]["auto_jac"]:
        problem["lin_dyn"] = convexify.generate_jacobians(
            lambda ts, zs, us: system_dynamics(ts, zs, us, problem),
            problem
        )
    else:
        problem["lin_dyn"] = lambda ts, zs, us: analytical_linsys(ts, zs, us, problem)

    # Nonconvex inequality constraints
    problem["path_lim"] = params["path_lim"]
    problem["P"] = lambda ts, zs, us, t_vec: nonlinear_inequality_constraints(ts, zs, us, problem)

    if params["bools"]["auto_jac_cnst"]:
        problem["lin_constr"] = convexify.generate_jacobians(
            lambda ts, zs, us: nonlinear_inequality_constraints(ts, zs, us, problem),
            problem
        )
    else:
        problem["lin_constr"] = lambda ts, zs, us: analytical_inequality_constraints(ts, zs, us, problem)

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
    params['system']                    = 'quad3dof'
    params['case_flag']                 = 1    # case1
    params['bools']['auto_jac']         = 0    # (1=symbolic jacobians for dynamics, 0=analytical)
    params['bools']['auto_jac_aero']    = 0    # (1=symbolic jacobians for aerodynamics, 0=analytical)
    params['bools']['auto_jac_cnst']    = 0    # (1=symbolic jacobians for constraints, 0=analytical)
    params['bools']['init_ctrl']        = 0

    # Physical constants
    params['ge']        = np.array([0, 0, -9.81]) # [m/s^2], grav accel at sea lvl

    # Problem params
    params['n']         = 6
    params['m']         = 3

    # Time of flight
    params['T_init']    = 7.

    # Define Cost Function
    params['cost']      = lambda t, z, u: np.dot(np.transpose(u), u) 

    ### Vehicle Parameters ###
    params['mass']          = 0.35;                 # [kg], quadrotor mass
    params['theta_max']     = np.deg2rad(100.);     # [rad], maximum tilt angle

    #======================
    # Path /NFZ constraints
    #======================
    # no fly zones, specified by position and radius [rad]
    if params['bools']['flag_nfz'] == 1:
        xc_dim = np.array([5]) 
        yc_dim = np.array([4])
        rc_dim = np.array([2])
    elif params['bools']['flag_nfz'] == 2:
        xc_dim = np.array([2.5, 5,  2.5, 5.5,  8,  5.5]) # 5
        yc_dim = np.array([2,   2.5,  5, 5.25, 5.5, 8]) # 4
        rc_dim = np.ones(xc_dim.size)# 2, 1
    else:
        xc_dim = np.array([])
        yc_dim = np.array([])
        rc_dim = np.array([])

    params['nfz_idx']       = np.arange(0, xc_dim.size)
    params['n_nfz']         = len(params['nfz_idx'])

    ### Set dim/nondim params based on flag ###
    # scaling values for nondim
    params                  = set_nondim_params(params)

    xc = xc_dim / params['nondim']['nd']
    yc = yc_dim / params['nondim']['nd']
    rc = rc_dim / params['nondim']['nd']

    params.setdefault('obs', {})['posc'] = np.array([xc, yc]) # xc and yc may be vectors
    params['obs']['rc']     = rc

    #====================
    # Boundary Conditions
    #====================
    # initial conditions
    params['z0_dim']           = np.array([0,0,5,0,0.5,0])

    # equality initial conditions
    params['zi']            = params['nondim']['M_state_d2nd'] @ params['z0_dim']
    params['zi_idx']        = np.arange(0, params['n'])

    # inequality initial conditions
    # none

    # equality terminal conditions
    params['zf']            = params['nondim']['M_state_d2nd'] @ np.array([10,10,0.5,0,0,0])
    params['zf_idx']        = np.arange(0,params['n'])

    # control boundary conditions
    params['ui']            = -params['ge']*params['mass'] / params['nondim']['nf']
    params['uf']            = -params['ge']*params['mass'] / params['nondim']['nf']

    #==============================
    # Control and state constraints
    #==============================
    # no state constraints
    params['z_min']         = np.array([0, 0, 0.25]) / params['nondim']['nd']
    params['z_min_idx']     = np.arange(0,3)
    params['z_max']         = np.array([12, 12, 10]) / params['nondim']['nd']
    params['z_max_idx']     = np.arange(0,3)

    params['u_norm_min']    = 0.21 / params['nondim']['nf']
    params['u_norm_max']    = 8.12 / params['nondim']['nf']

    params['udot_max']      = 5*np.ones(3) / (params['nondim']['nf'] / params['nondim']['nt'])# [N/s]
    params['udot_max_idx']  = np.arange(0,3)


    ### Time of flight constraints ###
    Ts_min                  = 1. / params['nondim']['nt']  # 50
    Ts_max                  = 20. / params['nondim']['nt']
    params['ddts_max']      = 5. / ((params['N'] - 1) * params['nondim']['nt'])  # 0.025
    params['dts_min']       = Ts_min / (params['N'] - 1)
    params['dts_max']       = Ts_max / (params['N'] - 1)

    ### Set default constraint data ###
    params                  = defaults.set_params_constraint_default(params)


    #======================================
    # Initialize trajectory (initial guess)
    #======================================
    if params['bools']['free_final_time'] and (params['bools'].get('buff_dyn')=='term'):
        us_range = np.ones((2, 1)) @ ((-params['ge'].reshape(1, -1) * params['mass'])+ np.array([0.08, 0.08, 0.0])) / params['nondim']['nf']
        
        # need to manually set the left-hand side vector to a column vector for multiplacation to work
        params              = guess.nonlinear_initial_guess(us_range, params)

    else:
        params              = guess.straight_line_initial_guess(params) 
        params['us_init']   =  np.tile(-params['ge'] * params['mass'], (params['N'], 1)) / params['nondim']['nf']

    if params['bools']['ctcs']:
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
    params['weights']['w_cost']         = 0.
    params['weights']['eps_nonzero1']   = 2e-1
    params['weights']['eps_nonzero2']   = 1e-10

    # === Trust region weights ===
    params['weights'].setdefault('alpha_z', 0.5)
    params['weights'].setdefault('alpha_u', np.inf)

    params['weights']['wtr_z']          = 1 / (2 * params['weights']['alpha_z'])
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
                    w_nfz   = 1e2 * wbuff / params['weights']['w_fac_N']
                    w_dyn   = 1e5 * wbuff / params['weights']['w_fac_Nm1']
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

        if params['bools']['free_final_time']:
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
    ctcs_mult_state         = 5e-1
    ctcs_mult_cnst          = 1e0
    eps_ctcs                = 1e-5

    params['conv']['setup']['ctcs_mult_state']                  = ctcs_mult_state
    params['conv']['setup']['ctcs_mult_cnst']                   = ctcs_mult_cnst

    params['eps_ctcs']                                          = eps_ctcs

    ### State convergence ###
    eps_d_state             = 1e-2  # [m]
    eps_v_state             = 1e-2   # [m/s]
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
    eps_nfz_cnst            = 1e-1
    params['conv']['setup']['eps_nfz']                          = eps_nfz_cnst * np.ones(params['n_nfz'])
    params['conv']['setup'].setdefault('cnst', {})['eps_nfz']   = eps_nfz_cnst

    ### Terminal constraint values ###
    eps_d_term              = 1e-1
    eps_v_term              = 1e-2

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

def system_dynamics(ts,zs,us,params,t_vec=None):
    """
    x1, x2: r (position)
    u1, u2: v (velocity)
    """
    # extracts params if "problem" parent struct is passed in
    if hasattr(params, 'params'):
        params = params['params']

    # extract constant param values
    m       = int( params['m'] )
    n       = int( params['n'] )
    mass    = params['mass'] / params['nondim']['nm']
    ge      = params['ge'] / params['nondim']['na']

    # extract states
    r = zs[0:3]
    v = zs[3:6]

    # extract controls 
    if t_vec is None:
        us2 = us
    else:
        us2 = np.array([np.interp(ts, t_vec, us[:, i]) for i in range(m)])
            
    # extract control
    T = us2

    # compute velocity and acceleration
    xDot        = np.empty(6) # initialize
    xDot[0:3]   = v
    xDot[3:6]   = T/mass + ge

    if np.issubdtype(r.dtype, np.number):
        if r[2] <= -1: # set xDot = 0 if the vehicle hits the ground
            xDot = np.zeros(n)
    elif np.issubdtype(r.dtype, np.nan) or any(np.isinf(r)):
        breakpoint()
        
    return xDot

def analytical_linsys(ts, zs, us, problem):
    
    # Extract parameters
    params  = problem.get("params", problem)
    n       = params["n"]
    m       = params["m"]
    mass    = params["mass"] / params['nondim']['nm']

    # Sanity check for vector shapes
    zs = np.asarray(zs).flatten()
    us = np.asarray(us).flatten()

    assert len(zs) == n, f"Expected state vector of length {n}, got {len(zs)}"
    assert len(us) == m, f"Expected control vector of length {m}, got {len(us)}"

    # Compute A matrix (Jacobian w.r.t. state)
    n2 = n // 2
    Ac = np.block([
        [np.zeros((n2, n2)), np.eye(n2)],
        [np.zeros((n2, n))]
    ])

    # Compute B matrix (Jacobian w.r.t. control)
    Bc = np.vstack([
        np.zeros((n2, m)),
        np.eye(m)
    ]) * (1.0 / mass)

    # Evaluate nonlinear dynamics
    fc = system_dynamics(ts, zs, us, params)

    # Return in dictionary format
    linsys = {
        "dfcn_dz": Ac,
        "dfcn_du": Bc,
        "fcn":     fc
    }

    return linsys

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
    return P


def analytical_cost(ts, zs, us, problem):

    # Extract params
    params = problem.get("params", problem)
    n               = params["n"]
    m               = params["m"]
    N               = params["N"]

    ts              = np.asarray(ts).flatten()
    zs              = np.asarray(zs)
    us              = np.asarray(us)
    dt              = np.diff(ts)

    # Preallocate outputs
    dcostdz         = np.zeros((N, 1, n))
    dcostdu         = np.zeros((N, 1, m))
    cost            = np.zeros((N, 1, 1))

    for k in range(N - 1):
        tk          = ts[k]
        zk          = zs[k]
        uk          = us[k]

        tkp         = ts[k + 1]
        zkp         = zs[k + 1]
        ukp         = us[k + 1]

        dcostdu[k]  = 2 * ((uk + ukp) / 2).reshape(1, m)
        avg_cost    = 0.5 * (problem["cost"](tk, zk, uk) + problem["cost"](tkp, zkp, ukp))
        cost[k]     = avg_cost * dt[k]

    # Last step (N)
    dcostdz[N - 1]  = 0
    dcostdu[N - 1]  = 0
    cost[N - 1]     = 0

    # Package into output dict
    lincost = {
        "dfcn_dz": dcostdz,
        "dfcn_du": dcostdu,
        "fcn":     cost     
    }

    return lincost

def analytical_inequality_constraints(ts, zs, us, problem):
    params    = problem.get("params", problem)
    N         = tools.num_timesteps(zs)
    n         = params["n"]
    m         = params["m"]
    n_path    = params["n_path"]
    n_nfz     = params["n_nfz"]
    path_idx  = params["path_idx"]

    # Scale path limits using nondimensional constraint weights
    scale = params["nondim"]["np_ineq"][:n_path]
    path_lim_scaled = np.linalg.solve(np.diag(scale), params["path_lim"])

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




####### ALGORITHM 

def custom_inputs(problem,local_vars):
    u_norm_min  = problem["params"]["u_norm_min"]
    u_norm_max  = problem["params"]["u_norm_max"]
    theta_max   = problem["params"]["theta_max"]
    mass        = problem["params"]["mass"] / problem['params']['nondim']['nm']
    m           = problem["params"]["m"]
    ehat_u      = np.eye(m)
    u1          = problem["params"]["ui"]
    uN          = problem["params"]["uf"]

    local_vars.update(locals())

    return local_vars 

def custom_subprob_variables(problem,local_vars): 
    
    N           = problem["params"]["N"]

    u_slack     = cp.Variable((N,1))  # 1×N variable
    w_jerk      = 1e-1

    local_vars.update(locals())

    return local_vars 

def custom_subprob_constraints(CNST,local_vars):

    us_ref     = local_vars["us_ref"]
    du         = local_vars["sol_vars"]["du"]
    u1         = local_vars["u1"]
    uN         = local_vars["uN"]
    u_slack    = local_vars["u_slack"]
    u_norm_min = local_vars["u_norm_min"]
    u_norm_max = local_vars["u_norm_max"]
    theta_max  = local_vars["theta_max"]
    mass       = local_vars["mass"]
    ehat_u     = local_vars["ehat_u"]
    N          = local_vars["N"]

    # Boundary constraints
    CNST.append(us_ref[0] + du[0] == u1)
    CNST.append(us_ref[N-1] + du[N-1] == uN)

    for k in range(N):
        u_k     = us_ref[k] + du[k]
        slack_k = u_slack[k]
        
        CNST.append(cp.norm(u_k) <= slack_k)
        CNST.append(slack_k >= u_norm_min)
        CNST.append(slack_k <= u_norm_max)
        CNST.append(np.cos(theta_max) * slack_k - (1 / mass) * ehat_u[:, 2].T @ u_k <= 0)

    return CNST

def custom_subprob_cost(PTR_COST,local_vars):

    # Extract variables from local_vars
    ts_ref    = local_vars["ts_ref"]
    N         = local_vars["N"]
    u_slack   = local_vars["u_slack"]
    us_ref    = local_vars["us_ref"]
    du        = local_vars["sol_vars"]["du"]
    # w_jerk  = local_vars["w_jerk"]  # Uncomment if you include JERK_COST term

    # Compute dts_ref (time step differences)
    dts_ref = np.diff(ts_ref)  # shape: (N-1,)

    TRUE_COST = 0
    JERK_COST = 0

    for k in range(N - 1):
        TRUE_COST   += cp.square(u_slack[k + 1]) * dts_ref[k]

        jerk        = (us_ref[k + 1] + du[k + 1] - us_ref[k] - du[k]) / dts_ref[k]
        # JERK_COST += w_jerk * cp.sum_squares(jerk)

    PTR_COST        = PTR_COST + TRUE_COST + JERK_COST

    return PTR_COST


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
        nm_dot = nm / nt
        nf = nm * na
        np_ineq = np.ones(n_nfz) * nd**2
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

    # params['nondim']['M_term_d2nd'] = np.diag(np.concatenate([
    #     nd_state[params['zf_idx']],
    #     nd_state[params['zf_min_idx']],
    #     nd_state[params['zf_max_idx']]
    # ])).copy()
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

# TESTING CONFIG_PARAMS
if __name__ == "__main__":
    print('..:: Testing config_params() ::..')
    # make dummy config
    config = config_main()
    params = config_params(config)
    print(f"function call successful... \n\tparams['save_var_names'] = {params['save_var_names']}")