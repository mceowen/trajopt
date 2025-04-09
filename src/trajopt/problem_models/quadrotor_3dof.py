import trajopt.utils.set_defaults           as defaults
import trajopt.algorithm.initial_guess      as guess
import trajopt.algorithm.convergence        as convergence
import trajopt.algorithm.convexification    as convexify
import trajopt.algorithm.discretization     as discretize

# TODO consolidate imports 
from scipy.interpolate  import interp1d
from scipy.integrate    import solve_ivp
import numpy as np


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
    problem["cost_init"] = problem["cost"](params["ts_init"], params["zs_init"], params["us_init"], problem)

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
    params['cost'] = lambda t, z, u: np.dot(np.transpose(u), u) #equivalent to dot product...TODO this will be a method at some point


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
    params = defaults.set_nondim_params(params)

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
    params['zf'] = params['nondim']['M_state_d2nd'] @ np.array([10,10,0.5,0,0,0])
    params['zf_idx'] = np.arange(0,params['n'])

    # control boundary conditions
    params['ui'] = -params['ge']*params['mass']
    params['uf'] = -params['ge']*params['mass']

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
    params = defaults.set_params_constraint_default(params)


    #======================================
    # Initialize trajectory (initial guess)
    #======================================
    if params['bools']['free_final_time'] and not params['bools']['buff_dyn']:
        us_range = ( -params['ge'].reshape(-1,1) * params['mass'] ) @ np.ones((1, 2)) + np.array([0.08, 0.08, 0]).reshape(-1,1)
        # need to manually set the left-hand side vector to a column vector for multiplacation to work
        params = guess.nonlinear_initial_guess(us_range, params)
    else:
        params = guess.waypoint_initial_guess(params) 
        params['us_init'] = ( -params['ge'].reshape(-1,1) * params['mass'] ) @ np.ones((1, params['N']))

    if params['bools']['ctcs']:
        params = guess.ctcs_initial_guess(params)

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
    params = convergence.set_convergence_tolerance(params)

    # Iterations
    params['conv']['iter_max'] = 20  # 14, 30
    # params['conv']['num_buffers'] = 4

    # Save variable names
    params['save_var_names'] = ['ts_opt', 'zs_opt', 'us_opt', 'params', 'O']

    return params

def analytical_linsys(ts, zs, us, problem):
    
    # Extract parameters
    params = problem.get("params", problem)
    n = params["n"]
    m = params["m"]
    mass = params["mass"]

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
    """
    Compute nonlinear inequality constraints: path constraints and no-fly zones.

    Parameters:
        ts     : (N,) time array
        zs     : (n, N) state array
        us     : (m, N) control array (unused in current logic)
        params : dict with fields:
                 - n_nfz
                 - obs["posc"], obs["rc"]
                 - path constraints (placeholder, unused)

    Returns:
        P : (n_ineq, N) constraint matrix, where each column is P[:, k]
    """
    # Extract nested params if needed
    if "params" in params:
        params = params["params"]

    N = zs.shape[1]
    n_nfz = params["n_nfz"]
    n_path = params.get("n_path", 0)  # currently unused

    P_path = []  # placeholder for path constraints
    P_nfz = []

    for k in range(N):
        rx_k = zs[0, k]
        ry_k = zs[1, k]

        # --- No Fly Zone constraints ---
        P_nfz_k = []
        if n_nfz > 0:
            for i in range(n_nfz):
                xc = params["obs"]["posc"][0, i]
                yc = params["obs"]["posc"][1, i]
                Rc = params["obs"]["rc"][i]

                val = Rc**2 - (rx_k - xc)**2 - (ry_k - yc)**2
                P_nfz_k.append(val)

        P_nfz.append(P_nfz_k)

    P_nfz = np.array(P_nfz).T if P_nfz else np.empty((0, N))
    P_path = np.empty((0, N))  # not implemented

    # Stack all inequality constraints
    P = np.vstack([P_path, P_nfz])
    return P


def analytical_cost(ts, zs, us, problem):

    # Extract params
    params = problem.get("params", problem)
    n = params["n"]
    m = params["m"]
    N = params["N"]

    ts = np.asarray(ts).flatten()
    zs = np.asarray(zs)
    us = np.asarray(us)
    dt = np.diff(ts)

    # Preallocate outputs
    dcostdz = np.zeros((1, n, N))
    dcostdu = np.zeros((1, m, N))
    cost    = np.zeros((1, 1, N))

    for k in range(N - 1):
        tk   = ts[k]
        zk   = zs[:, k]
        uk   = us[:, k]

        tkp  = ts[k + 1]
        zkp  = zs[:, k + 1]
        ukp  = us[:, k + 1]

        dcostdu[:, :, k] = 2 * ((uk + ukp) / 2).reshape(1, m)
        avg_cost = 0.5 * (problem["cost"](tk, zk, uk) + problem["cost"](tkp, zkp, ukp))
        cost[:, :, k] = avg_cost * dt[k]

    # Last step (N)
    dcostdz[:, :, N - 1] = 0
    dcostdu[:, :, N - 1] = 0
    cost[:, :, N - 1]    = 0

    # Package into output dict
    lincost = {
        "dfcn_dz": dcostdz,
        "dfcn_du": dcostdu,
        "fcn":     cost
    }

    return lincost

def analytical_inequality_constraints(ts, zs, us, problem):

    params = problem.get("params", problem)

    N = zs.shape[1]
    n = params["n"]
    m = params["m"]
    n_path = params["n_path"]
    n_nfz = params["n_nfz"]

    path_idx = params["path_idx"]
    path_lim_diag = np.diag(params["nondim"]["np_ineq"][:n_path])
    path_lim_scaled = np.linalg.solve(path_lim_diag, params["path_lim"])

    # Preallocate storage
    path_cnst = {"P": [], "Praw": [], "dPdz": [], "dPdu": []}
    nfz_cnst = {"P": [], "dPdz": [], "dPdu": []}

    for k in range(N):
        tk = ts[k]
        zk = zs[:, k]
        uk = us[:, k]
        rx_k, ry_k = zk[0], zk[1]

        # Evaluate full constraint vector
        P_full = nonlinear_inequality_constraints(tk, zk, uk, params)

        # ---- Path constraints ----
        if n_path > 0:
            P_path = P_full[path_idx]
            path_cnst["P"].append(P_path - path_lim_scaled)
            path_cnst["Praw"].append(P_path)

            # Use zero Jacobians as placeholders (same as original MATLAB)
            dPdz_path = np.zeros((n_path, n))
            dPdu_path = np.zeros((n_path, m))
            path_cnst["dPdz"].append(dPdz_path)
            path_cnst["dPdu"].append(dPdu_path)

        # ---- No-fly zone constraints ----
        if n_nfz > 0:
            P_nfz = P_full[n_path:n_path + n_nfz]
            dPdz_nfz = []
            dPdu_nfz = []

            for i in range(n_nfz):
                xc = params["obs"]["posc"][0, i]
                yc = params["obs"]["posc"][1, i]
                rc = params["obs"]["rc"][i]

                row_dz = np.zeros(n)
                row_dz[0] = 2 * (rx_k - xc)
                row_dz[1] = 2 * (ry_k - yc)
                dPdz_nfz.append(row_dz)

                dPdu_nfz.append(np.zeros(m))

            nfz_cnst["P"].append(P_nfz)
            nfz_cnst["dPdz"].append(np.vstack(dPdz_nfz))
            nfz_cnst["dPdu"].append(np.vstack(dPdu_nfz))

    # Stack outputs across time
    path_out = {
        "dfcn_dz": np.stack(path_cnst["dPdz"] + nfz_cnst["dPdz"], axis=2)
                   if n_path + n_nfz > 0 else np.array([]),
        "dfcn_du": np.stack(path_cnst["dPdu"] + nfz_cnst["dPdu"], axis=2)
                   if n_path + n_nfz > 0 else np.array([]),
        "fcn":     np.stack(path_cnst["P"] + nfz_cnst["P"], axis=1)
                   if n_path + n_nfz > 0 else np.array([]),
        "data": {
            "path": path_cnst,
            "nfz": nfz_cnst
        }
    }

    return path_out


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


# # testing defaults.set_nondim_params()
# if __name__ == "__main__":
#     print('..:: Testing defaults.set_nondim_params() ::..')
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
#     params = defaults.set_nondim_params(params)
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