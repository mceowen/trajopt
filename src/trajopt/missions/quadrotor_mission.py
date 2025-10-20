import numpy as np
import cvxpy as cp
import importlib

import trajopt.utils.set_defaults           as defaults
import trajopt.utils.tools                  as tools
import trajopt.algorithm.initial_guess      as guess
import trajopt.algorithm.convergence        as convergence
import trajopt.algorithm.convexification    as convexify
import trajopt.utils.nondim                 as nondim

def cost(ts, zs, us, mission):

    problem = mission.problem
    params = problem['params']
    
    return np.dot(np.transpose(us), us)

def analytical_cost(ts, zs, us, mission):

    problem = mission.problem
    params = problem['params']

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

        dcostdu[k]  = 2 * dt[k] * ((uk + ukp) / 2).reshape(1, m)
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

def mission_params(params):

    # Physical constants
    params['ge']        = np.array([0, 0, -9.81]) # [m/s^2], grav accel at sea lvl

    #======================
    # Path /NFZ constraints
    #======================
    # no fly zones, specified by position and radius [rad]
    if params['bools']['flag_nfz'] == 1:
        xc = np.array([5]) / params['nondim']['nd']
        yc = np.array([4]) / params['nondim']['nd']
        rc = np.array([2]) / params['nondim']['nd']
    elif params['bools']['flag_nfz'] == 2:
        xc = np.array([2.5, 5,  2.5, 5.5,  8,  5.5]) / params['nondim']['nd']# 5
        yc = np.array([2,   2.5,  5, 5.25, 5.5, 8]) / params['nondim']['nd']# 4
        rc = np.ones(xc.size) / params['nondim']['nd'] # 2, 1
    else:
        xc = np.array([])
        yc = np.array([])
        rc = np.array([])

    params['nfz_idx']       = np.arange(0, xc.size)
    params['n_nfz']         = len(params['nfz_idx'])

    params.setdefault('obs', {})['posc'] = np.array([xc, yc]) # xc and yc may be vectors
    params['obs']['rc']     = rc

    # set nondim for cost and constraints
    np_ineq = np.ones(params['n_nfz']) * params['nondim']['nd']**2
    ncost = params['nondim']['nf']**2 * params['nondim']['nt']

    params = nondim.set_cost_cnst_nondim_params(np_ineq, ncost, params)

    #====================
    # Boundary Conditions
    #====================
    # initial conditions
    params['z0_dim']           = np.array([0,0,5,0,0.5,0])

    # equality initial conditions
    params['zi']            = params['nondim']['M']['state']['d2nd'] @ params['z0_dim']
    params['zi_idx']        = np.arange(0, params['n'])

    # inequality initial conditions
    # none

    # equality terminal conditions
    params['zf']            = params['nondim']['M']['state']['d2nd'] @ np.array([10,10,0.5,0,0,0])
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

    return params

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

    params = local_vars['params']
    w_true = params['nondim']['ncost']

    TRUE_COST = 0
    JERK_COST = 0

    for k in range(N - 1):
        TRUE_COST   += cp.square(u_slack[k + 1]) * dts_ref[k]

        jerk        = (us_ref[k + 1] + du[k + 1] - us_ref[k] - du[k]) / dts_ref[k]
        # JERK_COST += w_jerk * cp.sum_squares(jerk)

    PTR_COST        = PTR_COST + w_true * TRUE_COST + JERK_COST

    return PTR_COST