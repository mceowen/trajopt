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
    params = problem["params"]
    
    return np.dot(np.transpose(us), us)

def analytical_cost(ts, zs, us, mission):

    problem = mission.problem
    params = problem["params"]

    n               = params["model"]["n"]
    m               = params["model"]["m"]
    N               = params["method"]["N"]

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

    #======================
    # Path /NFZ constraints
    #======================
    # no fly zones, specified by position and radius [rad]
    if params["mission"]["bools"]["flag_nfz"] == 1:
        xc = params["mission"]["nfz_1"]["xc"] / params["method"]["nondim"]["nd"]
        yc = params["mission"]["nfz_1"]["yc"] / params["method"]["nondim"]["nd"]
        rc = params["mission"]["nfz_1"]["rc"] / params["method"]["nondim"]["nd"]
    elif params["mission"]["bools"]["flag_nfz"] == 2:
        xc = params["mission"]["nfz_2"]["xc"] / params["method"]["nondim"]["nd"]
        yc = params["mission"]["nfz_2"]["yc"] / params["method"]["nondim"]["nd"]
        rc = params["mission"]["nfz_2"]["rc"] / params["method"]["nondim"]["nd"]
    else:
        xc = params["mission"]["nfz_0"]["xc"] / params["method"]["nondim"]["nd"]
        yc = params["mission"]["nfz_0"]["yc"] / params["method"]["nondim"]["nd"]
        rc = params["mission"]["nfz_0"]["rc"] / params["method"]["nondim"]["nd"]

    params["mission"]["nfz_idx"]       = np.arange(0, xc.size)
    params["mission"]["n_nfz"]         = len(params["mission"]["nfz_idx"])

    params["mission"].setdefault("obs", {})["posc"] = np.array([xc, yc]) # xc and yc may be vectors
    params["mission"]["obs"]["rc"]     = rc

    # set nondim for cost and constraints
    np_ineq = np.ones(params["mission"]["n_nfz"]) * params["method"]["nondim"]["nd"]**2
    ncost = params["method"]["nondim"]["nf"]**2 * params["method"]["nondim"]["nt"]

    params = nondim.set_cost_cnst_nondim_params(np_ineq, ncost, params)

    #====================
    # Boundary Conditions
    #====================
    # initial conditions

    # equality initial conditions
    params["mission"]["zi"]            = params["method"]["nondim"]["M"]["state"]["d2nd"] @ params["mission"]["zi"]
    params["mission"]["zi_idx"]        = np.arange(0, params["model"]["n"])

    # inequality initial conditions
    # none

    # equality terminal conditions
    params["mission"]["zf"]            = params["method"]["nondim"]["M"]["state"]["d2nd"] @ params["mission"]["zf"]  
    params["mission"]["zf_idx"]        = np.arange(0,params["model"]["n"])

    # control boundary conditions
    params["mission"]["ui"]            = -params["mission"]["ge"]*params["mission"]["mass"] / params["method"]["nondim"]["nf"]
    params["mission"]["uf"]            = -params["mission"]["ge"]*params["mission"]["mass"] / params["method"]["nondim"]["nf"]

    #==============================
    # Control and state constraints
    #==============================
    # no state constraints
    params["mission"]["z_min"]         = np.array([0, 0, 0.25]) / params["method"]["nondim"]["nd"]
    params["mission"]["z_min_idx"]     = np.arange(0,3)
    params["mission"]["z_max"]         = np.array([12, 12, 10]) / params["method"]["nondim"]["nd"]
    params["mission"]["z_max_idx"]     = np.arange(0,3)
    params["mission"]["u_norm_min"]    = 0.21 / params["method"]["nondim"]["nf"]
    params["mission"]["u_norm_max"]    = 8.12 / params["method"]["nondim"]["nf"]
    params["mission"]["udot_max"]      = 5*np.ones(3) / (params["method"]["nondim"]["nf"] / params["method"]["nondim"]["nt"])# [N/s]
    params["mission"]["udot_max_idx"]  = np.arange(0,3)


    ### Time of flight constraints ###
    Ts_min                  = 1. / params["method"]["nondim"]["nt"]  # 50
    Ts_max                  = 20. / params["method"]["nondim"]["nt"]
    params["method"]["ddts_max"]      = 5. / ((params["method"]["N"] - 1) * params["method"]["nondim"]["nt"])  # 0.025
    params["method"]["dts_min"]       = Ts_min / (params["method"]["N"] - 1)
    params["method"]["dts_max"]       = Ts_max / (params["method"]["N"] - 1)

    return params

def custom_inputs(problem,local_vars):
    u_norm_min  = problem["params"]["mission"]["u_norm_min"]
    u_norm_max  = problem["params"]["mission"]["u_norm_max"]
    theta_max   = np.deg2rad(problem["params"]["mission"]["theta_max"])
    mass        = problem["params"]["mission"]["mass"] / problem["params"]["method"]["nondim"]["nm"]
    m           = problem["params"]["model"]["m"]
    ehat_u      = np.eye(m)
    u1          = problem["params"]["mission"]["ui"]
    uN          = problem["params"]["mission"]["uf"]

    local_vars.update(locals())

    return local_vars 

def custom_subprob_variables(problem,local_vars): 
    
    N           = problem["params"]["method"]["N"]

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

    params = local_vars["params"]
    w_true = params["method"]["nondim"]["ncost"]

    TRUE_COST = 0
    JERK_COST = 0

    for k in range(N - 1):
        TRUE_COST   += cp.square(u_slack[k + 1]) * dts_ref[k]

        jerk        = (us_ref[k + 1] + du[k + 1] - us_ref[k] - du[k]) / dts_ref[k]
        # JERK_COST += w_jerk * cp.sum_squares(jerk)

    PTR_COST        = PTR_COST + w_true * TRUE_COST + JERK_COST

    return PTR_COST