import numpy as np
import cvxpy as cp

def cost(t, z, nu, problem):
    
    return np.dot(np.transpose(nu), nu)

def analytical_cost(t, z, nu, problem):

    mission = problem.mission
    model = problem.model
    method = problem.method

    n               = model.n
    m               = model.m
    N               = method.N

    ts              = np.asarray(ts).flatten()
    z            = np.asarray(z)
    nu            = np.asarray(nu)
    dt              = np.diff(ts)

    # Preallocate outputs
    dcostdz         = np.zeros((N, 1, n))
    dcostdnu         = np.zeros((N, 1, m))
    cost            = np.zeros((N, 1, 1))

    for k in range(N - 1):
        tk          = t[k]
        zk          = z[k]
        uk          = nu[k]

        tkp         = t[k + 1]
        zkp         = z[k + 1]
        ukp         = nu[k + 1]

        dcostdnu[k]  = 2 * dt[k] * ((uk + ukp) / 2).reshape(1, m)
        avg_cost    = 0.5 * (mission.cost(tk, zk, uk) + mission.cost(tkp, zkp, ukp))
        cost[k]     = avg_cost * dt[k]

    # Last step (N)
    dcostdz[N - 1]  = 0
    dcostdnu[N - 1]  = 0
    cost[N - 1]     = 0

    # Package into output dict
    lincost = {
        "dfcn_dz": dcostdz,
        "dfcn_du": dcostdnu,
        "fcn":     cost     
    }

    return lincost

def custom_inputs(problem,local_vars):

    mission = problem.mission
    model = problem.model
    method = problem.method

    u_norm_min  = mission.u_norm_min
    u_norm_max  = mission.u_norm_max
    theta_max   = np.deg2rad(mission.custom_input_dict["theta_max"])
    mass        = mission.vehicle["mass"] / method.nondim["nm"]
    m           = model.m
    ehat_u      = np.eye(m)
    u1          = mission.ui
    uN          = mission.uf

    ncost = method.nondim["ncost"]

    local_vars.update(locals())

    return local_vars 

def custom_variables(problem,local_vars):
    method = problem.method
    
    N           = method.N

    u_slack     = cp.Variable((N,1))  # 1×N variable
    w_jerk      = 1e-1

    local_vars.update(locals())

    return local_vars 

def custom_constraints(CNST,local_vars):

    nu_ref     = local_vars["us_ref"]
    du         = local_vars["sol_vars"]["du"]
    u_slack    = local_vars["u_slack"]
    u_norm_min = local_vars["u_norm_min"]
    u_norm_max = local_vars["u_norm_max"]
    theta_max  = local_vars["theta_max"]
    mass       = local_vars["mass"]
    ehat_u     = local_vars["ehat_u"]
    N          = local_vars["N"]

    for k in range(N):
        u_k     = nu_ref[k] + du[k]
        slack_k = u_slack[k]
        
        CNST.append(cp.norm(u_k) <= slack_k)
        CNST.append(slack_k >= u_norm_min)
        CNST.append(slack_k <= u_norm_max)
        CNST.append(np.cos(theta_max) * slack_k - (1 / mass) * ehat_u[:, 2].T @ u_k <= 0)

    return CNST

def custom_cost(PTR_COST,local_vars):

    # Extract variables from local_vars
    t_ref    = local_vars["t_ref"]
    N         = local_vars["N"]
    u_slack   = local_vars["u_slack"]
    nu_ref    = local_vars["us_ref"]
    du        = local_vars["sol_vars"]["du"]
    ncost     = local_vars["ncost"]
    # w_jerk  = local_vars["w_jerk"]  # Uncomment if you include JERK_COST term

    # Compute dt_ref (time step differences)
    dt_ref = np.diff(t_ref)  # shape: (N-1,)

    w_true = ncost

    TRUE_COST = 0
    JERK_COST = 0

    for k in range(N - 1):
        TRUE_COST   += cp.square(u_slack[k + 1]) * dt_ref[k]

        jerk        = (nu_ref[k + 1] + du[k + 1] - nu_ref[k] - du[k]) / dt_ref[k]
        # JERK_COST += w_jerk * cp.sum_squares(jerk)

    PTR_COST        = PTR_COST + w_true * TRUE_COST + JERK_COST

    return PTR_COST

def get_cost_cnstr_nondim(problem):
    mission = problem.mission
    method = problem.method

    ncost = method.nondim["nf"] ** 2 * method.nondim["nt"]
    np_ineq = np.ones(mission.n_nfz) * method.nondim["nd"] ** 2

    return ncost, np_ineq

def set_custom_params(problem):
    mission = problem.mission
    method  = problem.method

    mission.u_norm_min = mission.custom_input_dict["u_norm_min"] / method.nondim["nf"]
    mission.u_norm_max = mission.custom_input_dict["u_norm_max"] / method.nondim["nf"]