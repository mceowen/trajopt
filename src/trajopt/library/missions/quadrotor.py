import numpy as np
import cvxpy as cp

# =============================================================================
# terminal cost
# =============================================================================

def terminal_cost(t, z, nu, trajopt_obj):
    return np.dot(np.transpose(nu), nu)

def analytical_affine_approximation_terminal_cost(t, z, nu, trajopt_obj):
    model = trajopt_obj.model
    
    cost = terminal_cost(t, z, nu, trajopt_obj)

    dcostdz = np.zeros((1, model.n))
    dcostdnu = 2 * nu.reshape(1, -1)

    return cost, dcostdz, dcostdnu

# =============================================================================
# running cost
# =============================================================================

def running_cost(t, z, nu, trajopt_obj):
    return np.dot(np.transpose(nu), nu)

def analytical_affine_approximation_running_cost(t, z, nu, trajopt_obj):
    model = trajopt_obj.model
    
    cost = running_cost(t, z, nu, trajopt_obj)

    dcostdz = np.zeros((1, model.n))
    dcostdnu = 2 * nu.reshape(1, -1)

    return cost, dcostdz, dcostdnu

# =============================================================================
# custom constraints and cost (has direct access to subtrajopt_obj)
# =============================================================================

def custom_constraints(subtrajopt_obj):

    trajopt_obj = subtrajopt_obj.trajopt_obj
    constraints = subtrajopt_obj.constraints

    mission = trajopt_obj.mission
    method = trajopt_obj.method
    model = trajopt_obj.model

    subtrajopt_obj.u_slack = cp.Variable((method.index_map.N.time_grid,1))
    ehat_u = np.eye(model.m)

    for k in range(method.index_map.N.time_grid):
        u_k     = subtrajopt_obj.nu_ref[k] + subtrajopt_obj.dnu[k]
        slack_k = subtrajopt_obj.u_slack[k]
        
        constraints.append(cp.norm(u_k) <= slack_k)
        constraints.append(slack_k >= mission.u_norm_min)
        constraints.append(slack_k <= mission.u_norm_max)
        constraints.append(np.cos(mission.theta_max) * slack_k - (1 / (mission.vehicle["mass"] / method.nondim["nm"])) * ehat_u[:, 2].T @ u_k <= 0)

def custom_cost(subtrajopt_obj):

    trajopt_obj = subtrajopt_obj.trajopt_obj
    method = trajopt_obj.method

    w_true = 1.0
    t_indices = np.asarray(method.index_map.indices.z.time).reshape(-1)
    if t_indices.size != 1:
        raise ValueError("quadrotor.custom_cost expects exactly one time state in z")
    t_idx = int(t_indices[0])

    TRUE_COST = 0
    JERK_COST = 0

    for k in range(method.index_map.N.time_grid - 1):
        t_k = subtrajopt_obj.z_ref[k, t_idx] + subtrajopt_obj.dz[k, t_idx]
        t_kp1 = subtrajopt_obj.z_ref[k + 1, t_idx] + subtrajopt_obj.dz[k + 1, t_idx]
        dt_k = t_kp1 - t_k

        TRUE_COST   += cp.square(subtrajopt_obj.u_slack[k]) + dt_k

        jerk        = (subtrajopt_obj.nu_ref[k + 1] + subtrajopt_obj.dnu[k + 1] - subtrajopt_obj.nu_ref[k] - subtrajopt_obj.dnu[k])
        # JERK_COST += w_jerk * cp.sum_squares(jerk)

    subtrajopt_obj.cost_expr += w_true * (TRUE_COST + JERK_COST) / method.index_map.N.time_grid

def get_cost_cnstr_nondim(trajopt_obj):
    mission = trajopt_obj.mission
    method = trajopt_obj.method

    ncost = method.nondim["nf"] ** 2
    np_ineq = np.ones(mission.n_nfz) * 1.0 ** 2

    return ncost, np_ineq

def set_custom_params(trajopt_obj):
    mission = trajopt_obj.mission
    method  = trajopt_obj.method

    mission.u_norm_min = mission.custom_input_dict["u_norm_min"] / method.nondim["nf"]
    mission.u_norm_max = mission.custom_input_dict["u_norm_max"] / method.nondim["nf"]
    mission.theta_max = mission.custom_input_dict["theta_max"] / method.nondim["nang"]