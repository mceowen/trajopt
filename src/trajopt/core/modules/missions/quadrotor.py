import numpy as np
import cvxpy as cp

# =============================================================================
# terminal cost
# =============================================================================

def terminal_cost(ts, zs, us, problem):
    return np.dot(np.transpose(us), us)

def analytical_affine_approximation_terminal_cost(ts, zs, us, problem):
    model = problem.model
    
    cost = terminal_cost(ts, zs, us, problem)

    dcostdz = np.zeros((1, model.n))
    dcostdu = 2 * us.reshape(1, -1)

    return cost, dcostdz, dcostdu

# =============================================================================
# running cost
# =============================================================================

def running_cost(ts, zs, us, problem):
    return np.dot(np.transpose(us), us)

def analytical_affine_approximation_running_cost(ts, zs, us, problem):
    model = problem.model
    
    cost = running_cost(ts, zs, us, problem)

    dcostdz = np.zeros((1, model.n))
    dcostdu = 2 * us.reshape(1, -1)

    return cost, dcostdz, dcostdu

# =============================================================================
# custom constraints and cost (has direct access to subproblem)
# =============================================================================

def custom_constraints(subproblem):

    problem = subproblem.problem
    constraints = subproblem.constraints

    mission = problem.mission
    method = problem.method
    model = problem.model

    subproblem.u_slack = cp.Variable((method.N,1))
    ehat_u = np.eye(model.m)

    for k in range(method.N):
        u_k     = subproblem.us_ref[k] + subproblem.du[k]
        slack_k = subproblem.u_slack[k]
        
        constraints.append(cp.norm(u_k) <= slack_k)
        constraints.append(slack_k >= mission.custom_input_dict["u_norm_min"])
        constraints.append(slack_k <= mission.custom_input_dict["u_norm_max"])
        constraints.append(np.cos(mission.custom_input_dict["theta_max"]) * slack_k - (1 / (mission.vehicle["mass"] / method.nondim["nm"])) * ehat_u[:, 2].T @ u_k <= 0)

def custom_cost(subproblem):

    problem = subproblem.problem
    method = problem.method

    w_true = method.nondim["ncost"]

    TRUE_COST = 0
    JERK_COST = 0

    for k in range(method.N - 1):
        TRUE_COST   += cp.square(subproblem.u_slack[k + 1])

        jerk        = (subproblem.us_ref[k + 1] + subproblem.du[k + 1] - subproblem.us_ref[k] - subproblem.du[k])
        # JERK_COST += w_jerk * cp.sum_squares(jerk)

    subproblem.cost_expr += w_true * TRUE_COST + JERK_COST

def get_cost_cnstr_nondim(problem):
    mission = problem.mission
    method = problem.method

    ncost = method.nondim["nf"] ** 2
    np_ineq = np.ones(mission.n_nfz) * method.nondim["nd"] ** 2

    return ncost, np_ineq

def set_custom_params(problem):
    mission = problem.mission
    method  = problem.method

    mission.u_norm_min = mission.custom_input_dict["u_norm_min"] / method.nondim["nf"]
    mission.u_norm_max = mission.custom_input_dict["u_norm_max"] / method.nondim["nf"]