import numpy as np
import cvxpy as cp
import jax 
import jax.numpy as jnp

# =============================================================================
# terminal cost
# =============================================================================

def terminal_cost(t, z, nu, problem):
    return 0.0

def analytical_affine_approximation_terminal_cost(t, z, nu, problem):
    model = problem.model
    
    cost = terminal_cost(t, z, nu, problem)

    dcostdz = np.zeros((1, model.n))
    dcostdnu = 2 * 0.0 * nu.reshape(1, -1)

    return cost, dcostdz, dcostdnu

# =============================================================================
# running cost
# =============================================================================

def running_cost(t, z, nu, problem):
    return 0.0

def analytical_affine_approximation_running_cost(t, z, nu, problem):
    model = problem.model
    
    cost = running_cost(t, z, nu, problem)

    dcostdz = np.zeros((1, model.n))
    dcostdnu = 2 * 0.0 * nu.reshape(1, -1)

    return cost, dcostdz, dcostdnu

def custom_cost(subproblem):

    problem = subproblem.problem
    method = problem.method

    dt = subproblem.dt_ref + subproblem.dt

    subproblem.cost_expr +=  cp.sum(dt) / (method.N - 1)

def get_cost_cnstr_nondim(problem):
    mission = problem.mission
    method = problem.method

    ncost = method.nondim["nt"]
    np_ineq = np.array([1.0])

    return ncost, np_ineq


# TODO (CARLOS): we should only need to use custom constraints for very specific constraints in the near future
# leaving like this for now for the PDG problem
# the reason this is a custom constraint rn is due to the dependency of nu_ref. wasn't sure of the cleanest way to 
# allow for that in the general constraints list.

def custom_constraints(subproblem):

    problem = subproblem.problem
    mission = problem.mission
    method = problem.method

    for k in range(method.N):
        dnu_k = subproblem.dnu[k]
        nu_ref_k = subproblem.nu_ref[k]
        nu_ref_sq_k = subproblem.nu_ref_sq[k]

        subproblem.constraints.append(mission.custom_input_dict["min_thrust"] * cp.norm(nu_ref_k[:3]) <= nu_ref_sq_k + nu_ref_k[:3] @ dnu_k[:3])

def set_custom_params(problem):
    pass