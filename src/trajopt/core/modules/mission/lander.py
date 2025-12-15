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

# =============================================================================
# MISSION-SPECIFIC CONSTRAINT FUNCTIONS
# =============================================================================

# =============================================================================
# CUSTOM COST / CONSTRAINTS
# =============================================================================
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

def set_custom_params(problem):
    pass