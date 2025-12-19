import numpy as np
import cvxpy as cp
import jax 
import jax.numpy as jnp

# =============================================================================
# terminal cost
# =============================================================================

def terminal_cost(t, z, nu, trajopt_obj):
    return 0.0

def analytical_affine_approximation_terminal_cost(t, z, nu, trajopt_obj):
    model = trajopt_obj.model
    
    cost = terminal_cost(t, z, nu, trajopt_obj)

    dcostdz = np.zeros((1, model.n))
    dcostdnu = 2 * 0.0 * nu.reshape(1, -1)

    return cost, dcostdz, dcostdnu

# =============================================================================
# running cost
# =============================================================================

def running_cost(t, z, nu, trajopt_obj):
    return 0.0

def analytical_affine_approximation_running_cost(t, z, nu, trajopt_obj):
    model = trajopt_obj.model
    
    cost = running_cost(t, z, nu, trajopt_obj)

    dcostdz = np.zeros((1, model.n))
    dcostdnu = 2 * 0.0 * nu.reshape(1, -1)

    return cost, dcostdz, dcostdnu

# =============================================================================
# MISSION-SPECIFIC CONSTRAINT FUNCTIONS
# =============================================================================

# =============================================================================
# CUSTOM COST / CONSTRAINTS
# =============================================================================
def custom_cost(subtrajopt_obj):

    trajopt_obj = subtrajopt_obj.trajopt_obj
    method = trajopt_obj.method

    dt = subtrajopt_obj.dt_ref + subtrajopt_obj.dt

    subtrajopt_obj.cost_expr +=  cp.sum(dt) / (method.N - 1)

def get_cost_cnstr_nondim(trajopt_obj):
    mission = trajopt_obj.mission
    method = trajopt_obj.method

    ncost = method.nondim["nt"]
    np_ineq = np.array([1.0])

    return ncost, np_ineq

def set_custom_params(trajopt_obj):
    pass