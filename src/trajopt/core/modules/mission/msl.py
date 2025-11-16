import numpy as np
import trajopt.utils.tools as tools

import numpy as np
import jax 
import jax.numpy as jnp
import trajopt.utils.tools as tools
jax.config.update("jax_enable_x64", True)

# =============================== LOCAL AERO DATA: ===============================
# 'lookup' option for nonlinear_aero relies on local aero data not in repo

# import sys
# sys.path.append("/Users/carlosm/Documents/guidance/hypersonics/prototypes/local")
# import marsgram_dens_lut as dens
# ================================================================================

# =============================================================================
# terminal cost
# =============================================================================

def terminal_cost(t, z, nu, problem):
    return z[3]

def analytical_affine_approximation_terminal_cost(t, z, nu, problem):
    model = problem.model
    
    cost = terminal_cost(t, z, nu, problem)

    dcostdz = np.array([0, 0, 0, 1, 0, 0]).reshape(1, -1)
    dcostdnu = np.zeros((1, model.m))

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
    dcostdnu = np.zeros((1, model.m))

    return cost, dcostdz, dcostdnu

def get_cost_cnstr_nondim(problem):
    '''
    Returns the dimensional units of cost and constraint functions
    '''
    mission = problem.mission
    method = problem.method

    ncost = method.nondim["nv"]
    np_ineq = np.ones(mission.n_nfz) * method.nondim["nang"] ** 2

    return ncost, np_ineq

def atmosphere_model_jax(rs, problem):
    '''
    Returns density as a function of orbital radius

    TODO: (carlos)
    expand 'lookup' option to be higher dimensional
    remember LUT grow exponenetiall in size!
    '''

    mission = problem.mission
    model = problem.model
    method = problem.method

    # Compute altitude
    rdim = rs * method.nondim["nd"]
    hdim = rdim - mission.planet["r"]
    
    # TODO (carlos): add the remaining options for atmosphere model
    if mission.flags["aero_type"] == "lookup":
        rho = jnp.interp(hdim/1e3, dens.h_grid, dens.rho_vals)

    elif mission.flags["aero_type"] == "exponential":
        rho = mission.planet["rho"] * jnp.exp(-hdim / mission.planet["H"])

    return rho

def atmosphere_model_nonjax(rs, problem):
    '''
    Returns density as a function of orbital radius

    TODO: (carlos)
    expand 'lookup' option to be higher dimensional
    remember LUT grow exponenetiall in size!
    '''

    mission = problem.mission
    model = problem.model
    method = problem.method

    # Compute altitude
    rdim = rs * method.nondim["nd"]
    hdim = rdim - mission.planet["r"]
    
    # TODO (carlos): add the remaining options for atmosphere model
    if mission.flags["aero_type"] == "lookup":
        rho = np.interp(hdim/1e3, dens.h_grid, dens.rho_vals)

    elif mission.flags["aero_type"] == "exponential":
        rho = mission.planet["rho"] * np.exp(-hdim / mission.planet["H"])

    return rho    

def nonlinear_aero_jax(t, z, nu, problem):
    '''
    returns all aero data as a function of full state
    
    TODO: (carlos)
    this function currently only accepts a single time step!
    handle the general case? or make seperate function?
    '''

    mission = problem.mission
    model = problem.model
    method = problem.method

    # Extract states and controls
    r, _, _, v, _, _ = z

    rho = atmosphere_model_jax(r, problem)

    rho_s = rho / (method.nondim["nm"] / method.nondim["nd"] ** 3)
    sref_s = mission.vehicle["sref"] / method.nondim["nd"] ** 2
    bc_s = mission.vehicle["bc"] / (method.nondim["nm"] / (method.nondim["nd"] ** 2))

    D    = 0.5 * (1 / bc_s) * rho_s * v**2
    L    = D * mission.vehicle["LD"]

    alpha = 0

    return {"L": L, "D": D, "alpha": alpha, "rho": rho}

def nonlinear_aero_nonjax(t, z, nu, problem):
    '''
    returns all aero data as a function of full state
    
    TODO: (carlos)
    this function currently only accepts a single time step!
    handle the general case? or make seperate function?
    '''

    mission = problem.mission
    model = problem.model
    method = problem.method

    # Extract states and controls
    r, _, _, v, _, _ = z

    rho = atmosphere_model_nonjax(r, problem)

    rho_s = rho / (method.nondim["nm"] / method.nondim["nd"] ** 3)
    sref_s = mission.vehicle["sref"] / method.nondim["nd"] ** 2
    bc_s = mission.vehicle["bc"] / (method.nondim["nm"] / (method.nondim["nd"] ** 2))

    D    = 0.5 * (1 / bc_s) * rho_s * v**2
    L    = D * mission.vehicle["LD"]

    alpha = 0

    return {"L": L, "D": D, "alpha": alpha, "rho": rho}    

def custom_constraints(subproblem):
    pass

def custom_cost(subproblem):
    pass

def set_custom_params(problem):
    pass