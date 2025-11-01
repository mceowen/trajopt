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

def cost(ts, zs, us, problem):
    '''
    cost function: minimize terminal velocity
    '''
    
    return zs[-1, 3]

def analytical_cost(ts, zs, us, problem):
    '''
    analytical linearized cost function
    '''
    
    mission = problem.mission
    model = problem.model
    method = problem.method

    # Extract params
    n = model.n
    m = model.m
    N = method.N

    ts = np.asarray(ts).flatten()
    zs = np.asarray(zs)
    us = np.asarray(us)
    dt = np.diff(ts)

    # Preallocate outputs
    dcostdz = np.zeros((N, 1, n))
    dcostdu = np.zeros((N, 1, m))
    cost    = np.zeros((N, 1, 1))

    # Last step (N) (minimize terminal velocity)
    dcostdz[-1, 0, :] = np.array([0, 0, 0, 1, 0, 0])
    dcostdu[-1]       = 0
    cost[-1]          = zs[-1, 3]

    # Package into output dict
    lincost = {
        "dfcn_dz": dcostdz,
        "dfcn_du": dcostdu,
        "fcn":     cost     
    }

    return lincost

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
    if mission.bools["aero_type"] == "lookup":
        rho = jnp.interp(hdim/1e3, dens.h_grid, dens.rho_vals)

    elif mission.bools["aero_type"] == "exponential":
        rho = mission.planet["rho"] * jnp.exp(-hdim / mission.planet["H"])

    return rho

def nonlinear_aero_jax(ts, zs, us, problem):
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
    rs, _, _, vs, _, _ = zs

    rho = atmosphere_model_jax(rs, problem)

    rho_s = rho / (method.nondim["nm"] / method.nondim["nd"]**3)
    sref_s = mission.vehicle["sref"]   / method.nondim["nd"]**2
    bc_s = mission.vehicle["bc"]       / (method.nondim["nm"] / (method.nondim["nd"]**2))

    D    = 0.5 * (1 / bc_s) * rho_s * vs**2
    L    = D * mission.vehicle["LD"]

    alpha = 0

    return {"L": L, "D": D, "alpha": alpha, "rho": rho}

def custom_inputs(problem,local_vars):

    return local_vars 

def custom_variables(problem,local_vars): 

    return local_vars 

def custom_constraints(CNST,local_vars):

    return CNST

def custom_cost(PTR_COST,local_vars):

    return PTR_COST

def set_custom_params(problem):
    pass