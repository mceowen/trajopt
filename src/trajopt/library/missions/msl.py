import numpy as np
import trajopt.core.utils.tools as tools

import numpy as np
import jax 
import jax.numpy as jnp
import trajopt.core.utils.tools as tools
jax.config.update("jax_enable_x64", True)

def atmosphere_model_jax(r, params):
    '''
    Returns density as a function of orbital radius

    TODO: (carlos)
    expand 'lookup' option to be higher dimensional
    remember LUT grow exponenetiall in size!
    '''

    # Compute altitude
    h = r - params['mission']['planet']["r"]
    
    # TODO (carlos): add the remaining options for atmosphere model
    if params['mission']['flags']["aero_type"] == "lookup":
        rho = jnp.interp(h/1e3, dens.h_grid, dens.rho_vals)

    elif params['mission']['flags']["aero_type"] == "exponential":
        rho = params['mission']['planet']["rho"] * jnp.exp(-h / params['mission']['planet']["H"])

    return rho  

def nonlinear_aero_jax(t, z, nu, params):
    '''
    returns all aero data as a function of full state
    
    TODO: (carlos)
    this function currently only accepts a single time step!
    handle the general case? or make seperate function?
    '''

    # Extract states and controls
    r, _, _, v, _, _ = z

    rho = atmosphere_model_jax(r, params)

    D    = 0.5 * (1 / params['mission']['vehicle']["bc"]) * rho * v**2
    L    = D * params['mission']['vehicle']["LD"]

    alpha = 0

    return {"L": L, "D": D, "alpha": alpha, "rho": rho}