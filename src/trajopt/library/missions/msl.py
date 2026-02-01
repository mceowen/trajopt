import numpy as np
import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def atmosphere_model_jax(t, z, nu, params):
    '''
    Returns density as a function of orbital radius

    TODO: (carlos)
    expand 'lookup' option to be higher dimensional
    remember LUT grow exponenetiall in size!
    '''

    r = z[0]

    h = r - params['planet']["r"]
    
    if params['flags']["aero_type"] == "lookup":
        rho = jnp.interp(h/1e3, dens.h_grid, dens.rho_vals)

    elif params['flags']["aero_type"] == "exponential":
        rho = params['planet']["rho"] * jnp.exp(-h / params['planet']["H"])

    return rho  

def nonlinear_aero_jax(t, z, nu, params):
    '''
    returns all aero data as a function of full state
    
    TODO: (carlos)
    this function currently only accepts a single time step!
    handle the general case? or make seperate function?
    '''

    r = z[0]
    v = z[3]

    rho = atmosphere_model_jax(t, z, nu, params)

    D    = 0.5 * (1 / params['vehicle']["bc"]) * rho * v**2
    L    = D * params['vehicle']["LD"]

    alpha = 0

    return {"L": L, "D": D, "alpha": alpha, "rho": rho}