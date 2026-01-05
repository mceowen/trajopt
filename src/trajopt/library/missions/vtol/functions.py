import numpy as np
import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def atmosphere_model_jax(r, params):
    '''
    Returns density as a function of orbital radius

    TODO: (carlos)
    expand 'lookup' option to be higher dimensional
    remember LUT grow exponenetiall in size!
    '''

    h = r - params['mission']['planet']['r']
    
    # TODO (carlos): add the remaining options for atmosphere model
    if params['mission']['flags']['aero_type'] == "lookup":
        rho = jnp.interp(h/1e3, dens.h_grid, dens.rho_vals)

    elif params['mission']['flags']['aero_type'] == "exponential":
        rho = params['mission']['planet']['rho'] * jnp.exp(-h / params['mission']['planet']['H'])

    return rho

def atmosphere_model_nonjax(r, params):
    '''
    Returns density as a function of orbital radius
    ...rewritten from above for plain numpy
    '''
    # Compute altitude
    h = r - params['mission']['planet']['r']
    # TODO (carlos): add the remaining options for atmosphere model
    if params['mission']['flags']['aero_type'] == "lookup":
        rho = np.interp(h/1e3, dens.h_grid, dens.rho_vals)
    elif params['mission']['flags']['aero_type'] == "exponential":
        rho = params['mission']['planet']['rho'] * np.exp(-h / params['mission']['planet']['H'])
    return rho

def nonlinear_aero_jax(r, v, params):
    '''
    returns all aero data as a function of full state
    
    TODO: (carlos)
    this function currently only accepts a single time step!
    handle the general case? or make seperate function?ƒ
    '''


    ctrl_type = params['model']['flags']['ctrl_type']

    # Setup coefficient values
    kl1         = -0.041065
    kl2         = 0.016292
    kl3         = 0.0002602
    kd1         = 0.080505
    kd2         = -0.03026
    kd3         = 0.86495
    kalph       = 0.20705 / (340**2)
    vlim        = 4570
    alphlim_deg = 40

    if ctrl_type == 'bank_only':
        # Velocity-dependent polynomial coefficients
        Kd1     = kd1
        Kd2     = kd2
        Kd3     = kd3
        Kl1     = kl1 + kl2 * alphlim_deg + kl3 * alphlim_deg**2
        Kl2     = -kl2 * kalph - 2 * kl3 * alphlim_deg * kalph
        Kl3     = kl3 * kalph**2
    
    elif ctrl_type == 'bank_aoa':
        # AOA-dependent polynomial coefficients
        d2r     = 1  # /(pi/180)
        Kd1h    = kd1 + kd2 * kl1 + kd3 * kl1**2
        Kd2h    = (kd2 * kl2 + 2 * kd3 * kl1 * kl2) * d2r
        Kd3h    = (kd2 * kl3 + 2 * kd3 * kl1 * kl3 + kd3 * kl2**2) * d2r**2
        Kd4h    = (2 * kd3 * kl2 * kl3) * d2r**3
        Kd5h    = (kd3 * kl3**2) * d2r**4
        Kl1h    = kl1
        Kl2h    = kl2 * d2r**2
        Kl3h    = kl3 * d2r**3

    # compute v_sat with jnp
    v_sat = jnp.minimum(v, vlim)

    # compute Cl/Cd locally then set into arrays
    Cl = Kl1 + Kl2 * (v_sat - vlim)**2 + Kl3 * (v_sat - vlim)**4
    Cd = Kd1 + Kd2 * Cl + Kd3 * Cl**2
    alpha = jnp.deg2rad(alphlim_deg - kalph * (jnp.minimum(v, vlim) - vlim)**2)

    rho = atmosphere_model_jax(r, params)
    sref = params['mission']['vehicle']['sref']
    mass = params['mission']['vehicle']['mass']

    L = (0.5 / mass) * rho * sref * Cl * v**2
    D = (0.5 / mass) * rho * sref * Cd * v**2

    return {'L': L, 'D': D, 'Cl': Cl, 'Cd': Cd, 'alpha': alpha, 'rho': rho}

def nonlinear_aero_nonjax(r, v, params):
    '''
    returns all aero data as a function of full state
    ... rewritten from above without jax
    '''

    ctrl_type = params['model']['flags']['ctrl_type']

    # Setup coefficient values
    kl1         = -0.041065
    kl2         = 0.016292
    kl3         = 0.0002602
    kd1         = 0.080505
    kd2         = -0.03026
    kd3         = 0.86495
    kalph       = 0.20705 / (340**2)
    vlim        = 4570
    alphlim_deg = 40

    if ctrl_type == 'bank_only':
        # Velocity-dependent polynomial coefficients
        Kd1     = kd1
        Kd2     = kd2
        Kd3     = kd3
        Kl1     = kl1 + kl2 * alphlim_deg + kl3 * alphlim_deg**2
        Kl2     = -kl2 * kalph - 2 * kl3 * alphlim_deg * kalph
        Kl3     = kl3 * kalph**2
    
    elif ctrl_type == 'bank_aoa':
        # AOA-dependent polynomial coefficients
        d2r     = 1  # /(pi/180)
        Kd1h    = kd1 + kd2 * kl1 + kd3 * kl1**2
        Kd2h    = (kd2 * kl2 + 2 * kd3 * kl1 * kl2) * d2r
        Kd3h    = (kd2 * kl3 + 2 * kd3 * kl1 * kl3 + kd3 * kl2**2) * d2r**2
        Kd4h    = (2 * kd3 * kl2 * kl3) * d2r**3
        Kd5h    = (kd3 * kl3**2) * d2r**4
        Kl1h    = kl1
        Kl2h    = kl2 * d2r**2
        Kl3h    = kl3 * d2r**3

    # compute v_sat with jnp
    v_sat = np.minimum(v, vlim)
    # compute Cl/Cd locally then set into arrays
    Cl = Kl1 + Kl2 * (v_sat - vlim)**2 + Kl3 * (v_sat - vlim)**4
    Cd = Kd1 + Kd2 * Cl + Kd3 * Cl**2
    alpha = np.deg2rad(alphlim_deg - kalph * (np.minimum(v, vlim) - vlim)**2)
    
    rho = atmosphere_model_nonjax(r, params)
    sref = params['mission']['vehicle']['sref']
    mass = params['mission']['vehicle']['mass']
    
    L = (0.5 / mass) * rho * sref * Cl * v**2
    D = (0.5 / mass) * rho * sref * Cd * v**2
    
    return {'L': L, 'D': D, 'Cl': Cl, 'Cd': Cd, 'alpha': alpha, 'rho': rho}