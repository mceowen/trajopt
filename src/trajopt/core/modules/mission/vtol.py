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

def terminal_cost(t, z, nu, trajopt_obj):
    return z[3]

def analytical_affine_approximation_terminal_cost(t, z, nu, trajopt_obj):
    model = trajopt_obj.model
    
    cost = terminal_cost(t, z, nu, trajopt_obj)

    dcostdz = np.array([0, 0, 0, 1, 0, 0]).reshape(1, -1)
    dcostdnu = np.zeros((1, model.m))

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
    dcostdnu = np.zeros((1, model.m))

    return cost, dcostdz, dcostdnu

def atmosphere_model_jax(rs, trajopt_obj):
    '''
    Returns density as a function of orbital radius

    TODO: (carlos)
    expand 'lookup' option to be higher dimensional
    remember LUT grow exponenetiall in size!
    '''

    mission = trajopt_obj.mission
    model = trajopt_obj.model
    method = trajopt_obj.method

    # Compute altitude
    rdim = rs * method.nondim["nd"]
    hdim = rdim - mission.planet["r"]
    
    # TODO (carlos): add the remaining options for atmosphere model
    if mission.flags["aero_type"] == "lookup":
        rho = jnp.interp(hdim/1e3, dens.h_grid, dens.rho_vals)

    elif mission.flags["aero_type"] == "exponential":
        rho = mission.planet["rho"] * jnp.exp(-hdim / mission.planet["H"])

    return rho

def atmosphere_model_nonjax(rs, trajopt_obj):
    '''
    Returns density as a function of orbital radius
    ...rewritten from above for plain numpy
    '''
    mission = trajopt_obj.mission; model = trajopt_obj.model; method = trajopt_obj.method
    # Compute altitude
    rdim = rs * method.nondim["nd"]
    hdim = rdim - mission.planet["r"]
    # TODO (carlos): add the remaining options for atmosphere model
    if mission.flags["aero_type"] == "lookup":
        rho = np.interp(hdim/1e3, dens.h_grid, dens.rho_vals)
    elif mission.flags["aero_type"] == "exponential":
        rho = mission.planet["rho"] * np.exp(-hdim / mission.planet["H"])
    return rho



def nonlinear_aero_jax(t, z, nu, params):
    '''
    returns all aero data as a function of full state
    
    TODO: (carlos)
    this function currently only accepts a single time step!
    handle the general case? or make seperate function?ƒ
    '''


    ctrl_type = model.flags['ctrl_type']

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

    r = z[0]
    v = z[3]

    # compute v_sat with jnp
    v_sat = jnp.minimum(v * nv, vlim)

    # compute Cl/Cd locally then set into arrays
    Cl = Kl1 + Kl2 * (v_sat - vlim)**2 + Kl3 * (v_sat - vlim)**4
    Cd = Kd1 + Kd2 * Cl + Kd3 * Cl**2
    alpha = jnp.deg2rad(alphlim_deg - kalph * (jnp.minimum(v * nv, vlim) - vlim)**2)

    rho = atmosphere_model_jax(r, trajopt_obj)
    rho_s = rho / (method.nondim["nm"] / method.nondim["nd"] ** 3)
    sref_s = mission.vehicle["sref"] / method.nondim["nd"] ** 2

    L = (0.5 / mass_nd) * rho_s * sref_s * Cl * v**2
    D = (0.5 / mass_nd) * rho_s * sref_s * Cd * v**2

    return {'L': L, 'D': D, 'Cl': Cl, 'Cd': Cd, 'alpha': alpha, 'rho': rho}

def nonlinear_aero_nonjax(t, z, nu, trajopt_obj):
    '''
    returns all aero data as a function of full state
    ... rewritten from above without jax
    '''

    mission = trajopt_obj.mission; model = trajopt_obj.model; method = trajopt_obj.method
    ctrl_type = model.flags['ctrl_type']
    # Extract key params
    nv = method.nondim['nv']
    mass_nd = mission.vehicle['mass'] / method.nondim['nm']

    B = (method.nondim['nd'] / method.nondim['nm']) * (mission.vehicle['sref'] / 2)

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

    r = z[0]
    v = z[3]

    # compute v_sat with jnp
    v_sat = np.minimum(v * nv, vlim)
    # compute Cl/Cd locally then set into arrays
    Cl = Kl1 + Kl2 * (v_sat - vlim)**2 + Kl3 * (v_sat - vlim)**4
    Cd = Kd1 + Kd2 * Cl + Kd3 * Cl**2
    alpha = np.deg2rad(alphlim_deg - kalph * (np.minimum(v * nv, vlim) - vlim)**2)
    rho = atmosphere_model_nonjax(r, trajopt_obj)
    rho_s = rho / (method.nondim["nm"] / method.nondim["nd"] ** 3)
    sref_s = mission.vehicle["sref"] / method.nondim["nd"] ** 2
    L = (0.5 / mass_nd) * rho_s * sref_s * Cl * v**2
    D = (0.5 / mass_nd) * rho_s * sref_s * Cd * v**2
    return {'L': L, 'D': D, 'Cl': Cl, 'Cd': Cd, 'alpha': alpha, 'rho': rho}    

def custom_constraints(subtrajopt_obj):
    pass

def custom_cost(subtrajopt_obj):
    pass

def set_custom_params(trajopt_obj):
    pass