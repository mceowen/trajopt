import numpy as np

def compute_altitude(t, z, nu, params):  #dynamic pressure
    return (z[0] - params['mission']['planet']['r'])/1000;

def max_q_nonjax(t, z, nu, params):  #dynamic pressure

    r = z[0]; v = z[3]
    rho = atmosphere_model_nonjax(r, params)
    q_dim = 0.5 * rho * v ** 2 
    return q_dim

def max_Q_nonjax(t, z, nu, params): # heat rate
    r = z[0]; v = z[3]
    rho = atmosphere_model_nonjax(r, params)
    Q_dim = params['mission']['vehicle']['kQ'] * rho ** 0.5 * v ** 3
    return Q_dim #np.array([Q_dim / mission.path_limits["max_Q"] - 1.0])

def max_load_nonjax(t, z, nu, params): # normal load
    r = z[0]; v = z[3]
    aero = nonlinear_aero_nonjax(t, z , nu ,params)
    # aero = nonlinear_aero_nonjax(t, z, nu,trajopt_obj)
    L = aero["L"]; D = aero["D"]
    load_dim = np.sqrt(L ** 2 + D ** 2) / 9.81; # (CARLOS): IM DIVIDING BY G TO GET LOAD IN G'S FOR THE PLOTS
    return load_dim #np.array([load_dim / mission.path_limits["max_load"] - 1.0])







import numpy as np
import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# =============================== LOCAL AERO DATA: ===============================
# 'lookup' option for nonlinear_aero relies on local aero data not in repo

import sys
import trajopt.local.aero.marsgram_dens_lut_jax as dens
import trajopt.local.aero.cobra_aero_nonjax as aero
# ================================================================================

# ================================================================================
# helper functions
# ================================================================================

# Atmospheric temperature model
def mars_temperature_nonjax(h):

    T_0 = 249.75     # K (surface temperature on Mars), 210
    L   = 0.0025    # K/m (lapse rate), .0025
    T = np.maximum(T_0 - L * h, 150.0)
    
    return T  # Ensure temperature doesn't fall below ~150K

# Speed of sound calculation on Mars
def mars_speed_of_sound_nonjax(h):
    T = mars_temperature_nonjax(h)
    gamma_M = 1.29
    Rgas_M  = 188.92 
    return np.sqrt(gamma_M * Rgas_M * T)

# ================================================================================
# mission functions
# ================================================================================

def atmosphere_model_nonjax(r, params):
    '''
    Returns density as a function of orbital radius

    TODO: (carlos)
    expand 'lookup' option to be higher dimensional
    remember LUT grow exponenetiall in size!
    '''

    h = r - params["mission"]["planet"]["r"]

    # TODO (carlos): add the remaining options for atmosphere model
    if params["mission"]["flags"]["aero_type"] == "lookup":
        rho = np.interp(h/1e3, dens.h_grid, dens.rho_vals)

    elif params["mission"]["flags"]["aero_type"] == "exponential":
        rho = params["mission"]["planet"]["rho"] * np.exp(-h / params["mission"]["planet"]["H"])

    return rho

def nonlinear_aero_nonjax(t, z, nu, params):

    # Extract states and controls

    r = z[0]
    v = z[3]
    alpha = nu[1]

    h = r - params["mission"]["planet"]["r"]

    rho = atmosphere_model_nonjax(r, params)
    a = mars_speed_of_sound_nonjax(h)

    M = v / a 

    mass = params["mission"]["vehicle"]["mass"]

    Cd, Cl = aero.get_cd_cl(M, alpha)

    sref = params["mission"]["vehicle"]["sref"]

    L    = (1 / mass) * 0.5 * rho * v**2 * Cl * sref
    D    = (1 / mass) * 0.5 * rho * v**2 * Cd * sref

    return {'L': L, 'D': D, 'Cl': Cl, 'Cd': Cd, 'alpha': alpha,'rho': rho}