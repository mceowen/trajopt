import numpy as np
import jax 
import jax.numpy as jnp
import trajopt.utils.tools as tools
jax.config.update("jax_enable_x64", True)
import trajopt.core.modules.model.obstacles     as obstacles
import trajopt.local.modules.mission.aero.cobra_aero_nonjax as aero_nonjax


def compute_altitude(t, z, nu, problem):  #dynamic pressure
    mission = problem.mission; #method = problem.method
    return (z[0] - mission.planet['r'])/1000;

def max_q_nonjax(t, z, nu, problem):  #dynamic pressure
    mission = problem.mission; method = problem.method
    rs = z[0]; vs = z[3]
    rho = atmosphere_model_nonjax(rs/method.nondim['nd'], problem)
    nv = 1; #method.nondim["nv"]
    q_dim = 0.5 * rho * (vs * nv) ** 2 
    return q_dim #np.array([q_dim / mission.path_limits['max_q'] - 1.0])

def max_Q_nonjax(t, z, nu, problem): # heat rate
    mission = problem.mission; method = problem.method
    rs = z[0]; vs = z[3]
    rho = atmosphere_model_nonjax(rs/method.nondim['nd'], problem)
    nv = 1; #method.nondim["nv"]
    Q_dim = mission.vehicle["kQ"] * rho ** 0.5 * (vs * nv) ** 3
    return Q_dim #np.array([Q_dim / mission.path_limits["max_Q"] - 1.0])

def max_load_nonjax(t, z, nu, problem): # normal load
    mission = problem.mission; method = problem.method
    rs = z[0]; vs = z[3]
    Mstate = method.nondim['M']['state']['d2nd']
    stateidx = problem.indices.z['state']
    Mctrl = method.nondim['M']['ctrl']['d2nd']
    aero = nonlinear_aero_nonjax(t, z[stateidx] @ Mstate, nu @ Mctrl,problem)
    # aero = nonlinear_aero_nonjax(t, z, nu,problem)
    L = aero["L"]; D = aero["D"]
    load_dim = np.sqrt(L ** 2 + D ** 2) * method.nondim["na"] / mission.planet['g']; # (CARLOS): IM DIVIDING BY G TO GET LOAD IN G'S FOR THE PLOTS
    return load_dim #np.array([load_dim / mission.path_limits["max_load"] - 1.0])


# import numpy as np
# import trajopt.utils.tools as tools

# import numpy as np
# import jax 
# import jax.numpy as jnp
# import trajopt.utils.tools as tools
# jax.config.update("jax_enable_x64", True)

# =============================== LOCAL AERO DATA: ===============================
# 'lookup' option for nonlinear_aero relies on local aero data not in repo

import sys
import trajopt.local.modules.mission.aero.marsgram_dens_lut_jax as dens
import trajopt.local.modules.mission.aero.cobra_aero as aero
# ================================================================================

def terminal_cost(t, z, nu, problem):
    return z[3]

def atmosphere_model_nonjax(rs, problem):
    '''
    Returns density as a function of orbital radius

    TODO: (carlos)
    expand 'lookup' option to be higher dimensional
    remember LUT grow exponenetiall in size!
    '''

    mission = problem.mission
    model   = problem.model
    method  = problem.method

    rdim    = rs * method.nondim["nd"]
    hdim    = rdim - mission.planet["r"]

    # TODO (carlos): add the remaining options for atmosphere model
    if mission.flags["aero_type"] == "lookup":
        rho = np.interp(hdim/1e3, dens.h_grid, dens.rho_vals)

    elif mission.flags["aero_type"] == "exponential":
        rho = mission.planet["rho"] * np.exp(-hdim / mission.planet["H"])

    return rho

# Atmospheric temperature model
def mars_temperature_nonjax(hdim):

    T_0 = 249.75     # K (surface temperature on Mars), 210
    L   = 0.0025    # K/m (lapse rate), .0025
    T = np.maximum(T_0 - L * hdim, 150.0)
    
    return T  # Ensure temperature doesn't fall below ~150K

# Speed of sound calculation on Mars
def mars_speed_of_sound_nonjax(hdim):
    T = mars_temperature_nonjax(hdim)
    gamma_M = 1.29
    Rgas_M  = 188.92 
    return np.sqrt(gamma_M * Rgas_M * T)

def nonlinear_aero_nonjax(ts, zs, us, problem):

    mission = problem.mission
    model   = problem.model
    method  = problem.method

    # Extract states and controls

    rs = zs[0]
    vs = zs[3]

    alpha = us[1]
    alpha_deg = np.rad2deg(alpha)
    mass_s = mission.vehicle['mass'] / method.nondim['nm']

    rdim = rs * method.nondim["nd"]
    hdim = rdim - mission.planet["r"]

    rho = atmosphere_model_nonjax(rs, problem)
    a = mars_speed_of_sound_nonjax(hdim)

    M = vs / (a / method.nondim['nv'])

    Cd, Cl = aero_nonjax.get_cd_cl(M, alpha_deg)

    rho_s = rho / (method.nondim['nm'] / method.nondim['nd'] ** 3)
    sref_s = mission.vehicle['sref'] / method.nondim['nd'] ** 2

    L    = (1 / mass_s) * 0.5 * rho_s * vs**2 * Cl * sref_s
    D    = (1 / mass_s) * 0.5 * rho_s * vs**2 * Cd * sref_s

    return {'L': L, 'D': D, 'Cl': Cl, 'Cd': Cd, 'alpha': alpha,'rho': rho}