import numpy as np
import jax 
import jax.numpy as jnp
import trajopt.core.utils.tools as tools
jax.config.update("jax_enable_x64", True)
import trajopt.core.model.obstacles     as obstacles

def dynamics_jax(t, z, nu, trajopt_obj, t_vec=None):

    mission = trajopt_obj.mission
    model = trajopt_obj.model
    method = trajopt_obj.method

    ctrl_type = model.flags['ctrl_type']

    # Extract constant param values from struct
    Om = mission.planet["omega"] / (method.nondim["nang"] / method.nondim["nt"])
    Kg = mission.planet["mu"]    / (method.nondim["na"] * method.nondim["nd"] ** 2)

    # Extract states
    rs, theta, phi, vs, gamma, psi = z

    if ctrl_type == 'bank_aoa':
        alpha   = nu[1]
    
    sigma       = nu[0]
    
    # Determine lift and drag coefficients from velocity
    aero = mission.nonlinear_aero(t, z, nu)
    L    = aero["L"]
    D    = aero["D"]

    # Extract sines and cosines of various values
    cp  = jnp.cos(phi)
    sp  = jnp.sin(phi)
    tp  = jnp.tan(phi)
    cg  = jnp.cos(gamma)
    sg  = jnp.sin(gamma)
    tg  = jnp.tan(gamma)
    cps = jnp.cos(psi)
    sps = jnp.sin(psi)

    cs  = jnp.cos(sigma)
    ss  = jnp.sin(sigma)
    
    # state derivative function
    xDot = jnp.array([
        vs * sg,
        vs * cg * sps / (rs * cp),
        vs * cg * cps / rs, 
        - D - Kg * sg / rs**2 + Om**2 * rs * cp * (sg * cp - cg * sp * cps),
        (1 / vs) * ( L * cs + (vs**2 - Kg / rs) * cg / rs ) + 2 * Om * cp * sps + Om**2 * rs * (1 / vs) * cp * (cg * cp + sg * cps * sp),
        (1 / vs) * ( L * ss / cg + vs**2 * cg * sps * tp / rs ) - 2 * Om * (tg * cps * cp - sp) + Om**2 * rs * (1 / (vs * cg)) * sps * sp * cp
    ])

    return xDot

# def max_q(t, z, nu, trajopt_obj):  #dynamic pressure
#     mission = trajopt_obj.mission
#     method = trajopt_obj.method
    
#     rs = z[0]
#     vs = z[3]

#     rho = mission.mission_module.atmosphere_model_jax(rs, trajopt_obj)
#     nv = method.nondim["nv"]

#     q_dim = 0.5 * rho * (vs * nv) ** 2 

#     return jnp.array([q_dim / mission.path_limits['max_q'] - 1.0])

# def max_Q(t, z, nu, trajopt_obj): # heat rate
#     mission = trajopt_obj.mission
#     method = trajopt_obj.method

#     rs = z[0]
#     vs = z[3]

#     rho = mission.mission_module.atmosphere_model_jax(rs, trajopt_obj)
#     nv = method.nondim["nv"]

#     Q_dim = mission.vehicle["kQ"] * rho ** 0.5 * (vs * nv) ** 3

#     return jnp.array([Q_dim / mission.path_limits["max_Q"] - 1.0])

# def max_load(t, z, nu, trajopt_obj): # normal load
#     mission = trajopt_obj.mission
#     method = trajopt_obj.method

#     rs = z[0]
#     vs = z[3]

#     aero = mission.nonlinear_aero(t, z, nu)

#     L = aero["L"]
#     D = aero["D"]

#     load_dim = jnp.sqrt(L ** 2 + D ** 2) * method.nondim["na"]

#     return jnp.array([load_dim / mission.path_limits["max_load"] - 1.0])



def max_q_nonjax(t, z, nu, trajopt_obj):  #dynamic pressure
    mission = trajopt_obj.mission; method = trajopt_obj.method
    rs = z[0]; vs = z[3]
    rho = atmosphere_model_nonjax(rs/method.nondim['nd'], trajopt_obj)
    nv = 1; #method.nondim["nv"]
    q_dim = 0.5 * rho * (vs * nv) ** 2 
    return q_dim #np.array([q_dim / mission.path_limits['max_q'] - 1.0])

def max_Q_nonjax(t, z, nu, trajopt_obj): # heat rate
    mission = trajopt_obj.mission; method = trajopt_obj.method
    rs = z[0]; vs = z[3]
    rho = atmosphere_model_nonjax(rs/method.nondim['nd'], trajopt_obj)
    nv = 1; #method.nondim["nv"]
    Q_dim = mission.vehicle["kQ"] * rho ** 0.5 * (vs * nv) ** 3
    return Q_dim #np.array([Q_dim / mission.path_limits["max_Q"] - 1.0])

def max_load_nonjax(t, z, nu, trajopt_obj): # normal load
    mission = trajopt_obj.mission; method = trajopt_obj.method
    rs = z[0]; vs = z[3]
    Mstate = method.nondim['M']['state']['d2nd']
    stateidx = trajopt_obj.indices.z['state']
    aero = nonlinear_aero_nonjax(t, z[stateidx] @ Mstate, nu,trajopt_obj)
    # aero = nonlinear_aero_nonjax(t, z, nu,trajopt_obj)
    L = aero["L"]; D = aero["D"]
    load_dim = np.sqrt(L ** 2 + D ** 2); #* method.nondim["na"]
    return load_dim #np.array([load_dim / mission.path_limits["max_load"] - 1.0])


# import numpy as np
# import trajopt.core.utils.tools as tools

# import numpy as np
# import jax 
# import jax.numpy as jnp
# import trajopt.core.utils.tools as tools
# jax.config.update("jax_enable_x64", True)

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

def get_cost_cnstr_nondim(trajopt_obj):
    '''
    Returns the dimensional units of cost and constraint functions
    '''
    mission = trajopt_obj.mission
    method = trajopt_obj.method

    nd = method.nondim["nd"]
    nt = method.nondim["nt"]
    nm = method.nondim["nm"]

    ncost = method.nondim["nv"]
    np_ineq = np.array([1, 1, nm / (nd * nt**2), 1, 1])

    return ncost, np_ineq


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

