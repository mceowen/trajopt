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
    handle the general case? or make seperate function?ƒ
    '''

    mission = problem.mission
    model = problem.model
    method = problem.method

    ctrl_type = model['ctrl_type']

    # Extract key params
    nv = method['nv']
    rhoe = mission['planet']['rho']
    re = mission['planet']['r']
    N = extract_N(ts)
    mass_nd = mission['vehicle']['mass'] / method['nondim']['nm']

    B = (mission['planet']['r'] / mission['vehicle']['mass']) * (mission['vehicle']['sref'] / 2)
    
    # Initialize output vectors
    Cl      = jnp.zeros(N)
    Cd      = jnp.zeros(N)
    alpha   = jnp.zeros(N)
    L       = jnp.zeros(N)
    D       = jnp.zeros(N)

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

    for k in range(N):
        # Extract states and controls
        tk = ts if N == 1 else ts[k]
        zk = zs if N == 1 else zs[k]
        uk = us if N == 1 else us[k]
        r, theta, phi, v, gamma, psi = zs if N == 1 else zs[k]


        # Extract control
        if ctrl_type == 'bank_only':
            # Velocity-dependent coefficients
            v_sat       = min(v * nv, vlim)
            Cl[k]       = Kl1 + Kl2 * (v_sat - vlim)**2 + Kl3 * (v_sat - vlim)**4
            Cd[k]       = Kd1 + Kd2 * Cl[k] + Kd3 * Cl[k]**2
            alpha[k]    = jnp.deg2rad(alphlim_deg - kalph * (min(v * nv, vlim) - vlim)**2)
        elif ctrl_type in 'bank_aoa':
            alpha[k]    = us if N == 1 else us[k,1]
            alpha_deg   = jnp.rad2deg(alpha[k])
            Cl[k]       = Kl1h + Kl2h * alpha_deg + Kl3h * alpha_deg**2
            Cd[k]       = Kd1h + Kd2h * alpha_deg + Kd3h * alpha_deg**2 + Kd4h * alpha_deg**3 + Kd5h * alpha_deg**4

        # Compute lift and drag
        rho = atmosphere_model_jax(rs, problem)
        L[k]    = (B / mass_nd) * rho * Cl[k] * v**2
        D[k]    = (B / mass_nd) * rho * Cd[k] * v**2

    return {
        'L': L,
        'D': D,
        'Cl': Cl,
        'Cd': Cd,
        'alpha': alpha,
        'rho': rho
    }

def extract_N(ts):
    N = 1 if isinstance(ts, float) else (ts.shape[0] if ts.ndim == 1 else ts.shape[1])
    return N

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