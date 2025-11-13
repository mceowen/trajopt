import numpy as np
import trajopt.utils.tools as tools

import numpy as np
import jax 
import jax.numpy as jnp
import trajopt.utils.tools as tools
jax.config.update("jax_enable_x64", True)

# =============================================================================
# terminal cost
# =============================================================================

def terminal_cost(t, z, nu, problem):
    return -z[2]

def analytical_affine_approximation_terminal_cost(t, z, nu, problem):
    model = problem.model
    
    cost = terminal_cost(t, z, nu, problem)

    dcostdz = np.array([0, 0, -1, 0, 0, 0]).reshape(1, -1)
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

    ncost = method.nondim["nang"]
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


def nonlinear_aero_jax(t, z, nu, problem):
    '''
    returns all aero data as a function of full state
    
    TODO: (carlos)
    this function currently only accepts a single time step!
    handle the general case? or make seperate function?ƒ
    '''

    mission = problem.mission
    model = problem.model
    method = problem.method


    ctrl_type = model.flags['ctrl_type']

    # Extract key params
    nv = method.nondim['nv']
    mass_nd = mission.vehicle['mass'] / method.nondim['nm']

    rs = z[0]
    vs = z[3]

    # Extract control
    if ctrl_type == 'bank_only':
        alpha_deg = 15
        alpha = jnp.deg2rad(alpha_deg)

    elif ctrl_type == 'bank_aoa':
        alpha = nu[1]
        alpha_deg = jnp.rad2deg(alpha)

    # COEFFICIENTS

    M   = vs * nv / ((1.4 * 287 * 239)**0.5)
    cl0 = 0.0052  * jnp.log(M) - 0.0334
    cl1 = 0.03    * (M**(-0.49))
    cd0 = 0.0577  * jnp.exp(-0.042*M)
    cd1 = 0.00879 * jnp.log(M) - 0.0192
    cd2 = 0.4521  * (M**(0.4856))

    # AoA-DEPENDENT AERO COEFFICIENTS
    Cl = cl0 + cl1 * alpha_deg
    Cd = cd0 + (cd1 * Cl) + (cd2 * (Cl**2))

    rho = atmosphere_model_jax(rs, problem)
    rho_s = rho / (method.nondim["nm"] / method.nondim["nd"] ** 3)
    sref_s = mission.vehicle["sref"] / method.nondim["nd"] ** 2
    
    L = 0.5 * (1 / mass_nd) * rho_s * sref_s * Cl * vs**2
    D = 0.5 * (1 / mass_nd) * rho_s * sref_s * Cd * vs**2

    return {'L': L, 'D': D, 'Cl': Cl, 'Cd': Cd, 'alpha': alpha, 'rho': rho}

def custom_constraints(subproblem):
    pass

def custom_cost(subproblem):
    pass

def set_custom_params(problem):
    pass