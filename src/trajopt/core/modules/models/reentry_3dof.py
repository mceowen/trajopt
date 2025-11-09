import numpy as np
import jax 
import jax.numpy as jnp
import trajopt.utils.tools as tools
jax.config.update("jax_enable_x64", True)
import trajopt.core.modules.models.obstacles     as obstacles

def dynamics_jax(ts, zs, us, problem, t_vec=None):

    mission = problem.mission
    model = problem.model
    method = problem.method

    ctrl_type = model.flags['ctrl_type']

    # Extract constant param values from struct
    Om = mission.planet["omega"] / (method.nondim["nang"] / method.nondim["nt"])
    Kg = mission.planet["mu"]    / (method.nondim["na"] * method.nondim["nd"] ** 2)

    # Extract states
    rs, theta, phi, vs, gamma, psi = zs

    # Determine lift and drag coefficients from velocity
    aero = mission.nonlinear_aero(ts, zs, us)
    L    = aero["L"]
    D    = aero["D"]

    # extract controls 
    if t_vec is None:
        us2 = us
    else:
        us2 = np.array([jnp.interp(ts, t_vec, us[:, i]) for i in range(model.m)])

    # Extract bank angle
    if ctrl_type == 'bank_aoa':
        alpha   = us2[1]
    
    sigma       = us2[0]

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

def max_q(ts, zs, us, problem):
    mission = problem.mission
    method = problem.method
    
    rs = zs[0]
    vs = zs[3]

    rho = mission.mission_module.atmosphere_model_jax(rs, problem)
    nv = method.nondim["nv"]

    return  0.5 * rho * (vs * nv) ** 2 - mission.path_limits['qmax']

def max_Q(ts, zs, us, problem):
    mission = problem.mission
    method = problem.method

    rs = zs[0]
    vs = zs[3]

    rho = mission.mission_module.atmosphere_model_jax(rs, problem)
    nv = method.nondim["nv"]

    return mission.vehicle["kQ"] * rho ** 0.5 * (vs * nv) ** 3 - mission.path_limits["max_Q"]