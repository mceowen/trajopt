import numpy as np
import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# import local aero data
import sys
sys.path.append("/Users/carlosm/Documents/guidance/hypersonics/prototypes/local")
import marsgram_dens_lut as dens


def nonlinear_aero_jax(ts, zs, us, problem):

    mission = problem.mission
    model = problem.model
    method = problem.method

    # Extract states and controls
    rs, _, _, vs, _, _ = zs

    # Compute altitude
    rdim = rs * method.nondim["nd"]
    hdim = rdim - mission.planet["re"]
    
    rho = jnp.interp(hdim/1e3, dens.h_grid, dens.rho_vals)

    rho_s = rho / (method.nondim["nm"] / method.nondim["nd"]**3)
    sref_s = mission.vehicle["sref"] / method.nondim["nd"]**2
    bc_s = mission.vehicle["bc"] / (method.nondim["nm"] / (method.nondim["nd"]**2))

    D    = 0.5 * (1 / bc_s) * rho_s * vs**2
    L    = D * mission.vehicle["LD"]

    alpha = 0

    return {"L": L, "D": D, "alpha": alpha, "rho": rho}

def system_dynamics_jax(ts, zs, us, problem, t_vec=None):

    mission = problem.mission
    model = problem.model
    method = problem.method

    # Extract constant param values from struct
    Om = mission.planet["omega_s"]
    Kg = mission.planet["kg"]

    # Extract states
    rs, theta, phi, vs, gamma, psi = zs

    # Determine lift and drag coefficients from velocity
    aero = nonlinear_aero_jax(ts, zs, us, problem)
    L    = aero["L"]
    D    = aero["D"]

    # extract controls 
    if t_vec is None:
        us2 = us
    else:
        us2 = np.array([jnp.interp(ts, t_vec, us[:, i]) for i in range(model.m)])

    # Extract bank angle
    sigma   = us2[0]
    alpha   = us2[1]

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