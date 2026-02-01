import numpy as np
import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
    
def dynamics(t, z, nu, params, fcns):

    Om = jnp.deg2rad(params['planet']['omega'])
    mu = params['planet']['mu']

    # Extract states
    r, theta, phi, v, gamma, psi = z

    sigma = nu[0]
    alpha = nu[1]

    theta_rad = jnp.deg2rad(theta)
    phi_rad = jnp.deg2rad(phi)
    gamma_rad = jnp.deg2rad(gamma)
    psi_rad = jnp.deg2rad(psi)

    sigma_rad = jnp.deg2rad(sigma)
    alpha_rad = jnp.deg2rad(alpha)

    # Determine lift and drag coefficients from velocity
    aero = fcns['nonlinear_aero_jax'](t, z, nu, params)
    L    = aero["L"]
    D    = aero["D"]

    # Extract sines and cosines of various values
    cp  = jnp.cos(phi_rad)
    sp  = jnp.sin(phi_rad)
    tp  = jnp.tan(phi_rad)
    cg  = jnp.cos(gamma_rad)
    sg  = jnp.sin(gamma_rad)
    tg  = jnp.tan(gamma_rad)
    cps = jnp.cos(psi_rad)
    sps = jnp.sin(psi_rad)

    cs  = jnp.cos(sigma_rad)
    ss  = jnp.sin(sigma_rad)
    
    # state derivative function
    x_dot = jnp.array([
        v * sg,
        jnp.rad2deg(v * cg * sps / (r * cp)),
        jnp.rad2deg(v * cg * cps / r), 
        - D - mu * sg / r**2 + Om**2 * r * cp * (sg * cp - cg * sp * cps),
        jnp.rad2deg((1 / v) * ( L * cs + (v**2 - mu / r) * cg / r ) + 2 * Om * cp * sps + Om**2 * r * (1 / v) * cp * (cg * cp + sg * cps * sp)),
        jnp.rad2deg((1 / v) * ( L * ss / cg + v**2 * cg * sps * tp / r ) - 2 * Om * (tg * cps * cp - sp) + Om**2 * r * (1 / (v * cg)) * sps * sp * cp)
    ])

    return x_dot

def heat_rate(t, z, nu, params, fcns): # heat rate

    r = z[0]
    v = z[3]

    rho = fcns['atmosphere_model_jax'](t, z, nu, params)

    return jnp.array([params['vehicle']['kQ'] * rho ** 0.5 * v ** 3])

def dynamic_pressure(t, z, nu, params, fcns):  #dynamic pressure
    
    r = z[0]
    v = z[3]

    rho = fcns['atmosphere_model_jax'](t, z, nu, params)

    return jnp.array([0.5 * rho * (v) ** 2])

def aero_load(t, z, nu, params, fcns): # normal load

    aero = fcns['nonlinear_aero_jax'](t, z, nu, params)

    L = aero["L"]
    D = aero["D"]

    return jnp.array([jnp.sqrt(L ** 2 + D ** 2)])