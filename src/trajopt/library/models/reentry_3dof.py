import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# ===============================
# JAX MODEL
# ===============================
    
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
    aero = fcns['nonlinear_aero'](t, z, nu, params, fcns)
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

def heat_rate(t, z, nu, params, fcns):

    r = z[0]
    v = z[3]

    rho = fcns['density_model'](t, z, nu, params, fcns)

    return jnp.array([params['vehicle']['kQ'] * rho ** 0.5 * v ** 3])

def dynamic_pressure(t, z, nu, params, fcns):
    
    r = z[0]
    v = z[3]

    rho = fcns['density_model'](t, z, nu, params, fcns)

    return jnp.array([0.5 * rho * (v) ** 2])

def aero_load(t, z, nu, params, fcns):

    aero = fcns['nonlinear_aero'](t, z, nu, params, fcns)

    L = aero["L"]
    D = aero["D"]

    return jnp.array([jnp.sqrt(L ** 2 + D ** 2)])

def long_lat(t, z, nu, params, fcns):
    theta = z[1]
    phi = z[2]

    return jnp.array([theta, phi])

def long_lat_alt(t, z, nu, params, fcns):
    r = z[0]
    theta = z[1]
    phi = z[2]

    return jnp.array([theta, phi, r - params['planet']['r']])

def altitude(t, z, nu, params, fcns):

    r = z[0]

    return jnp.array([r - params['planet']['r']])

def longitude(t, z, nu, params, fcns):
    
    phi = z[1]

    return jnp.array([phi])

def latitude(t, z, nu, params, fcns):
    theta = z[2]

    return jnp.array([theta])

def velocity(t, z, nu, params, fcns):
    v = z[3]

    return jnp.array([v])

def fpa(t, z, nu, params, fcns):
    gamma = z[4]

    return jnp.array([gamma])

def heading(t, z, nu, params, fcns):
    psi = z[5]

    return jnp.array([psi])

# ===============================
# CASADI MODEL
# ===============================