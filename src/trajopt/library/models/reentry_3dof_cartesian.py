import numpy as np
import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# skew symmetric cross product matrix function
def cr(v):
    return jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    
def dynamics(t, z, nu, params, fcns):

    r = z[0:3]
    v = z[3:6]

    sigma = nu[0]
    alpha = nu[1]

    sigma_rad = jnp.deg2rad(sigma)
    alpha_rad = jnp.deg2rad(alpha)

    mass = params['mission']['vehicle']['mass']
    mu = params['mission']['planet']['mu']

    a_grav = -mu * r / jnp.linalg.norm(r) ** 3

    # unit vectors for velocity, right, down (velocity frame)
    e_v = v / jnp.linalg.norm(v)
    e_r = cr(e_v) @ r / jnp.linalg.norm(cr(e_v) @ r)
    e_d = cr(v) @ e_r / jnp.linalg.norm(cr(v) @ e_r)

    aero = fcns['nonlinear_aero_jax'](t, z, nu, params)

    L_mag = aero["L"]
    D_mag = aero["D"]

    L = - L_mag * (jnp.cos(sigma_rad) * e_d - jnp.sin(sigma_rad) * e_r)

    D = - D_mag * e_v

    a_aero = (L + D)

    # state derivative function
    xDot = jnp.concatenate([v, a_aero + a_grav])

    return xDot

def heat_rate(t, z, nu, params, fcns): # heat rate

    r = jnp.linalg.norm(z[0:3])
    v = jnp.linalg.norm(z[3:6])

    rho = fcns['atmosphere_model_jax'](t, z, nu, params)

    return jnp.array([params['mission']['vehicle']['kQ'] * rho ** 0.5 * v ** 3])

def dynamic_pressure(t, z, nu, params, fcns):  #dynamic pressure
    
    r = jnp.linalg.norm(z[0:3])
    v = jnp.linalg.norm(z[3:6])

    rho = fcns['atmosphere_model_jax'](t, z, nu, params)

    return jnp.array([0.5 * rho * v ** 2])

def aero_load(t, z, nu, params, fcns): # normal load

    aero = fcns['nonlinear_aero_jax'](t, z, nu, params)

    L = aero["L"]
    D = aero["D"]

    return jnp.array([jnp.sqrt(L ** 2 + D ** 2)])

def dynamic_pressure_nonjax(t, z, nu, params, fcns):
    r = jnp.linalg.norm(z[0:3])
    v = jnp.linalg.norm(z[3:6])

    rho = fcns['atmosphere_model_nonjax'](t, z, nu, params)

    return 0.5 * rho * v ** 2

def heat_rate_nonjax(t, z, nu, params, fcns):
    r = jnp.linalg.norm(z[0:3])
    v = jnp.linalg.norm(z[3:6])

    rho = fcns['atmosphere_model_nonjax'](t, z, nu, params)

    return params['mission']['vehicle']['kQ'] * rho ** 0.5 * v ** 3

def aero_load_nonjax(t, z, nu, params, fcns):
    r = jnp.linalg.norm(z[0:3])
    v = jnp.linalg.norm(z[3:6])

    aero = fcns['nonlinear_aero_nonjax'](t, z, nu, params)

    L = aero["L"]
    D = aero["D"]

    return (L ** 2 + D ** 2) ** 0.5