import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def exp_density_jax(t, z, nu, params, fcns=None):

    r = z[0]

    h = r - params['planet']["r"]

    rho = params['planet']["rho"] * jnp.exp(-h / params['planet']["H"])

    return rho

def nonlinear_aero_jax(t, z, nu, params, fcns):

    r = z[0]
    v = z[3]

    rho = fcns['density_model'](t, z, nu, params)

    D    = 0.5 * (1 / params['vehicle']["bc"]) * rho * v**2
    L    = D * params['vehicle']["LD"]

    alpha = 0

    return {"L": L, "D": D, "alpha": alpha, "rho": rho}