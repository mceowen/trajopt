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

    mass = params['vehicle']['mass']
    sref_shell = params['vehicle']['sref_shell']
    sref_chute = params['vehicle']['sref_chute']
    LD = params['vehicle']['LD']
    bc = params['vehicle']['bc']

    Cd_entry = mass / (bc * sref_shell)
    Cl_entry = Cd_entry * LD

    Cd_chute = 0.55
    Cl_chute = 0.0

    p = params.get('p', 0)

    Cd   = (1 - p) * Cd_entry   + p * Cd_chute
    Cl   = (1 - p) * Cl_entry   + p * Cl_chute
    sref = (1 - p) * sref_shell + p * sref_chute

    L = (1 / mass) * 0.5 * rho * v**2 * Cl * sref
    D = (1 / mass) * 0.5 * rho * v**2 * Cd * sref

    return {"L": L, "D": D, "alpha": 0, "rho": rho}

def lift_drag(t, z, nu, params, fcns):

    aero = nonlinear_aero_jax(t, z, nu, params, fcns)

    L = aero["L"]
    D = aero["D"]
    return jnp.array([L, D])