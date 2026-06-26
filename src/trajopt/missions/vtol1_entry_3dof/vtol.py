import jax
import jax.numpy as jnp
from jax import Array
from trajopt.utils.tools import AttrDict

jax.config.update("jax_enable_x64", True)

def nonlinear_aero_jax(x: Array, u: Array, t: float, params: dict, fcns: dict) -> dict:
    """Nonlinear aerodynamic force coefficients and state for VTOL."""
    v = x[3]

    # Setup coefficient values
    kl1 = -0.041065
    kl2 = 0.016292
    kl3 = 0.0002602
    kd1 = 0.080505
    kd2 = -0.03026
    kd3 = 0.86495
    kalph = 0.20705 / (340**2)
    vlim = 4570
    alphlim_deg = 40

    vehicle = params.vehicle
    rho = fcns.density_model(x, u, t, params, fcns)

    # Velocity-dependent polynomial coefficients
    Kd1 = kd1
    Kd2 = kd2
    Kd3 = kd3
    Kl1 = kl1 + kl2 * alphlim_deg + kl3 * alphlim_deg**2
    Kl2 = -kl2 * kalph - 2 * kl3 * alphlim_deg * kalph
    Kl3 = kl3 * kalph**2

    # compute v_sat with jnp
    v_sat = jnp.minimum(v, vlim)

    # compute Cl/Cd locally then set into arrays
    Cl = Kl1 + Kl2 * (v_sat - vlim) ** 2 + Kl3 * (v_sat - vlim) ** 4
    Cd = Kd1 + Kd2 * Cl + Kd3 * Cl**2
    alpha = jnp.deg2rad(alphlim_deg - kalph * (jnp.minimum(v, vlim) - vlim) ** 2)

    
    sref = vehicle.sref
    mass = vehicle.mass

    L = (0.5 / mass) * rho * sref * Cl * v**2
    D = (0.5 / mass) * rho * sref * Cd * v**2

    return AttrDict({"L": L, "D": D, "Cl": Cl, "Cd": Cd, "alpha": alpha, "rho": rho})
