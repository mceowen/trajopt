import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_enable_x64", True)


def exp_density_jax(t: float, z: Array, nu: Array, params: dict, fcns: dict | None = None) -> Array:
    """Exponential atmosphere density model (kg/m³)."""
    h = z[0] - params["planet"]["r"]

    return params["planet"]["rho"] * jnp.exp(-h / params["planet"]["H"])


def lut_density_jax(t: float, z: Array, nu: Array, params: dict) -> Array:
    """Lookup-table atmosphere density model (kg/m³)."""
    h = z[0] - params["planet"]["r"]

    return jnp.interp(h / 1e3, dens.h_grid, dens.rho_vals)


def nonlinear_aero_jax(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> dict:
    """Nonlinear aerodynamic force coefficients and state for VTOL."""
    r = z[0]
    v = z[3]

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

    rho = fcns["density_model"](t, z, nu, params)
    sref = params["vehicle"]["sref"]
    mass = params["vehicle"]["mass"]

    L = (0.5 / mass) * rho * sref * Cl * v**2
    D = (0.5 / mass) * rho * sref * Cd * v**2

    return {"L": L, "D": D, "Cl": Cl, "Cd": Cd, "alpha": alpha, "rho": rho}
