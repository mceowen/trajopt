import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_enable_x64", True)


def exp_density_jax(t: float, z: Array, nu: Array, params: dict, fcns: dict | None = None) -> Array:
    """Exponential atmosphere density model (kg/m³), JAX version."""
    h = z[0] - params["planet"]["r"]

    return params["planet"]["rho"] * jnp.exp(-h / params["planet"]["H"])


def nonlinear_aero_jax(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> dict:
    """Nonlinear aerodynamic force coefficients and state for MSL, JAX version."""
    v = z[3]
    rho = fcns["density_model"](t, z, nu, params)

    D = 0.5 * (1 / params["vehicle"]["bc"]) * rho * v**2
    L = D * params["vehicle"]["LD"]

    return {"L": L, "D": D, "alpha": 0, "rho": rho}
