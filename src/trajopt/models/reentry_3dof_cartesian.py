import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_enable_x64", True)


def cr(v: Array) -> Array:
    """Skew-symmetric cross-product matrix for vector v."""
    return jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def dynamics(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """3-DoF atmospheric reentry dynamics in Cartesian coordinates."""
    r = z[0:3]
    v = z[3:6]

    sigma_rad = jnp.deg2rad(nu[0])
    mu = params["planet"]["mu"]

    a_grav = -mu * r / jnp.linalg.norm(r) ** 3

    # unit vectors for velocity, right, down (velocity frame)
    e_v = v / jnp.linalg.norm(v)
    e_r = cr(e_v) @ r / jnp.linalg.norm(cr(e_v) @ r)
    e_d = cr(v) @ e_r / jnp.linalg.norm(cr(v) @ e_r)

    aero = fcns["nonlinear_aero_jax"](t, z, nu, params)
    L_mag = aero["L"]
    D_mag = aero["D"]

    L = -L_mag * (jnp.cos(sigma_rad) * e_d - jnp.sin(sigma_rad) * e_r)
    D = -D_mag * e_v
    a_aero = L + D

    # state derivative function
    return jnp.concatenate([v, a_aero + a_grav])


def heat_rate(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Convective heat rate (W/m²). Uses JAX atmosphere model."""
    v = jnp.linalg.norm(z[3:6])
    rho = fcns["atmosphere_model_jax"](t, z, nu, params)

    return jnp.array([params["vehicle"]["kQ"] * rho**0.5 * v**3])


def dynamic_pressure(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Dynamic pressure q-bar (Pa). Uses JAX atmosphere model."""
    r = jnp.linalg.norm(z[0:3])
    v = jnp.linalg.norm(z[3:6])
    rho = fcns["atmosphere_model_jax"](t, z, nu, params)

    return jnp.array([0.5 * rho * v**2])


def aero_load(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Aerodynamic load magnitude (m/s²). Uses JAX aero model."""
    aero = fcns["nonlinear_aero_jax"](t, z, nu, params)
    L = aero["L"]
    D = aero["D"]

    return jnp.array([jnp.sqrt(L**2 + D**2)])


def heat_rate_nonjax(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> float:
    """Convective heat rate (W/m²). Uses non-JAX atmosphere model."""
    r = jnp.linalg.norm(z[0:3])
    v = jnp.linalg.norm(z[3:6])
    rho = fcns["atmosphere_model_nonjax"](t, z, nu, params)

    return params["vehicle"]["kQ"] * rho**0.5 * v**3


def dynamic_pressure_nonjax(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> float:
    """Dynamic pressure q-bar (Pa). Uses non-JAX atmosphere model."""
    r = jnp.linalg.norm(z[0:3])
    v = jnp.linalg.norm(z[3:6])
    rho = fcns["atmosphere_model_nonjax"](t, z, nu, params)

    return 0.5 * rho * v**2


def aero_load_nonjax(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> float:
    """Aerodynamic load magnitude sqrt (m/s²). Uses non-JAX aero model."""
    r = jnp.linalg.norm(z[0:3])
    v = jnp.linalg.norm(z[3:6])
    aero = fcns["nonlinear_aero_nonjax"](t, z, nu, params)
    L = aero["L"]
    D = aero["D"]

    return (L**2 + D**2) ** 0.5
