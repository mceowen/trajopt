import jax
import jax.numpy as jnp
from jax import Array

from trajopt.utils.tools import AttrDict
jax.config.update("jax_enable_x64", True)

# ===============================
# JAX MODEL
# ===============================
def dynamics(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """3-DoF atmospheric reentry dynamics."""

    Om = jnp.deg2rad(params.planet.omega)
    mu = params.planet.mu

    r, theta, phi, v, gamma, psi = z

    phi_rad = jnp.deg2rad(phi)
    gamma_rad = jnp.deg2rad(gamma)
    psi_rad = jnp.deg2rad(psi)

    sigma_rad = jnp.deg2rad(nu[0])

    # Determine lift and drag coefficients from velocity
    aero = fcns.nonlinear_aero(t, z, nu, params, fcns)
    L = aero.L
    D = aero.D

    # Extract sines and cosines of various values
    cp = jnp.cos(phi_rad)
    sp = jnp.sin(phi_rad)
    tp = jnp.tan(phi_rad)
    cg = jnp.cos(gamma_rad)
    sg = jnp.sin(gamma_rad)
    tg = jnp.tan(gamma_rad)
    cps = jnp.cos(psi_rad)
    sps = jnp.sin(psi_rad)
    cs = jnp.cos(sigma_rad)
    ss = jnp.sin(sigma_rad)

    return jnp.array(
        [
            v * sg,
            jnp.rad2deg(v * cg * sps / (r * cp)),
            jnp.rad2deg(v * cg * cps / r),
            -D - mu * sg / r**2 + Om**2 * r * cp * (sg * cp - cg * sp * cps),
            jnp.rad2deg(
                (1 / v) * (L * cs + (v**2 - mu / r) * cg / r)
                + 2 * Om * cp * sps
                + Om**2 * r * (1 / v) * cp * (cg * cp + sg * cps * sp),
            ),
            jnp.rad2deg(
                (1 / v) * (L * ss / cg + v**2 * cg * sps * tp / r)
                - 2 * Om * (tg * cps * cp - sp)
                + Om**2 * r * (1 / (v * cg)) * sps * sp * cp,
            ),
        ],
    )


def nonlinear_aero(t: float, x: Array, u: Array, params: dict, fcns: dict) -> dict:
    """Nonlinear aerodynamic acceleration coefficients and state"""
    v = x[3]
    rho = fcns.density_model(t, x, u, params, fcns)

    vehicle = params.vehicle
    mass = vehicle.mass
    sref = vehicle.sref

    coeffs = fcns.coeffs(t, x, u, params, fcns)

    L = (1 / mass) * 0.5 * rho * v**2 * coeffs.cl * sref
    D = (1 / mass) * 0.5 * rho * v**2 * coeffs.cd * sref

    return AttrDict({"L": L, "D": D})


def heat_rate(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Convective heat rate (W/m²)."""
    v = z[3]
    rho = fcns.density_model(t, z, nu, params, fcns)

    return jnp.array([params.vehicle.kQ * rho**0.5 * v**3])


def dynamic_pressure(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Dynamic pressure q-bar (Pa)."""
    v = z[3]
    rho = fcns.density_model(t, z, nu, params, fcns)

    return jnp.array([0.5 * rho * v**2])


def aero_load(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Aerodynamic load magnitude (m/s²)."""
    aero = fcns.nonlinear_aero(t, z, nu, params, fcns)

    return jnp.array([jnp.sqrt(aero.L**2 + aero.D**2)])


def vel_heat_rate(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """[velocity (m/s), heat rate (W/m²)]."""
    v = z[3]
    rho = fcns.density_model(t, z, nu, params, fcns)

    return jnp.array([v, params.vehicle.kQ * rho**0.5 * v**3])


def vel_dynamic_pressure(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """[velocity (m/s), dynamic pressure (Pa)]."""
    v = z[3]
    rho = fcns.density_model(t, z, nu, params, fcns)

    return jnp.array([v, 0.5 * rho * v**2])


def vel_aero_load(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """[velocity (m/s), aero load (m/s²)]."""
    v = z[3]
    aero = fcns.nonlinear_aero(t, z, nu, params, fcns)

    return jnp.array([v, jnp.sqrt(aero.L**2 + aero.D**2)])


def vel_altitude(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """[velocity (m/s), altitude (m)]."""
    return jnp.array([z[3], z[0] - params.planet.r])


def long_lat(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Longitude and latitude [theta, phi] (deg, deg)."""
    return jnp.array([z[1], z[2]])


def long_lat_alt(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Longitude, latitude, and altitude output [theta, phi, h] (deg, deg, m)."""
    return jnp.array([z[1], z[2], z[0] - params.planet.r])


def altitude(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Altitude above planet surface (m)."""
    return jnp.array([z[0] - params.planet.r])


def longitude(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Longitude (deg)."""
    return jnp.array([z[1]])


def latitude(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Latitude (deg)."""
    return jnp.array([z[2]])


def velocity(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Velocity (m/s)."""
    return jnp.array([z[3]])


def fpa(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Flight-path angle (deg)."""
    return jnp.array([z[4]])


def heading(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Heading angle (deg)."""
    return jnp.array([z[5]])


def bank(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Bank angle (deg)."""
    return jnp.array([nu[0]])


def aoa(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Angle of attack (deg)."""
    return jnp.array([nu[1]])


def exp_density(t: float, z: Array, nu: Array, params: dict, fcns: dict | None = None) -> Array:
    """Exponential atmosphere density model (kg/m³), JAX version."""
    h = z[0] - params.planet.r

    return params.planet.rho * jnp.exp(-h / params.planet.H)


def lift_drag(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Lift and drag forces for MSL."""
    aero = fcns.nonlinear_aero(t, z, nu, params, fcns)

    return jnp.array([aero.L, aero.D])

# ===============================
# CASADI MODEL
# ===============================
