"""Solar-sail dynamics ported from riccardo-branch (85ab54a).

State: [p - r_GEO, f, g, h, k, elapsed_time], in km and seconds.
The independent variable is true longitude in radians. Earth gravity and
solar radiation pressure act on the sail; the Sun rotates uniformly about Z.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _scalar_longitude(longitude):
    """Accept trajopt's one-element time vector as well as a scalar."""
    return jnp.ravel(jnp.asarray(longitude))[0]


def orbital_frame(h, k, longitude):
    """Radial, transverse and orbit-normal unit vectors in ECI."""
    s_squared = 1.0 + h**2 + k**2
    equinoctial_x = jnp.array([1.0 - k**2 + h**2, 2.0 * h * k, -2.0 * k]) / s_squared
    equinoctial_y = jnp.array([2.0 * h * k, 1.0 + k**2 - h**2, 2.0 * h]) / s_squared
    cosine = jnp.cos(longitude)
    sine = jnp.sin(longitude)
    radial = cosine * equinoctial_x + sine * equinoctial_y
    transverse = -sine * equinoctial_x + cosine * equinoctial_y
    return (radial, transverse, jnp.cross(radial, transverse))


def semi_latus_rectum(x, params):
    """Absolute semi-latus rectum from the offset carried in the state."""
    return x[0] + params.geo_radius_km


def position_eci(x, longitude, params):
    """Spacecraft position in ECI coordinates, in km."""
    longitude = _scalar_longitude(longitude)
    p = semi_latus_rectum(x, params)
    f, g = (x[1], x[2])
    q = 1.0 + f * jnp.cos(longitude) + g * jnp.sin(longitude)
    radial, _, _ = orbital_frame(x[3], x[4], longitude)
    return p / q * radial


def _unit_vector(vector, fallback):
    squared_norm = jnp.dot(vector, vector)
    safe_norm = jnp.sqrt(squared_norm + 1e-24)
    return jnp.where(squared_norm > 1e-24, vector / safe_norm, fallback)


def sun_direction(x, params):
    """Earth-to-Sun unit vector at the spacecraft's elapsed time."""
    base = jnp.asarray(params.sun_direction_eci, dtype=float)
    base = base / jnp.linalg.norm(base)
    angle = params.sun_angular_rate_rad_s * x[5]
    cos_a, sin_a = (jnp.cos(angle), jnp.sin(angle))
    return jnp.array(
        [cos_a * base[0] - sin_a * base[1], sin_a * base[0] + cos_a * base[1], base[2]]
    )


def light_direction(x, params):
    """Unit vector along photon travel, i.e. pointing away from the Sun."""
    return -sun_direction(x, params)


def sail_normal_from_primer(x, u, longitude, params):
    """Locally optimal sail normal and cone angle, from Equations (23)-(25)."""
    longitude = _scalar_longitude(longitude)
    radial, transverse, _ = orbital_frame(x[3], x[4], longitude)
    primer = jnp.cos(u[0]) * radial + jnp.sin(u[0]) * transverse
    sunward = light_direction(x, params)
    cos_gamma_raw = jnp.clip(jnp.dot(sunward, primer), -1.0, 1.0)
    cos_gamma = jnp.clip(cos_gamma_raw, -1.0 + 1e-10, 1.0 - 1e-10)
    sin_gamma = jnp.sqrt(jnp.maximum(0.0, 1.0 - cos_gamma**2))
    numerator = jnp.sqrt(cos_gamma**2 + 8.0) - 3.0 * cos_gamma
    denominator = 4.0 * sin_gamma
    alpha = jnp.arctan2(numerator, denominator)
    reference = jnp.where(
        jnp.abs(sunward[2]) < 0.9, jnp.array([0.0, 0.0, 1.0]), jnp.array([0.0, 1.0, 0.0])
    )
    fallback = _unit_vector(jnp.cross(reference, sunward), transverse)
    e2 = _unit_vector(primer - cos_gamma_raw * sunward, fallback)
    sail_normal = jnp.cos(alpha) * sunward + jnp.sin(alpha) * e2
    return (sail_normal, alpha)


def solar_sail_acceleration_lvlh(x, u, longitude, params):
    """Sail acceleration resolved in the local radial/transverse/normal frame."""
    longitude = _scalar_longitude(longitude)
    radial, transverse, normal = orbital_frame(x[3], x[4], longitude)
    sail_normal, alpha = sail_normal_from_primer(x, u, longitude, params)
    pressure_acceleration = params.solar_pressure_n_m2 * params.area_to_mass_m2_kg / 1000.0
    epsilon = params.reflectivity
    sunward = light_direction(x, params)
    cosine = jnp.cos(alpha)
    acceleration_eci = (
        params.illumination
        * pressure_acceleration
        * cosine
        * ((1.0 - epsilon) * sunward + 2.0 * epsilon * cosine * sail_normal)
    )
    return jnp.array(
        [
            jnp.dot(radial, acceleration_eci),
            jnp.dot(transverse, acceleration_eci),
            jnp.dot(normal, acceleration_eci),
        ]
    )


def dynamics(x, u, longitude, params, fcns):
    """Modified-equinoctial dynamics with longitude as the independent variable."""
    del fcns
    longitude = _scalar_longitude(longitude)
    p = semi_latus_rectum(x, params)
    f, g, h, k = (x[1], x[2], x[3], x[4])
    mu = params.earth_mu_km3_s2
    cosine = jnp.cos(longitude)
    sine = jnp.sin(longitude)
    q = 1.0 + f * cosine + g * sine
    s_squared = 1.0 + h**2 + k**2
    out_of_plane_factor = h * sine - k * cosine
    radial_accel, transverse_accel, normal_accel = solar_sail_acceleration_lvlh(
        x, u, longitude, params
    )
    common = jnp.sqrt(p / mu) / q
    p_dot = common * (2.0 * p) * transverse_accel
    f_dot = common * (
        q * sine * radial_accel
        + ((q + 1.0) * cosine + f) * transverse_accel
        - g * out_of_plane_factor * normal_accel
    )
    g_dot = common * (
        -q * cosine * radial_accel
        + ((q + 1.0) * sine + g) * transverse_accel
        + f * out_of_plane_factor * normal_accel
    )
    h_dot = common * (0.5 * s_squared * cosine) * normal_accel
    k_dot = common * (0.5 * s_squared * sine) * normal_accel
    longitude_dot = common * out_of_plane_factor * normal_accel + jnp.sqrt(mu * p) * (q / p) ** 2
    return jnp.array([p_dot, f_dot, g_dot, h_dot, k_dot, 1.0]) / longitude_dot
