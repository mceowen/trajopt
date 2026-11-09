"""Shadow boundaries and the GEO graveyard terminal conditions."""

import cvxpy as cp
import jax.numpy as jnp
from trajopt.models.solar_sail.solar_sail import (
    _scalar_longitude,
    position_eci,
    light_direction,
    semi_latus_rectum,
)


def shadow_boundary(x, u, longitude, params):
    """Zero exactly on the cylindrical Earth-shadow terminator."""
    del u
    position = position_eci(x, longitude, params)
    sunward = light_direction(x, params)
    axial = jnp.dot(position, sunward)
    transverse_squared = jnp.dot(position, position) - axial**2
    radius_squared = params.earth_radius_km**2
    return jnp.atleast_1d((transverse_squared - radius_squared) / radius_squared)


def phase_end_longitude(x, u, longitude, params):
    """Expose the true longitude so an arc endpoint can be bounded."""
    del x, u, params
    return jnp.atleast_1d(_scalar_longitude(longitude))


def terminal_eccentricity(x, u, longitude, params):
    """Final eccentricity, for a buffered terminal inequality."""
    del u, longitude, params
    return jnp.atleast_1d(jnp.linalg.norm(x[1:3]))


def terminal_perigee_radius(x, u, longitude, params):
    """Final perigee radius in km, for a buffered terminal inequality."""
    del u, longitude
    eccentricity = jnp.linalg.norm(x[1:3])
    return jnp.atleast_1d(semi_latus_rectum(x, params) / (1.0 + eccentricity))


def terminal_eccentricity_cone(x, u, params):
    """Exact convex form of ``e_f <= e_max``."""
    del u
    return cp.norm(x[:, 1:3], axis=1) - float(params.final_eccentricity_max)


def terminal_perigee_cone(x, u, params):
    """Exact convex form of ``r_p,f >= r_target``."""
    del u
    target = float(params.final_perigee_radius_km)
    return target * (1.0 + cp.norm(x[:, 1:3], axis=1)) - (x[:, 0] + float(params.geo_radius_km))
