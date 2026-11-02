"""Scenario costs, constraints, and output callbacks."""
import cvxpy as cp
import jax.numpy as jnp


def thrust_accel_mag(x, u, t, params, fcns):
    """Thrust acceleration magnitude (m/s^2), smoothed near zero."""
    return jnp.array([jnp.sqrt(u[2]**2 + u[3]**2 + u[4]**2 + 1e-6)])


def thrust_accel_sq(x, u, t, params, fcns):
    """Squared thrust acceleration magnitude, the fuel proxy."""
    return jnp.array([u[2]**2 + u[3]**2 + u[4]**2])


def max_thrust_accel_cone(x, u, params):
    """||aT|| <= vehicle.aT_max, as a CVXPY second-order cone."""
    return cp.norm(u[:, 2:5], axis=1) - float(params.vehicle.aT_max)


def downrange_crossrange(x, u, t, params, fcns):
    """[downrange, crossrange] (m) to the touchdown target PCPF.

    Downrange is the great-circle range to the target resolved along the
    horizontal velocity direction, crossrange the component normal to it
    (positive when the target is right of track). Target longitude/latitude
    come from params.target (deg).
    """
    theta = jnp.deg2rad(x[1])
    phi = jnp.deg2rad(x[2])
    psi = jnp.deg2rad(x[5])

    theta_t = jnp.deg2rad(params.target.lon)
    phi_t = jnp.deg2rad(params.target.lat)

    # unit position vectors of vehicle and target in PCPF
    r_veh = jnp.array([
        jnp.cos(theta) * jnp.cos(phi),
        jnp.sin(theta) * jnp.cos(phi),
        jnp.sin(phi),
    ])
    r_tgt = jnp.array([
        jnp.cos(theta_t) * jnp.cos(phi_t),
        jnp.sin(theta_t) * jnp.cos(phi_t),
        jnp.sin(phi_t),
    ])

    # local North/East unit vectors at the vehicle in PCPF
    n_hat = jnp.array([
        -jnp.cos(theta) * jnp.sin(phi),
        -jnp.sin(theta) * jnp.sin(phi),
        jnp.cos(phi),
    ])
    e_hat = jnp.array([-jnp.sin(theta), jnp.cos(theta), 0.0])

    # bearing to target, clockwise from north
    bearing = jnp.arctan2(jnp.dot(r_tgt, e_hat), jnp.dot(r_tgt, n_hat))

    # atan2 form of the great-circle central angle rather than acos(dot)
    central = jnp.arctan2(
        jnp.linalg.norm(jnp.cross(r_veh, r_tgt)),
        jnp.dot(r_veh, r_tgt),
    )
    R = params.planet.r * central

    return jnp.array([R * jnp.cos(bearing - psi), R * jnp.sin(bearing - psi)])


def deploy_range_bias(x, u, t, params, fcns):
    """Deploy-surface residual [downrange - range_bias, crossrange] (m).

    Zero on the arc params.target.range_bias (m) uprange of the touchdown
    target with heading aligned toward it (deploy range bias, Mendeck & Craig
    AIAA 2011-6639, Fig. 3).
    """
    dr_cr = downrange_crossrange(x, u, t, params, fcns)
    return dr_cr - jnp.array([params.target.range_bias, 0.0])


def target_miss_distance_sq(x, u, t, params, fcns):
    """Squared great-circle miss distance (m^2) to the touchdown target."""
    theta = jnp.deg2rad(x[1])
    phi = jnp.deg2rad(x[2])

    theta_t = jnp.deg2rad(params.target.lon)
    phi_t = jnp.deg2rad(params.target.lat)

    r_veh = jnp.array([
        jnp.cos(theta) * jnp.cos(phi),
        jnp.sin(theta) * jnp.cos(phi),
        jnp.sin(phi),
    ])
    r_tgt = jnp.array([
        jnp.cos(theta_t) * jnp.cos(phi_t),
        jnp.sin(theta_t) * jnp.cos(phi_t),
        jnp.sin(phi_t),
    ])

    central = jnp.arctan2(
        jnp.linalg.norm(jnp.cross(r_veh, r_tgt)),
        jnp.dot(r_veh, r_tgt),
    )

    return jnp.array([(params.planet.r * central)**2])


def deploy_position_error_sq(x, u, t, params, fcns):
    """Squared local-horizontal position error (m^2) to params.deploy_target."""
    theta_d = jnp.deg2rad(params.deploy_target.lon)
    phi_d = jnp.deg2rad(params.deploy_target.lat)

    e_dr = params.planet.r * (jnp.deg2rad(x[1]) - theta_d) * jnp.cos(phi_d)
    e_cr = params.planet.r * (jnp.deg2rad(x[2]) - phi_d)

    return jnp.array([e_dr**2 + e_cr**2])
