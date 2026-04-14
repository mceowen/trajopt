import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_enable_x64", True)


def dynamics(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """6-DoF atmospheric reentry dynamics."""
    r = z[0:3]
    v_body = z[3:6]
    q = z[6:10]
    w = jnp.deg2rad(z[10:13])

    torque = nu[:3]

    veh = params["vehicle"]
    planet = params["planet"]
    mu = planet["mu"]
    Jbvec = jnp.array([veh["Jb11"], veh["Jb22"], veh["Jb33"]])
    Jb = jnp.diag(Jbvec)
    Jbinv = jnp.diag(1 / Jbvec)
    mass = veh["mass"]

    r_norm = jnp.linalg.norm(r)
    a_grav_inertial = -mu * r / r_norm**3

    # aero forces and moments
    aero = fcns["nonlinear_aero"](t, z, nu, params, fcns)
    a_aero_trans = (1 / mass) * aero["f_trans"]

    v_inertial = DCM(q).T @ v_body

    # rotational kinematics and dynamics (body sees control + aero moments)
    q_dot = (1/2) * omega(w) @ q
    w_dot = jnp.rad2deg(Jbinv @ (torque - cr(w) @ Jb @ w))

    # translational accelerations
    v_body_dot = a_aero_trans + DCM(q) @ a_grav_inertial - cr(w) @ v_body

    # state derivative
    return jnp.concatenate([v_inertial, v_body_dot, q_dot, w_dot])


def control_torques_dt(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Constraint: control torques must equal the required aero moments (== 0)."""
    aero = fcns["nonlinear_aero"](t, z, nu, params, fcns)

    return nu[:3] - aero["m_rot"]  # == 0


def heat_rate(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Convective heat rate (W/m²)."""
    v = jnp.linalg.norm(z[3:6])
    rho = fcns["density_model"](t, z, nu, params, fcns)

    return jnp.array([params["vehicle"]["kQ"] * rho**0.5 * v**3])


def dynamic_pressure(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Dynamic pressure q-bar (Pa)."""
    v = jnp.linalg.norm(z[3:6])
    rho = fcns["density_model"](t, z, nu, params, fcns)

    return jnp.array([0.5 * rho * v**2])


def aero_load(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Aerodynamic translational force magnitude (N)."""
    aero = fcns["nonlinear_aero"](t, z, nu, params, fcns)

    return jnp.array([jnp.linalg.norm(aero["f_trans"])])


def quaternion_norm(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Quaternion norm (should be 1 for a unit quaternion)."""
    return jnp.array([jnp.linalg.norm(z[6:10])])


def velocity(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Body-frame velocity (m/s)."""
    return jnp.array([jnp.linalg.norm(z[3:6])])


def aoa(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Angle of attack (deg)."""
    v = z[3:6]
    v_norm = jnp.maximum(jnp.linalg.norm(v), 1e-10)
    e_v = v / v_norm
    alpha = jnp.rad2deg(jnp.arctan2(e_v[2], e_v[0]))

    return jnp.array([alpha])


def sideslip(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Sideslip angle (deg)."""
    v = z[3:6]
    v_norm = jnp.linalg.norm(v)
    e_v = v / v_norm
    beta = jnp.rad2deg(jnp.arcsin(e_v[1]))

    return jnp.array([beta])


def altitude(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Altitude above planet surface (m)."""
    return jnp.array([(jnp.linalg.norm(z[0:3]) - params["planet"]["r"])])


def DCM(q: Array) -> Array:
    """Direction cosine matrix (inertial → body) from unit quaternion [q0, q1, q2, q3]."""
    return jnp.array(
        [
            [
                1 - 2 * (q[2] ** 2 + q[3] ** 2),
                2 * (q[1] * q[2] + q[0] * q[3]),
                2 * (q[1] * q[3] - q[0] * q[2]),
            ],
            [
                2 * (q[1] * q[2] - q[0] * q[3]),
                1 - 2 * (q[1] ** 2 + q[3] ** 2),
                2 * (q[2] * q[3] + q[0] * q[1]),
            ],
            [
                2 * (q[1] * q[3] + q[0] * q[2]),
                2 * (q[2] * q[3] - q[0] * q[1]),
                1 - 2 * (q[1] ** 2 + q[2] ** 2),
            ],
        ],
    )


def omega(w: Array) -> Array:
    """Skew-symmetric quaternion kinematic matrix for angular velocity w."""
    return jnp.array(
        [
            [0, -w[0], -w[1], -w[2]],
            [w[0], 0, w[2], -w[1]],
            [w[1], -w[2], 0, w[0]],
            [w[2], w[1], -w[0], 0],
        ],
    )


def cr(v: Array) -> Array:
    """Skew-symmetric cross-product matrix for vector v."""
    return jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def R_vec_theta(n: Array, theta_deg: float) -> Array:
    """Rotation matrix for a rotation of theta_deg degrees about unit vector n."""
    theta_rad = jnp.deg2rad(theta_deg)
    n_hat = n / jnp.linalg.norm(n)

    return (
        jnp.outer(n_hat, n_hat)
        + jnp.cos(theta_rad) * (jnp.eye(3) - jnp.outer(n_hat, n_hat))
        + jnp.sin(theta_rad) * cr(n_hat)
    )


def quat_from_dcm(C: Array) -> Array:
    """Quaternion [q0, q1, q2, q3] from a direction cosine matrix C."""
    q_0 = 0.5 * jnp.sqrt(C[0, 0] + C[1, 1] + C[2, 2] + 1)
    q_1 = (C[1, 2] - C[2, 1]) / (4 * q_0)
    q_2 = (C[2, 0] - C[0, 2]) / (4 * q_0)
    q_3 = (C[0, 1] - C[1, 0]) / (4 * q_0)

    return jnp.array([q_0, q_1, q_2, q_3])


def bank_aoa_to_quat(v_vec: Array, r_vec: Array, sigma_deg: float, alpha_deg: float) -> Array:
    """Convert bank angle and angle of attack to a body-frame quaternion."""
    # always target zero sideslip angle
    beta_deg = 0.0
    r_hat = r_vec / jnp.linalg.norm(r_vec)

    # unit vectors for velocity, right, down (velocity frame)
    v_hat = v_vec / jnp.linalg.norm(v_vec)
    v_right_hat = jnp.cross(v_hat, r_hat) / jnp.linalg.norm(jnp.cross(v_hat, r_hat))
    v_down_hat = jnp.cross(v_hat, v_right_hat) / jnp.linalg.norm(jnp.cross(v_hat, v_right_hat))

    # initialize B frame to be aligned with velocity frame
    B_frame = jnp.column_stack((v_hat, v_right_hat, v_down_hat))

    # apply rotation for (beta, alpha, sigma) to get body frame from velocity frame
    B_frame = R_vec_theta(v_down_hat, -beta_deg) @ B_frame
    B_frame = R_vec_theta(B_frame[:, 1], alpha_deg) @ B_frame
    B_frame = R_vec_theta(v_hat, sigma_deg) @ B_frame

    # quaternion from DCM
    return quat_from_dcm(B_frame.T)


def quat_to_bank_aoa(q: Array, v_vec: Array, r_vec: Array) -> tuple[Array, Array, Array]:
    """Extract bank angle, AoA, and sideslip from a quaternion."""
    DCM_I2B = DCM(q)

    # alpha and beta from body-frame velocity direction
    v_body = DCM_I2B @ v_vec
    v_body_hat = v_body / jnp.linalg.norm(v_body)
    alpha_deg = jnp.rad2deg(jnp.arctan2(v_body_hat[2], v_body_hat[0]))
    beta_deg = jnp.rad2deg(jnp.arcsin(v_body_hat[1]))

    # velocity frame (same construction as bank_aoa_to_quat)
    r_hat = r_vec / jnp.linalg.norm(r_vec)
    v_hat = v_vec / jnp.linalg.norm(v_vec)
    v_right_hat = jnp.cross(v_hat, r_hat) / jnp.linalg.norm(jnp.cross(v_hat, r_hat))
    v_down_hat = jnp.cross(v_hat, v_right_hat) / jnp.linalg.norm(jnp.cross(v_hat, v_right_hat))

    # zero-bank body frame: apply only beta and alpha rotations to velocity frame
    B0 = jnp.column_stack((v_hat, v_right_hat, v_down_hat))
    B0 = R_vec_theta(v_down_hat, -beta_deg) @ B0
    B0 = R_vec_theta(B0[:, 1], alpha_deg) @ B0

    # actual body frame columns (inertial coordinates)
    B_actual = DCM_I2B.T

    # bank angle = rotation from B0 to B_actual about v_hat
    z0 = B0[:, 2]
    z1 = B_actual[:, 2]
    z0_perp = z0 - jnp.dot(z0, v_hat) * v_hat
    z1_perp = z1 - jnp.dot(z1, v_hat) * v_hat
    z0_perp = z0_perp / jnp.linalg.norm(z0_perp)
    z1_perp = z1_perp / jnp.linalg.norm(z1_perp)

    cos_sigma = jnp.dot(z0_perp, z1_perp)
    sin_sigma = jnp.dot(jnp.cross(z0_perp, z1_perp), v_hat)
    sigma_deg = jnp.rad2deg(jnp.arctan2(sin_sigma, cos_sigma))

    return sigma_deg, alpha_deg, beta_deg


def quaternion_error(q_des: Array, q: Array) -> Array:
    """Error quaternion between desired and current quaternions: q_conj * q_des."""
    q_conj = jnp.array([q[0], -q[1], -q[2], -q[3]])
    return quat_mult(q_conj, q_des)


def quat_mult(q1: Array, q2: Array) -> Array:
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return jnp.array([w, x, y, z])


def long_lat(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Longitude and latitude from Cartesian position [theta, phi] (deg, deg)."""
    theta = jnp.rad2deg(jnp.atan2(z[1], z[0]))
    phi = jnp.rad2deg(jnp.atan2(z[2], jnp.sqrt(z[0] ** 2 + z[1] ** 2)))

    return jnp.array([theta, phi])


def long_lat_alt(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Longitude, latitude, and altitude from Cartesian position [theta, phi, h] (deg, deg, m)."""
    r = jnp.linalg.norm(z[0:3])

    theta = jnp.rad2deg(jnp.atan2(z[1], z[0]))
    phi = jnp.rad2deg(jnp.atan2(z[2], jnp.sqrt(z[0] ** 2 + z[1] ** 2)))
    alt = r - params["planet"]["r"]

    return jnp.array([theta, phi, alt])


def r_v(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Radial distance and velocity [r, v] (m, m/s)."""
    r = jnp.linalg.norm(z[0:3])
    v = jnp.linalg.norm(z[3:6])

    return jnp.array([r, v])


def polar_radius(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Polar radius / distance from planet center (m)."""
    return jnp.array([jnp.linalg.norm(z[0:3])])


def polar_longitude(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Polar longitude (deg)."""
    return jnp.array([jnp.rad2deg(jnp.arctan2(z[1], z[0]))])


def polar_latitude(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Polar latitude (deg)."""
    return jnp.array([jnp.rad2deg(jnp.arctan2(z[2], jnp.sqrt(z[0] ** 2 + z[1] ** 2)))])


def polar_velocity(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Polar velocity magnitude (m/s)."""
    return jnp.array([jnp.linalg.norm(z[3:6])])


def polar_fpa(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Polar flight-path angle (deg), derived from 6-DoF Cartesian state."""
    r = z[0:3]
    v_body = z[3:6]
    q = z[6:10]

    v_inertial = DCM(q).T @ v_body
    r_hat = r / jnp.linalg.norm(r)
    v_mag = jnp.maximum(jnp.linalg.norm(v_inertial), 1e-10)
    gamma = jnp.rad2deg(jnp.arcsin(jnp.dot(v_inertial, r_hat) / v_mag))

    return jnp.array([gamma])


def polar_heading(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Polar heading angle (deg), measured from north (increasing latitude)."""
    r = z[0:3]
    v_body = z[3:6]
    q = z[6:10]

    v_inertial = DCM(q).T @ v_body
    r_hat = r / jnp.linalg.norm(r)

    v_horiz = v_inertial - jnp.dot(v_inertial, r_hat) * r_hat

    theta = jnp.arctan2(r[1], r[0])
    phi = jnp.arctan2(r[2], jnp.sqrt(r[0] ** 2 + r[1] ** 2))
    e_north = jnp.array(
        [-jnp.sin(phi) * jnp.cos(theta), -jnp.sin(phi) * jnp.sin(theta), jnp.cos(phi)]
    )
    e_east = jnp.array([-jnp.sin(theta), jnp.cos(theta), 0.0])

    psi = jnp.rad2deg(jnp.arctan2(jnp.dot(v_horiz, e_east), jnp.dot(v_horiz, e_north)))

    return jnp.array([psi])
