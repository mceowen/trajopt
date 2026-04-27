import jax.numpy as jnp
from jax import Array


# Direction Cosine Matrix Function
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


# =============================================================================
# dynamics
# =============================================================================


def dynamics(t: float, z: Array, nu: Array, params: dict) -> Array:
    """6-DoF powered descent dynamics."""
    veh = params.vehicle

    x_dot = jnp.zeros(len(z))

    g = jnp.array([-params.planet.g, 0, 0])

    Jb = jnp.diag(jnp.array([veh.Jb11, veh.Jb22, veh.Jb33]))
    Jbinv = jnp.diag(jnp.array([veh.Jbinv11, veh.Jbinv22, veh.Jbinv33]))
    rt = jnp.array([veh["rt1"], veh["rt2"], veh["rt3"]])

    x_dot = x_dot.at[0].set(-veh["alpha"] * jnp.linalg.norm(nu))
    x_dot = x_dot.at[1:4].set(z[4:7])
    x_dot = x_dot.at[4:7].set((1 / z[0]) * DCM(z[7:11]).T @ nu[:3] + g)
    x_dot = x_dot.at[7:11].set((1 / 2) * omega(z[11:14]) @ z[7:11])
    x_dot = x_dot.at[11:14].set(Jbinv @ (cr(rt) @ nu[:3] - cr(z[11:14]) @ Jb @ z[11:14]))

    return x_dot


# =============================================================================
# nonconvex inequality constraint functions
# (used with type: nonconvex_inequality constraints)
# =============================================================================


def thrust(t: float, z: Array, nu: Array, params: dict) -> Array:
    """Thrust magnitude."""
    return jnp.array([jnp.linalg.norm(nu[:3])])


def glideslope(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Glideslope cone constraint: vehicle must stay above the glideslope angle."""
    r_i = z[0:3]
    theta_gs = params.theta_gs

    return jnp.array([jnp.tan(jnp.deg2rad(theta_gs)) * jnp.linalg.norm(r_i[1:3]) - r_i[0]])


def tilt(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Tilt angle constraint: limits the vehicle tilt from vertical."""
    theta_tilt = jnp.deg2rad(params.theta_tilt)
    q2 = z[8]
    q3 = z[9]

    return jnp.array([jnp.cos(theta_tilt) - 1.0 + 2 * (q2**2 + q3**2)])


def altitude(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Altitude state (x-position)."""
    return jnp.array([z[0]])


def speed(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Speed (inertial velocity magnitude)."""
    eps = 0.000001
    return jnp.array([jnp.sqrt(z[3] ** 2 + z[4] ** 2 + z[5] ** 2 + eps)])


# =============================================================================
# trajectory helper functions
# =============================================================================

def mass(t: float, z: Array, nu: Array, params: dict) -> Array:
    return jnp.array([z[0]])

def pos_x(t: float, z: Array, nu: Array, params: dict) -> Array:
    return jnp.array([z[1]])

def pos_y(t: float, z: Array, nu: Array, params: dict) -> Array:
    return jnp.array([z[2]])

def pos_z(t: float, z: Array, nu: Array, params: dict) -> Array:
    return jnp.array([z[3]])

def vel_x(t: float, z: Array, nu: Array, params: dict) -> Array:
    return jnp.array([z[4]])

def vel_y(t: float, z: Array, nu: Array, params: dict) -> Array:
    return jnp.array([z[5]])

def vel_z(t: float, z: Array, nu: Array, params: dict) -> Array:
    return jnp.array([z[6]])

def q0(t: float, z: Array, nu: Array, params: dict) -> Array:
    return jnp.array([z[7]])

def q1(t: float, z: Array, nu: Array, params: dict) -> Array:
    return jnp.array([z[8]])

def q2(t: float, z: Array, nu: Array, params: dict) -> Array:
    return jnp.array([z[9]])

def q3(t: float, z: Array, nu: Array, params: dict) -> Array:
    return jnp.array([z[10]])

def ang_vel_x(t: float, z: Array, nu: Array, params: dict) -> Array:
    return jnp.array([z[11]])

def ang_vel_y(t: float, z: Array, nu: Array, params: dict) -> Array:
    return jnp.array([z[12]])

def ang_vel_z(t: float, z: Array, nu: Array, params: dict) -> Array:
    return jnp.array([z[13]])

def thrust_x(t: float, z: Array, nu: Array, params: dict) -> Array:
    return jnp.array([nu[0]])

def thrust_y(t: float, z: Array, nu: Array, params: dict) -> Array:
    return jnp.array([nu[1]])

def thrust_z(t: float, z: Array, nu: Array, params: dict) -> Array:
    return jnp.array([nu[2]])

def thrust_mag(t: float, z: Array, nu: Array, params: dict) -> Array:
    return jnp.array([jnp.linalg.norm(nu[:3])])

def pos_vec(t: float, z: Array, nu: Array, params: dict) -> Array:
    return z[1:4]

def vel_vec(t: float, z: Array, nu: Array, params: dict) -> Array:
    return z[4:7]

def quat_vec(t: float, z: Array, nu: Array, params: dict) -> Array:
    return z[7:11]

def ang_vel_vec(t: float, z: Array, nu: Array, params: dict) -> Array:
    return z[11:14]

def traj_yz(t: float, z: Array, nu: Array, params: dict) -> Array:
    """Y-Z horizontal plane (no altitude)."""
    return z[2:4]

def traj_y_alt(t: float, z: Array, nu: Array, params: dict) -> Array:
    """Y downrange vs altitude: col0=Y, col1=altitude."""
    return jnp.array([z[2], z[1]])

def traj_3d(t: float, z: Array, nu: Array, params: dict) -> Array:
    """3D trajectory: col0=Y, col1=Z, col2=altitude (so altitude appears on z-axis)."""
    return jnp.array([z[2], z[3], z[1]])


# =============================================================================
# quiver direction functions  (return direction in plot-frame coordinates)
#
# Inertial frame:  x = altitude, y = Y, z = Z
# 3D plot columns: (Y, Z, altitude) = (inertial y, inertial z, inertial x)
# 2D plot columns: (Y, altitude)    = (inertial y, inertial x)
#
# Body axis k in inertial frame = DCM.T @ e_k  (k-th column of C_IB = DCM.T)
# =============================================================================

def _body_axis_3d(q: Array, k: int) -> Array:
    """k-th body axis in 3D plot coordinates (Y, Z, altitude)."""
    C_IB = DCM(q).T
    v_inertial = C_IB[:, k]           # [alt, Y, Z] in inertial
    return jnp.array([v_inertial[1], v_inertial[2], v_inertial[0]])

def body_x_dir_3d(t: float, z: Array, nu: Array, params: dict) -> Array:
    return _body_axis_3d(z[7:11], 0)

def body_y_dir_3d(t: float, z: Array, nu: Array, params: dict) -> Array:
    return _body_axis_3d(z[7:11], 1)

def body_z_dir_3d(t: float, z: Array, nu: Array, params: dict) -> Array:
    return _body_axis_3d(z[7:11], 2)

def thrust_dir_3d(t: float, z: Array, nu: Array, params: dict) -> Array:
    """Thrust unit vector in 3D plot coordinates (Y, Z, altitude)."""
    T_body = nu[:3]
    T_hat  = T_body / (jnp.linalg.norm(T_body) + 1e-8)
    v_inertial = DCM(z[7:11]).T @ T_hat   # [alt, Y, Z]
    return jnp.array([v_inertial[1], v_inertial[2], v_inertial[0]])

def body_x_dir_y_alt(t: float, z: Array, nu: Array, params: dict) -> Array:
    """Body x-axis in 2D Y-vs-altitude plot coordinates (Y, altitude)."""
    C_IB = DCM(z[7:11]).T
    v    = C_IB[:, 0]
    return jnp.array([v[1], v[0]])

def thrust_dir_y_alt(t: float, z: Array, nu: Array, params: dict) -> Array:
    """Thrust unit vector in 2D Y-vs-altitude plot coordinates (Y, altitude)."""
    T_hat      = nu[:3] / (jnp.linalg.norm(nu[:3]) + 1e-8)
    v_inertial = DCM(z[7:11]).T @ T_hat
    return jnp.array([v_inertial[1], v_inertial[0]])

def body_x_dir_yz(t: float, z: Array, nu: Array, params: dict) -> Array:
    """Body x-axis projected onto the Y-Z plot plane (Y, Z)."""
    C_IB = DCM(z[7:11]).T
    v    = C_IB[:, 0]              # [alt, Y, Z] in inertial
    return jnp.array([v[1], v[2]])

def thrust_dir_yz(t: float, z: Array, nu: Array, params: dict) -> Array:
    """Thrust unit vector projected onto the Y-Z plot plane (Y, Z)."""
    T_hat      = nu[:3] / (jnp.linalg.norm(nu[:3]) + 1e-8)
    v_inertial = DCM(z[7:11]).T @ T_hat   # [alt, Y, Z]
    return jnp.array([v_inertial[1], v_inertial[2]])
