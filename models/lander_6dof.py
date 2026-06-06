import cvxpy as cp
import jax.numpy as jnp
import numpy as np
from jax import Array

from . import rotations
from trajopt.utils.tools import AttrDict

# =============================================================================
# dynamics
# =============================================================================

def dynamics(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """6-DoF powered descent dynamics."""
    veh = params.vehicle

    x_dot = jnp.zeros(len(x))

    g = jnp.array([-params.planet.g, 0, 0])

    Jb    = jnp.diag(jnp.array([veh.Jb11, veh.Jb22, veh.Jb33]))
    Jbinv = jnp.diag(jnp.array([veh.Jbinv11, veh.Jbinv22, veh.Jbinv33]))
    rt    = jnp.array([veh.rt1, veh.rt2, veh.rt3])

    mass   = x[0]
    v      = x[4:7]
    q      = x[7:11]
    w      = x[11:14]
    thrust = u[:3]

    # mass dynamics
    x_dot = x_dot.at[0].set(-veh.alpha * jnp.linalg.norm(thrust))

    # translational kinematics
    x_dot = x_dot.at[1:4].set(v)

    # translational dynamics
    x_dot = x_dot.at[4:7].set((1 / mass) * rotations.DCM(q).T @ thrust + g)

    # quaternion kinematics
    x_dot = x_dot.at[7:11].set(0.5 * rotations.omega(w) @ q)

    # rotational dynamics
    x_dot = x_dot.at[11:14].set(Jbinv @ (rotations.cr(rt) @ thrust - rotations.cr(w) @ Jb @ w))

    return x_dot

def u_squared_cost(x, u, t, params, fcns):
    return jnp.atleast_1d(jnp.sum(u**2))

# =============================================================================
# nonconvex inequality constraint functions
# =============================================================================

def thrust(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """Thrust magnitude."""
    return jnp.array([jnp.linalg.norm(u[:3])])


def altitude(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """Altitude state (x-position)."""
    return jnp.array([x[1]])


def speed(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """Speed (inertial velocity magnitude)."""
    eps = 1e-6
    return jnp.array([jnp.sqrt(x[4] ** 2 + x[5] ** 2 + x[6] ** 2 + eps)])


def angular_speed(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """Angular speed (angular velocity magnitude, rad/s)."""
    eps = 1e-6
    return jnp.array([jnp.sqrt(x[11] ** 2 + x[12] ** 2 + x[13] ** 2 + eps)])


def glide_slope(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """Glide slope angle from vertical (degrees)."""
    alt = x[1] + 0.5
    # eps = 1e-6
    horiz = x[2] ** 2 + x[3] ** 2
    return jnp.array([jnp.tan(params.theta_gs)**2 * horiz - alt**2])


def tilt(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """Tilt angle from vertical (degrees) via quaternion."""
    q2 = x[9]
    q3 = x[10]
    cos_tilt = 1.0 - 2 * (q2**2 + q3**2)
    cos_theta_limit = jnp.cos(jnp.deg2rad(params.theta_tilt))
    return jnp.array([cos_theta_limit - cos_tilt])


def los(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """Line-of-sight angle: angle between body x-axis and vertical (degrees)."""
    body_x_inertial = rotations.DCM(x[7:11]).T[:, 0]
    cos_angle = jnp.clip(body_x_inertial[0], -1.0, 1.0)
    return jnp.array([jnp.rad2deg(jnp.arccos(cos_angle))])


# =============================================================================
# convex inequality constraint functions (CVXPY expressions)
# g(x, u, params) <= 0
# Vectorized: x is (N, n_state), u is (N, n_ctrl), returns (N,) expression
# =============================================================================

def cvx_gimbal_limit(x, u, params):
    """Gimbal angle limit: thrust vector within theta_gimbal of body x-axis."""
    cos_theta = np.cos(np.deg2rad(float(params.theta_gimbal)))
    return cos_theta * cp.norm(u[:, :3], axis=1) - u[:, 0]


def cvx_glide_slope(x, u, params):
    """Glide slope cone: horizontal distance bounded by altitude."""
    tan_theta = np.tan(np.deg2rad(float(params.theta_gs)))
    return tan_theta * cp.norm(x[:, 2:4], axis=1) - x[:, 1]


def cvx_max_thrust(x, u, params):
    """Maximum thrust magnitude."""
    return cp.norm(u[:, :3], axis=1) - float(params.max_thrust)


def cvx_max_angular_velocity(x, u, params):
    """Maximum angular velocity magnitude."""
    return cp.norm(x[:, 11:14], axis=1) - float(params.angular_speed_limit)


def cvx_tilt_limit(x, u, params):
    """Tilt angle limit via quaternion components q2, q3."""
    rhs = np.sqrt((1.0 - np.cos(np.deg2rad(float(params.theta_tilt)))) / 2.0)
    return cp.norm(x[:, 9:11], axis=1) - rhs


# =============================================================================
# trajectory helper functions
# =============================================================================

def mass(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    return jnp.array([x[0]])

def pos_vec(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    return x[1:4]

def vel_vec(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    return x[4:7]

def quat_vec(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    return x[7:11]

def ang_vel_vec(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    return x[11:14]

def thrust_x(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    return jnp.array([u[0]])

def thrust_y(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    return jnp.array([u[1]])

def thrust_z(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    return jnp.array([u[2]])

def thrust_mag(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    return jnp.array([jnp.linalg.norm(u[:3])])

def traj_yz(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """Y-Z horizontal plane (no altitude)."""
    return x[2:4]

def traj_y_alt(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """Y downrange vs altitude: col0=Y, col1=altitude."""
    return jnp.array([x[2], x[1]])

def traj_3d(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """3D trajectory: col0=Y, col1=Z, col2=altitude (so altitude appears on z-axis)."""
    return jnp.array([x[2], x[3], x[1]])


# =============================================================================
# quiver direction functions
# =============================================================================

def _body_axis_3d(q: Array, k: int) -> Array:
    """k-th body axis in 3D plot coordinates (Y, Z, altitude)."""
    C_IB = rotations.DCM(q).T
    v_inertial = C_IB[:, k]
    return jnp.array([v_inertial[1], v_inertial[2], v_inertial[0]])

def body_x_dir_3d(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    return _body_axis_3d(x[7:11], 0)

def body_x_dir_y_alt(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """Body x-axis in 2D Y-vs-altitude plot coordinates (Y, altitude)."""
    C_IB = rotations.DCM(x[7:11]).T
    v    = C_IB[:, 0]
    return jnp.array([v[1], v[0]])

def thrust_dir_y_alt(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """Thrust unit vector in 2D Y-vs-altitude plot coordinates (Y, altitude)."""
    T_hat      = u[:3] / (jnp.linalg.norm(u[:3]) + 1e-8)
    v_inertial = rotations.DCM(x[7:11]).T @ T_hat
    return jnp.array([v_inertial[1], v_inertial[0]])

def body_x_dir_yz(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """Body x-axis projected onto the Y-Z plot plane (Y, Z)."""
    C_IB = rotations.DCM(x[7:11]).T
    v    = C_IB[:, 0]
    return jnp.array([v[1], v[2]])

def thrust_dir_yz(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """Thrust unit vector projected onto the Y-Z plot plane (Y, Z)."""
    T_hat      = u[:3] / (jnp.linalg.norm(u[:3]) + 1e-8)
    v_inertial = rotations.DCM(x[7:11]).T @ T_hat
    return jnp.array([v_inertial[1], v_inertial[2]])

def thrust_dir_3d(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """Thrust unit vector in 3D plot coordinates (Y, Z, altitude)."""
    T_body = u[:3]
    T_hat  = T_body / (jnp.linalg.norm(T_body) + 1e-8)
    v_inertial = rotations.DCM(x[7:11]).T @ T_hat
    return jnp.array([v_inertial[1], v_inertial[2], v_inertial[0]])


# =============================================================================
# STL constraint trajectory helpers (for plotting)
# =============================================================================

def altitude_traj(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    return jnp.array([x[1]])

def speed_traj(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    return jnp.array([jnp.sqrt(x[4] ** 2 + x[5] ** 2 + x[6] ** 2 + 1e-6)])

def angular_speed_traj(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    return jnp.array([jnp.sqrt(x[11] ** 2 + x[12] ** 2 + x[13] ** 2 + 1e-6)])

def glide_slope_traj(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    alt = x[1] + 0.5
    eps = 1e-6
    horiz = jnp.sqrt(x[2] ** 2 + x[3] ** 2 + eps)
    return jnp.array([jnp.rad2deg(jnp.arctan2(horiz, alt))])

def tilt_traj(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    q2 = x[9]
    q3 = x[10]
    cos_tilt = 1.0 - 2 * (q2**2 + q3**2)
    return jnp.array([jnp.rad2deg(jnp.arccos(jnp.clip(cos_tilt, -1.0, 1.0)))])

def los_traj(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    body_x_inertial = rotations.DCM(x[7:11]).T[:, 0]
    cos_angle = jnp.clip(body_x_inertial[0], -1.0, 1.0)
    return jnp.array([jnp.rad2deg(jnp.arccos(cos_angle))])