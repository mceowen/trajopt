import numpy as np
import jax.numpy as jnp
from jax import Array
import cvxpy as cp
import trajopt.models.rotations as rotations

# =============================================================================
# dynamics
# =============================================================================

def dynamics(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """6-DoF powered descent dynamics."""
    veh = params.vehicle

    x_dot = jnp.zeros(len(z))

    g = jnp.array([-params.planet.g, 0, 0])

    Jb    = jnp.diag(jnp.array([veh.Jb11, veh.Jb22, veh.Jb33]))
    Jbinv = jnp.diag(jnp.array([veh.Jbinv11, veh.Jbinv22, veh.Jbinv33]))
    rt    = jnp.array([veh.rt1, veh.rt2, veh.rt3])

    mass   = z[0]
    v      = z[4:7]
    q      = z[7:11]
    w      = z[11:14]
    thrust = nu[:3]

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

# =============================================================================
# nonconvex inequality constraint functions
# =============================================================================

def thrust(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Thrust magnitude."""
    return jnp.array([jnp.linalg.norm(nu[:3])])


def altitude(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Altitude state (x-position)."""
    return jnp.array([z[1]])


def speed(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Speed (inertial velocity magnitude)."""
    eps = 1e-6
    return jnp.array([jnp.sqrt(z[4] ** 2 + z[5] ** 2 + z[6] ** 2 + eps)])


def angular_speed(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Angular speed (angular velocity magnitude, rad/s)."""
    eps = 1e-6
    return jnp.array([jnp.sqrt(z[11] ** 2 + z[12] ** 2 + z[13] ** 2 + eps)])


def glide_slope(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Glide slope angle from vertical (degrees)."""
    alt = z[1] + 0.5
    # eps = 1e-6
    horiz = z[2] ** 2 + z[3] ** 2
    return jnp.array([jnp.tan(params.theta_gs)**2 * horiz - alt**2])


def tilt(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Tilt angle from vertical (degrees) via quaternion."""
    q2 = z[9]
    q3 = z[10]
    cos_tilt = 1.0 - 2 * (q2**2 + q3**2)
    cos_theta_limit = jnp.cos(jnp.deg2rad(params.theta_tilt))
    return jnp.array([cos_theta_limit - cos_tilt])


def los(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Line-of-sight angle: angle between body x-axis and vertical (degrees)."""
    body_x_inertial = rotations.DCM(z[7:11]).T[:, 0]
    cos_angle = jnp.clip(body_x_inertial[0], -1.0, 1.0)
    return jnp.array([jnp.rad2deg(jnp.arccos(cos_angle))])


# =============================================================================
# convex inequality constraint functions (CVXPY expressions)
# g(t, x, u, params) <= 0
# =============================================================================

def cvx_gimbal_limit(t, x, u, params):
    """Gimbal angle limit: thrust vector within theta_gimbal of body x-axis."""
    cos_theta = np.cos(np.deg2rad(float(params.theta_gimbal)))
    return cos_theta * cp.norm(u[:3]) - u[0]


def cvx_glide_slope(t, x, u, params):
    """Glide slope cone: horizontal distance bounded by altitude."""
    tan_theta = np.tan(np.deg2rad(float(params.theta_gs)))
    return tan_theta * cp.norm(x[2:4]) - x[1]


def cvx_max_thrust(t, x, u, params):
    """Maximum thrust magnitude."""
    return cp.norm(u[:3]) - float(params.max_thrust)


def cvx_max_angular_velocity(t, x, u, params):
    """Maximum angular velocity magnitude."""
    return cp.norm(x[11:14]) - float(params.angular_speed_limit)


def cvx_tilt_limit(t, x, u, params):
    """Tilt angle limit via quaternion components q2, q3."""
    rhs = np.sqrt((1.0 - np.cos(np.deg2rad(float(params.theta_tilt)))) / 2.0)
    return cp.norm(x[9:11]) - rhs


# =============================================================================
# trajectory helper functions
# =============================================================================

def mass(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([z[0]])

def pos_vec(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return z[1:4]

def vel_vec(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return z[4:7]

def quat_vec(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return z[7:11]

def ang_vel_vec(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return z[11:14]

def thrust_x(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([nu[0]])

def thrust_y(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([nu[1]])

def thrust_z(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([nu[2]])

def thrust_mag(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([jnp.linalg.norm(nu[:3])])

def traj_yz(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Y-Z horizontal plane (no altitude)."""
    return z[2:4]

def traj_y_alt(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Y downrange vs altitude: col0=Y, col1=altitude."""
    return jnp.array([z[2], z[1]])

def traj_3d(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """3D trajectory: col0=Y, col1=Z, col2=altitude (so altitude appears on z-axis)."""
    return jnp.array([z[2], z[3], z[1]])


# =============================================================================
# quiver direction functions
# =============================================================================

def _body_axis_3d(q: Array, k: int) -> Array:
    """k-th body axis in 3D plot coordinates (Y, Z, altitude)."""
    C_IB = rotations.DCM(q).T
    v_inertial = C_IB[:, k]
    return jnp.array([v_inertial[1], v_inertial[2], v_inertial[0]])

def body_x_dir_3d(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return _body_axis_3d(z[7:11], 0)

def body_x_dir_y_alt(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Body x-axis in 2D Y-vs-altitude plot coordinates (Y, altitude)."""
    C_IB = rotations.DCM(z[7:11]).T
    v    = C_IB[:, 0]
    return jnp.array([v[1], v[0]])

def thrust_dir_y_alt(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Thrust unit vector in 2D Y-vs-altitude plot coordinates (Y, altitude)."""
    T_hat      = nu[:3] / (jnp.linalg.norm(nu[:3]) + 1e-8)
    v_inertial = rotations.DCM(z[7:11]).T @ T_hat
    return jnp.array([v_inertial[1], v_inertial[0]])

def body_x_dir_yz(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Body x-axis projected onto the Y-Z plot plane (Y, Z)."""
    C_IB = rotations.DCM(z[7:11]).T
    v    = C_IB[:, 0]
    return jnp.array([v[1], v[2]])

def thrust_dir_yz(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Thrust unit vector projected onto the Y-Z plot plane (Y, Z)."""
    T_hat      = nu[:3] / (jnp.linalg.norm(nu[:3]) + 1e-8)
    v_inertial = rotations.DCM(z[7:11]).T @ T_hat
    return jnp.array([v_inertial[1], v_inertial[2]])

def thrust_dir_3d(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Thrust unit vector in 3D plot coordinates (Y, Z, altitude)."""
    T_body = nu[:3]
    T_hat  = T_body / (jnp.linalg.norm(T_body) + 1e-8)
    v_inertial = rotations.DCM(z[7:11]).T @ T_hat
    return jnp.array([v_inertial[1], v_inertial[2], v_inertial[0]])


# =============================================================================
# STL constraint trajectory helpers (for plotting)
# =============================================================================

def altitude_traj(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([z[1]])

def speed_traj(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([jnp.sqrt(z[4] ** 2 + z[5] ** 2 + z[6] ** 2 + 1e-6)])

def angular_speed_traj(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([jnp.sqrt(z[11] ** 2 + z[12] ** 2 + z[13] ** 2 + 1e-6)])

def glide_slope_traj(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    alt = z[1] + 0.5
    eps = 1e-6
    horiz = jnp.sqrt(z[2] ** 2 + z[3] ** 2 + eps)
    return jnp.array([jnp.rad2deg(jnp.arctan2(horiz, alt))])

def tilt_traj(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    q2 = z[9]
    q3 = z[10]
    cos_tilt = 1.0 - 2 * (q2**2 + q3**2)
    return jnp.array([jnp.rad2deg(jnp.arccos(jnp.clip(cos_tilt, -1.0, 1.0)))])

def los_traj(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    body_x_inertial = rotations.DCM(z[7:11]).T[:, 0]
    cos_angle = jnp.clip(body_x_inertial[0], -1.0, 1.0)
    return jnp.array([jnp.rad2deg(jnp.arccos(cos_angle))])


# =============================================================================
# spatial overlay functions
# =============================================================================

def glideslope_cone(params, ax) -> np.ndarray:
    """Glideslope cone boundary for Y-vs-altitude plot: two lines from apex."""
    tan_gs = np.tan(np.deg2rad(float(params.theta_gs)))
    offset = float(params.alt_trigger)
    alts   = np.array([0.0, 1e4])
    horiz  = (alts + offset) * tan_gs
    nan    = np.full(1, np.nan)
    xs = np.concatenate([ horiz, nan, -horiz])
    ys = np.concatenate([ alts,  nan,  alts])
    return np.column_stack([xs, ys])

def glideslope_cone_3d(params, ax) -> np.ndarray:
    """Glideslope cone surface for 3D trajectory plot (Y, Z, altitude axes)."""
    tan_gs = np.tan(np.deg2rad(float(params.theta_gs)))
    offset = float(params.alt_trigger)
    nan    = np.full((1, 3), np.nan)

    theta  = np.linspace(0, 2 * np.pi, 80)
    r0     = offset * tan_gs
    circle = np.column_stack([r0 * np.cos(theta), r0 * np.sin(theta), np.zeros(80)])

    r_top   = (1e4 + offset) * tan_gs
    n_lines = 12
    phi     = np.linspace(0, 2 * np.pi, n_lines, endpoint=False)
    segs    = []
    apex    = np.array([[0.0, 0.0, -offset]])
    for p in phi:
        tip = np.array([[r_top * np.cos(p), r_top * np.sin(p), 1e4]])
        segs.append(np.concatenate([apex, tip, nan]))

    return np.concatenate([circle, nan] + segs, axis=0)
