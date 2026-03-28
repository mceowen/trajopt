import numpy as np
import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
    
def dynamics(t, z, nu, params, fcns):
    # extract states
    r      = z[0:3]
    v_body = z[3:6]
    q      = z[6:10]
    w      = jnp.deg2rad(z[10:13])

    # extract controls
    torque = nu[:3]

    # extract parameters
    veh    = params['vehicle']
    planet = params['planet']
    mu     = planet['mu']
    Jbvec  = jnp.array([veh["Jb11"], veh["Jb22"], veh["Jb33"]])
    Jb     = jnp.diag(Jbvec)
    Jbinv  = jnp.diag(1 / Jbvec)
    mass   = veh["mass"]
    
    r_norm = jnp.linalg.norm(r)
    a_grav_inertial = -mu * r / r_norm ** 3

    # aero forces and moments
    aero = fcns['nonlinear_aero'](t, z, nu, params, fcns)
    a_aero_trans = (1 / mass) * aero["f_trans"]

    m_rot = aero["m_rot"]

    v_inertial = DCM(q).T @ v_body

    # rotational kinematics and dynamics (body sees control + aero moments)
    q_dot = (1/2) * omega(w) @ q
    w_dot = jnp.rad2deg(Jbinv @ ( torque - cr(w) @ Jb @ w))

    # translational accelerations
    v_body_dot = a_aero_trans + DCM(q) @ a_grav_inertial - cr(w) @ v_body

    # state derivative
    x_dot = jnp.concatenate([v_inertial, v_body_dot, q_dot, w_dot])

    return x_dot

def control_torques_dt(t, z, nu, params, fcns):

    aero = fcns["nonlinear_aero"](t, z, nu, params, fcns)
    m_rot = aero["m_rot"]

    moment = nu[:3]

    return moment - m_rot # == 0

def control_torques_ct(t, z, nu, params, fcns):

    aero = fcns["nonlinear_aero"](t, z, nu, params, fcns)
    m_rot = aero["m_rot"]

    moment = nu[:3]
    difference = moment - m_rot

    ub = jnp.array([800.0, 5000.0, 5000.0])
    lb = jnp.array([-800.0, -5000.0, -5000.0])

    # ub = jnp.array([0, 0.0, 0.0])
    # lb = jnp.array([-0.0, -0.0, -0.0])

    top    = difference - ub
    bottom = lb - difference

    g_x = jnp.concatenate([top, bottom])

    # alpha = 50.0
    # # numerically stable log-sum-exp
    # m = jnp.max(g_x)
    # lse = m + (1.0 / alpha) * jnp.log(jnp.sum(jnp.exp(alpha * (g_x - m))))

    return jnp.max(g_x).reshape(1,)


def heat_rate(t, z, nu, params, fcns): # heat rate

    r = jnp.linalg.norm(z[0:3])
    v = jnp.linalg.norm(z[3:6])

    rho = fcns['density_model'](t, z, nu, params, fcns)

    return jnp.array([params['vehicle']['kQ'] * rho ** 0.5 * v ** 3])

def dynamic_pressure(t, z, nu, params, fcns):  #dynamic pressure
    
    r = jnp.linalg.norm(z[0:3])
    v = jnp.linalg.norm(z[3:6])

    rho = fcns['density_model'](t, z, nu, params, fcns)

    return jnp.array([0.5 * rho * v ** 2])

def aero_load(t, z, nu, params, fcns): # normal load

    aero = fcns['nonlinear_aero'](t, z, nu, params, fcns)

    return jnp.linalg.norm(aero["f_trans"])

def quaternion_norm(t, z, nu, params, fcns):
    return jnp.array([jnp.linalg.norm(z[6:10])])

def velocity(t, z, nu, params, fcns):
    return jnp.array([jnp.linalg.norm(z[3:6])])

def aoa(t, z, nu, params, fcns):
    # Extract states and controls
    v = z[3:6]
    v_norm = jnp.linalg.norm(v)
    
    # Prevent division by zero
    v_norm = jnp.maximum(v_norm, 1e-10)

    q = z[6:10]
    q = q

    # get AoA and sideslip angles
    DCM_q = DCM(q)
    e_v = v / v_norm

    e_v_body = e_v

    alpha = jnp.rad2deg(jnp.arctan2(e_v_body[2], e_v_body[0]))

    return jnp.array([alpha])

def sideslip(t, z, nu, params, fcns):
    # Extract states and controls
    v = z[3:6]
    v_norm = jnp.linalg.norm(v)

    q = z[6:10]

    # get AoA and sideslip angles
    e_v = v / v_norm

    e_v_body = e_v

    beta = jnp.rad2deg(jnp.arcsin(e_v_body[1]))

    return jnp.array([beta])


def altitude(t, z, nu, params, fcns):
    return jnp.array([(jnp.linalg.norm(z[0:3]) - params['planet']['r'])])

# Direction Cosine Matrix Function (converts vectors from intertial to body frame)
def DCM(q): 
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
        ]
    )

# skew symmetric quaternion matrix
def omega(w):
    return jnp.array(
    [
        [0, -w[0], -w[1], -w[2]],
        [w[0], 0, w[2], -w[1]],
        [w[1], -w[2], 0, w[0]],
        [w[2], w[1], -w[0], 0],
    ]
)

# skew symmetric cross product matrix function
def cr(v):
    return jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def R_vec_theta(n, theta_deg):

    theta_rad = jnp.deg2rad(theta_deg)
    n_hat = n / jnp.linalg.norm(n)

    # rotation matrix for rotation about vec_hat by theta
    R = jnp.outer(n_hat, n_hat) + jnp.cos(theta_rad) * (jnp.eye(3) - jnp.outer(n_hat, n_hat)) + jnp.sin(theta_rad) * cr(n_hat)

    return R

def quat_from_dcm(C):

    q_0 = 0.5 * jnp.sqrt(C[0, 0] + C[1, 1] + C[2, 2] + 1)
    q_1 = (C[1, 2] - C[2, 1]) / (4*q_0)
    q_2 = (C[2, 0] - C[0, 2]) / (4*q_0)
    q_3 = (C[0, 1] - C[1, 0]) / (4*q_0)

    return jnp.array([q_0, q_1, q_2, q_3])

# converts from polar coordinates (r, theta, phi, v, gamma, psi, sigma, alpha, beta) to Cartesian 6-DoF
def bank_aoa_to_quat(v_vec, r_vec, sigma_deg, alpha_deg):

    # define the inertial frame unit vectors (resolved in inertial frame)
    x_hat = np.array([1, 0, 0])
    y_hat = np.array([0, 1, 0])
    z_hat = np.array([0, 0, 1])

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

    # DCM_I2B 
    DCM_I2B = B_frame.T

    # quaternion from DCM
    q = quat_from_dcm(DCM_I2B)

    return q

def quat_to_bank_aoa(q, v_vec, r_vec):

    DCM_I2B = DCM(q)

    # alpha and beta from body-frame velocity direction
    v_body = DCM_I2B @ v_vec
    v_body_hat = v_body / jnp.linalg.norm(v_body)
    alpha_deg = jnp.rad2deg(jnp.arctan2(v_body_hat[2], v_body_hat[0]))
    beta_deg  = jnp.rad2deg(jnp.arcsin(v_body_hat[1]))

    # velocity frame (same construction as bank_aoa_to_quat)
    r_hat = r_vec / jnp.linalg.norm(r_vec)
    v_hat = v_vec / jnp.linalg.norm(v_vec)
    v_right_hat = jnp.cross(v_hat, r_hat) / jnp.linalg.norm(jnp.cross(v_hat, r_hat))
    v_down_hat  = jnp.cross(v_hat, v_right_hat) / jnp.linalg.norm(jnp.cross(v_hat, v_right_hat))

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

def quaternion_error(q_des, q):
    q_conj = jnp.array([q[0], -q[1], -q[2], -q[3]])
    q_error = quat_mult(q_conj, q_des)
    return q_error

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return jnp.array([w, x, y, z])

def long_lat(t, z, nu, params, fcns):

    r = jnp.linalg.norm(z[0:3])
    x = z[0]
    y = z[1]
    z = z[2]

    theta = jnp.rad2deg(jnp.atan2(y, x))
    phi = jnp.rad2deg(jnp.atan2(z, jnp.sqrt(x**2 + y**2)))
    
    return jnp.array([theta, phi])

def long_lat_alt(t, z, nu, params, fcns):

    r = jnp.linalg.norm(z[0:3])
    x = z[0]
    y = z[1]
    z = z[2]

    theta = jnp.rad2deg(jnp.atan2(y, x))
    phi = jnp.rad2deg(jnp.atan2(z, jnp.sqrt(x**2 + y**2)))

    alt = r - params['planet']['r']
    
    return jnp.array([theta, phi, alt])

def r_v(t, z, nu, params, fcns):
    r = jnp.linalg.norm(z[0:3])
    v = jnp.linalg.norm(z[3:6])

    return jnp.array([r, v])

