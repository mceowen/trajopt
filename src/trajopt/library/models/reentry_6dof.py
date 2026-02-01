import numpy as np
import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# Direction Cosine Matrix Function
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
    
def dynamics(t, z, nu, params, fcns):
    """
    6DOF reentry dynamics with PD attitude controller tracking reference quaternion.
    
    State z[0:16]:
      - z[0:3]   = position (m)
      - z[3:6]   = velocity (m/s)  
      - z[6:10]  = quaternion (scalar-first)
      - z[10:13] = angular velocity (deg/s)
      - z[13:16] = reference quaternion vector part (optimized by SCP)
    
    Control nu[0:3] = rate of reference quaternion vector
    """
    # extract states
    r = z[0:3]
    v = z[3:6]
    q = z[6:10]
    w = jnp.deg2rad(z[10:13]) 
    q_ref_vec = z[13:16]
    q_ref_vec_dot = nu[:3]

    veh = params['vehicle']
    planet = params['planet']
    mu = planet['mu']
    Jbvec = jnp.array([veh["Jb11"], veh["Jb22"], veh["Jb33"]])
    Jb = jnp.diag(Jbvec)
    Jbinv = jnp.diag(1 / Jbvec)
    mass = veh["mass"]

    # === SIMPLE PD CONTROLLER ===
    q_sign = jnp.sign(q[0] + 1e-10)  # +1 if q[0] >= 0, -1 if q[0] < 0
    q_vec = q_sign * q[1:4]
    q_err = q_vec - q_ref_vec
    
    wn = 5.0
    zeta = 2.0
    Kp = wn**2 * Jbvec
    Kd = 2.0 * zeta * wn * Jbvec
    
    moment = -Kp * q_err - Kd * w
    # ============================

    r_norm = jnp.linalg.norm(r)
    a_grav = -mu * r / r_norm ** 3

    # aero forces and moments
    aero = fcns['nonlinear_aero_jax'](t, z, nu, params)
    a_aero_trans = 1 / mass * DCM(q).T @ aero["f_trans"]
    m_aero_rot = aero["m_rot"]

    # rotational kinematics and dynamics
    q_dot = (1/2) * omega(w) @ q
    w_dot = jnp.rad2deg(Jbinv @ (m_aero_rot + moment - cr(w) @ Jb @ w))

    # state derivative
    x_dot = jnp.concatenate([v, a_aero_trans + a_grav, q_dot, w_dot, q_ref_vec_dot])

    return x_dot

def heat_rate(t, z, nu, params, fcns): # heat rate

    r = jnp.linalg.norm(z[0:3])
    v = jnp.linalg.norm(z[3:6])

    rho = fcns['atmosphere_model_jax'](t, z, nu, params)

    return jnp.array([params['vehicle']['kQ'] * rho ** 0.5 * v ** 3])

def dynamic_pressure(t, z, nu, params, fcns):  #dynamic pressure
    
    r = jnp.linalg.norm(z[0:3])
    v = jnp.linalg.norm(z[3:6])

    rho = fcns['atmosphere_model_jax'](t, z, nu, params)

    return jnp.array([0.5 * rho * v ** 2])

def aero_load(t, z, nu, params, fcns): # normal load

    aero = fcns['nonlinear_aero_jax'](t, z, nu, params)

    return jnp.linalg.norm(aero["f_trans"])

def dynamic_pressure_nonjax(t, z, nu, params, fcns):
    r = jnp.linalg.norm(z[0:3])
    v = jnp.linalg.norm(z[3:6])

    rho = fcns['atmosphere_model_nonjax'](t, z, nu, params)

    return 0.5 * rho * v ** 2

def heat_rate_nonjax(t, z, nu, params, fcns):
    r = jnp.linalg.norm(z[0:3])
    v = jnp.linalg.norm(z[3:6])

    rho = fcns['atmosphere_model_nonjax'](t, z, nu, params)

    return params['vehicle']['kQ'] * rho ** 0.5 * v ** 3

def aero_load_nonjax(t, z, nu, params, fcns):
    r = jnp.linalg.norm(z[0:3])
    v = jnp.linalg.norm(z[3:6])

    aero = fcns['nonlinear_aero_nonjax'](t, z, nu, params)

    return jnp.linalg.norm(aero["f_trans"])

def quaternion_norm(t, z, nu, params):
    return jnp.array([jnp.linalg.norm(z[6:10])])

def minimum_quaternion_norm(t, z, nu, params):
    return jnp.array([-jnp.linalg.norm(z[6:10])])

def minimum_velocity(t, z, nu, params):
    return jnp.array([- jnp.linalg.norm(z[3:6])])

def minimum_altitude(t, z, nu, params):
    return jnp.array([-(jnp.linalg.norm(z[0:3]) - params['planet']['r'])])

def aoa(t, z, nu, params):
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

    e_v_body = DCM_q @ e_v

    alpha = jnp.rad2deg(jnp.arctan2(e_v_body[2], e_v_body[0]))  # AoA: angle in x-z plane

    return jnp.array([-alpha, alpha])

def sideslip(t, z, nu, params):
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

    e_v_body = DCM_q @ e_v

    beta = jnp.rad2deg(jnp.arctan2(e_v_body[1], e_v_body[0]))

    return jnp.array([-beta, beta])