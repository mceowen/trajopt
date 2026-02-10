import numpy as np
import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
    
def dynamics(t, z, nu, params, fcns):
    """
    6DOF reentry dynamics with PD attitude controller tracking reference quaternion.
    
    State z[0:16]:
      - z[0:3]   = position (m)
      - z[3:6]   = velocity (m/s)  
      - z[6:10]  = quaternion (scalar-first)
      - z[10:13] = angular velocity (deg/s)
    
    Control nu[0:3] = reference quaternion vector
    """
    # extract states
    r = z[0:3]
    v = z[3:6]
    q = z[6:10]
    w = jnp.deg2rad(z[10:13]) 
    quat_vec_ref = nu[:3]


    # extract parameters
    veh    = params['vehicle']
    planet = params['planet']
    mu     = planet['mu']
    Jbvec  = jnp.array([veh["Jb11"], veh["Jb22"], veh["Jb33"]])
    Jb     = jnp.diag(Jbvec)
    Jbinv  = jnp.diag(1 / Jbvec)
    mass   = veh["mass"]

    # Dynamic pressure for gain scheduling (inverse scaling: high q -> lower gains)
    rho = fcns['density_model'](t, z, nu, params)
    v_norm = jnp.linalg.norm(v)
    q_dyn = 0.5 * rho * v_norm**2
    q_ref = veh.get("ctrl_q_ref", 1e4)       # Pa, reference dynamic pressure
    q_min = veh.get("ctrl_q_min", 1.0)       # Pa, lower bound to avoid huge gains
    scale_lo = veh.get("ctrl_scale_lo", 0.2)
    scale_hi = veh.get("ctrl_scale_hi", 5.0)
    q_dyn_safe = jnp.maximum(q_dyn, q_min)
    scale = jnp.clip(q_ref / q_dyn_safe, scale_lo, scale_hi)

    # === PD CONTROLLER (gain scheduled by dynamic pressure) ===
    q_sign = jnp.sign(q[0] + 1e-10)
    q_vec = q_sign * q[1:4]
    q_err = q_vec - quat_vec_ref

    wn = 5.0
    zeta = 1.0
    Kp = wn**2 * Jbvec * scale
    Kd = 2.0 * zeta * wn * Jbvec * scale

    moment = -Kp * q_err - Kd * w
    # ==========================================================
    
    r_norm = jnp.linalg.norm(r)
    a_grav = -mu * r / r_norm ** 3

    # aero forces and moments
    aero = fcns['nonlinear_aero'](t, z, nu, params, fcns)
    a_aero_trans = 1 / mass * DCM(q).T @ aero["f_trans"]
    m_aero_rot = aero["m_rot"]

    # rotational kinematics and dynamics
    q_dot = (1/2) * omega(w) @ q
    w_dot = jnp.rad2deg(Jbinv @ (m_aero_rot + moment - cr(w) @ Jb @ w))

    # state derivative
    x_dot = jnp.concatenate([v, a_aero_trans + a_grav, q_dot, w_dot])

    return x_dot

def heat_rate(t, z, nu, params, fcns): # heat rate

    r = jnp.linalg.norm(z[0:3])
    v = jnp.linalg.norm(z[3:6])

    rho = fcns['density_model'](t, z, nu, params)

    return jnp.array([params['vehicle']['kQ'] * rho ** 0.5 * v ** 3])

def dynamic_pressure(t, z, nu, params, fcns):  #dynamic pressure
    
    r = jnp.linalg.norm(z[0:3])
    v = jnp.linalg.norm(z[3:6])

    rho = fcns['density_model'](t, z, nu, params)

    return jnp.array([0.5 * rho * v ** 2])

def aero_load(t, z, nu, params, fcns): # normal load

    aero = fcns['nonlinear_aero'](t, z, nu, params, fcns)

    return jnp.linalg.norm(aero["f_trans"])

def quaternion_norm(t, z, nu, params):
    return jnp.array([jnp.linalg.norm(z[6:10])])

def velocity(t, z, nu, params):
    return jnp.array([jnp.linalg.norm(z[3:6])])

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

    return jnp.array([alpha])

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

    return jnp.array([beta])

def altitude(t, z, nu, params):
    return jnp.array([(jnp.linalg.norm(z[0:3]) - params['planet']['r'])])

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