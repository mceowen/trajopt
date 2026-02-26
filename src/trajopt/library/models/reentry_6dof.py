import numpy as np
import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
    
def dynamics(t, z, nu, params, fcns):
    # extract states
    r = z[0:3]
    v_body = z[3:6]
    q = z[6:10]
    w = jnp.deg2rad(z[10:13]) 
    moment_coeffs = nu[:3]

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
    a_aero_trans_body = (1 / mass) * aero["f_trans"]
    
    # not using for now
    # m_aero_rot_body = aero["m_rot"]

    rho = fcns['density_model'](t, z, nu, params)
    v_inertial = DCM(q).T @ v_body
    v_norm = jnp.linalg.norm(v_body)

    sref = params["vehicle"]["sref"]
    lref = params["vehicle"]["lref"]

    q_dyn_press = 0.5 * rho * v_norm**2
    # moment = q_dyn_press * sref * lref * moment_coeffs
    moment = moment_coeffs

    # rotational kinematics and dynamics
    q_dot = (1/2) * omega(w) @ q
    w_dot = jnp.rad2deg(Jbinv @ ( moment - cr(w) @ Jb @ w))

    # translational accelerations
    v_body_dot = a_aero_trans_body + DCM(q) @ a_grav_inertial - cr(w) @ v_body

    # state derivative
    x_dot = jnp.concatenate([v_inertial, v_body_dot, q_dot, w_dot])

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

    e_v_body = e_v

    alpha = jnp.rad2deg(jnp.arctan2(e_v_body[2], e_v_body[0]))

    return jnp.array([alpha])

def sideslip(t, z, nu, params):
    # Extract states and controls
    v = z[3:6]
    v_norm = jnp.linalg.norm(v)

    q = z[6:10]

    # get AoA and sideslip angles
    e_v = v / v_norm

    e_v_body = e_v

    beta = jnp.rad2deg(jnp.arcsin(e_v_body[1]))

    return jnp.array([beta])

def control_torques(t, z, nu, params, fcns):

    aero = fcns["nonlinear_aero"](t, z, nu, params, fcns)
    m_rot = aero["m_rot"]

    moment_coeffs = nu[:3]
    v_body = z[3:6]

    v_norm = jnp.linalg.norm(v_body)

    sref = params["vehicle"]["sref"]
    lref = params["vehicle"]["lref"]

    rho = fcns['density_model'](t, z, nu, params)

    q_dyn_press = 0.5 * rho * v_norm**2
    moment = q_dyn_press * sref * lref * moment_coeffs

    return moment

def altitude(t, z, nu, params):
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