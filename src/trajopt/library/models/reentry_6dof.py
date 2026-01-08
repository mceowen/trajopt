# copying over the lander 6dof and replacing the control inputs with deflection rates

import jax 
import jax.numpy as jnp
import cvxpy as cp

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

# =============================================================================
# dynamics
# =============================================================================

def dynamics(z, nu, params, fcns):
    """
    z[0] : mass [kg]
    z[1:4] : position [m]
    z[4:7] : velocity [m/s] (v_x, v_y, v_z)
    z[7:11] : quaternion [none] (q0, q1, q2, q3), scalar first
    z[11:14] : angular velocity [rad/s] (p, q, r)
    z[14:16] : control surface deflection rates (delta_a, delta_e)
    """

    veh = params['mission']["vehicle"]

    x_dot = jnp.zeros(params['model']['dimensions']['n'])

    g = jnp.array([-params['mission']["planet"]["g"], 0, 0])

    Jb = jnp.diag(jnp.array([veh["Jb11"], veh["Jb22"], veh["Jb33"]]))
    Jbinv = jnp.diag(jnp.array([veh["Jbinv11"], veh["Jbinv22"], veh["Jbinv33"]]))
    rt = jnp.array([veh["rt1"], veh["rt2"], veh["rt3"]])
    
    x_dot = x_dot.at[0].set(-params['mission']["vehicle"]["alpha"] * jnp.linalg.norm(nu))
    x_dot = x_dot.at[1:4].set(z[4:7])
    x_dot = x_dot.at[4:7].set((1/z[0]) * DCM(z[7:11]).T @ nu[:3] + g)
    x_dot = x_dot.at[7:11].set((1/2) * omega(z[11:14]) @ z[7:11])
    x_dot = x_dot.at[11:14].set(Jbinv @ (cr(rt) @ nu[:3] - cr(z[11:14]) @ Jb @ z[11:14]))

    return x_dot