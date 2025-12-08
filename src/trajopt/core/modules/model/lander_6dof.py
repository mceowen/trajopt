import jax 
import jax.numpy as jnp
import cvxpy as cp

# =============================================================================
# helper functions
# =============================================================================

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

def dynamics_jax(t, z, nu, problem):
        
        mission = problem.mission
        model = problem.model

        veh = mission.vehicle

        x_dot = jnp.zeros(model.n)

        g = jnp.array([-mission.planet["g"], 0, 0])

        Jb = jnp.diag(jnp.array([veh["Jb11"], veh["Jb22"], veh["Jb33"]]))
        Jbinv = jnp.diag(jnp.array([veh["Jbinv11"], veh["Jbinv22"], veh["Jbinv33"]]))
        rt = jnp.array([veh["rt1"], veh["rt2"], veh["rt3"]])
        
        x_dot = x_dot.at[0].set(-mission.vehicle["alpha"] * jnp.linalg.norm(nu))
        x_dot = x_dot.at[1:4].set(z[4:7])
        x_dot = x_dot.at[4:7].set((1/z[0]) * DCM(z[7:11]).T @ nu[:3] + g)
        x_dot = x_dot.at[7:11].set((1/2) * omega(z[11:14]) @ z[7:11])
        x_dot = x_dot.at[11:14].set(Jbinv @ (cr(rt) @ nu[:3] - cr(z[11:14]) @ Jb @ z[11:14]))

        return x_dot

# =============================================================================
# nonconvex inequality constraints
# =============================================================================

def tilt_cone_jax(t, z, nu, problem):

    mission = problem.mission

    return jnp.array([mission.path_limits["cos_theta_max"] + 2 * jnp.sum(z[9:11] ** 2) - 1.0])

def glideslope_jax(t, z, nu, problem):

    mission = problem.mission

    return jnp.array([mission.path_limits["tan_gamma_gs"] * jnp.linalg.norm(z[2:4] + 0.000001 * jnp.ones(2)) - z[1]])

def angular_velocity_jax(t, z, nu, problem):

    mission = problem.mission

    return jnp.array([jnp.linalg.norm(z[11:14]) - mission.path_limits["w_max"]])


# STL CONSTRAINTS

def height_triggered_pitch(t, z, nu, problem):
    mission = problem.mission

    # f > 0 ==> g >= 0

    f = 2.0 - z[1]
    g = 1.0 - (mission.path_limits["cos_theta_max"] + 2 * jnp.sum(z[9:11] ** 2))

    return jnp.array([jnp.maximum(f, 0.0) * jnp.maximum(-g, 0.0)])

# =============================================================================
# convex constraints
# =============================================================================

def gimbal_cone(t, z, nu, problem):
    mission = problem.mission
     
    return mission.convex_limits["cos_delta_max"] * cp.norm(nu) <= nu[0]

def max_norm_thrust(t, z, nu, problem):
    mission = problem.mission

    return cp.norm(nu[:3]) <= mission.convex_limits["max_thrust"]

def tilt_cone(t, z, nu, problem):
    mission = problem.mission

    return mission.convex_limits["cos_theta_max"] + 2 * cp.sum_squares(z[9:11]) <= 1.0

def glideslope(t, z, nu, problem):
    mission = problem.mission

    return mission.convex_limits["tan_gamma_gs"] * cp.norm(z[2:4]) <= z[1]

def angular_velocity(t, z, nu, problem):
    mission = problem.mission

    return cp.norm(z[11:14]) <= mission.convex_limits["w_max"]