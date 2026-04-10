import jax.numpy as jnp

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

def dynamics(t, z, nu, params):

        veh = params["vehicle"]

        x_dot = jnp.zeros(len(z))

        g = jnp.array([-params["planet"]["g"], 0, 0])

        Jb = jnp.diag(jnp.array([veh["Jb11"], veh["Jb22"], veh["Jb33"]]))
        Jbinv = jnp.diag(jnp.array([veh["Jbinv11"], veh["Jbinv22"], veh["Jbinv33"]]))
        rt = jnp.array([veh["rt1"], veh["rt2"], veh["rt3"]])
        
        x_dot = x_dot.at[0].set(-veh["alpha"] * jnp.linalg.norm(nu))
        x_dot = x_dot.at[1:4].set(z[4:7])
        x_dot = x_dot.at[4:7].set((1/z[0]) * DCM(z[7:11]).T @ nu[:3] + g)
        x_dot = x_dot.at[7:11].set((1/2) * omega(z[11:14]) @ z[7:11])
        x_dot = x_dot.at[11:14].set(Jbinv @ (cr(rt) @ nu[:3] - cr(z[11:14]) @ Jb @ z[11:14]))

        return x_dot

# =============================================================================
# nonconvex inequality constraint functions
# (used with type: nonconvex_inequality constraints)
# =============================================================================

def thrust(t, z, nu, params):
    return jnp.array([jnp.linalg.norm(nu[:3])])

def glideslope(t, z, nu, params, fcns):
    r_i = z[0:3]

    theta_gs = params["theta_gs"]

    return jnp.array([jnp.tan(jnp.deg2rad(theta_gs))*jnp.linalg.norm(r_i[1:3]) - r_i[0]])
     
def tilt(t, z, nu, params, fcns):
     
     theta_tilt = jnp.deg2rad(params["theta_tilt"])

     q2 = z[8]
     q3 = z[9]

     return jnp.array([jnp.cos(theta_tilt) - 1.0 + 2*(q2**2 + q3**2)])
     
# def los(t, z, nu, params, fcns):
     
def altitude(t, z, nu, params, fcns):
    return jnp.array([z[0]])

def speed(t, z, nu, params, fcns):
     
     eps = 0.000001
     
     return jnp.array([jnp.sqrt(z[3]**2 + z[4]**2 + z[5]**2 + eps)])