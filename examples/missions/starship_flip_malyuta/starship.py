# source: https://github.com/UW-ACL/SCPToolbox.jl/tree/master/test/examples/starship_flip

import numpy as np
import jax.numpy as jnp
import cvxpy as cp

def u_squared_cost(x, u, t, params, fcns):
    return jnp.atleast_1d(jnp.sum(u**2))

def dynamics(x, u, t, params, fcns):

    v         = x[2:4]
    theta_dot = x[5]

    T     = u[0]
    delta = u[1]

    leng  = -params.lcg
    lcp   = params.lcp - params.lcg
    cd    = params.cd
    mass  = params.mass
    g     = params.g
    g_vec = jnp.array([0, -g])
    
    ei = fcns.ei(x, u, t, params, fcns)
    ej = fcns.ej(x, u, t, params, fcns)

    eps = 1e-10
    norm_v = jnp.sqrt(v[0]**2 + v[1]**2 + eps)

    Tv = T * (-jnp.sin(delta) * ei + jnp.cos(delta) * ej)
    MT = leng * T * jnp.sin(delta)
    D  = -cd * norm_v * v
    MD = -lcp * jnp.dot(D, ei)
    
    v_dot      = (Tv + D) / mass + g_vec
    theta_ddot = (MT + MD) / params.J
    
    return jnp.concatenate([v, v_dot, jnp.atleast_1d(theta_dot), jnp.atleast_1d(theta_ddot)])

def position(x, u, t, params, fcns):
    return jnp.array([x[0], x[1]])

def body_dir(x, u, t, params, fcns):
    theta = x[4]
    return jnp.array([-jnp.sin(theta), jnp.cos(theta)])

def velocity(x, u, t, params, fcns):
    return jnp.array([x[2], x[3]])

def angular(x, u, t, params, fcns):
    return jnp.array([x[4], x[5]])

def control(x, u, t, params, fcns):
    return jnp.array([u[0], u[1]])

def thrust_magnitude(x, u, t, params, fcns):
    return jnp.array([u[0]])

def gimbal_angle(x, u, t, params, fcns):
    return jnp.array([u[1]])

def thrust_dir(x, u, t, params, fcns):
    delta = u[1]
    e_i = fcns.ei(x, u, t, params, fcns)
    e_j = fcns.ej(x, u, t, params, fcns)
    return -jnp.sin(delta) * e_i + jnp.cos(delta) * e_j

def engine_offset(x, u, t, params, fcns):
    """Offset from CG to engine (bottom of rocket): -lcg * ej."""
    e_j = fcns.ej(x, u, t, params, fcns)
    return -params.lcg * e_j

def cvx_glide_slope(x, u, params):
    """Glide slope cone: tan(gamma) * |x1| <= x2."""
    tan_gamma = np.tan(np.deg2rad(float(params.gamma_gs)))
    return tan_gamma * cp.abs(x[:, 0]) - x[:, 1]

def glideslope_overlay(params, ax):
    tan_gamma = np.tan(np.deg2rad(float(params.gamma_gs)))
    y = np.array([0, 800])
    x_bound = y / tan_gamma
    return np.array([[0, 0], [x_bound[-1], y[-1]],
                     [np.nan, np.nan],
                     [0, 0], [-x_bound[-1], y[-1]]])

def ei(x, u, t, params, fcns):
    theta = x[4]

    e1 = jnp.array([1.0, 0.0])
    e2 = jnp.array([0.0, 1.0])

    return jnp.cos(theta) * e1 + jnp.sin(theta) * e2

def ej(x, u, t, params, fcns):
    theta = x[4]

    e1 = jnp.array([1.0, 0.0])
    e2 = jnp.array([0.0, 1.0])

    return -jnp.sin(theta) * e1 + jnp.cos(theta) * e2