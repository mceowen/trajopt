# source: https://github.com/UW-ACL/SCPToolbox.jl/tree/master/test/examples/starship_flip

import numpy as np
import jax.numpy as jnp
import cvxpy as cp

def u_squared_cost(t, x, u, params):
    return cp.sum_squares(u)

def dynamics(t, x, u, params, fcns):

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
    
    ei = fcns.ei(t, x, u, params, fcns)
    ej = fcns.ej(t, x, u, params, fcns)

    eps = 1e-10
    norm_v = jnp.sqrt(v[0]**2 + v[1]**2 + eps)

    Tv = T * (-jnp.sin(delta) * ei + jnp.cos(delta) * ej)
    MT = leng * T * jnp.sin(delta)
    D  = -cd * norm_v * v
    MD = -lcp * jnp.dot(D, ei)
    
    v_dot      = (Tv + D) / mass + g_vec
    theta_ddot = (MT + MD) / params.J
    
    return jnp.concatenate([v, v_dot, jnp.atleast_1d(theta_dot), jnp.atleast_1d(theta_ddot)])

def position(t, x, u, params, fcns):
    return jnp.array([x[0], x[1]])

def body_dir(t, x, u, params, fcns):
    theta = x[4]
    return jnp.array([-jnp.sin(theta), jnp.cos(theta)])

def velocity(t, x, u, params, fcns):
    return jnp.array([x[2], x[3]])

def angular(t, x, u, params, fcns):
    return jnp.array([x[4], x[5]])

def control(t, x, u, params, fcns):
    return jnp.array([u[0], u[1]])

def thrust_dir(t, x, u, params, fcns):
    delta = u[1]
    e_i = fcns.ei(t, x, u, params, fcns)
    e_j = fcns.ej(t, x, u, params, fcns)
    return -jnp.sin(delta) * e_i + jnp.cos(delta) * e_j

def engine_offset(t, x, u, params, fcns):
    """Offset from CG to engine (bottom of rocket): -lcg * ej."""
    e_j = fcns.ej(t, x, u, params, fcns)
    return -params.lcg * e_j

def cvx_glide_slope(t, x, u, params):
    """Glide slope cone: tan(gamma) * |x1| <= x2."""
    tan_gamma = np.tan(np.deg2rad(float(params.gamma_gs)))
    return tan_gamma * cp.abs(x[0]) - x[1] <= 0.0

def ei(t, x, u, params, fcns):
    theta = x[4]

    e1 = jnp.array([1.0, 0.0])
    e2 = jnp.array([0.0, 1.0])

    return jnp.cos(theta) * e1 + jnp.sin(theta) * e2

def ej(t, x, u, params, fcns):
    theta = x[4]

    e1 = jnp.array([1.0, 0.0])
    e2 = jnp.array([0.0, 1.0])

    return -jnp.sin(theta) * e1 + jnp.cos(theta) * e2