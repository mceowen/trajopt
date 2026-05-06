
# sources: 
# https://underactuated.mit.edu/acrobot.html#cart_pole 
# https://youtu.be/wlkRYMVUZTs?si=ZE_MSqaa0uizo3cF

import jax.numpy as jnp
import cvxpy as cp

def u_squared_cost(t, x, u, params):
    return cp.sum_squares(u)

def dynamics(t, x, u, params, fcns):
    # shortcuts for the cosine and the sine of theta
    # they might be handy
    ctheta = jnp.cos(x[1])
    stheta = jnp.sin(x[1])

    # parameters
    g  = params.g
    mp = params.mp
    mc = params.mc
    l  = params.l
    
    fx1       = u[0]
    x1dot     = x[2]
    thetadot  = x[3]
    x1ddot    = (1 / (mc + mp * stheta**2)) * (fx1 + mp * stheta * (l*thetadot**2 + g * ctheta))
    thetaddot = (1 / (l * (mc + mp*stheta**2))) * (-fx1 * ctheta - mp * l * thetadot**2*ctheta*stheta - (mc + mp)*g*stheta)

    return jnp.array([x1dot, thetadot, x1ddot, thetaddot])

def state_pos(t, x, u, params, fcns):
    return jnp.array([x[0], x[1]])

def state_vel(t, x, u, params, fcns):
    return jnp.array([x[2], x[3]])

def control_force(t, x, u, params, fcns):
    return jnp.array([u[0]])

def cart_pos(t, x, u, params, fcns):
    return jnp.array([x[0], 0.0])

def pole_dir(t, x, u, params, fcns):
    theta = x[1]
    return jnp.array([jnp.sin(theta), -jnp.cos(theta)])