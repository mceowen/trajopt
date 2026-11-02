"""Reusable physical model functions."""

# sources: 
# https://underactuated.mit.edu/acrobot.html#cart_pole 
# https://youtu.be/wlkRYMVUZTs?si=ZE_MSqaa0uizo3cF
import jax.numpy as jnp


def dynamics(x, u, t, params, fcns):
    # shortcuts for the cosine and the sine of theta
    # they might be handy
    ctheta = jnp.cos(x[1])
    stheta = jnp.sin(x[1])

    # parameters
    g  = params.g
    mp = params.mp
    mc = params.mc
    l  = params.l
    
    f_pos1    = u[0]
    pos1dot   = x[2]
    thetadot  = x[3]
    pos1ddot  = (1 / (mc + mp * stheta**2)) * (f_pos1 + mp * stheta * (l*thetadot**2 + g * ctheta))
    thetaddot = (1 / (l * (mc + mp*stheta**2))) * (-f_pos1 * ctheta - mp * l * thetadot**2*ctheta*stheta - (mc + mp)*g*stheta)

    return jnp.array([pos1dot, thetadot, pos1ddot, thetaddot])
