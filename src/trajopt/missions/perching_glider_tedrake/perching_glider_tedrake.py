# source: https://underactuated.mit.edu/trajopt.html#perching

import jax.numpy as jnp
import cvxpy as cp

def u_squared_cost(x, u, t, params, fcns):
    return jnp.atleast_1d(jnp.sum(u**2))

def dynamics(x, u, t, params, fcns):

    # parameters
    Sw = params.Sw
    Se = params.Se
    lw = params.lw
    le = params.le
    lh = params.lh
    inertia = params.inertia
    m = params.m
    rho = params.rho
    g = params.g

    # extract states and control
    pitch = x[2]
    elevator = x[3]
    pos1dot = x[4]
    pos2dot = x[5]
    pitchdot = x[6]
    elevatordot = u[0]

    eps = 1e-10
    pos1wdot = pos1dot + lw * pitchdot * jnp.sin(pitch)
    pos2wdot = pos2dot + lw * pitchdot * jnp.cos(pitch)
    vw = jnp.sqrt(pos2wdot**2 + pos1wdot**2 + eps)
    
    fw = -rho * Sw * (jnp.sin(pitch) * pos1wdot + jnp.cos(pitch) * pos2wdot) * vw

    e = pitch + elevator
    edot = pitchdot + elevatordot
    pos1edot = (pos1dot + lh * pitchdot * jnp.sin(pitch) + le * edot * jnp.sin(e))
    pos2edot = (pos2dot + lh * pitchdot * jnp.cos(pitch) + le * edot * jnp.cos(e))
    ve = jnp.sqrt(pos2edot**2 + pos1edot**2 + eps)
    
    fe = -rho * Se * (jnp.sin(e) * pos1edot + jnp.cos(e) * pos2edot) * ve

    pos1ddot = (fw * jnp.sin(pitch) + fe * jnp.sin(e)) / m
    pos2ddot = (fw * jnp.cos(pitch) + fe * jnp.cos(e)) / m - g
    pitchddot = (fw * lw + fe * (lh * jnp.cos(elevator) + le)) / inertia

    x_dot = jnp.array([pos1dot, pos2dot, pitchdot, elevatordot, pos1ddot, pos2ddot, pitchddot])

    return x_dot

def terminal_box(x, u, t, params, fcns):
    return jnp.array([x[4], x[5], x[2]])

# functions for trajectory plots
def x1x2(x, u, t, params, fcns):
    return jnp.array([x[0], x[1]])

def body_dir(x, u, t, params, fcns):
    pitch = x[2]
    return jnp.array([jnp.cos(pitch), -jnp.sin(pitch)])