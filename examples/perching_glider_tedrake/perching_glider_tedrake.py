# source: https://underactuated.mit.edu/trajopt.html#perching

import jax.numpy as jnp
import cvxpy as cp

def u_squared_cost(t, x, u, params):
    return jnp.atleast_1d(jnp.sum(u**2))

def dynamics(t, x, u, params, fcns):

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
    x1dot = x[4]
    x2dot = x[5]
    pitchdot = x[6]
    elevatordot = u[0]

    eps = 1e-10
    x1wdot = x1dot + lw * pitchdot * jnp.sin(pitch)
    x2wdot = x2dot + lw * pitchdot * jnp.cos(pitch)
    vw = jnp.sqrt(x2wdot**2 + x1wdot**2 + eps)
    
    fw = -rho * Sw * (jnp.sin(pitch) * x1wdot + jnp.cos(pitch) * x2wdot) * vw

    e = pitch + elevator
    edot = pitchdot + elevatordot
    x1edot = (x1dot + lh * pitchdot * jnp.sin(pitch) + le * edot * jnp.sin(e))
    x2edot = (x2dot + lh * pitchdot * jnp.cos(pitch) + le * edot * jnp.cos(e))
    ve = jnp.sqrt(x2edot**2 + x1edot**2 + eps)
    
    fe = -rho * Se * (jnp.sin(e) * x1edot + jnp.cos(e) * x2edot) * ve

    x1ddot = (fw * jnp.sin(pitch) + fe * jnp.sin(e)) / m
    x2ddot = (fw * jnp.cos(pitch) + fe * jnp.cos(e)) / m - g
    pitchddot = (fw * lw + fe * (lh * jnp.cos(elevator) + le)) / inertia

    x_dot = jnp.array([x1dot, x2dot, pitchdot, elevatordot, x1ddot, x2ddot, pitchddot])

    return x_dot

def terminal_box(t, x, u, params):
    return jnp.array([x[4], x[5], x[2]])

# functions for trajectory plots
def x1x2(t, x, u, params, fcns):
    return jnp.array([x[0], x[1]])

def body_dir(t, x, u, params, fcns):
    pitch = x[2]
    return jnp.array([jnp.cos(pitch), -jnp.sin(pitch)])