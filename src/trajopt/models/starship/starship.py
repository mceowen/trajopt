"""Reusable physical model functions."""

# source: https://github.com/UW-ACL/SCPToolbox.jl/tree/master/test/examples/starship_flip
import jax.numpy as jnp


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
