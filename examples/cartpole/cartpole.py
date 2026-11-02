"""Scenario costs, constraints, and output callbacks."""
import jax.numpy as jnp


def u_squared_cost(x, u, t, params, fcns):
    return jnp.atleast_1d(jnp.sum(u**2))


def state_pos(x, u, t, params, fcns):
    return jnp.array([x[0], x[1]])


def state_vel(x, u, t, params, fcns):
    return jnp.array([x[2], x[3]])


def cart_pos(x, u, t, params, fcns):
    return jnp.array([x[0], 0.0])


def pole_dir(x, u, t, params, fcns):
    theta = x[1]
    return jnp.array([jnp.sin(theta), -jnp.cos(theta)])
