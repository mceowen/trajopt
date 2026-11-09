"""Scenario costs, constraints, and output callbacks."""
import jax.numpy as jnp


def u_squared_cost(x, u, t, params, fcns):
    return jnp.atleast_1d(jnp.sum(u**2))


def terminal_box(x, u, t, params, fcns):
    return jnp.array([x[4], x[5], x[2]])
