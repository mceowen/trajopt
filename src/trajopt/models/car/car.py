"""Reusable physical model functions."""

# source: https://github.com/UW-ACL/SCPToolbox.jl/tree/master/test/examples/starship_flip
import jax.numpy as jnp


def dynamics(x, u, t, params, fcns):

    # extract state
    pos1  = x[0]
    pos2  = x[1]
    theta = x[2]

    # translational and angular speeds as direct control
    v     = u[0]
    w     = u[1]

    # dynamics
    pos1_dot  = v * jnp.cos(theta)
    pos2_dot  = v * jnp.sin(theta)
    theta_dot = w

    x_dot = jnp.array([pos1_dot, pos2_dot, theta_dot])

    return x_dot
