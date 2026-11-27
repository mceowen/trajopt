"""Descent in up/downrange/crossrange coordinates; scalar-first quaternion."""

import jax
import jax.numpy as jnp

from trajopt.models.lander_6dof.lander_6dof import dynamics as powered

jax.config.update("jax_enable_x64", True)

def passive(x, u, t, params, fcns):
    dx = jnp.zeros_like(x).at[1:4].set(x[4:7])
    acceleration = jnp.array([-params.planet.g, 0.0, 0.0])
    rho = params.rho_surface * jnp.exp(-x[1] / params.scale_height)
    velocity = x[4:7]
    speed = jnp.sqrt(jnp.dot(velocity, velocity) + 1e-12)
    acceleration -= 0.5 * rho * params.drag_area * speed * velocity / x[0]
    return dx.at[4:7].set(acceleration)
