# source: https://github.com/UW-ACL/SCPToolbox.jl/tree/master/test/examples/starship_flip

import numpy as np
import jax.numpy as jnp
import cvxpy as cp

def u_squared_cost(x, u, t, params, fcns):
    return jnp.atleast_1d(jnp.sum(u**2))

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

def obstacle(x, u, t, params, fcns):
    pos1  = x[0]
    pos2  = x[1]

    eps = 1e-6
    d1 = jnp.sqrt((pos1 - params.obs1_x)**2 + (pos2 - params.obs1_y)**2 + eps)
    d2 = jnp.sqrt((pos1 - params.obs2_x)**2 + (pos2 - params.obs2_y)**2 + eps)

    return jnp.array([d1, d2])

def x1x2(x, u, t, params, fcns):
    pos1  = x[0]
    pos2  = x[1]

    return jnp.array([pos1, pos2])

def body_dir(x, u, t, params, fcns):
    theta = x[2]
    return jnp.array([jnp.cos(theta), jnp.sin(theta)])

def obstacle_circles(params, ax) -> np.ndarray:
    th  = np.linspace(0, 2 * np.pi, 200)
    nan = np.full((1, 2), np.nan)
    r1  = float(params.obs1_r)
    r2  = float(params.obs2_r)
    c1  = np.column_stack([float(params.obs1_x) + r1 * np.cos(th),
                           float(params.obs1_y) + r1 * np.sin(th)])
    c2  = np.column_stack([float(params.obs2_x) + r2 * np.cos(th),
                           float(params.obs2_y) + r2 * np.sin(th)])
    return np.vstack([c1, nan, c2])