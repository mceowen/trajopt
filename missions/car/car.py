# source: https://github.com/UW-ACL/SCPToolbox.jl/tree/master/test/examples/starship_flip

import numpy as np
import jax.numpy as jnp
import cvxpy as cp

def u_squared_cost(t, x, u, params):
    return jnp.atleast_1d(jnp.sum(u**2))

def dynamics(t, x, u, params, fcns):

    # extract state
    x1    = x[0]
    x2    = x[1]
    theta = x[2]

    # translational and angular speeds as direct control
    v     = u[0]
    w     = u[1]

    # dynamics
    x1_dot    = v * jnp.cos(theta)
    x2_dot    = v * jnp.sin(theta)
    theta_dot = w 

    x_dot = jnp.array([x1_dot, x2_dot, theta_dot]) 

    return x_dot

def obstacle(t, x, u, params, fcns):
    x1    = x[0]
    x2    = x[1]

    eps = 1e-6
    d1 = jnp.sqrt((x1 - params.obs1_x)**2 + (x2 - params.obs1_y)**2 + eps)
    d2 = jnp.sqrt((x1 - params.obs2_x)**2 + (x2 - params.obs2_y)**2 + eps)

    return jnp.array([d1, d2])

def x1x2(t, x, u, params, fcns):
    x1    = x[0]
    x2    = x[1]

    return jnp.array([x1, x2])

def body_dir(t, x, u, params, fcns):
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