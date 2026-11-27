"""Double-integrator unicycle: state [x, y, v, theta, theta_dot], control [a, Omega]."""
import jax.numpy as jnp
import numpy as np


def dynamics(x, u, t, params, fcns):
    v, theta, theta_dot = x[2], x[3], x[4]
    return jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), u[0], theta_dot, u[1]])


def obstacle(x, u, t, params, fcns):
    """Distance from the planar position to each circular no-fly-zone center."""
    d = []
    for c in params.obstacles:
        d.append(jnp.sqrt((x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2 + 1e-9))
    return jnp.array(d)


def xy(x, u, t, params, fcns):
    return x[0:2]


def heading(x, u, t, params, fcns):
    return jnp.array([jnp.cos(x[3]), jnp.sin(x[3])])


def obstacle_circles(params, ax) -> np.ndarray:
    th = np.linspace(0, 2 * np.pi, 200)
    parts = []
    for c in params.obstacles:
        parts.append(np.column_stack([c[0] + c[2] * np.cos(th), c[1] + c[2] * np.sin(th)]))
        parts.append(np.full((1, 2), np.nan))
    return np.vstack(parts[:-1])
