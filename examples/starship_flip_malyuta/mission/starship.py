"""Scenario costs, constraints, and output callbacks."""
import numpy as np
import jax.numpy as jnp
import cvxpy as cp


def u_squared_cost(x, u, t, params, fcns):
    return jnp.atleast_1d(jnp.sum(u**2))


def position(x, u, t, params, fcns):
    return jnp.array([x[0], x[1]])


def body_dir(x, u, t, params, fcns):
    theta = x[4]
    return jnp.array([-jnp.sin(theta), jnp.cos(theta)])


def velocity(x, u, t, params, fcns):
    return jnp.array([x[2], x[3]])


def angular(x, u, t, params, fcns):
    return jnp.array([x[4], x[5]])


def control(x, u, t, params, fcns):
    return jnp.array([u[0], u[1]])


def thrust_dir(x, u, t, params, fcns):
    delta = u[1]
    e_i = fcns.ei(x, u, t, params, fcns)
    e_j = fcns.ej(x, u, t, params, fcns)
    return -jnp.sin(delta) * e_i + jnp.cos(delta) * e_j


def engine_offset(x, u, t, params, fcns):
    """Offset from CG to engine (bottom of rocket): -lcg * ej."""
    e_j = fcns.ej(x, u, t, params, fcns)
    return -params.lcg * e_j


def cvx_glide_slope(x, u, params):
    """Glide slope cone: tan(gamma) * |x1| <= x2."""
    tan_gamma = np.tan(np.deg2rad(float(params.gamma_gs)))
    return tan_gamma * cp.abs(x[:, 0]) - x[:, 1]


def glideslope_overlay(params, ax):
    tan_gamma = np.tan(np.deg2rad(float(params.gamma_gs)))
    y = np.array([0, 800])
    x_bound = y / tan_gamma
    return np.array([[0, 0], [x_bound[-1], y[-1]],
                     [np.nan, np.nan],
                     [0, 0], [-x_bound[-1], y[-1]]])
