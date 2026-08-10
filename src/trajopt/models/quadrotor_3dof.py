import cvxpy as cp
import jax.numpy as jnp
import numpy as np
from jax import Array

from trajopt.utils.tools import AttrDict


def dynamics_jax(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """3-DoF quadrotor dynamics (double integrator with gravity)."""
    g = params.planet.g
    mass = params.vehicle.mass
    g_vec = jnp.array([0, 0, -g])
    v = x[3:6]
    T = u

    return jnp.concatenate([v, T / mass + g_vec])


def thrust_norm(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """Thrust magnitude."""
    return jnp.array([jnp.linalg.norm(u)])


def obstacle(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """Distance from circular obstacle centered at (5,5) in xy-plane."""
    r = x[0:2]
    pos_obs = jnp.array([5, 5])
    return jnp.array([jnp.linalg.norm(r - pos_obs)])


def max_thrust_cone(x, u, params):
    """||T|| <= T_max as a CVXPY SOC constraint."""
    T_max = float(params.vehicle.T_max)
    return cp.norm(u[:, 0:3], axis=1) - T_max


def xy(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """xy-position."""
    return x[0:2]


def xz(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """xz-position."""
    return jnp.array([x[0], x[2]])


def yz(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """yz-position."""
    return jnp.array([x[1], x[2]])


def xyz(x: Array, u: Array, t: float, params: AttrDict, fcns: AttrDict) -> Array:
    """xyz-position."""
    return x[0:3]


def one(x, u, t, params, fcns):
    return jnp.array([1.0])

def u_squared(x, u, t, params, fcns):
    return jnp.array([jnp.sum(jnp.square(u))])

def obstacle_xy(params, ax) -> np.ndarray:
    """Circle boundary of the obstacle in the xy-plane (center (5,5), radius 4)."""
    th = np.linspace(0, 2 * np.pi, 200)
    return np.column_stack([5.0 + 4.0 * np.cos(th), 5.0 + 4.0 * np.sin(th)])