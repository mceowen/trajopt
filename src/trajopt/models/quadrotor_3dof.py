import cvxpy as cp
import jax.numpy as jnp
from jax import Array


def dynamics_jax(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """3-DoF quadrotor dynamics (double integrator with gravity)."""
    g = params.planet.g
    mass = params.vehicle.mass
    g_vec = jnp.array([0, 0, -g])
    v = z[3:6]
    T = nu

    return jnp.concatenate([v, T / mass + g_vec])


def thrust_norm(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Thrust magnitude."""
    return jnp.array([jnp.linalg.norm(nu)])


def obstacle(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Distance from circular obstacle centered at (5,5) in xy-plane."""
    r = z[0:2]
    pos_obs = jnp.array([5, 5])
    return jnp.array([jnp.linalg.norm(r - pos_obs)])


def max_thrust_cone(x, u, params):
    """||T|| <= T_max as a CVXPY SOC constraint."""
    T_max = params.vehicle.T_max
    return cp.norm(u[0:3]) <= T_max


def pos_x(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """x-position."""
    return jnp.array([z[0]])


def pos_y(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """y-position."""
    return jnp.array([z[1]])


def height(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """Height (z-position)."""
    return jnp.array([z[2]])


def xy(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """xy-position."""
    return z[0:2]


def xz(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """xz-position."""
    return jnp.array([z[0], z[2]])


def yz(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """yz-position."""
    return jnp.array([z[1], z[2]])


def xyz(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """xyz-position."""
    return z[0:3]


def vel_x(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([z[3]])

def vel_y(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([z[4]])

def vel_z(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([z[5]])

def thrust_x(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([nu[0]])

def thrust_y(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([nu[1]])

def thrust_z(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([nu[2]])
