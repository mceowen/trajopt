import numpy as np
import jax 
import jax.numpy as jnp
import sympy as sp

def dynamics_jax(t, z, nu, params, fcns):

    # extract parameters
    g     = params["planet"]["g"]
    mass  = params["vehicle"]["mass"]
    g_vec = jnp.array([0,0, -g])


    r = z[0:3]
    v = z[3:6]
    
    T = nu

    x_dot = jnp.concatenate([v, T/mass + g_vec])

    return x_dot

def thrust_norm(t, z, nu, params, fcns):
    return jnp.array([jnp.linalg.norm(nu)])

def obstacle(t, z, nu, params, fcns):
    r = z[0:2]
    pos_obs = jnp.array([5, 5])
    return jnp.array([jnp.linalg.norm(r - pos_obs)])

def xy(t, z, nu, params, fcns):
    pos_xy = z[0:2]
    return pos_xy

def xyz(t, z, nu, params, fcns):
    pos_xyz = z[0:3]
    return pos_xyz