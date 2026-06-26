import jax
import jax.numpy as jnp
from jax import Array
from trajopt.utils.tools import AttrDict
jax.config.update("jax_enable_x64", True)

def coeffs_entry(x, u, t, params, fcns):
    vehicle = params.vehicle
    mass = vehicle.mass
    sref = vehicle.sref

    LD = vehicle.LD
    bc = vehicle.bc

    cd_entry = mass / (bc * sref)
    cl_entry = cd_entry * LD

    return AttrDict({"cd": cd_entry, "cl": cl_entry})

def coeffs_descent(x, u, t, params, fcns):

    return AttrDict({"cd": 0.55, "cl": 0.0})