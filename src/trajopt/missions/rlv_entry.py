import jax.numpy as jnp
from jax import Array
from trajopt.utils.tools import AttrDict


def nonlinear_aero(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> AttrDict:
    aoa_rad = jnp.deg2rad(nu[1])
    v       = z[3]
    rho     = fcns.density_model(t, z, nu, params, fcns)
    cl      = params.vehicle.cl
    cd      = params.vehicle.cd
    CL      = cl[0] + cl[1] * aoa_rad
    CD      = cd[0] + cd[1] * aoa_rad + cd[2] * aoa_rad**2
    q       = 0.5 * rho * v**2
    L       = q * params.vehicle.sref * CL / params.vehicle.mass
    D       = q * params.vehicle.sref * CD / params.vehicle.mass
    return AttrDict({"L": L, "D": D})
