import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from trajopt.utils.tools import AttrDict

jax.config.update("jax_enable_x64", True)

def nonlinear_aero(t, x, u, params, fcns):
    """Nonlinear aerodynamic force coefficients and state for GHAME, JAX version."""

    vehicle = params.vehicle
    rho = fcns.density_model(t, x, u, params, fcns)
    mass = vehicle.mass
    v = x[3]

    alpha_deg = u[1]

    # COEFFICIENTS
    M = v / ((1.4 * 287 * 239) ** 0.5)
    cl0 = 0.0052 * jnp.log(M) - 0.0334
    cl1 = 0.03 * (M ** (-0.49))
    cd0 = 0.0577 * jnp.exp(-0.042 * M)
    cd1 = 0.00879 * jnp.log(M) - 0.0192
    cd2 = 0.4521 * (M ** (0.4856))

    # AoA-DEPENDENT AERO COEFFICIENTS
    Cl = cl0 + cl1 * alpha_deg
    Cd = cd0 + (cd1 * Cl) + (cd2 * (Cl**2))

    sref = vehicle.sref

    L = 0.5 * (1 / mass) * rho * sref * Cl * v**2
    D = 0.5 * (1 / mass) * rho * sref * Cd * v**2

    return AttrDict({"L": L, "D": D, "Cl": Cl, "Cd": Cd, "alpha": alpha_deg, "rho": rho})