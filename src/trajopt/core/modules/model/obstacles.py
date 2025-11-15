"""
constraints.py
---------------
Shared constraint utilities for trajectory optimization models.
"""

import numpy as np
import jax.numpy as jnp

# =============================================================================
# NFZ Nonlinear Constraint Block
# =============================================================================

def nfz_nonlinear(t, z, nu, problem):
    mission = problem.mission
    model = problem.model
    method = problem.method
    
    x_idx = model.obs["x_idx"]
    y_idx = model.obs["y_idx"]

    # TODO: need a cleaner way to get the nondim for nfz based on model
    nondim_nfz = method.nondim["M"]["state"]["nd2d"][1, 1]

    if method.flags["jax_dyn"]:
        xc = jnp.asarray(mission.obs["xc"]) / nondim_nfz
        yc = jnp.asarray(mission.obs["yc"]) / nondim_nfz
        rc = jnp.asarray(mission.obs["rc"]) / nondim_nfz
    else:
        xc = mission.obs["xc"] / nondim_nfz
        yc = mission.obs["yc"] / nondim_nfz
        rc = mission.obs["rc"] / nondim_nfz
    
    return (1.0 - (z[x_idx] - xc)**2 / rc**2 - (z[y_idx] - yc)**2 / rc**2)

# =============================================================================
# NFZ Analytical Linearization Block
# =============================================================================

def nfz_analytical_affine_approximation(t, z, nu, problem):
    xc = problem.mission.obs["xc"]
    yc = problem.mission.obs["yc"]
    rc = problem.mission.obs["rc"]
    
    x_idx = problem.model.obs["x_idx"]
    y_idx = problem.model.obs["y_idx"]

    P_nfz = nfz_nonlinear(t, z, nu, problem)
    
    dP_dz = np.zeros((problem.mission.n_nfz, problem.model.n))
    dP_dz[:, x_idx] = -2 * (z[x_idx] - xc) / (rc**2)
    dP_dz[:, y_idx] = -2 * (z[y_idx] - yc) / (rc**2)

    dP_du = np.zeros((problem.mission.n_nfz, problem.model.m))
    
    return P_nfz, dP_dz, dP_du

