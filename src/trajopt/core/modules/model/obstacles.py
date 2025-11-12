"""
constraints.py
---------------
Shared constraint utilities for trajectory optimization models.
"""

import numpy as np

# =============================================================================
# NFZ Nonlinear Constraint Block
# =============================================================================

def nfz_nonlinear(ts, zs, us, problem):
    xc = problem.mission.obs["xc"]
    yc = problem.mission.obs["yc"]
    rc = np.asarray(problem.mission.obs["rc"])

    x_idx = problem.model.obs["x_idx"]
    y_idx = problem.model.obs["y_idx"]
    
    return rc**2 - (zs[x_idx] - xc)**2 - (zs[y_idx] - yc)**2

# =============================================================================
# NFZ Analytical Linearization Block
# =============================================================================

def nfz_analytical_affine_approximation(ts, zs, us, problem):
    xc = problem.mission.obs["xc"]
    yc = problem.mission.obs["yc"]
    
    x_idx = problem.model.obs["x_idx"]
    y_idx = problem.model.obs["y_idx"]

    P_nfz = nfz_nonlinear(ts, zs, us, problem)
    
    dP_dz = np.zeros((problem.mission.n_nfz, problem.model.n))
    dP_dz[:, x_idx] = -2 * (zs[x_idx] - xc)
    dP_dz[:, y_idx] = -2 * (zs[y_idx] - yc)

    dP_du = np.zeros((problem.mission.n_nfz, problem.model.m))
    
    return P_nfz, dP_dz, dP_du

