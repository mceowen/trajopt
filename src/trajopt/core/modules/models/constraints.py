"""
constraints.py
---------------
Shared constraint utilities for trajectory optimization models.
"""

import numpy as np
from trajopt.utils import tools


# =============================================================================
# NFZ Nonlinear Constraint Block
# =============================================================================

def nfz_nonlinear(ts, zs, us, problem):
    """
    Compute nonlinear NFZ inequality constraint block for stacking.

    Each constraint has the form (shape N, n_nfz):
        P_nfz_j = rc_j^2 - distance_to_center_j^2

    Satisfied when P_nfz >= 0.
    """
    mission = problem.mission
    N = tools.num_timesteps(zs)
    n_nfz = getattr(mission, "n_nfz", 0)

    if n_nfz == 0:
        return np.empty((N, 0))

    posc = np.atleast_2d(mission.obs["posc"])   # shape (dim, n_nfz)
    rc   = np.asarray(mission.obs["rc"])        # (n_nfz,)

    # Planar NFZs (x, y)
    if posc.shape[0] == 2:
        if zs.ndim == 1:
            rx, ry = zs[0], zs[1]
            P_nfz = rc**2 - (rx - posc[0])**2 - (ry - posc[1])**2
        else:
            rx = zs[:, 0][:, None]
            ry = zs[:, 1][:, None]
            P_nfz = rc[None, :]**2 - (rx - posc[0][None, :])**2 - (ry - posc[1][None, :])**2

    # Spherical NFZs (theta, phi)
    elif posc.shape[0] == 3:
        if zs.ndim == 1:
            theta, phi = zs[1], zs[2]
            P_nfz = rc**2 - (theta - posc[1])**2 - (phi - posc[2])**2
        else:
            theta = zs[:, 1][:, None]
            phi   = zs[:, 2][:, None]
            P_nfz = rc[None, :]**2 - (theta - posc[1][None, :])**2 - (phi - posc[2][None, :])**2

    else:
        raise ValueError(f"Unsupported NFZ posc shape: {posc.shape}")

    return P_nfz


# =============================================================================
# NFZ Analytical Linearization Block
# =============================================================================

def nfz_analytical(ts, zs, us, problem):
    """
    Compute analytical partial derivatives of NFZ constraints.

    Returns
    -------
    block : dict
        {
          "fcn"      : P_nfz  (N, n_nfz)
          "dfcn_dz"  : dP/dz  (N, n_nfz, n)
          "dfcn_du"  : dP/du  (N, n_nfz, m)
        }
    """
    mission = problem.mission
    model   = problem.model
    N       = tools.num_timesteps(zs)
    n       = model.n
    m       = model.m
    n_nfz   = getattr(mission, "n_nfz", 0)

    if n_nfz == 0:
        return {
            "fcn": np.empty((N, 0)),
            "dfcn_dz": np.zeros((N, 0, n)),
            "dfcn_du": np.zeros((N, 0, m))
        }

    posc = np.atleast_2d(mission.obs["posc"])  # (dim, n_nfz)
    rc   = np.asarray(mission.obs["rc"])       # (n_nfz,)

    P_nfz = nfz_nonlinear(ts, zs, us, problem)
    dPdz = np.zeros((N, n_nfz, n))
    dPdu = np.zeros((N, n_nfz, m))

    # Planar (x,y)
    if posc.shape[0] == 2:
        if zs.ndim == 1:
            rx, ry = zs[0], zs[1]
            dPdz[:, :, 0] = -2 * (rx - posc[0])
            dPdz[:, :, 1] = -2 * (ry - posc[1])
        else:
            rx = zs[:, 0][:, None]
            ry = zs[:, 1][:, None]
            dPdz[:, :, 0] = -2 * (rx - posc[0][None, :])
            dPdz[:, :, 1] = -2 * (ry - posc[1][None, :])

    # Spherical (theta, phi)
    elif posc.shape[0] == 3:
        if zs.ndim == 1:
            theta, phi = zs[1], zs[2]
            dPdz[:, :, 1] = -2 * (theta - posc[1])
            dPdz[:, :, 2] = -2 * (phi - posc[2])
        else:
            theta = zs[:, 1][:, None]
            phi   = zs[:, 2][:, None]
            dPdz[:, :, 1] = -2 * (theta - posc[1][None, :])
            dPdz[:, :, 2] = -2 * (phi - posc[2][None, :])

    else:
        raise ValueError(f"Unsupported NFZ posc shape: {posc.shape}")

    return {"fcn": P_nfz, "dfcn_dz": dPdz, "dfcn_du": dPdu}
