import numpy as np

import trajopt.utils.tools                  as tools
import trajopt.core.modules.models.obstacles     as obstacles


def system_dynamics(ts, zs, us, problem, t_vec=None):
    """
    x1, x2: r (position)
    u1, u2: v (velocity)
    """
    # extracts params if "problem" parent struct is passed in
    mission = problem.mission
    model = problem.model
    method = problem.method

    # extract constant param values
    m       = model.m
    n       = model.n
    mass    = mission.vehicle["mass"] / method.nondim["nm"]
    g_vec      = np.array([0,0, -mission.planet["g"]]) / method.nondim["na"]

    # extract states
    r = zs[0:3]
    v = zs[3:6]

    # extract controls 
    if t_vec is None:
        us2 = us
    else:
        us2 = np.array([np.interp(ts, t_vec, us[:, i]) for i in range(m)])
            
    # extract control
    T = us2

    # compute velocity and acceleration
    xDot        = np.empty(6) # initialize
    xDot[0:3]   = v
    xDot[3:6]   = T/mass + g_vec

    if np.issubdtype(r.dtype, np.number):
        if r[2] <= -1: # set xDot = 0 if the vehicle hits the ground
            xDot = np.zeros(n)
    elif np.issubdtype(r.dtype, np.nan) or any(np.isinf(r)):
        breakpoint()
        
    return xDot

def analytical_linsys(ts, zs, us, problem):

    mission = problem.mission
    model = problem.model
    method = problem.method
    
    # Extract parameters

    n       = model.n
    m       = model.m
    mass    = mission.vehicle["mass"] / method.nondim["nm"]

    # Sanity check for vector shapes
    zs = np.asarray(zs).flatten()
    us = np.asarray(us).flatten()

    assert len(zs) == n, f"Expected state vector of length {n}, got {len(zs)}"
    assert len(us) == m, f"Expected control vector of length {m}, got {len(us)}"

    # Compute A matrix (Jacobian w.r.t. state)
    n2 = n // 2
    Ac = np.block([
        [np.zeros((n2, n2)), np.eye(n2)],
        [np.zeros((n2, n))]
    ])

    # Compute B matrix (Jacobian w.r.t. control)
    Bc = np.vstack([
        np.zeros((n2, m)),
        np.eye(m)
    ]) * (1.0 / mass)

    # Evaluate nonlinear dynamics
    fc = system_dynamics(ts, zs, us, problem)

    # Return in dictionary format
    linsys = {
        "dfcn_dz": Ac,
        "dfcn_du": Bc,
        "fcn":     fc
    }

    return linsys

def nonlinear_inequality_constraints(ts, zs, us, problem):
    N = tools.num_timesteps(zs)
    P_blocks = []

    # path constraints (placeholder)
    P_path = np.empty((N, 0))
    P_blocks.append(P_path)

    # add NFZ block
    P_blocks.append(obstacles.nfz_nonlinear(ts, zs, us, problem))

    # stack all constraint blocks horizontally
    P = np.hstack([P for P in P_blocks if P.size > 0])

    return P

def analytical_inequality_constraints(ts, zs, us, problem):
    N = tools.num_timesteps(zs)
    model = problem.model
    n = model.n
    m = model.m

    # NFZ block
    nfz = obstacles.nfz_analytical(ts, zs, us, problem)

    # preallocate total constraint arrays
    fcn_all   = nfz["fcn"]
    dPdz_all  = nfz["dfcn_dz"]
    dPdu_all  = nfz["dfcn_du"]

    return {"fcn": fcn_all, "dfcn_dz": dPdz_all, "dfcn_du": dPdu_all}