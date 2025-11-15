import numpy as np
import jax 
import jax.numpy as jnp
import trajopt.utils.tools                      as tools
import trajopt.core.modules.model.obstacles    as obstacles

def dynamics(t, z, nu, problem):
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
    g_vec   = np.array([0,0, -mission.planet["g"]]) / method.nondim["na"]

    # extract states
    r = z[0:3]
    v = z[3:6]
    
    T = nu

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

def analytical_linsys(t, z, nu, problem):

    mission = problem.mission
    model = problem.model
    method = problem.method
    
    # Extract parameters

    n       = model.n
    m       = model.m
    mass    = mission.vehicle["mass"] / method.nondim["nm"]

    # Sanity check for vector shapes
    z =  np.asarray(z).flatten()
    nu = np.asarray(nu).flatten()

    assert len(z) == n, f"Expected state vector of length {n}, got {len(z)}"
    assert len(nu) == m, f"Expected control vector of length {m}, got {len(nu)}"

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
    fc = dynamics(t, z, nu, problem)

    return fc, Ac, Bc

def dynamics_jax(t, z, nu, problem):
    mission = problem.mission
    model = problem.model
    method = problem.method

    mass    = mission.vehicle["mass"] / method.nondim["nm"]
    g_vec   = np.array([0,0, -mission.planet["g"]]) / method.nondim["na"]

    r = z[0:3]
    v = z[3:6]
    
    T = nu

    xDot = jnp.concatenate([v, T/mass + g_vec])

    return xDot