import numpy as np
import trajopt.utils.tools                      as tools

# ===============================================================
# MISSION
# ===============================================================

def cost2(t, z, nu, problem):
    return np.dot(np.transpose(nu), nu)

# ===============================================================
# MODEL
# ===============================================================

def dynamics(t, z, nu, problem, t_vec=None):
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
    r = z[0:3]
    v = z[3:6]

    # extract control (us is now a single control vector, not a trajectory)
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

def dynamics2(t, z, nu, problem, t_vec=None):
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