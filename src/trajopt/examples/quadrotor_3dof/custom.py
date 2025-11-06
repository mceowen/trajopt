import numpy as np

import trajopt.utils.tools                      as tools

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

def system_dynamics2(ts, zs, us, problem, t_vec=None):
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