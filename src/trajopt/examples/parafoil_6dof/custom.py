import numpy as np
import trajopt.utils.tools                      as tools

# ===============================================================
# MISSION
# ===============================================================

def cost2(t, z, nu, trajopt_obj):
    return np.dot(np.transpose(nu), nu)

# ===============================================================
# MODEL
# ===============================================================

def dynamics(t, z, nu, trajopt_obj, t_vec=None):
    """
    x1, x2: r (position)
    u1, u2: v (velocity)
    """
    # extracts params if 'trajopt_obj' parent struct is passed in
    mission = trajopt_obj.mission
    model = trajopt_obj.model
    method = trajopt_obj.method

    # extract constant param values
    m       = model.m
    n       = model.n
    mass    = mission.vehicle["mass"] / method.nondim["nm"]
    g_vec      = np.array([0,0, -mission.planet["g"]]) / method.nondim["na"]

    # extract states
    r = z[0:3]
    v = z[3:6]

    # extract controls 
    if t_vec is None:
        us2 = nu
    else:
        us2 = np.array([np.interp(t, t_vec, nu[:, i]) for i in range(m)])
            
    # extract control
    T = nu2

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