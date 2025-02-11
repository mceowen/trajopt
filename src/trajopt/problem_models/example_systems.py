def test_func():
    return "Hello, World!"

# Skye Mceowen
# Feb. 17th, 2024
# 2D circular NFZ
# Single integrator linear dynamics

# TODO consolidate imports 
from scipy.interpolate import interp1d
import numpy as np


def system_dynamics(ts,zs,us,params,t_vec=None):
    '''
    x1, x2: r (position)
    u1, u2: v (velocity)
    '''

    # extracts params if "problem" parent struct is passed in
    if hasattr(params, 'params'):
        params = params.params

    # extract constant param values
    m       = params['m']
    n       = params['n']
    mass    = params['mass']
    ge      = params['ge']


    # extract states
    r = np.array( zs[0:3, 0] ).reshape(-1,1) # turn into a column vector (works if zs is already a col. or a row vec)
    v = np.array( zs[3:6, 0] ).reshape(-1,1)

    # extract controls 
    if t_vec is None:
        us2 = np.array( us )    # got an error that said us2 was an array.array for some reason
                                # (and therefore I coudn't do element-wise operations)
    else:
        for i in range(m):
            interp = interp1d(t_vec, us[i,:]) # this doesn't work
            us2[i,:] = interp(ts)
            
    # extract control
    T = us2

    # UPDATE STATES 
    # initialize state vector
    xDot = np.full((6, 1), np.nan)

    # r_dot 
    xDot[0:3] = v

    # v_dot
    xDot[3:6] = T/mass + ge

    if np.issubdtype(r.dtype, np.number):
        if r[2,0] <= -1: # set xDot = 0 if the vehicle hits the ground
            xDot = np.zeros(n,1)
    elif np.issubdtype(r.dtype, np.nan) or any(np.isinf(r)):
        breakpoint()

    return xDot


# ts = 0
# us = np.array([0, 0, 3.4]).reshape(-1,1)
# zs = np.zeros((12,1))
# zs[2] = 0.5
# t_vec = np.linspace(0, 5, 100).reshape(-1,1)

# params = {
#     'm'     : 3,
#     'n'     : 6,
#     'mass'  : 0.35,
#     'ge'    : np.array([0, 0, -9.81]).reshape(-1,1)
# }


# args to be defined by MATLAB
# FIXME can't pass t_vec in via matlab
# xDot = system_dynamics(ts, zs, us, params)
