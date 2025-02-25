def test_func():
    return "Hello, World!"

# Skye Mceowen
# Feb. 17th, 2024
# 2D circular NFZ
# Single integrator linear dynamics

# TODO consolidate imports 
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
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
    m       = int( params['m'] )
    n       = int( params['n'] )
    mass    = params['mass']
    ge      = params['ge']

    # extract states
    r = zs[0:3]
    v = zs[3:6]

    # extract controls 
    if t_vec is None:
        us2 = us
    else:
        us2 = np.empty(m)
        for i in range(m):
            interp = interp1d(t_vec, us[i,:]) # does this work?
            us2[i] = interp(ts)
            
    # extract control
    T = us2

    # compute velocity and acceleration
    xDot = np.empty(6) # initialize
    xDot[0:3] = v
    xDot[3:6] = T/mass + ge

    if np.issubdtype(r.dtype, np.number):
        if r[2] <= -1: # set xDot = 0 if the vehicle hits the ground
            xDot = np.zeros(n)
    elif np.issubdtype(r.dtype, np.nan) or any(np.isinf(r)):
        breakpoint()
        
    return xDot


def init_params_struct():
    
    pass


# Potential class structure ?
class ocp:
    def __init__(self, config):
        self.config = config
        pass

    def init_params(self):
        # make params dict
        params = {}
        return params

    def nonlinear_initial_guess(self):
        pass

