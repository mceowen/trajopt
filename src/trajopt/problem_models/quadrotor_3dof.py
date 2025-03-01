from trajopt.defaults.set_defaults import set_params_default
from trajopt.algorithm.cost_func import cost_func

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


def init_params_struct(config):
    # init
    params = set_params_default(config)
    params['case_flag']             = 1    # case1: bank angle only
    params['bools.auto_jac']        = 0    # (1=symbolic jacobians for dynamics, 0=analytical)
    params['bools.auto_jac_aero']   = 0    # (1=symbolic jacobians for aerodynamics, 0=analytical)
    params['bools.auto_jac_cnst']   = 0    # (1=symbolic jacobians for constraints, 0=analytical)
    params['bools.init_ctrl']       = 0

    # physical constants
    params['ge'] = np.array([0, 0, -9.81]) # [m/s^2], grav accel at sea lvl

    # problem params
    params['n'] = 6
    params['m'] = 3

    # time of flight
    params['T_init'] = 10

    ## Cost ##
    params['cost'] = lambda t, z, u: np.sum(u**2) #equivalent to dot product...TODO this will be a method at some point

    ## Path /NFZ constraints ##

    # no fly zones, specified by position and radius [rad]
    if params['bools']['flag_nfz'] == 0:
        xc = np.array([])
        yc = np.array([])
        rc = np.array([])
    elif params['bools']['flag_nfz'] == 1:
        xc = np.array([5])
        yc = np.array([4])
        rc = 2 * np.ones(xc.size)
    elif params['bools']['flag_nfz'] == 2:
        xc = np.array([2.5, 5,  2.5, 5.5,  8,  5.5])  # 5
        yc = np.array([2,   2.5,  5, 5.25, 5.5, 8])   # 4
        rc = np.ones(xc.size)  # 2, 1
    else:
        xc = np.array([])
        yc = np.array([])
        rc = np.array([])

    params['obs']['posc'] = np.array([xc, 
                                      yc])
    params['obs']['rc'] = rc
    

    return params


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

