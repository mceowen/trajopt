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


def nonlinear_initial_guess(us_range, params):
    """
    Generate a nonlinear initial guess for trajectory and control.

    Parameters:
    us_range (numpy.ndarray): Range of control inputs.
    params (dict): Dictionary containing parameters.

    Returns:
    dict: Updated params with initial guesses for trajectory and control.
    """
    # Initialization trajectory
    params['dt_init'] = (params['T_init'] / (params['N'] - 1)) * np.ones(params['N'] - 1)
    params['Ts_init'] = params['T_init'] / params['nondim']['nt']
    params['dts_init'] = params['dt_init'] / params['nondim']['nt']
    ts_init = np.cumsum(np.concatenate(([0], params['dts_init'])))
    
    # Initial control
    us_init = np.array([np.linspace(us_range[i, 0], us_range[i, 1], params['N']) for i in range(len(us_range))])
    
    # Propagate initial trajectory from nonlinear simulation
    odesettings = {'atol': 1E-12, 'rtol': 1E-12}
    # sol = solve_ivp(lambda t, x: system_dynamics(t, x, us_init, params, ts_init), [ts_init[0], ts_init[-1]], params['z0s'], t_eval=ts_init, **odesettings)
    sol = solve_ivp(system_dynamics, [ts_init[0], ts_init[-1]], params['z0s'], args=(us_init, params, ts_init),t_eval=ts_init, **odesettings)

    zs_init = sol.y
    
    # Create initial state and control vector
    params['ts_init'] = ts_init
    params['zs_init'] = zs_init
    params['us_init'] = us_init
    
    return params


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

