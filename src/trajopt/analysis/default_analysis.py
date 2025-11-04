import numpy as np
from scipy.integrate import solve_ivp
import types

def initialize_plot_struct():
    plt = {}
    plt['scenario1'] = {}
    plt['scenario1']['method1'] = {}
    plt['scenario1']['method1']['params'] = {}
    plt['scenario1']['method1']['runs'] = []
    return plt

def extract_non_function_params(obj):
    """
    Extract all non-function parameters from an object.
    Returns a dictionary of attribute name -> value pairs, excluding:
    - Methods/functions
    - Private attributes starting with '__'
    - The 'problem' attribute 
    """
    params = {}
    for attr_name in dir(obj):
        # skip private attributes and methods
        if attr_name.startswith('__'):
            continue
        
        # skip the 'problem' attribute to avoid circular references
        if attr_name == 'problem':
            continue
        
        try:
            attr_value = getattr(obj, attr_name)
            # skip if it's a function/method
            if isinstance(attr_value, types.MethodType) or isinstance(attr_value, types.FunctionType):
                continue
            # skip if it's a callable
            if callable(attr_value) and not isinstance(attr_value, (np.ndarray, list, dict, str, int, float)):
                continue
            
            params[attr_name] = attr_value
        except:
            # skip attributes that can't be accessed
            continue
    
    return params

def perform_default_analysis(problem, plt=None, scenario='scenario1', method='method1'):
    if plt is None:
        plt = initialize_plot_struct()
    
    if scenario not in plt:
        plt[scenario] = {}
    if method not in plt[scenario]:
        plt[scenario][method] = {'params': {}, 'runs': []}
    
    iter_data = problem.method.subprob.iter_data
    n = problem.model.n
    m = problem.model.m
    N = problem.method.N
    nondim = problem.method.nondim

    # Extract non-function parameters from mission, model, and method
    mission_params = extract_non_function_params(problem.mission)
    model_params = extract_non_function_params(problem.model)
    method_params = extract_non_function_params(problem.method)
    
    model_params['n'] = n
    model_params['m'] = m
    
    method_params['N'] = N
    method_params['nondim'] = nondim

    plt[scenario][method]['params'] = {
        'mission': mission_params,
        'model': model_params,
        'method': method_params
    }

    odesettings = {"atol": 1e-12, "rtol": 1e-12}
    N_dense = 20 * N

    iters = []
    for data in iter_data:
        iter_dict = {}
        
        # get reference trajectory for this iteration (in nondimensional coordinates)
        ts_ref = np.asarray(data['ts_ref'])
        zs_ref = np.asarray(data['zs_ref'])
        us_ref = np.asarray(data['us_ref'])
        
        # create dense time grid for this iteration based on its reference trajectory time span
        ts_dense = np.linspace(ts_ref[0], ts_ref[-1], N_dense)
        t_dense = ts_dense * nondim['nt']
        
        # create dense control interpolation for this iteration
        us_ref_dense = np.hstack([np.interp(ts_dense, ts_ref, us_ref[:, i]).reshape((-1, 1)) for i in range(m)])
        u_ref_dense = us_ref_dense @ nondim['M']['ctrl']['nd2d']
        
        # nonlinear propagation
        zs_ref_np = np.asarray(zs_ref)
        sol = solve_ivp(
            problem.model.dynamics,
            [ts_ref[0], ts_ref[-1]],
            zs_ref_np[0, :n],
            args=(us_ref, ts_ref),
            t_eval=ts_dense,
            method='RK45',
            **odesettings
        )
        
        iter_dict['t_nl'] = t_dense
        iter_dict['z_nl'] = sol.y.T @ nondim['M']['state']['nd2d']
        iter_dict['u_nl'] = u_ref_dense

        iter_dict['t_ref'] = data['ts_ref'] * nondim['nt']
        iter_dict['z_ref'] = data['zs_ref'][:, :n] @ nondim['M']['state']['nd2d']
        iter_dict['u_ref'] = data['us_ref'] @ nondim['M']['ctrl']['nd2d']

        if 'ts' in data:
            iter_dict['t'] = data['ts'] * nondim['nt']
            iter_dict['z'] = data['zs'][:, :n] @ nondim['M']['state']['nd2d']
            iter_dict['u'] = data['us'] @ nondim['M']['ctrl']['nd2d']
            iter_dict['weights'] = data['weights']
            iter_dict['conv_data'] = {k: v for k, v in data['conv_data'].items() if k != 'soln'}

        iters.append(iter_dict)

    plt[scenario][method]['runs'].append({'iters': iters, 'delta_params': {}})
    return plt
