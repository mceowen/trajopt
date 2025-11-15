import numpy as np
from scipy.integrate import solve_ivp
import trajopt.utils.tools as tools

'''
outline of plt_data structure
scenario_data = {
    "method1": {
        "run_data": [{"iters": {}, "params": {}}, {"iters": {}, "params": {}}, ...]
    },

    "method2": {
        "run_data": []
    }, 
}
'''

def perform_default_analysis(problem):

    iter_data = problem.method.subprob.iter_data
    n = problem.model.n
    m = problem.model.m
    N = problem.method.N
    nondim = problem.method.nondim

    mission_params_exclude_list = ['_nonlinear_aero', 'costs', 'custom_modules', 'mission_module', '_get_cost_cnstr_nondim', '_set_custom_params', '_custom_constraints', '_custom_cost', 'problem'] 
    model_params_list = ['constraint_config_list', 'flags', 'm', 'n', 'name', 'nz', 'obs', 'u_types', 'z_types']
    method_params_list = ['N', 'N_dens', 'Npm', 'T_init', 'T_max', 'T_min', 'Ts_init', 'conv', 'conv_data', 'cost_init', 
                          'dT_max', 'ddt_max', 'dt_init', 'dt_init', 'dt_max', 'dt_min', 'flags', 'line_guess_u_init',
                          'name', 'n_minus', 'n_plus', 'nl_guess_u_start', 'nl_guess_u_stop', 'solver_opts',
                          'nondim', 't_init', 'nu_init', 'weights', 'z_ind', 'z_init']

    mission_params = tools.extract_attributes_exclude(problem.mission, exclude=mission_params_exclude_list)
    model_params   = tools.extract_attributes(problem.model, model_params_list)
    method_params  = tools.extract_attributes(problem.method, method_params_list)

    model_params['n'] = n
    model_params['m'] = m
    
    method_params['N'] = N
    method_params['nondim'] = nondim

    params_dict = {
        'mission': mission_params,
        'model': model_params,
        'method': method_params
    }

    odesettings = {"atol": 1e-12, "rtol": 1e-12}
    N_dense = 20 * N

    for data in iter_data:
        
        # get reference trajectory for this iteration (in nondimensional coordinates)
        t_ref = np.asarray(data['t_ref'])
        z_ref = np.asarray(data['z_ref'])
        nu_ref = np.asarray(data['us_ref'])
        
        # create dense time grid for this iteration based on its reference trajectory time span
        t_dense = np.linspace(t_ref[0], t_ref[-1], N_dense)
        t_dense = t_dense * nondim['nt']
        
        # create dense control interpolation for this iteration
        nu_ref_dense = np.hstack([np.interp(t_dense, t_ref, nu_ref[:, i]).reshape((-1, 1)) for i in range(m)])
        u_ref_dense = nu_ref_dense @ nondim['M']['ctrl']['nd2d']
        
        def FOH_dynamics(t, z, nu_ref, t_ref):
            """First-order hold dynamics for RK45 integration."""
            # Interpolate control at time t (each control dimension separately)
            u_t = np.array([np.interp(t, t_ref, nu_ref[:, i]) for i in range(m)])
            # Call model dynamics
            return problem.model.dynamics(t, z, u_t)
        
        # nonlinear propagation
        z_ref_np = np.asarray(z_ref)
        sol = solve_ivp(
            FOH_dynamics,
            [t_ref[0], t_ref[-1]],
            z_ref_np[0, :n],
            args=(nu_ref, t_ref),
            t_eval=t_dense,
            method='RK45',
            **odesettings
        )
        
        data['t_nl'] = t_dense
        data['z_nl'] = sol.y.T @ nondim['M']['state']['nd2d']
        data['u_nl'] = u_ref_dense

        data['t_ref'] = data['t_ref'] * nondim['nt']
        data['z_ref'] = data['z_ref'][:, :n] @ nondim['M']['state']['nd2d']
        data['u_ref'] = data['us_ref'] @ nondim['M']['ctrl']['nd2d']

        if 'ts' in data:
            data['t'] = data['ts'] * nondim['nt']
            data['z'] = data['zs'][:, :n] @ nondim['M']['state']['nd2d']
            data['u'] = data['us'] @ nondim['M']['ctrl']['nd2d']

    return {'iters': iter_data, 'params': params_dict}