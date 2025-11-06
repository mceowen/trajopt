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

    # Extract non-function parameters from mission, model, and method
    mission_params = tools.extract_non_function_params(problem.mission, exclude=['mission_module'])
    model_params = tools.extract_non_function_params(problem.model, exclude=['model_module'])
    
    method_exclude = ['method_module', 'subprob', 'Ak', 'Ak_ind', 'Ak_ind_jax', 'Bk', 'Bk_ind', 'Bk_ind_jax', 'Bkp', 'Bkp_ind', 'Bkp_ind_jax',
                      'Sk', 'Sk_ind', 'Sk_ind_jax', 'lds0', 'lds0_size', 'lds0_size_jax', 'z_ind_jax']
    method_params = tools.extract_non_function_params(problem.method, exclude=method_exclude)

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

    return {'iters': iters, 'params': params_dict}


# contents of iters dict from subproblem
# self.iter_data: List[Dict[str, Any]] = [{
#     "iter_num": 0,  # init only (no outputs yet)
#     "zs_ref": problem.method.zs_init,
#     "us_ref": problem.method.us_init,
#     "dts_ref": problem.method.dts_init,
#     "ts_ref": problem.method.ts_init,
#     "conv_data": {
#         "vb_path": np.zeros((self.N, mission.n_path)),
#         "vb_nfz":  np.zeros((self.N, mission.n_nfz)),
#         "vb_aux":  np.zeros((self.N, self.n_aux)),
#         "vb_dyn":  np.zeros((self.N - 1, self.nz)),
#         "vb_term": np.zeros((self.n_term, 1)),
#     },
#     "weights": problem.method.weights,
# }]
