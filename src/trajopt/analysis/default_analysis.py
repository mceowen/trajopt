import numpy as np
from scipy.integrate import solve_ivp
import trajopt.utils.tools as tools
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import trajopt.core.modules.method.integrators as integrators

'''
outline of plt_data structure
scenario_data = {
    "method1": {
        "mc_data": [{"iters": {}, "params": {}}, {"iters": {}, "params": {}}, ...]
    },

    "method2": {
        "run_data": []
    }, 
}
'''


def perform_default_analysis(trajopt_obj):

    iter_data = trajopt_obj.method.subprob.iter_data
    n = trajopt_obj.model.n
    m = trajopt_obj.model.m
    N = trajopt_obj.method.N
    nondim = trajopt_obj.method.nondim

    mission_params_exclude_list = ['_nonlinear_aero', 'costs', 'custom_modules', 'mission_module', '_get_cost_cnstr_nondim', '_set_custom_params', '_custom_constraints', '_custom_cost', 'trajopt_obj'] 
    model_params_list = ['constraint_config_list', 'flags', 'm', 'n', 'name', 'nz', 'obs', 'u_types', 'z_types']
    method_params_list = ['N', 'N_dens', 'Npm', 'T_init', 'T_max', 'T_min', 'Ts_init', 'conv', 'conv_data', 'cost_init', 
                          'dT_max', 'ddt_max', 'dt_init', 'dt_init', 'dt_max', 'dt_min', 'flags', 'line_guess_u_init',
                          'name', 'n_minus', 'n_plus', 'nl_guess_u_start', 'nl_guess_u_stop', 'solver_opts',
                          'nondim', 't_init', 'nu_init', 'weights', 'z_ind', 'z_init']

    mission_params = tools.extract_attributes_exclude(trajopt_obj.mission, exclude=mission_params_exclude_list)
    model_params   = tools.extract_attributes(trajopt_obj.model, model_params_list)
    method_params  = tools.extract_attributes(trajopt_obj.method, method_params_list)

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

    for data in iter_data[1:]:
        
        # get reference trajectory for this iteration (in nondimensional coordinates)
        t_opt = np.asarray(data['t_opt'])
        z_opt = np.asarray(data['z_opt'])
        nu_opt = np.asarray(data["nu_opt"])
        
        # create dense time grid for this iteration based on its reference trajectory time span
        t_dense = np.linspace(t_opt[0], t_opt[-1], N_dense)
        
        # create dense control interpolation for this iteration
        nu_opt_dense = np.hstack([np.interp(t_dense, t_opt, nu_opt[:, i]).reshape((-1, 1)) for i in range(m)])
        u_ref_dense = nu_opt_dense
        
        # TODO: need to move this to an integrator module
        # choose integrator based on jax_dyn flag
        use_jax = trajopt_obj.method.flags.get("jax_dyn", 0)
        
        z_opt_np = np.asarray(z_opt)
        
        if use_jax:
            # use JAX-based RK4 propagation
            z_nl = integrators.propagate_rk4_dense(z_opt_np[0, :n], nu_opt, t_opt, t_dense, trajopt_obj)
            
            data['t_nl'] = t_dense * nondim['nt']
            data['z_nl'] = z_nl @ nondim['M']['state']['nd2d']
            data['nu_nl'] = u_ref_dense @ nondim['M']['ctrl']['nd2d']
        else:
            # use scipy solve_ivp
            def FOH_dynamics(t, z, nu_opt, t_opt):
                """First-order hold dynamics for RK45 integration."""
                # Interpolate control at time t (each control dimension separately)
                u_t = np.array([np.interp(t, t_opt, nu_opt[:, i]) for i in range(m)])
                # Call model dynamics
                return trajopt_obj.model.dynamics(t, z, u_t)
            
            sol = solve_ivp(
                FOH_dynamics,
                [t_opt[0], t_opt[-1]],
                z_opt_np[0, :n],
                args=(nu_opt, t_opt),
                t_eval=t_dense,
                method='RK45',
                **odesettings
            )
            
            data['t_nl'] = t_dense * nondim['nt']
            data['z_nl'] = sol.y.T @ nondim['M']['state']['nd2d']
            data['nu_nl'] = u_ref_dense @ nondim['M']['ctrl']['nd2d']

        # data['t_opt'] = data['t_opt'] * nondim['nt']
        # data['z_opt'] = data['z_opt'][:, :n] @ nondim['M']['state']['nd2d']
        # data['u_ref'] = data["nu_opt"] @ nondim['M']['ctrl']['nd2d']

        data['t_init'] = trajopt_obj.method.t_init * nondim['nt']
        data['z_init'] = trajopt_obj.method.z_init[:, :n] @ nondim['M']['state']['nd2d']
        data['nu_init'] = trajopt_obj.method.nu_init @ nondim['M']['ctrl']['nd2d']

        data['t_opt'] = data["t_opt"] * nondim['nt']
        data['z_opt'] = data["z_opt"][:, :n] @ nondim['M']['state']['nd2d']
        data['nu_opt'] = data["nu_opt"] @ nondim['M']['ctrl']['nd2d']

    return {'iters': iter_data, 'params': params_dict}