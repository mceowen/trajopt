import numpy as np
from scipy.integrate import solve_ivp
import trajopt.utils.tools as tools
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import trajopt.library.methods.integrators as integrators

'''
outline of solution_data structure:
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
    problem = trajopt_obj.problem
    method = trajopt_obj.method

    iter_data = method.subprob.iter_data

    #================== KEEP THESE AS REFERENCE FOR NOW TO KNOW WHAT WE NEED FOR PLOTTING PLS ==========================
    mission_params_exclude_list = ['_nonlinear_aero', 'costs', 'custom_modules', 'mission_module', '_get_cost_cnstr_nondim', '_set_custom_params', '_custom_constraints', '_custom_cost', 'trajopt_obj'] 
    model_params_list = ['constraint_config_list', 'flags', 'm', 'n', 'name', 'nz', 'obs', 'u_types', 'z_types']
    method_params_list = ['N', 'N_dens', 'Npm', 'T_init', 'T_max', 'T_min', 'Ts_init', 'conv', 'conv_data', 'cost_init', 
                          'dT_max', 'ddt_max', 'dt_init', 'dt_init', 'dt_max', 'dt_min', 'flags', 'line_guess_u_init',
                          'name', 'n_minus', 'n_plus', 'nl_guess_u_start', 'nl_guess_u_stop', 'solver_opts',
                          'nondim', 't_init', 'nu_init', 'weights', 'z_ind', 'z_init']
    #===================================================================================================================
    n = problem.n
    nondim = method.nondim

    problem_params = problem.params
    method_params = tools.extract_attributes(method, method_params_list)
    params_dict = {**problem_params, **method_params}

    for data in iter_data[1:]:
        
        # get reference trajectory for this iteration (in nondimensional coordinates)
        t_opt = np.asarray(data['t_opt'])
        z_opt = np.asarray(data['z_opt'])
        nu_opt = np.asarray(data["nu_opt"])

        # nonlinear propagation
        t_nl, z_nl, nu_nl = integrators.nonlinear_propagation(t_opt, z_opt, nu_opt, problem, method)

        t_init = method.t_init
        z_init = method.z_init
        nu_init = method.nu_init

        # compute constraints for z_nl, z_opt, name = SUBPLOT , TYPE, group = FIGURE, units
        constraint_data = {}

        for constraint in problem.constraints.get("all"):
            if hasattr(constraint, "compute_constraint_values"):
                name  = constraint.name
                type  = constraint.implement_type
                group = constraint.group

                if group == None:
                    group = name
                
                opt_vals  = constraint.compute_constraint_values(t_opt, z_opt, nu_opt)
                nl_vals   = constraint.compute_constraint_values(t_nl, z_nl, nu_nl)
                init_vals = constraint.compute_constraint_values(t_init, z_init, nu_init)

                output = {
                    "name": name,
                    "type": type,
                    "opt_vals": opt_vals,
                    "nl_vals": nl_vals,
                    "init_vals": init_vals
                }

                constraint_data[group] = {}
                constraint_data[group][name] = output

        # re-dimensionalize all the data (the goal is to keep nondim data internal, user should only have
        # to worry about dimensional data)
        data['t_nl']  = t_nl * nondim.nt
        data['z_nl']  = z_nl @ nondim.M["state"]["nd2d"]
        data['nu_nl'] = nu_nl @ nondim.M["ctrl"]["nd2d"]

        data['t_init']  = t_init * nondim.nt
        data['z_init']  = z_init[:, :n] @ nondim.M["state"]["nd2d"]
        data['nu_init'] = nu_init @ nondim.M["ctrl"]["nd2d"]

        data['t_opt']  = t_opt * nondim.nt
        data['z_opt']  = z_opt[:, :n] @ nondim.M["state"]["nd2d"]
        data['nu_opt'] = nu_opt @ nondim.M["ctrl"]["nd2d"]
        data['constraint_data'] = constraint_data

    return {'iters': iter_data, 'params': params_dict}