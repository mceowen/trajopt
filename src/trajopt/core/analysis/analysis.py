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

    method_params_list = ['N', 'N_dens', 'Npm', 'T_init', 'T_max', 'T_min', 'Ts_init', 'conv', 'conv_data', 'cost_init', 
                          'dT_max', 'ddt_max', 'dt_init', 'dt_init', 'dt_max', 'dt_min', 'flags', 'line_guess_u_init',
                          'name', 'n_minus', 'n_plus', 'nl_guess_u_start', 'nl_guess_u_stop', 'solver_opts',
                          'nondim', 't_init', 'nu_init', 'weights', 'z_ind', 'z_init']
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

                if constraint_data.get(group) is None:
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


# ======================================================================
# STANDALONE ANALYSIS
# ======================================================================

def run_standalone_analysis(trajopt_obj):

    # perform the default analysis
    data = perform_default_analysis(trajopt_obj)
    
    # populate scenario_data dict for plotting
    scenario_data = {"autotune": {"mc_data": [data]}}

    return scenario_data

# ======================================================================
# MONTE CARLO ANALYSIS
# ======================================================================

def add_monte_carlo_dispersions(mission_dict, realization):
        for mc_var, mc_disp in realization.items():
            mission_dict[mc_var] = mission_dict[mc_var] + mc_disp

def run_mc_analysis(example_name, nominal_config, gen_mc_variations=1, save_mc_variations=0, save_scenario_data=0, mc_name="mc1", local=False):


    mv_variations = cfg.load_mv_variations(example_name, local=local)

    if gen_mc_variations:
        mc_variations = cfg.gen_mc_variations(example_name, local=local)

        if save_mc_variations:
            np.save(f"data/mc_variations/{mc_name}", mc_variations)
    else:
        mc_variations = np.load(f"data/mc_variations/{mc_name}.npy", allow_pickle=True).item()

    variations = {
        "method": mv_variations,
        "mission": mc_variations
    }

    scenario_data = {}

    # loop through method variations
    for name, method_variation in variations["method"].items():
        
        # initialize method sub-dictionary for scenario_data dict
        scenario_data[name] = {"method_params": {},
                                    'mc_data': [None] * (variations["mission"]["num_variations"] + 1),
                                    }

        cached_subprob = None
        
        # loop through monte-carlo mission parameter realizations (number of runs)
        for run_idx, realization in enumerate(variations["mission"]["realizations"]):
            
            # take in nominal configs
            run_config = copy.deepcopy(nominal_config)

            # set method variations
            run_config["method"] = tools.deep_update(run_config["method"], method_variation)

            # set monte carlo mission variations
            add_monte_carlo_dispersions(run_config["mission"], realization)

            # create trajopt_obj instance
            trajopt_obj = traj.Problem(run_config, cached_subprob)
            
            # run SCP
            trajopt_obj = scp.run_scp(trajopt_obj)

            # perform default analysis on this mc run and store related params
            scenario_data[name]["mc_data"][run_idx] = default_analysis.perform_default_analysis(trajopt_obj)

            # store total time for scp (used to calculate time to converge)
            scenario_data[name]['mc_data'][run_idx]['t_full'] = trajopt_obj.solution['t_full']
            
            # cache subproblem graph to speed up solves
            cached_subprob = None # trajopt_obj.method.subprob

    if save_scenario_data:
        np.save(f"data/scenario_data/{example_name}_{mc_name}", scenario_data)

    return scenario_data, trajopt_obj