import copy
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

    problem_config = problem.config
    method_config = tools.extract_attributes(method, method_params_list)
    params = problem.params
    all_params_dict = {**problem_config, **params, **method_config}

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
                
                # slice state to original dimension (n) to handle augmented states from 
                # continuous time reformulation
                opt_vals  = constraint.compute_constraint_values(t_opt, z_opt[:, :n], nu_opt, params)
                nl_vals   = constraint.compute_constraint_values(t_nl, z_nl[:, :n], nu_nl, params)
                init_vals = constraint.compute_constraint_values(t_init, z_init[:, :n], nu_init, params)

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

    return {'iters': iter_data, 'params': all_params_dict}


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

def get_nested(d, path):
    for part in path.split('.'):
        d = d.get(part, {}) if isinstance(d, dict) else None
        if d is None:
            return None
    return d

def set_nested(d, path, value):
    parts = path.split('.')
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value

def run_mc_analysis(trajopt_obj):
    cfg = trajopt_obj.variation_config
    mission_vars = cfg.get('mission_variations', {})
    mission_config = trajopt_obj.problem.config['mission']
    num = cfg.get('num_mission_variations', 1)
    
    if 'seed' in cfg:
        np.random.seed(cfg['seed'])
    
    nominal = {}
    for path in mission_vars:
        val = get_nested(mission_config, path)
        nominal[path] = np.array(val).copy() if val is not None else None
    
    scenario_data = {"autotune": {"mc_data": [perform_default_analysis(trajopt_obj)]}}
    
    for i in range(num):
        print(f"\n=== MC Run {i+1}/{num} ===")
        for path, spec in mission_vars.items():
            if spec.get("type") == "uniform":
                delta = np.random.uniform(np.array(spec["lb"]), np.array(spec["ub"]))
            else:
                delta = np.random.normal(np.array(spec["mu"]), np.array(spec["sigma"]))
            new_val = nominal[path] + delta
            set_nested(mission_config, path, new_val.tolist() if hasattr(new_val, 'tolist') else new_val)
        
        trajopt_obj.problem.update_from_config(mission_vars.keys(), trajopt_obj.method.nondim)
        trajopt_obj.method.get_initial_guess(trajopt_obj.problem)
        m = trajopt_obj.method
        problem = trajopt_obj.problem
        subprob = m.subprob
        trajopt_obj.method.subprob.iter_data = [{
            "iter_num": 0,
            "z_ref": m.z_init,
            "nu_ref": m.nu_init,
            "dt_ref": m.dt_init,
            "t_ref": m.t_init,
            "conv_data": {
                "vb_ineq": np.zeros((subprob.N, problem.n_ineq)),
                "vb_dyn":  np.zeros((subprob.N - 1, subprob.n_dyn)),
                "vb_term": np.zeros((problem.n_term_total, 1)),
            },
            "weights": copy.deepcopy(m.weights),
        }]
        trajopt_obj.solve()
        scenario_data["autotune"]["mc_data"].append(perform_default_analysis(trajopt_obj))
    
    for path, val in nominal.items():
        if val is not None:
            set_nested(mission_config, path, val.tolist() if hasattr(val, 'tolist') else val)
    
    return scenario_data