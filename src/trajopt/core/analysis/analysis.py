import copy
import numpy as np
import trajopt.utils.tools as tools
import jax
import trajopt.methods.scp.integrators as integrators
import trajopt.methods.scp.initial_guess as guess
from trajopt.core.indexing.index_map import IndexMap
from trajopt.core.problem import Problem
from trajopt.core.solution_method import SolutionMethod
from trajopt.utils.tools import recursive_attrdict, AttrDict
jax.config.update("jax_enable_x64", True)


def _eval_per_phase(eval_fn, z, nu, problem, params_dict, t_nodes=None):
    if problem.phases is None:
        return eval_fn(z, nu, params_dict)

    N_pts = z.shape[0]
    _, t, _ = problem.index_map.unpack_z(z)
    t = np.asarray(t).reshape(-1)

    parts = []
    for phase in problem.phases:
        if t_nodes is None:
            s, e = phase.start, min(phase.end, N_pts)
        else:
            t_s = t_nodes[phase.start]
            t_e = t_nodes[min(phase.end, len(t_nodes) - 1)] if phase.end < len(t_nodes) else t[-1] + 1
            s   = int(np.searchsorted(t, t_s))
            e   = int(np.searchsorted(t, t_e)) if phase.end < len(t_nodes) else N_pts

        if e <= s:
            continue
        parts.append((s, e, eval_fn(z[s:e], nu[s:e], {**params_dict, **phase.params})))

    if not parts:
        return eval_fn(z, nu, params_dict)

    first, N0 = parts[0][2], parts[0][1] - parts[0][0]
    stitched = {}
    for key, val in first.items():
        if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == N0:
            full = np.zeros((N_pts, *val.shape[1:]))
            for s, e, r in parts:
                full[s:e] = r[key]
            stitched[key] = full
        else:
            stitched[key] = val

    return stitched


'''
outline of solution_data structure:
results = {
    "method1": {
        "runs": [{"iters": {}, "params": {}}, {"iters": {}, "params": {}}, ...]
    },

    "method2": {
        "runs": []
    }, 
}
'''

ITER_DATA_KEYS_TO_KEEP = {
    "iter_num", "converged", "cost", "solve_time", "prop_time", "parse_time", "t_full",
    "t_opt", "z_opt", "nu_opt", "dt_opt", "T_opt",
    "t_nl", "z_nl", "nu_nl", "t_init", "z_init", "nu_init",
    "constraint_data", "trajectory_data", "conv_data", "W", "dual", "penalty",
}

METHOD_DATA_KEYS_TO_KEEP = {
    'time_grid', 'N_dens', 'Npm', 'T_init', 'T_max', 'T_min', 'Ts_init', 'conv', 'conv_data', 'cost_init', 
    'dT_max', 'ddt_max', 'dt_max', 'dt_min', 'flags', 'line_guess_u_init',
    'name', 'n_minus', 'n_plus', 'nl_guess_u_start', 'nl_guess_u_stop', 'solver_opts',
    'nondim', 'initial_guess', 'penalty', 'z_ind'
}

def perform_analysis(trajopt_obj, trim=True, compute_iters=False):
    problem     = trajopt_obj.problem
    method      = trajopt_obj.method

    n_x         = problem.index_map.n.state
    n_u        = problem.index_map.n.control
    params      = problem.params
    params_dict = tools.recursive_to_dict(params)
    iter_data   = method.subproblem.iter_data
    nondim      = method.nondim

    if compute_iters == True:
        selected_iter_data = iter_data[1:]
    else:
        selected_iter_data = [iter_data[-1]]

    for data in selected_iter_data:
        
        z_opt = np.asarray(data['z_opt'])
        nu_opt = np.asarray(data['nu_opt'])

        idx = problem.index_map.indices
        x_opt = z_opt[:, idx.z.state]
        t_opt = z_opt[:, idx.z.time].squeeze(-1)
        u_opt = nu_opt[:, idx.nu.control]

        t_nl, z_nl, nu_nl = integrators.propagate_znu(z_opt[0], nu_opt, problem, method)
        x_nl = z_nl[:, idx.z.state]
        u_nl = nu_nl[:, idx.nu.control]

        z_init = method.initial_guess.z
        nu_init = method.initial_guess.nu
        x_init = z_init[:, idx.z.state]
        t_init = z_init[:, idx.z.time].squeeze(-1)
        u_init = nu_init[:, idx.nu.control]

        # compute constraints for z_nl, z_opt, name = SUBPLOT , TYPE, group = FIGURE, units
        constraint_data = AttrDict({})
        trajectory_data = AttrDict({})

        # compute constraint values
        for constraint in problem.constraints.get():
            if hasattr(constraint, "compute_constraint_values"):
                name  = constraint.name
                cnstr_type  = constraint.type
                group = constraint.group

                if group == None:
                    group = name
                
                eval_fn   = constraint.compute_constraint_values
                opt_vals  = _eval_per_phase(eval_fn,z_opt,  nu_opt,  problem, params_dict)
                nl_vals   = _eval_per_phase(eval_fn, z_nl,   nu_nl,   problem, params_dict, t_nodes=t_opt)
                init_vals = _eval_per_phase(eval_fn,z_init, nu_init, problem, params_dict)

                output = AttrDict({
                    "name": name,
                    "type": cnstr_type,
                    "opt_vals": opt_vals,
                    "nl_vals": nl_vals,
                    "init_vals": init_vals
                })

                if constraint_data.get(group) is None:
                    constraint_data[group] = AttrDict({})
                
                constraint_data[group][name] = output
        
        # dimensional augmented z/nu for trajectory value evaluation
        z_opt_d  = z_opt.copy()
        z_opt_d[:, idx.z.state] = x_opt @ nondim.M.state.nd2d
        z_opt_d[:, idx.z.time]  = z_opt[:, idx.z.time] * nondim.time_scale
        nu_opt_d = nu_opt.copy()
        nu_opt_d[:, idx.nu.control] = u_opt @ nondim.M.control.nd2d

        z_nl_d  = z_nl.copy()
        z_nl_d[:, idx.z.state] = x_nl @ nondim.M.state.nd2d
        z_nl_d[:, idx.z.time]  = z_nl[:, idx.z.time] * nondim.time_scale
        nu_nl_d = nu_nl.copy()
        nu_nl_d[:, idx.nu.control] = u_nl @ nondim.M.control.nd2d

        z_init_d  = z_init.copy()
        z_init_d[:, idx.z.state] = x_init @ nondim.M.state.nd2d
        z_init_d[:, idx.z.time]  = z_init[:, idx.z.time] * nondim.time_scale
        nu_init_d = nu_init.copy()
        nu_init_d[:, idx.nu.control] = u_init @ nondim.M.control.nd2d

        t_opt_d = t_opt * nondim.time_scale

        # compute general trajectory values (not necessarily constraints as specified in the problem)
        for trajectory in problem.trajectories.get():
            if hasattr(trajectory, "compute_trajectory_values"):
                name = trajectory.name
                traj_type = trajectory.type
                group = trajectory.group

                if group == None:
                    group = name

                eval_fn   = trajectory.compute_trajectory_values
                opt_vals  = _eval_per_phase(eval_fn, z_opt_d,  nu_opt_d,  problem, params_dict)
                nl_vals   = _eval_per_phase(eval_fn, z_nl_d,   nu_nl_d,   problem, params_dict, t_nodes=t_opt_d)
                init_vals = _eval_per_phase(eval_fn, z_init_d, nu_init_d, problem, params_dict)

                output = AttrDict({
                    "name": name,
                    "type": traj_type,
                    "opt_vals": opt_vals,
                    "nl_vals": nl_vals,
                    "init_vals": init_vals,
                    "title": getattr(trajectory, "title", None),
                    "xlabel": getattr(trajectory, "xlabel", None),
                    "ylabel": getattr(trajectory, "ylabel", None),
                    "tick_nbins": getattr(trajectory, "tick_nbins", None),
                    "markers": getattr(trajectory, "markers", None),
                })

                if trajectory_data.get(group) is None:
                    trajectory_data[group] = AttrDict({})
                
                trajectory_data[group][name] = output

        # re-dimensionalize all the data
        data['t_nl']  = t_nl * nondim.time_scale
        data['z_nl']  = x_nl @ nondim.M.state.nd2d
        data['nu_nl'] = u_nl @ nondim.M.control.nd2d

        data['t_init']  = t_init * nondim.time_scale
        data['z_init']  = x_init @ nondim.M.state.nd2d
        data['nu_init'] = u_init @ nondim.M.control.nd2d

        data['t_opt']  = t_opt * nondim.time_scale
        data['z_opt']  = x_opt @ nondim.M.state.nd2d
        data['nu_opt'] = u_opt @ nondim.M.control.nd2d
        data['constraint_data'] = constraint_data
        data['trajectory_data'] = trajectory_data

    if trim:
        iters_out = [tools.trim_dict(rec, ITER_DATA_KEYS_TO_KEEP) for rec in iter_data]
    else:
        iters_out = iter_data
    
    return AttrDict({'iters': iters_out})

# ======================================================================
# STANDALONE ANALYSIS
# ======================================================================

def run_standalone_analysis(trajopt_obj, show_iters = True):
    config  = trajopt_obj.config.method
    name = config.get("name", "method1")
    return recursive_attrdict({name: {"runs": [perform_analysis(trajopt_obj, compute_iters=show_iters)]}})

# ======================================================================
# MISSION AND METHOD VARIATION ANALYSIS
# ======================================================================

def update_problem_with_variations(problem, realized_mission_variations_flat, config, nondim):
    for path, value in realized_mission_variations_flat.items():
        
        if path.startswith("constraints."):
            # constraints.initial_state.value: 
            constraint_name = path.split(".")[1]
            path_to_spec = ".".join(path.split(".")[2:])

            constraint = problem.constraints.get(name=constraint_name)[0]
            tools.set_attr_from_path(constraint, path_to_spec, value)
        
        if path.startswith("costs."):
            cost_name = path.split(".")[1]

            if cost_name in problem.costs.names:
                cost = problem.costs.get(name=cost_name)[0]
                path_to_spec = ".".join(path.split(".")[2:])
                tools.set_attr_from_path(cost, path_to_spec, value)
        
        if path.startswith("params."):
            path_to_param = path.split("params.")[1]
            tools.set_attr_from_path(problem.params, path_to_param, value)
        
    # Nondimensionalize constraints that were just updated 
    updated_constraint_names = set()
    for path in realized_mission_variations_flat.keys():
        if path.startswith("constraints."):
            constraint_name = path.split(".")[1]
            updated_constraint_names.add(constraint_name)
    for constraint in problem.constraints.get():
        if constraint.name in updated_constraint_names:
            constraint.nondim_constraint(nondim)

def run_mc_analysis(trajopt_obj):

    nominal_config = trajopt_obj.config
    
    seed = nominal_config.get("seed", 0)

    orig_problem = trajopt_obj.problem
    orig_method  = trajopt_obj.method
    results = AttrDict({})

    for method_name, method_var_config in nominal_config.variations.method.items():

        np.random.seed(seed)

        # start with nominal config
        config_for_current_method = copy.deepcopy(nominal_config)

        # extract method variations
        method_var_config_flat = tools.flatten_dict(method_var_config)

        # apply method variations to current config
        for path, val in method_var_config_flat.items():
            if path.startswith("constraints.") or path.startswith("costs."):
                tools.set_from_path(config_for_current_method.problem, path, val)
            else:
                tools.set_from_path(config_for_current_method.method, path, val)

        # build a new problem and method for method variations
        index_map = IndexMap(config_for_current_method)
        problem   = Problem(config_for_current_method, index_map=index_map)
        method    = SolutionMethod(problem, config_for_current_method, index_map=index_map)
        
        trajopt_obj.index_map = index_map
        trajopt_obj.problem = problem
        trajopt_obj.method  = method
        trajopt_obj.solve()

        subprob = method.subproblem
        runs = [perform_analysis(trajopt_obj)]

        mission_var_config      = config_for_current_method.variations.mission
        mission_var_config_flat = tools.flatten_dict(mission_var_config)

        # mission variations
        realized_mission_variations_flat = AttrDict({})
        for i in range(config_for_current_method.variations.mission.num):
            print(f"\n=== method: {method_name} | run: {i+1} / {config_for_current_method.variations.mission.num} ===")
            
            for path_to_spec, spec in mission_var_config_flat.items():
                if spec == "uniform":
                    path = path_to_spec.replace(".type", "")
                    random_variable_spec = tools.get_from_path(mission_var_config, path)

                    lb = random_variable_spec["lb"]
                    ub = random_variable_spec["ub"]
                    
                    delta = np.random.uniform(lb, ub)

                    new_val = tools.get_from_path(config_for_current_method.problem.mission, path) + delta
                    realized_mission_variations_flat[path] = new_val
                
                if spec == "normal":
                    path = path_to_spec.replace(".type", "")
                    random_variable_spec = tools.get_from_path(mission_var_config, path)
                    mu = random_variable_spec["mu"]
                    sigma = random_variable_spec["sigma"]

                    delta = np.random.normal(mu, sigma)

            update_problem_with_variations(problem, realized_mission_variations_flat, config_for_current_method, method.nondim)
            guess.set_initial_guess(problem, method)

            n_N    = problem.index_map.N.time_grid
            n_neq  = problem.index_map.n.nonconvex_inequality
            n_dyn  = problem.index_map.n.dynamics
            n_term = problem.index_map.n.term_total

            # Reset iter_data like the first run: W and dual None so first iteration uses configure_penalty_weights
            subprob.iter_data = [tools.recursive_attrdict({
                "iter_num": 0,
                "z_opt":  method.initial_guess.z,
                "nu_opt": method.initial_guess.nu,
                "conv_data": {
                    "vb_ineq": np.zeros((n_N, n_neq)),
                    "vb_dyn":  np.zeros((n_N - 1, n_dyn)),
                    "vb_terminal": np.zeros((n_term, 1)),
                },
                "W": None,
                "dual": None,
                "penalty": copy.deepcopy(method.penalty),
            })]
            trajopt_obj.solve()
            runs.append(perform_analysis(trajopt_obj, compute_iters=False))

        results[method_name] = AttrDict({"runs": runs})

    trajopt_obj.problem = orig_problem
    trajopt_obj.method  = orig_method
    
    return results