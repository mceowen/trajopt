import copy

import jax
import jax.numpy as jnp
import numpy as np

import trajopt.methods.scp.initial_guess as guess
from trajopt.core.indexing.index_map import IndexMap
from trajopt.core.problem import Problem
import trajopt.methods.scp.scp as scp
from trajopt.methods.scp import discretize, integrators
from trajopt.utils import tools
from trajopt.utils.tools import AttrDict, recursive_attrdict

jax.config.update("jax_enable_x64", True)

"""
outline of solution_data structure:
results = {
    "method1": {
        "runs": [{"iters": {}, "params": {}}, {"iters": {}, "params": {}}, ...]
    },

    "method2": {
        "runs": []
    },
}
"""

def perform_analysis(trajopt_obj, compute_iters=False):
    problem          = trajopt_obj.problem
    method           = trajopt_obj.method

    params           = problem.params
    iter_data_list   = method.iter_data_list
    nondim           = problem.nondim

    if compute_iters:
        selected_iter_data = iter_data_list
    else:
        selected_iter_data = [iter_data_list[-1]]

    for iter_idx, iter_data in enumerate(selected_iter_data):
        z_opt = np.asarray(iter_data["z_opt"])
        nu_opt = np.asarray(iter_data["nu_opt"])

        idx = problem.index_map.indices
        x_opt = z_opt[:, idx.z.state]
        t_opt = z_opt[:, idx.z.time].squeeze(-1)
        u_opt = nu_opt[:, idx.nu.control]

        N = z_opt.shape[0]
        if getattr(method.flags, 'discretize', 'ms') == 'ps':
            _, etau, _, _ = discretize.compute_ps_differentiation_matrix(N - 1)
            tau_nodes = (etau + 1.0) / 2.0
        else:
            tau_nodes = np.linspace(0.0, 1.0, N)

        tau_ref = jnp.asarray(tau_nodes)
        nu_ref  = jnp.asarray(nu_opt)
        def nu_fn(z, tau):
            k = jnp.clip(jnp.searchsorted(tau_ref, tau, side='right') - 1, 0, N - 2)
            a = (tau - tau_ref[k]) / (tau_ref[k + 1] - tau_ref[k])
            return (1 - a) * nu_ref[k] + a * nu_ref[k + 1]

        dynamics = problem.constraints.get(type="dynamics")[0].fcn
        _, z_nl, nu_nl = integrators.propagate_from_nodes(
            z_opt, tau_nodes, nu_fn, dynamics, problem.params,
        )
        t_nl  = z_nl[:, idx.z.time].squeeze(-1)
        x_nl  = z_nl[:, idx.z.state]
        u_nl  = nu_nl[:, idx.nu.control]

        z_init = method.initial_guess.z
        nu_init = method.initial_guess.nu
        x_init = z_init[:, idx.z.state]
        t_init = z_init[:, idx.z.time].squeeze(-1)
        u_init = nu_init[:, idx.nu.control]

        z_init_dense = method.initial_guess.z_dense
        nu_init_dense = method.initial_guess.nu_dense
        x_init_dense = z_init_dense[:, idx.z.state]
        t_init_dense = z_init_dense[:, idx.z.time].squeeze(-1)
        u_init_dense = nu_init_dense[:, idx.nu.control]

        # compute constraints for z_nl, z_opt, name = SUBPLOT , TYPE, group = FIGURE, units
        trajectory_data = AttrDict({})
        
        # dimensional augmented z/nu for trajectory value evaluation
        z_opt_d = z_opt.copy()
        z_opt_d[:, idx.z.state] = x_opt @ nondim.M.state.nd2d
        z_opt_d[:, idx.z.time] = z_opt[:, idx.z.time] * nondim.time_scale
        nu_opt_d = nu_opt.copy()
        nu_opt_d[:, idx.nu.control] = u_opt @ nondim.M.control.nd2d

        z_nl_d = z_nl.copy()
        z_nl_d[:, idx.z.state] = x_nl @ nondim.M.state.nd2d
        z_nl_d[:, idx.z.time] = z_nl[:, idx.z.time] * nondim.time_scale
        nu_nl_d = nu_nl.copy()
        nu_nl_d[:, idx.nu.control] = u_nl @ nondim.M.control.nd2d

        z_init_d = z_init.copy()
        z_init_d[:, idx.z.state] = x_init @ nondim.M.state.nd2d
        z_init_d[:, idx.z.time] = z_init[:, idx.z.time] * nondim.time_scale
        nu_init_d = nu_init.copy()
        nu_init_d[:, idx.nu.control] = u_init @ nondim.M.control.nd2d

        z_init_dense_d = z_init_dense.copy()
        z_init_dense_d[:, idx.z.state] = x_init_dense @ nondim.M.state.nd2d
        z_init_dense_d[:, idx.z.time] = z_init_dense[:, idx.z.time] * nondim.time_scale
        nu_init_dense_d = nu_init_dense.copy()
        nu_init_dense_d[:, idx.nu.control] = u_init_dense @ nondim.M.control.nd2d

        # compute general trajectory values (not necessarily constraints as specified in the problem)
        for trajectory in problem.trajectories.get():
            if hasattr(trajectory, "compute_trajectory_values"):
                name = trajectory.name
                traj_type = trajectory.type
                group = trajectory.group

                opt_vals      = trajectory.compute_trajectory_values(z_opt_d,        nu_opt_d,        params)
                nl_vals       = trajectory.compute_trajectory_values(z_nl_d,         nu_nl_d,         params)
                init_vals     = trajectory.compute_trajectory_values(z_init_d,       nu_init_d,       params)
                init_nl_vals  = trajectory.compute_trajectory_values(z_init_dense_d, nu_init_dense_d, params)

                output = AttrDict(
                    {
                        "name": name,
                        "type": traj_type,
                        "opt_vals": opt_vals,
                        "nl_vals": nl_vals,
                        "init_vals": init_vals,
                        "init_nl_vals": init_nl_vals,
                        "title": getattr(trajectory, "title", None),
                        "xlabel": getattr(trajectory, "xlabel", None),
                        "ylabel": getattr(trajectory, "ylabel", None),
                        "zlabel": getattr(trajectory, "zlabel", None),
                        "tick_nbins": getattr(trajectory, "tick_nbins", None),
                        "markers": getattr(trajectory, "markers", None),
                        "invert_x": getattr(trajectory, "invert_x", False),
                        "show_iters": getattr(trajectory, "show_iters", None),
                    },
                )

                if group not in trajectory_data:
                    trajectory_data[group] = AttrDict({})
                trajectory_data[group][name] = output

        # re-dimensionalize all the data
        iter_data['t_nl'] = t_nl * nondim.time_scale
        iter_data['z_nl'] = x_nl @ nondim.M.state.nd2d
        iter_data['nu_nl'] = u_nl @ nondim.M.control.nd2d

        iter_data['t_init'] = t_init * nondim.time_scale
        iter_data['z_init'] = x_init @ nondim.M.state.nd2d
        iter_data['nu_init'] = u_init @ nondim.M.control.nd2d

        iter_data["t_init_nl"] = t_init_dense * nondim.time_scale
        iter_data["z_init_nl"] = x_init_dense @ nondim.M.state.nd2d
        iter_data["nu_init_nl"] = u_init_dense @ nondim.M.control.nd2d

        iter_data['t_opt']   = t_opt * nondim.time_scale
        iter_data['z_opt']   = x_opt @ nondim.M.state.nd2d
        iter_data['nu_opt']  = u_opt @ nondim.M.control.nd2d
        iter_data['trajectory_data'] = trajectory_data

        iter_data = tools.trim_dict(iter_data, tools.ITER_DATA_KEYS_TO_KEEP) 
    
    return AttrDict({'iter_data_list': iter_data_list})

# ======================================================================
# STANDALONE ANALYSIS
# ======================================================================

def run_standalone_analysis(trajopt_obj, show_iters = True):
    config  = trajopt_obj.config.method
    method_name = config.get("name", "method1")

    run_0_data = perform_analysis(trajopt_obj, compute_iters=show_iters)

    method_data = {"runs": [run_0_data]}
    
    return recursive_attrdict({method_name: method_data})


# ======================================================================
# MONTE CARLO ANALYSIS
# ======================================================================

def run_mc_analysis(trajopt_obj):

    nominal_config = trajopt_obj._raw_config
    
    seed = nominal_config.get("seed", 0)
    np.random.seed(seed)

    orig_problem = trajopt_obj.problem
    orig_method = trajopt_obj.method
    results = AttrDict({})

    for method_name, method_var_config in nominal_config.variations.method.items():

        # start with nominal config
        config_for_current_method = copy.deepcopy(nominal_config)

        # extract method variations
        method_var_config_flat = tools.flatten_dict(method_var_config)

        # apply method variations to current config
        for path, val in method_var_config_flat.items():
            if path.startswith(("constraints.", "costs.")):
                tools.set_from_path(config_for_current_method.problem, path, val)
            else:
                tools.set_from_path(config_for_current_method.method, path, val)

        # build a new problem and method for method variations
        index_map = IndexMap(config_for_current_method)
        problem   = Problem(config_for_current_method, index_map=index_map)

        SolutionMethod = getattr(scp, "SCP")
        method = SolutionMethod(problem, config_for_current_method, index_map)

        method.initialize()
        
        trajopt_obj.index_map = index_map
        trajopt_obj.problem   = problem
        trajopt_obj.method    = method
        trajopt_obj.solve()

        subprob = method
        runs = [perform_analysis(trajopt_obj)]

        mission_var_config = config_for_current_method.variations.mission
        mission_var_config_flat = tools.flatten_dict(mission_var_config)

        # mission variations
        for i in range(config_for_current_method.variations.mission.num):
            print(f"\n=== method: {method_name} | run: {i+1} / {config_for_current_method.variations.mission.num} ===")
            
            updated_config_vals_flat = get_variations(config_for_current_method, mission_var_config, mission_var_config_flat)

            tools.update_problem_from_config(problem, updated_config_vals_flat, problem.nondim)
            guess.set_initial_guess(problem, method)

            n_N = problem.index_map.N.time_grid
            n_neq = getattr(problem.index_map.n, 'nonconvex_inequality', 0)
            n_dyn = problem.index_map.n.dynamics
            n_term = problem.index_map.n.term_total

            # Reset iter_data like the first run: W and dual None so first iteration uses configure_penalty_weights
            subprob.iter_data = [
                tools.recursive_attrdict(
                    {
                        "iter_num": 0,
                        "z_opt": method.initial_guess.z,
                        "nu_opt": method.initial_guess.nu,
                        "conv_data": {
                            "vb_ineq": np.zeros((n_N, n_neq)),
                            "vb_dyn": np.zeros((n_N - 1, n_dyn)),
                            "vb_terminal": np.zeros((n_term, 1)),
                        },
                        "W": None,
                        "dual": None,
                        "penalty": copy.deepcopy(method.penalty),
                    },
                ),
            ]
            trajopt_obj.solve()
            runs.append(perform_analysis(trajopt_obj, compute_iters=False))

        results[method_name] = AttrDict({"runs": runs})

    trajopt_obj.problem = orig_problem
    trajopt_obj.method  = orig_method
    
    return results

def get_variations(config_for_current_method, mission_var_config, mission_var_config_flat):
    updated_config_vals_flat = AttrDict({})
    for path_to_spec, spec in mission_var_config_flat.items():
        if spec == "uniform":
            path = path_to_spec.replace(".type", "")
            random_variable_spec = tools.get_from_path(mission_var_config, path)
            lb = random_variable_spec["lb"]
            ub = random_variable_spec["ub"]
            delta = np.random.uniform(lb, ub)
        elif spec == "normal":
            path = path_to_spec.replace(".type", "")
            random_variable_spec = tools.get_from_path(mission_var_config, path)
            mu = random_variable_spec["mu"]
            sigma = random_variable_spec["sigma"]
            delta = np.random.normal(mu, sigma)
        else:
            continue

        new_val = tools.get_from_path(config_for_current_method.problem.mission, path) + delta
        updated_config_vals_flat[path] = new_val

    return updated_config_vals_flat
