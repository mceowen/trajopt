import copy
import numpy as np
import trajopt.utils.tools as tools
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import trajopt.library.methods.integrators as integrators
from trajopt.core.indexing.index_map import IndexMap
from trajopt.core.problem import Problem
from trajopt.core.solution_method import SolutionMethod

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

ITER_DATA_KEYS_KEEP = {
    "iter_num", "converged", "cost", "solve_time", "prop_time", "parse_time", "t_full",
    "t_opt", "z_opt", "nu_opt", "dt_opt", "T_opt",
    "t_nl", "z_nl", "nu_nl", "t_init", "z_init", "nu_init",
    "constraint_data", "conv_data", "W", "dual", "penalty",
}

METHOD_DATA_KEYS_KEEP = {
    'N', 'N_dens', 'Npm', 'T_init', 'T_max', 'T_min', 'Ts_init', 'conv', 'conv_data', 'cost_init', 
    'dT_max', 'ddt_max', 'dt_init', 'dt_init', 'dt_max', 'dt_min', 'flags', 'line_guess_u_init',
    'name', 'n_minus', 'n_plus', 'nl_guess_u_start', 'nl_guess_u_stop', 'solver_opts',
    'nondim', 't_init', 'nu_init', 'penalty', 'z_ind', 'z_init'
}

def _trim_iter_record(rec):
    return {k: rec[k] for k in ITER_DATA_KEYS_KEEP if k in rec}

def perform_default_analysis(trajopt_obj, trim=True):
    problem     = trajopt_obj.problem
    method      = trajopt_obj.method

    n_x         = problem.index_map.n['state']
    n_nu        = problem.index_map.n['control']
    params      = problem.params
    iter_data   = method.subprob.iter_data
    nondim      = method.nondim

    problem_config = problem.config
    method_data = tools.extract_attributes(method, METHOD_DATA_KEYS_KEEP)
    
    all_params_dict = {**problem_config, **method_data}

    for data in iter_data[1:]:
        
        # get reference trajectory for this iteration (in nondimensional coordinates)
        t_opt = np.asarray(data['t_opt'])
        z_opt = np.asarray(data['z_opt'])
        nu_opt = np.asarray(data["nu_opt"])

        t_nl = np.linspace(t_opt[0], t_opt[-1], 1000)

        # nonlinear propagation
        t_nl, z_nl, nu_nl = integrators.propagate_jax_rk4_dense(z_opt[0, :n_x], nu_opt[:, :n_nu], t_opt, t_nl, problem, method)

        t_init = method.t_init
        z_init = method.z_init
        nu_init = method.nu_init

        # compute constraints for z_nl, z_opt, name = SUBPLOT , TYPE, group = FIGURE, units
        constraint_data = {}

        for constraint in problem.constraints.get():
            if hasattr(constraint, "compute_constraint_values"):
                name  = constraint.name
                type  = constraint.type
                group = constraint.group

                if group == None:
                    group = name
                
                opt_vals  = constraint.compute_constraint_values(t_opt, z_opt[:, :n_x], nu_opt, params)
                nl_vals   = constraint.compute_constraint_values(t_nl, z_nl[:, :n_x], nu_nl, params)
                init_vals = constraint.compute_constraint_values(t_init, z_init[:, :n_x], nu_init, params)

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
        data['z_init']  = z_init[:, :n_x] @ nondim.M["state"]["nd2d"]
        data['nu_init'] = nu_init @ nondim.M["ctrl"]["nd2d"]

        data['t_opt']  = t_opt * nondim.nt
        data['z_opt']  = z_opt[:, :n_x] @ nondim.M["state"]["nd2d"]
        data['nu_opt'] = nu_opt @ nondim.M["ctrl"]["nd2d"]
        data['constraint_data'] = constraint_data

    if trim:
        iters_out = [_trim_iter_record(rec) for rec in iter_data]
    else:
        iters_out = iter_data
    return {'iters': iters_out, 'params': all_params_dict}


# ======================================================================
# STANDALONE ANALYSIS
# ======================================================================
def run_standalone_analysis(trajopt_obj):
    cfg = trajopt_obj.method_config or {}
    name = cfg.get("name", "method1")
    return {name: {"mc_data": [perform_default_analysis(trajopt_obj)]}}

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

def _method_names(cfg):
    method_vars = cfg.get("method_variations", {})
    if not method_vars:
        return ["default"]
    seen, out = set(), []
    for path in method_vars:
        n = path.split(".")[0]
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out

def _merge_method_config(base, method_vars, method_name):
    prefix = method_name + "."
    merged = copy.deepcopy(base)
    for path, val in method_vars.items():
        if path.startswith(prefix):
            set_nested(merged, path[len(prefix):], val)
    return merged

def run_mc_analysis(trajopt_obj):
    cfg = trajopt_obj.variation_config
    mission_vars = cfg.get('mission_variations', {})
    num = cfg.get('num_mission_variations', 1)
    method_vars = cfg.get("method_variations", {})
    method_names = _method_names(cfg)
    problem_config = copy.deepcopy(trajopt_obj.problem_config)
    base_config = trajopt_obj.method_config

    if 'seed' in cfg:
        np.random.seed(cfg['seed'])

    nominal = {}
    mission_config = problem_config['config']['mission']
    for path in mission_vars:
        val = get_nested(mission_config, path)
        nominal[path] = np.array(val).copy() if val is not None else None

    def _apply_delta(mission_cfg, path, spec):
        delta = np.random.uniform(np.array(spec["lb"]), np.array(spec["ub"])) if spec.get("type") == "uniform" else np.random.normal(np.array(spec["mu"]), np.array(spec["sigma"]))
        new_val = np.real(np.asarray(nominal[path], dtype=float) + np.asarray(delta, dtype=float))
        set_nested(mission_cfg, path, new_val.tolist() if new_val.size > 1 else float(new_val))

    def _run_mc_for_method(name):
        merged = _merge_method_config(base_config, method_vars, name)
        index_map = IndexMap(
            model_config=problem_config['model'],
            mission_config=problem_config['mission'],
            method_config=merged,
        )
        problem = Problem(copy.deepcopy(problem_config), index_map=index_map)
        method = SolutionMethod(problem, merged, index_map=index_map)
        method.get_initial_guess(problem)
        
        trajopt_obj.problem = problem
        trajopt_obj.method = method
        trajopt_obj.solve()
        
        m, problem, subprob = trajopt_obj.method, trajopt_obj.problem, trajopt_obj.method.subprob
        mc_data = [perform_default_analysis(trajopt_obj)]
        
        for i in range(num):
            print(f"\n=== {name} MC Run {i+1}/{num} ===")
            mission_cfg = problem.config['mission']
            for path, spec in mission_vars.items():
                _apply_delta(mission_cfg, path, spec)
            problem.update_from_config(mission_vars.keys(), m.nondim)
            m.get_initial_guess(problem)
            
            W_stack, dual_stack = subprob.constraints.stack_W_and_dual(problem, m)
            subprob.iter_data = [{
                "iter_num": 0,
                "z_ref": m.z_init,
                "nu_ref": m.nu_init,
                "dt_ref": m.dt_init,
                "t_ref": m.t_init,
                "conv_data": {"vb_ineq": np.zeros((subprob.N, problem.index_map.n['nonconvex_inequality'])), "vb_dyn": np.zeros((subprob.N - 1, subprob.index_map.n['dynamics'])), "vb_terminal": np.zeros((problem.index_map.n['term_total'], 1))},
                "W": copy.deepcopy(W_stack),
                "dual": copy.deepcopy(dual_stack),
                "penalty": copy.deepcopy(m.penalty),
            }]

            trajopt_obj.solve()
            mc_data.append(perform_default_analysis(trajopt_obj))
        for path, val in nominal.items():
            if val is not None:
                set_nested(problem.config['mission'], path, val.tolist() if hasattr(val, 'tolist') else val)
        return mc_data

    scenario_data = {}
    orig_problem, orig_method = trajopt_obj.problem, trajopt_obj.method
    try:
        for name in method_names:
            scenario_data[name] = {"mc_data": _run_mc_for_method(name)}
    finally:
        trajopt_obj.problem = orig_problem
        trajopt_obj.method = orig_method

    return scenario_data
