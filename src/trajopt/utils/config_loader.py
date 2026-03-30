"""
YAML config loader:
  - Dot notation: 'foo.bar.x: 1' expands to '{foo: {bar: {x: 1}}}'
  - Inheritance: 'foo.inherit: path' or top-level 'inherit: path'
  - Expressions: '${expr}' evaluates Python with access to config values and numpy
  - Functions: 'path:func' loads a function from a Python file
  - Array conversion: lists of numbers become numpy arrays
"""

import yaml
import numpy as np
import importlib.resources
import importlib.util
from pathlib import Path
import re

from trajopt.utils.tools import AttrDict, recursive_attrdict, expand_dot_keys, deep_merge, flatten_dict

# =============================================================================
# MAIN CONFIG LOADER
# =============================================================================

def load_trajopt_config(mission_path, model_path, method_path, variations_path=None):
    """Load and combine mission, model, and method configs."""
    
    # load mission and model configs and merge mission into model to form problem config
    mission_config = load_yaml(mission_path)
    mission_config = resolve_inheritance(mission_config)

    model_config   = load_yaml(model_path)
    model_config   = resolve_inheritance(model_config)

    problem_config = deep_merge(model_config, mission_config)
    
    method_config  = load_yaml(method_path)
    method_config  = resolve_inheritance(method_config)

    # optionally load variations config if provided
    if variations_path is not None:
        variations_config = load_yaml(variations_path)
    else:
        variations_config = AttrDict({})
    
    # setup a a context dictionary so that we can evaluate the expressions
    # specified by  ${ }
    # we do this by flattening problem and method configs into one dict so we can access
    # via dot notation like:
    # ${params.vehicle.mass + 3} -> eval_context["params"]["vehicle"]["mass"] + 3
    
    problem_config_flat = flatten_dict(problem_config)
    method_config_flat  = flatten_dict(method_config)
    eval_context        = {**problem_config_flat, **method_config_flat, "np": np}

    # evaluate expressions
    problem_config = eval_values(problem_config, eval_context)
    method_config  = eval_values(method_config, eval_context)
    
    mission_config = eval_values(mission_config, eval_context)
    model_config   = eval_values(model_config, eval_context)

    # remove inactive constraints from the problem config
    constraint_config  = problem_config.constraints
    cost_config        = problem_config.costs
    trajectory_config  = problem_config.get("trajectories", {})

    # update constraints and cost with any method-specific specfications
    constraint_config = deep_merge(constraint_config, method_config.get("constraints", {}))
    cost_config       = deep_merge(cost_config, method_config.get("costs", {}))
    
    # only keep active constraints specified in the mission config
    active_constraint_list = problem_config.constraint_list
    active_cost_list       = problem_config.cost_list

    constraint_config = AttrDict({name: {'name': name, **constraint_config[name]} for name in active_constraint_list})
    cost_config       = AttrDict({name: {'name': name, **cost_config[name]} for name in active_cost_list})
    trajectory_config = AttrDict({name: {'name': name, **trajectory_config[name]} for name in trajectory_config.keys()})

    # extract parameters and functions
    params_config = problem_config.get('params', {})
    fcns_config   = problem_config.get('fcns', {})

    # exctract state, control, and time config
    state_config = problem_config.state
    control_config = problem_config.control
    time_config = problem_config.time

    config = recursive_attrdict({
        'problem': {
            'state': state_config,
            'control': control_config,
            'time': time_config,
            'constraints': constraint_config,
            'trajectories': trajectory_config,
            'costs': cost_config,
            'params': params_config,
            'fcns': fcns_config,
            'mission': mission_config,
            'model': model_config
        },
        
        'method': method_config, 
        'variations': variations_config
    })

    return config

# =============================================================================
# PATH AND FUNCTION RESOLUTION
# =============================================================================

def resolve_path(path_str):
    """Resolve a path string to a Path object. Handles local and package paths."""
    
    # Package path: trajopt/library/... -> importlib.resources
    if path_str.startswith('trajopt/') or path_str.startswith('/trajopt/'):
        path_str = path_str.lstrip('/')
        parts    = path_str.split('/')
        
        path = importlib.resources.files('.'.join(parts[:-1])).joinpath(parts[-1])
    else:
        path = Path(path_str)
    
    return path

def resolve_function_from_path(path_str):
    """Load a function from 'path/to/file.py:function_name' format."""
    
    file_path_str, func_name_str = path_str.rsplit(':', 1)
    file_path = resolve_path(file_path_str)
    
    spec   = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    function = getattr(module, func_name_str)
    
    return function

def resolve_inheritance(d):
    if not isinstance(d, dict):
        return d
    d = {k: resolve_inheritance(v) for k, v in d.items()}
    
    if "inherit" in d:
        parent = resolve_inheritance(load_yaml(d["inherit"]))
        d = deep_merge(parent, d)
        d.pop("inherit", None)
    
    return recursive_attrdict(d)

# =============================================================================
# CONFIG LOADING AND MERGING
# =============================================================================

def load_yaml(path_str):
    """Load YAML to nested AttrDict""" 
    
    path = resolve_path(path_str)
    
    with open(path, 'r') as f:
        raw_config = recursive_attrdict(yaml.safe_load(f))

    config = expand_dot_keys(raw_config)
    
    return config

# =============================================================================
# EXPRESSION EVALUATION
# =============================================================================

def eval_expr(expr_str, ctx):
    """Evaluate a Python expression with dotted references resolved."""
    # Find dotted references like params.planet.r
    refs  = re.findall(r'\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)\b', expr_str)
    local = ctx
    
    for ref in sorted(refs, key=len, reverse=True):
        val = ctx.get(ref) or ctx.get(ref.replace('.', '_'))
        
        if val is not None:
            safe_name = ref.replace('.', '_')
            local[safe_name] = val
            expr_str = expr_str.replace(ref, safe_name)
    
    return eval(expr_str, local)

def eval_values(obj, ctx, key=None):
    """Recursively evaluate expressions in a config object."""
    if isinstance(obj, dict):
        return AttrDict({k: eval_values(v, ctx, key=k) for k, v in obj.items()})

    if isinstance(obj, list):
        results = [eval_values(item, ctx) for item in obj]
        if all(isinstance(x, (int, float, np.number, np.ndarray)) for x in results):
            arr = np.array(results)
            if key and "idx" in key and arr.dtype.kind == "f" and np.all(arr == np.round(arr)):
                arr = np.round(arr).astype(np.intp)
            return arr
        return results
    
    if isinstance(obj, str):
        # Expression: ${...}
        if '${' in obj:
            expression = obj.split('${', 1)[1].rsplit('}', 1)[0]
            result = eval_expr(expression, ctx)
            return eval_values(result, ctx)
    
    return obj