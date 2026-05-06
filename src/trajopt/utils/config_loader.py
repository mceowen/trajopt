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
from pathlib import Path
import re

from trajopt.utils.tools import AttrDict, recursive_attrdict, expand_dot_keys, deep_merge, flatten_dict

# =============================================================================
# MAIN CONFIG LOADER
# =============================================================================

def load_trajopt_config(mission_path, model_path, method_path, variations_path=None):
    """Load and combine mission, model, and method configs."""

    _files = f"mission='{mission_path}', model='{model_path}', method='{method_path}'"
    
    # load mission and model configs and merge mission into model to form problem config
    mission_config = load_yaml(mission_path)
    mission_config = resolve_inheritance(mission_config, _source=mission_path)

    model_config   = load_yaml(model_path)
    model_config   = resolve_inheritance(model_config, _source=model_path)

    try:
        problem_config = deep_merge(model_config, mission_config)
    except Exception as e:
        raise type(e)(f"error merging model and mission configs ({_files}): {e}") from None
    
    method_config  = load_yaml(method_path)
    method_config  = resolve_inheritance(method_config, _source=method_path)

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
    try:
        problem_config = eval_values(problem_config, eval_context)
        method_config  = eval_values(method_config, eval_context)
        mission_config = eval_values(mission_config, eval_context)
        model_config   = eval_values(model_config, eval_context)
    except Exception as e:
        raise type(e)(f"error evaluating expressions ({_files}): {e}") from None

    # remove inactive constraints from the problem config
    try:
        constraint_config  = problem_config.constraints
    except (KeyError, AttributeError):
        raise KeyError(f"'constraints' not found — check model ('{model_path}') and mission ('{mission_path}')") from None

    try:
        cost_config = problem_config.costs
    except (KeyError, AttributeError):
        raise KeyError(f"'costs' not found — check model ('{model_path}') and mission ('{mission_path}')") from None

    trajectory_config  = problem_config.get("trajectories", {})

    # update constraints and cost with any method-specific specfications
    constraint_config = deep_merge(constraint_config, method_config.get("constraints", {}))
    cost_config       = deep_merge(cost_config, method_config.get("costs", {}))
    
    # only keep active constraints specified in the mission config
    try:
        active_constraint_list = problem_config.constraint_list
    except (KeyError, AttributeError):
        raise KeyError(f"'constraint_list' not found — check mission ('{mission_path}')") from None

    try:
        active_cost_list = problem_config.cost_list
    except (KeyError, AttributeError):
        raise KeyError(f"'cost_list' not found — check mission ('{mission_path}')") from None

    try:
        constraint_config = AttrDict({name: {'name': name, **constraint_config[name]} for name in active_constraint_list})
    except KeyError as e:
        raise KeyError(f"constraint {e} is in 'constraint_list' but not defined in 'constraints' — check model ('{model_path}') and mission ('{mission_path}')") from None

    for name in constraint_config:
        if 'ct' not in constraint_config[name]:
            constraint_config[name]['ct'] = 0

    try:
        cost_config = AttrDict({name: {'name': name, **cost_config[name]} for name in active_cost_list})
    except KeyError as e:
        raise KeyError(f"cost {e} is in 'cost_list' but not defined in 'costs' — check model ('{model_path}') and mission ('{mission_path}')") from None

    trajectory_config = AttrDict({name: {'name': name, **trajectory_config[name]} for name in trajectory_config.keys() if trajectory_config[name] is not None})

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
            'constraint_list': active_constraint_list,
            'trajectories': trajectory_config,
            'costs': cost_config,
            'cost_list': active_cost_list,
            'params': params_config,
            'fcns': fcns_config,
            'mission': mission_config,
            'model': model_config,
            'plot_config': problem_config.get('plot_config', {}),
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
    
    # Package path: trajopt/... -> importlib.resources
    if path_str.startswith('trajopt/') or path_str.startswith('/trajopt/'):
        path_str = path_str.lstrip('/')
        parts    = path_str.split('/')
        
        path = importlib.resources.files('.'.join(parts[:-1])).joinpath(parts[-1])
    else:
        path = Path(path_str)
    
    return path

def resolve_inheritance(d, _source="unknown"):
    if not isinstance(d, dict):
        return d
    d = {k: resolve_inheritance(v, _source=_source) for k, v in d.items()}
    
    if "inherit" in d:
        parent_path = d["inherit"]
        try:
            parent = resolve_inheritance(load_yaml(parent_path), _source=parent_path)
            d = deep_merge(parent, d)
        except Exception as e:
            raise type(e)(f"error resolving 'inherit: {parent_path}' (referenced from '{_source}'): {e}") from None
        d.pop("inherit", None)
    
    return recursive_attrdict(d)

# =============================================================================
# CONFIG LOADING AND MERGING
# =============================================================================

def load_yaml(path_str):
    """Load YAML to nested AttrDict""" 
    
    path = resolve_path(path_str)

    try:
        with open(path, 'r') as f:
            raw_config = recursive_attrdict(yaml.safe_load(f))
    except FileNotFoundError:
        raise FileNotFoundError(f"file not found: '{path_str}' (resolved to '{path}')") from None
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML syntax error in '{path_str}': {e}") from None

    try:
        config = expand_dot_keys(raw_config)
    except Exception as e:
        raise type(e)(f"error expanding dot-keys in '{path_str}': {e}") from None
    
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

def eval_values(obj, ctx, key=None, _path=""):
    """Recursively evaluate expressions in a config object."""
    if isinstance(obj, dict):
        return AttrDict({k: eval_values(v, ctx, key=k, _path=f"{_path}.{k}") for k, v in obj.items()})

    if isinstance(obj, list):
        results = [eval_values(item, ctx, _path=f"{_path}[{i}]") for i, item in enumerate(obj)]
        if all(isinstance(x, (int, float, np.number, np.ndarray)) for x in results):
            arr = np.array(results)
            if key and "idx" in key and arr.dtype.kind == "f" and np.all(arr == np.round(arr)):
                arr = np.round(arr).astype(np.intp)
            return arr
        return results
    
    if isinstance(obj, str):
        # Expression: ${...}
        if '${' in obj:
            # If the whole string is a single ${...}, evaluate and return the result directly
            m = re.fullmatch(r'\$\{([^}]+)\}', obj.strip())
            if m:
                try:
                    result = eval_expr(m.group(1), ctx)
                except Exception as e:
                    raise type(e)(f"error evaluating '${{  {m.group(1)}  }}' at key '{_path}': {e}") from None
                return eval_values(result, ctx, _path=_path)
            # Mixed expression (e.g. "${params.x_ub} * fcns.q_s") — substitute the
            # template parts but leave the rest as a string for resolve_fcn to handle
            def _sub(match):
                try:
                    return str(eval_expr(match.group(1), ctx))
                except Exception as e:
                    raise type(e)(f"error evaluating '${{  {match.group(1)}  }}' at key '{_path}': {e}") from None
            return re.sub(r'\$\{([^}]+)\}', _sub, obj)
    
    return obj