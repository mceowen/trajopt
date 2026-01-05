import yaml
import numpy as np
import importlib.resources
import importlib.util
from pathlib import Path


def resolve_path(path_str):
    """
    Resolve a config path to an actual file path.
    
    Unified format: use / for paths, : for function names
      - trajopt/library/models/lander_6dof/model.yaml  -> package file
      - ./local_config.yaml                            -> relative file
      - /abs/path/config.yaml                          -> absolute file
    """
    if not path_str:
        return None
    
    # Check if it's an absolute path (not starting with trajopt)
    if path_str.startswith('/') and not path_str.startswith('/trajopt'):
        return Path(path_str)
    
    # Check if it's a local file
    local_path = Path(path_str)
    if local_path.exists():
        return local_path
    
    # Package path: starts with trajopt/ or /trajopt/
    if path_str.startswith('trajopt/') or path_str.startswith('/trajopt/'):
        path_str = path_str.lstrip('/')
        parts = path_str.split('/')
        file_name = parts[-1]
        pkg_path = '.'.join(parts[:-1])
        return importlib.resources.files(pkg_path).joinpath(file_name)
    
    # Fallback: treat as relative path
    return Path(path_str)


def resolve_function(path_str):
    """
    Resolve a function path to an actual function object.
    
    Format: path/to/file.py:function_name
      - trajopt/library/models/reentry_3dof/functions.py:dynamics  -> package function
      - ./custom.py:my_func                                        -> relative file function
      - /abs/path/file.py:func                                     -> absolute file function
    """
    if not path_str or ':' not in path_str:
        return None
    
    file_path, func_name = path_str.rsplit(':', 1)
    
    # Package path: starts with trajopt/
    if file_path.startswith('trajopt/') or file_path.startswith('/trajopt/'):
        file_path = file_path.lstrip('/')
        # Get actual file path from package
        parts = file_path.split('/')
        pkg_path = '.'.join(parts[:-1])
        file_name = parts[-1]
        actual_path = importlib.resources.files(pkg_path).joinpath(file_name)
        file_path = actual_path
    else:
        file_path = Path(file_path)
    
    # Load from file (works without __init__.py)
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)


def load_yaml(path):
    """Load a yaml file and convert lists to numpy arrays."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return _convert_lists(data) if data else {}


def _convert_lists(obj):
    """Recursively convert lists to numpy arrays (except string lists)."""
    if isinstance(obj, dict):
        return {k: _convert_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        if not obj:
            return np.array([], dtype=int)
        if all(isinstance(x, dict) for x in obj):
            return [_convert_lists(x) for x in obj]
        if all(isinstance(x, str) for x in obj):
            return obj
        return np.array(obj)
    elif obj == "inf":
        return np.inf
    elif obj == "-inf":
        return -np.inf
    return obj


def deep_merge(base, override):
    """Recursively merge override into base. Override wins on conflicts."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path_or_dict):
    """
    Load a config with inheritance support.
    
    For any dict containing 'inherit', loads that base first
    and merges the current dict on top of it. Recurses into subdicts.
    """
    # If given a path string, load it first
    if isinstance(path_or_dict, (str, Path)):
        config = load_yaml(resolve_path(path_or_dict))
    else:
        config = path_or_dict
    
    return _resolve_inheritance(config)


def _resolve_inheritance(config):
    """Recursively resolve inheritance in a dict."""
    if not isinstance(config, dict):
        return config
    
    # If this dict has a inherit, load and merge
    if 'inherit' in config:
        base_path = config.pop('inherit')
        base = load_config(base_path)
        config = deep_merge(base, config)
    
    # Recurse into subdicts
    for key, val in config.items():
        if isinstance(val, dict):
            config[key] = _resolve_inheritance(val)
        elif isinstance(val, list) and all(isinstance(x, dict) for x in val):
            config[key] = [_resolve_inheritance(x) for x in val]
    
    return config


def merge_constraints_with_model(mission_constraints, model_constraint_defs):
    """
    Merge mission constraint params with model constraint definitions.
    
    mission_constraints: list of dicts from mission.yaml 'constraints'
                         (contains name + mission-specific params)
    model_constraint_defs: list of dicts from model.yaml 'constraint_models'
                           (contains name + type + fcn + defaults)
    
    Returns: list of fully merged constraint dicts
    """
    # Index model definitions by name
    model_by_name = {c['name']: c for c in model_constraint_defs if 'name' in c}
    
    merged = []
    for mission_cnst in mission_constraints:
        name = mission_cnst.get('name')
        if name and name in model_by_name:
            # Start with model definition, merge mission params on top
            merged.append(deep_merge(model_by_name[name], mission_cnst))
        else:
            # No model definition found - use mission params as-is
            # (allows defining simple constraints entirely in mission.yaml)
            merged.append(mission_cnst)
    
    return merged


def build_params(problem):
    """
    Build a unified params dict from mission, model, and constraint params.
    Constraint 'mission_params' are merged directly into params['mission'].
    """
    params = {}
    
    if 'mission' in problem:
        params['mission'] = problem['mission'].copy() if isinstance(problem['mission'], dict) else {}
    else:
        params['mission'] = {}
    
    if 'model' in problem:
        params['model'] = problem['model']
    
    # Merge mission_params from constraints into mission
    for cnst in problem.get('constraints', []):
        if 'mission_params' in cnst and isinstance(cnst['mission_params'], dict):
            params['mission'].update(cnst['mission_params'])
    
    # Merge mission_params from costs into mission
    for cost in problem.get('costs', []):
        if 'mission_params' in cost and isinstance(cost['mission_params'], dict):
            params['mission'].update(cost['mission_params'])
    
    return params


def eval_expressions(obj, context=None):
    """Recursively evaluate 'eval:' prefixed strings in a dict."""
    if context is None:
        context = {}
    
    if isinstance(obj, dict):
        for key, val in obj.items():
            if isinstance(val, str) and val.startswith("eval:"):
                expr = val.split(":", 1)[1].strip()
                obj[key] = eval(expr, {"np": np, **context})
            elif isinstance(val, (dict, list)):
                eval_expressions(val, context)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str) and item.startswith("eval:"):
                expr = item.split(":", 1)[1].strip()
                obj[i] = eval(expr, {"np": np, **context})
            elif isinstance(item, (dict, list)):
                eval_expressions(item, context)


def load_trajopt_config(path):
    """
    Main entry point. Loads a trajopt config and returns the full config dict.
    
    - Resolves all inherit inheritance
    - Uses mission['constraints'] as the list of active constraints
    - Merges model constraint_models definitions with mission constraint params
    - Builds problem['params'] with full mission and model dicts
    - Evaluates all 'eval:' expressions
    """
    config = load_config(path)
    problem = config.get('problem', {})
    
    # Get model and mission
    model = problem.get('model', {})
    mission = problem.get('mission', {})
    
    # --- CONSTRAINTS ---
    # Get constraint definitions from model (type, fcn, defaults)
    model_constraint_defs = model.get('constraint_models', [])
    
    # Get active constraints from mission (with mission-specific params)
    # Use .get() instead of .pop() to keep constraints in mission dict
    mission_constraints = mission.get('constraints', [])
    
    # Merge: model definitions + mission params
    problem['constraints'] = merge_constraints_with_model(mission_constraints, model_constraint_defs)
    
    # --- COSTS ---
    # Same pattern for costs
    model_cost_defs = model.get('cost_models', [])
    
    # Check if costs are specified in problem (trajopt.yaml) or mission
    # Use .get() instead of .pop() to keep costs in mission dict
    mission_costs = mission.get('costs', [])
    problem_costs = problem.get('costs', [])
    
    # Use problem costs if specified, otherwise use mission costs
    active_costs = problem_costs if problem_costs else mission_costs
    
    if active_costs:
        # If just names, look up full definitions from model
        costs_by_name = {c['name']: c for c in model_cost_defs if 'name' in c}
        merged_costs = []
        for cost in active_costs:
            name = cost.get('name') if isinstance(cost, dict) else cost
            if isinstance(cost, dict) and name in costs_by_name:
                merged_costs.append(deep_merge(costs_by_name[name], cost))
            elif name in costs_by_name:
                merged_costs.append(costs_by_name[name])
            elif isinstance(cost, dict):
                merged_costs.append(cost)
        problem['costs'] = merged_costs
    else:
        problem['costs'] = []
    
    # Build params - contains full mission and model dicts
    problem['params'] = {
        'mission': mission,  # Full mission dict including constraints, costs, etc.
        'model': model,      # Full model dict including constraint_models, cost_models, etc.
    }
    
    # Merge mission_params from constraints into mission (for backward compat)
    for cnst in problem.get('constraints', []):
        if 'mission_params' in cnst and isinstance(cnst['mission_params'], dict):
            problem['params']['mission'].update(cnst['mission_params'])
    
    # Merge mission_params from costs into mission (for backward compat)
    for cost in problem.get('costs', []):
        if 'mission_params' in cost and isinstance(cost['mission_params'], dict):
            problem['params']['mission'].update(cost['mission_params'])
    
    # Create eval context with params as top-level for easy access
    eval_context = {
        **config,
        'params': problem['params'],
    }
    
    # Evaluate all 'eval:' expressions with full config as context
    eval_expressions(config, context=eval_context)
    
    return config


def gen_mc_variations(example_name, local=False):

    mc_variations = {}

    if local:
        mc_variations = load_config(f"trajopt.local.examples.{example_name}.variations.mission.yaml")
    else:
        mc_variations = load_config(f"trajopt.examples.{example_name}.variations.mission.yaml")

    # generate the realizations for the mission variations
    mc_variations["realizations"] = [{}]
    for i in range(0, mc_variations["num_variations"]):
        realization_dict = {}
        for rv_name, rv_properties in mc_variations["random_vars"].items():
            
            rv_var_type = rv_properties.get("variation_type", "uniform")

            if rv_var_type == "uniform":
                lb = rv_properties["lb"]
                ub = rv_properties["ub"]
                
                realization = lb + (ub - lb) * np.random.random(lb.shape)
            
            # TODO (carlos): add normal dispersions as well
            # elif rv_var_type == "normal":
            #     realization = 
            
            realization_dict[rv_name] = realization
        
        mc_variations["realizations"].append(realization_dict)
    return mc_variations


def load_mv_variations(example_name, local=False):

    if local:
        mv_variations = load_config(f"trajopt.local.examples.{example_name}.variations.method.yaml")
    else:
        mv_variations = load_config(f"trajopt.examples.{example_name}.variations.method.yaml")

    return mv_variations
