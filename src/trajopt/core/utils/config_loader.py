import yaml
import numpy as np
import importlib.resources
from pathlib import Path


def resolve_path(path_str):
    """
    Resolve a config path to an actual file path.
    Supports:
      - Absolute paths: /full/path/to/config.yaml
      - Relative paths / filenames: config.yaml (in cwd)
      - Package paths: core.missions.configs.lander -> trajopt/core/missions/configs/lander.yaml
    """
    if not path_str:
        return None
    
    # Check if it's an absolute path
    if path_str.startswith('/'):
        return Path(path_str)
    
    # Check if it's a file in the current directory
    local_path = Path(path_str)
    if local_path.exists():
        return local_path
    
    # Otherwise, treat as package path (e.g., core.missions.configs.lander)
    parts = path_str.split('.')
    file_name = parts[-1] + '.yaml'
    pkg_path = 'trajopt.' + '.'.join(parts[:-1])
    
    return importlib.resources.files(pkg_path).joinpath(file_name)


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
    
    For any dict containing 'base_config', loads that base first
    and merges the current dict on top of it. Recurses into subdicts.
    
    For constraints list items, uses 'base_config_mission' to find
    the constraint params from the mission config.
    """
    # If given a path string, load it first
    if isinstance(path_or_dict, (str, Path)):
        config = load_yaml(resolve_path(path_or_dict))
    else:
        config = path_or_dict
    
    return _resolve_inheritance(config)


def _resolve_inheritance(config):
    """Recursively resolve base_config inheritance in a dict."""
    if not isinstance(config, dict):
        return config
    
    # If this dict has a base_config, load and merge
    if 'base_config' in config:
        base_path = config.pop('base_config')
        base = load_config(base_path)
        config = deep_merge(base, config)
    
    # Recurse into subdicts
    for key, val in config.items():
        if isinstance(val, dict):
            config[key] = _resolve_inheritance(val)
        elif isinstance(val, list) and all(isinstance(x, dict) for x in val):
            config[key] = [_resolve_inheritance(x) for x in val]
    
    return config


def merge_constraints(model_constraints, mission_params):
    """
    Merge constraint_models from model with constraint_params from mission.
    Matches by 'name' field and merges mission params on top of model definition.
    """
    # Index mission params by name
    mission_by_name = {p['name']: p for p in mission_params if 'name' in p}
    
    merged = []
    for model_cnst in model_constraints:
        name = model_cnst.get('name')
        if name and name in mission_by_name:
            merged.append(deep_merge(model_cnst, mission_by_name[name]))
        else:
            merged.append(model_cnst)
    
    return merged


def build_params(problem):
    """
    Build a unified params dict from mission, model, and constraint params.
    """
    params = problem.get('params', {}).copy() if isinstance(problem.get('params'), dict) else {}
    
    if 'mission' in problem:
        params['mission'] = problem['mission']
    if 'model' in problem:
        params['model'] = problem['model']
    
    for cnst in problem.get('constraints', []):
        if 'params' in cnst and 'name' in cnst:
            params[cnst['name']] = cnst['params']
    
    for cost in problem.get('costs', []):
        if 'params' in cost and 'name' in cost:
            params[cost['name']] = cost['params']
    
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
    
    - Resolves all base_config inheritance
    - Merges model constraint_models with mission constraint_params
    - Filters to only requested constraints/costs
    - Builds problem['params']
    - Evaluates all 'eval:' expressions
    """
    config = load_config(path)
    problem = config.get('problem', {})
    
    # Get model and mission
    model = problem.get('model', {})
    mission = problem.get('mission', {})
    
    # Load planet and vehicle configs if they're string references
    if 'planet' in mission and isinstance(mission['planet'], str):
        mission['planet'] = load_config(f"core.missions.configs.planet.{mission['planet']}")
    if 'vehicle' in mission and isinstance(mission['vehicle'], str):
        mission['vehicle'] = load_config(f"core.missions.configs.vehicle.{mission['vehicle']}")
    
    # Merge constraints: model definitions + mission params
    model_constraints = model.get('constraint_models', [])
    mission_params = mission.get('constraint_params', [])
    all_constraints = merge_constraints(model_constraints, mission_params)
    
    # Filter to requested constraints (by name)
    requested = [c['name'] for c in problem.get('constraints', []) if 'name' in c]
    if requested:
        by_name = {c['name']: c for c in all_constraints}
        problem['constraints'] = [by_name[n] for n in requested if n in by_name]
    else:
        problem['constraints'] = all_constraints
    
    # Same for costs
    model_costs = model.get('cost_models', [])
    mission_cost_params = mission.get('cost_params', [])
    all_costs = merge_constraints(model_costs, mission_cost_params)
    
    requested_costs = [c['name'] for c in problem.get('costs', []) if 'name' in c]
    if requested_costs:
        costs_by_name = {c['name']: c for c in all_costs}
        problem['costs'] = [costs_by_name[n] for n in requested_costs if n in costs_by_name]
    else:
        problem['costs'] = all_costs
    
    # Build params
    problem['params'] = build_params(problem)
    
    # Evaluate all 'eval:' expressions with full config as context
    eval_expressions(config, context=config)
    
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
