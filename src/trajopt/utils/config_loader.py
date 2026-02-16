"""
Simple YAML config loader with:
  - Dot notation: `foo.bar.x: 1` expands to `{foo: {bar: {x: 1}}}`
  - Inheritance: `foo.inherit: path` or top-level `inherit: path`
  - Expressions: `${expr}` evaluates Python with access to config values and numpy
  - Functions: `path:func` loads a function from a Python file
  - Auto-arrays: lists of numbers become numpy arrays
"""

import yaml
import numpy as np
import importlib.resources
import importlib.util
from pathlib import Path
import re


# =============================================================================
# PATH AND FUNCTION RESOLUTION
# =============================================================================

def resolve_path(path_str):
    """Resolve a path string to a Path object. Handles local and package paths."""
    if not path_str:
        return None
    # Absolute path (not package path)
    if path_str.startswith('/') and not path_str.startswith('/trajopt'):
        return Path(path_str)
    # Local path exists
    local = Path(path_str)
    if local.exists():
        return local
    # Package path: trajopt/library/... -> importlib.resources
    if path_str.startswith('trajopt/') or path_str.startswith('/trajopt/'):
        path_str = path_str.lstrip('/')
        parts = path_str.split('/')
        return importlib.resources.files('.'.join(parts[:-1])).joinpath(parts[-1])
    return Path(path_str)


def resolve_function(path_str):
    """Load a function from 'path/to/file.py:function_name' format."""
    if not path_str or ':' not in path_str:
        return None
    file_path, func_name = path_str.rsplit(':', 1)
    resolved = resolve_path(file_path)
    spec = importlib.util.spec_from_file_location("dynamic_module", resolved)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)


# =============================================================================
# CONFIG LOADING AND MERGING
# =============================================================================

def deep_merge(base, override):
    """Recursively merge override into base, returning new dict."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def set_nested(d, path, value):
    """Set a value in nested dict using dot path: 'a.b.c' -> d['a']['b']['c']."""
    parts = path.split('.')
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def expand_dotted_keys(raw):
    """Expand dotted keys and handle .inherit directives."""
    expanded = {}
    inherits = []

    for key, val in raw.items():
        if key.endswith('.inherit'):
            inherits.append((key[:-8], val))
        elif '.' in key:
            set_nested(expanded, key, val)
        else:
            expanded[key] = val

    # Process inherits: load base config and merge with any existing values
    for base_key, path in inherits:
        inherited = load_config(path)
        # Navigate to the target location
        parts = base_key.split('.')
        target = expanded
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        existing = target.get(parts[-1], {})
        target[parts[-1]] = deep_merge(inherited, existing) if isinstance(existing, dict) else inherited

    return expanded


def load_config(path):
    """Load YAML, expand dotted keys, and handle inheritance."""
    with open(resolve_path(path), 'r') as f:
        raw = yaml.safe_load(f) or {}
    
    config = expand_dotted_keys(raw)
    
    # Handle top-level inherit
    if 'inherit' in config:
        base = load_config(config.pop('inherit'))
        config = deep_merge(base, config)
    
    return config


# =============================================================================
# EXPRESSION EVALUATION
# =============================================================================

def flatten_to_context(config):
    """Flatten nested config into a flat dict for expression evaluation."""
    ctx = {'np': np}
    
    def recurse(obj, prefix=''):
        if isinstance(obj, dict):
            for key, val in obj.items():
                full = f"{prefix}.{key}" if prefix else key
                ctx[full] = val
                ctx[full.replace('.', '_')] = val
                recurse(val, full)
    
    recurse(config)
    return ctx


def eval_expr(expr_str, ctx):
    """Evaluate a Python expression with dotted references resolved."""
    # Find dotted references like params.planet.r
    refs = re.findall(r'\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)\b', expr_str)
    local = dict(ctx)
    
    for ref in sorted(refs, key=len, reverse=True):
        val = ctx.get(ref) or ctx.get(ref.replace('.', '_'))
        if val is not None:
            safe_name = ref.replace('.', '_')
            local[safe_name] = val
            expr_str = expr_str.replace(ref, safe_name)
    
    return eval(expr_str, local)


def eval_value(obj, ctx):
    """Recursively evaluate expressions in a config object."""
    if isinstance(obj, dict):
        return {k: eval_value(v, ctx) for k, v in obj.items()}
    
    if isinstance(obj, list):
        results = [eval_value(item, ctx) for item in obj]
        # Convert numeric lists to numpy arrays
        if results and all(isinstance(x, (int, float, np.number, np.ndarray)) for x in results):
            return np.array(results)
        return results
    
    if isinstance(obj, str):
        # Function reference
        if obj.startswith("precompute-"):
            return resolve_function(obj.split("-", 1)[1].strip())
        # Expression: ${...}
        if '${' in obj:
            if obj.startswith('${') and obj.endswith('}') and obj.count('${') == 1:
                result = eval_expr(obj[2:-1], ctx)
                return eval_value(result, ctx)
            # Multiple expressions in string
            def replace(m):
                r = eval_expr(m.group(1), ctx)
                return str(r) if not isinstance(r, (int, float)) else repr(r)
            return re.sub(r'\$\{([^}]+)\}', replace, obj)
    
    return obj


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def load_trajopt_config(mission_path, model_path, method_path):
    """Load and combine mission, model, and method configs."""
    mission = load_config(mission_path)
    model = load_config(model_path)
    method = load_config(method_path)

    # Model is base, mission overrides
    combined = deep_merge(model, mission)
    
    # Build context and add method values
    ctx = flatten_to_context(combined)
    ctx['method'] = method
    for key, val in method.items():
        if key not in ctx:
            ctx[key] = val

    # Evaluate expressions
    combined = eval_value(combined, ctx)
    method = eval_value(method, ctx)
    model_eval = eval_value(model, ctx)

    # Extract constraints and costs by name
    all_constraints = combined.get('constraints', {})
    all_costs = combined.get('costs', {})
    
    constraint_list = ['dynamics'] + list(combined.get('constraint_list', []))
    cost_list = combined.get('cost_list', [])

    constraints = []
    for name in constraint_list:
        if name in all_constraints:
            c = dict(all_constraints[name])
            c['name'] = name
            constraints.append(c)

    costs = []
    for name in cost_list:
        if name in all_costs:
            c = dict(all_costs[name])
            c['name'] = name
            costs.append(c)

    # Params only contains what's explicitly under params.* in YAMLs
    params = combined.get('params', {})

    return {
        'problem': {
            'mission': combined,
            'model': model_eval,
            'constraints': constraints,
            'costs': costs,
            'params': params,
            'config': {'mission': combined, 'model': model_eval}
        },
        'method': method
    }