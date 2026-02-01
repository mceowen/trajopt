import yaml
import numpy as np
import importlib.resources
import importlib.util
from pathlib import Path
import re


def resolve_path(path_str):
    if not path_str:
        return None
    if path_str.startswith('/') and not path_str.startswith('/trajopt'):
        return Path(path_str)
    local_path = Path(path_str)
    if local_path.exists():
        return local_path
    if path_str.startswith('trajopt/') or path_str.startswith('/trajopt/'):
        path_str = path_str.lstrip('/')
        parts = path_str.split('/')
        return importlib.resources.files('.'.join(parts[:-1])).joinpath(parts[-1])
    return Path(path_str)


def resolve_function(path_str):
    if not path_str or ':' not in path_str:
        return None
    file_path, func_name = path_str.rsplit(':', 1)
    if file_path.startswith('trajopt/') or file_path.startswith('/trajopt/'):
        file_path = file_path.lstrip('/')
        parts = file_path.split('/')
        file_path = importlib.resources.files('.'.join(parts[:-1])).joinpath(parts[-1])
    else:
        file_path = Path(file_path)
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def deep_merge(base, override):
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def set_nested(d, path, value):
    parts = path.split('.')
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def get_nested(d, path):
    parts = path.split('.')
    for part in parts:
        if isinstance(d, dict) and part in d:
            d = d[part]
        else:
            return None
    return d


def expand_config(raw):
    expanded = {}
    inherit_keys = []

    for key, val in raw.items():
        if key.endswith('.inherit'):
            inherit_keys.append((key[:-8], val))
        elif '.' in key:
            set_nested(expanded, key, val)
        else:
            expanded[key] = val

    for base_key, path in inherit_keys:
        inherited = load_config(path)
        existing = get_nested(expanded, base_key)
        if existing and isinstance(existing, dict):
            merged = deep_merge(inherited, existing)
        else:
            merged = inherited
        set_nested(expanded, base_key, merged)

    return expanded


def load_config(path):
    raw = load_yaml(resolve_path(path))
    config = expand_config(raw)

    if 'inherit' in config:
        base = load_config(config.pop('inherit'))
        config = deep_merge(base, config)

    return config


def build_context(config):
    ctx = {'np': np}

    def flatten(obj, prefix=''):
        if isinstance(obj, dict):
            for key, val in obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                ctx[full_key] = val
                ctx[full_key.replace('.', '_')] = val
                flatten(val, full_key)

    def add_aliases(obj, prefix=''):
        if isinstance(obj, dict):
            for key, val in obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                parts = full_key.split('.')
                if len(parts) > 1:
                    short_key = '.'.join(parts[1:])
                    if short_key not in ctx:
                        ctx[short_key] = val
                        ctx[short_key.replace('.', '_')] = val
                add_aliases(val, full_key)

    flatten(config)
    add_aliases(config)
    return ctx


def eval_expr(expr_str, ctx):
    ref_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)\b'
    refs = re.findall(ref_pattern, expr_str)
    local_ctx = dict(ctx)

    for ref in sorted(refs, key=len, reverse=True):
        val = ctx.get(ref) or ctx.get(ref.replace('.', '_'))
        if val is not None:
            var_name = ref.replace('.', '_')
            local_ctx[var_name] = val
            expr_str = expr_str.replace(ref, var_name)

    return eval(expr_str, local_ctx)


def eval_value(obj, ctx, param_refs=None):
    if param_refs is None:
        param_refs = {}
    
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            result[k] = eval_value(v, ctx, param_refs)
            if k == 'x' and isinstance(v, str) and v.startswith('${params.'):
                key = v[9:-1]
                result['param_key'] = key
        return result

    if isinstance(obj, list):
        result = [eval_value(item, ctx, param_refs) for item in obj]
        if result and all(isinstance(x, (int, float, np.number, np.ndarray)) for x in result):
            return np.array(result)
        return result

    if isinstance(obj, str):
        if obj.startswith("precompute-"):
            return resolve_function(obj.split("-", 1)[1].strip())

        if obj.startswith("eval:"):
            return eval_expr(obj.split(":", 1)[1].strip(), ctx)

        if '${' in obj:
            if obj.startswith('${') and obj.endswith('}') and obj.count('${') == 1:
                return eval_expr(obj[2:-1], ctx)

            def replace_expr(match):
                result = eval_expr(match.group(1), ctx)
                return str(result) if not isinstance(result, (int, float)) else repr(result)

            return re.sub(r'\$\{([^}]+)\}', replace_expr, obj)

    return obj


def load_trajopt_config(mission_path, model_path, method_path):
    mission = load_config(mission_path)
    model = load_config(model_path)
    method = load_config(method_path)

    combined = deep_merge(model, mission)
    ctx = build_context(combined)
    ctx['method'] = method
    
    # Add method keys directly to context so N, flags, etc. are accessible
    for key, val in method.items():
        if key not in ctx:
            ctx[key] = val

    combined = eval_value(combined, ctx)
    method = eval_value(method, ctx)
    model_evaluated = eval_value(model, ctx)

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

    exclude = {'constraints', 'costs', 'constraint_list', 'cost_list', 'fcns'}
    params = {}
    for k, v in combined.items():
        if k == 'params':
            params.update(v)
        elif k not in exclude:
            params[k] = v
    
    return {
        'problem': {
            'mission': combined,
            'model': model_evaluated,
            'constraints': constraints,
            'costs': costs,
            'params': params,
            'config': {'mission': combined, 'model': model_evaluated}
        },
        'method': method
    }
