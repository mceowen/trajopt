import inspect
from functools import partial

import numpy as np
import cvxpy as cp
import jax


class _FcnExpr:
    """Lightweight wrapper enabling arithmetic on (t, x, u, params) callables."""
    def __init__(self, fcn):
        self._fcn = fcn

    def __mul__(self, other):
        f = self._fcn
        if isinstance(other, _FcnExpr):
            g = other._fcn
            return _FcnExpr(lambda t, x, u, params: f(t, x, u, params) * g(t, x, u, params))
        if isinstance(other, list):
            other = np.array(other)
        return _FcnExpr(lambda t, x, u, params: f(t, x, u, params) * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        f = self._fcn
        if isinstance(other, _FcnExpr):
            g = other._fcn
            return _FcnExpr(lambda t, x, u, params: f(t, x, u, params) / g(t, x, u, params))
        return _FcnExpr(lambda t, x, u, params: f(t, x, u, params) / other)

    def __add__(self, other):
        f = self._fcn
        if isinstance(other, _FcnExpr):
            g = other._fcn
            return _FcnExpr(lambda t, x, u, params: f(t, x, u, params) + g(t, x, u, params))
        return _FcnExpr(lambda t, x, u, params: f(t, x, u, params) + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        f = self._fcn
        if isinstance(other, _FcnExpr):
            g = other._fcn
            return _FcnExpr(lambda t, x, u, params: f(t, x, u, params) - g(t, x, u, params))
        return _FcnExpr(lambda t, x, u, params: f(t, x, u, params) - other)

    def __neg__(self):
        f = self._fcn
        return _FcnExpr(lambda t, x, u, params: -f(t, x, u, params))

class AttrDict(dict):
    """Dictionary that allows attribute access to keys.

    Example: d = AttrDict({'a':1}); d.a == 1; d['a'] == 1
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

# register the AttrDict class with jax to make it traceable

jax.tree_util.register_pytree_node(AttrDict, 
                                   flatten_func=lambda d: (list(d.values()), list(d.keys())),
                                   unflatten_func=lambda keys, vals: AttrDict(zip(keys, vals))
                                   )
    
def recursive_attrdict(d):
    if isinstance(d, dict):
        return AttrDict({k: recursive_attrdict(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [recursive_attrdict(i) for i in d]
    else:
        return d
    
def recursive_to_dict(d):
    if isinstance(d, AttrDict):
        return AttrDict({k: recursive_to_dict(v) for k, v in d.items()})
    elif isinstance(d, dict):
        return {k: recursive_to_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [recursive_to_dict(i) for i in d]
    else:
        return d
            
def get_from_path(d, path):
    keys = path.split(".")
    current_dict = d
    
    for key in keys:
        current_dict = current_dict[key]
        
    return current_dict

def set_from_path(d, path, value):
    keys = path.split(".")
    current_dict = d
    
    for key in keys[:-1]:
        current_dict = current_dict[key]
    
    current_dict[keys[-1]] = value

def set_attr_from_path(obj, path, value):
    parts = path.split(".")
    current = obj

    for part in parts[:-1]:
        current = getattr(current, part)

    setattr(current, parts[-1], value)

def get_attr_from_path(obj, path):
    parts = path.split(".")
    current = obj

    for part in parts:
        current = getattr(current, part)

    return current

def deep_merge(base, override):
    current_dict = recursive_attrdict(base.copy())
    for key, val in override.items():
        if key in current_dict and isinstance(current_dict[key], dict) and isinstance(val, dict):
            current_dict[key] = deep_merge(current_dict[key], val)
        else:
            current_dict[key] = val
    return current_dict

def expand_dot_keys(d):
    result = {}

    for key, value in d.items():

        # If value is a dictionary, expand it first
        if isinstance(value, dict):
            value = expand_dot_keys(value)

        parts = key.split(".")
        current = result

        # Walk down the path
        for part in parts[:-1]:
            
            if part not in current:
                current[part] = {}

            current = current[part]

        if (
            parts[-1] in current
            and isinstance(current[parts[-1]], dict)
            and isinstance(value, dict)
        ):
            current[parts[-1]].update(value)
        else:
            current[parts[-1]] = value

    return result

def flatten_dict(d, parent_key=''):
    items = AttrDict({})

    for key, value in d.items():
        new_key = f"{parent_key}.{key}" if parent_key else key

        if isinstance(value, dict) and value:
            items.update(flatten_dict(value, new_key))
        else:
            items[new_key] = value

    return items

def trim_dict(d, keys):
    return {k: d[k] for k in keys if k in d}

def extract_attributes(obj, keys):
    return {k: getattr(obj, k) for k in keys if hasattr(obj, k)}

def extract_attributes_exclude(obj, exclude=()):
    excl = set(exclude)
    return {k: v for k, v in vars(obj).items() if k not in excl}

def expand_to_array_if_scalar(x, n):
    x = np.asarray(x)

    if x.ndim == 0 or x.size == 1:
        return np.full(n, x)
    
    return x

def resolve_fcn(fcn_string, fcns=None):
    if fcns and isinstance(fcn_string, str):
        if any(op in fcn_string for op in ('<=', '>=', ' and ', ' or ', ' implies ')):
            from trajopt.core.constraints.stl import parse_stl_expression
            return parse_stl_expression(fcn_string, fcns)

        if 'fcns.' in fcn_string:
            ns = {}
            for name, fn in fcns.items():
                if 'fcns' in inspect.signature(fn).parameters:
                    fn = partial(fn, fcns=fcns)
                ns[name] = _FcnExpr(fn)
            result = eval(fcn_string.replace('fcns.', ''), {'__builtins__': {}}, ns)
            return result._fcn if isinstance(result, _FcnExpr) else result

        from trajopt.utils.config_loader import resolve_function_from_path
        return resolve_function_from_path(fcn_string)