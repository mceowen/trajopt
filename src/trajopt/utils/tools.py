import inspect
from functools import partial

import numpy as np
import cvxpy as cp


class _FcnExpr:
    """Lightweight wrapper enabling arithmetic on (t, x, u, params) callables."""
    def __init__(self, fcn):
        self._fcn = fcn

    def __mul__(self, other):
        f = self._fcn
        if isinstance(other, _FcnExpr):
            g = other._fcn
            return _FcnExpr(lambda t, x, u, params: f(t, x, u, params) * g(t, x, u, params))
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
    
def recursive_attrdict(d):
    if isinstance(d, dict):
        return AttrDict({k: recursive_attrdict(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [recursive_attrdict(i) for i in d]
    else:
        return d
    
def recursive_to_dict(d):
    if isinstance(d, dict):
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

def safe_val(var, rows=1, cols=1, fallback=0.0):
    """
    Safely extract the numeric value from a cvxpy expression or return a fallback.

    Parameters:
    -----------
    var : any
        The input value to extract. Can be a cvxpy Expression, a scalar, or None.
    
    rows : int, optional (default=1)
        Number of rows for fallback matrix if needed.
    
    cols : int, optional (default=1)
        Number of columns for fallback matrix if needed.
    
    fallback : float, optional (default=0.0)
        The default value to return if `var` is None or has no numeric value.

    Returns:
    --------
    numeric or np.ndarray
        - If `var` is not a cvxpy Expression and is not None, returns `var` directly.
        - If `var` is a cvxpy Expression with a valid `.value`, returns `var.value`.
        - If `var` is None or has no valid `.value`, returns:
            - `fallback` if (rows == 1 and cols == 1)
            - `np.full((rows, cols), fallback)` otherwise

    """
    if not isinstance(var, cp.Expression):
        # pass value through if it is not a cvxpy object
        if var is not None:
            return var
    else:
        # use the value or fallback if it is a cvxpy object
        if var.value is not None:
            return var.value
    # fallback if var or var.value is None
    return fallback if (rows == 1 and cols == 1) else np.full((rows, cols), fallback)

def ensure_shape(M, shape):
    """
    Safely broadcast, pad, or trim an array to the requested shape.
    Works with scalars, empty arrays, and mismatched shapes.
    """
    if M is None:
        return np.zeros(shape)

    if np.isscalar(M):
        return np.full(shape, float(M))

    M = np.asarray(M)

    # Empty or zero-column arrays → zeros
    if M.size == 0 or 0 in M.shape:
        return np.zeros(shape)

    # Perfect match → return directly
    if M.shape == shape:
        return M

    # Oversized array → safely trim
    if np.prod(M.shape) > np.prod(shape):
        return M[: shape[0], : shape[1]] if M.ndim == 2 else M[: shape[0]]

    # Broadcast smaller array up to shape
    try:
        return np.broadcast_to(M, shape)
    except ValueError:
        # Fallback reshape + broadcast
        return np.broadcast_to(M.reshape(-1, 1) if M.ndim == 1 else M, shape)
    
def get_val(var, rows=1, cols=1, fallback=0.0):
    if hasattr(var, "value"):
        val = var.value
        if val is not None:
            return val
        return safe_val(var, fallback=fallback, rows=rows, cols=cols)
    return var

def safe_array(M):
    return np.array([0.0]) if M is None or np.size(M) == 0 else M

def reshape_1d(x):
    return np.asarray(x).reshape(-1)