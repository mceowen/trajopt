import numpy as np
import cvxpy as cp

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