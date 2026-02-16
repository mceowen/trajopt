import numpy as np
import cvxpy as cp

# TODO: just condense into a single function (not both get_val, safe_val)

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
    
 
def get_val(var, rows=1, cols=1, fallback=0.0):
    if hasattr(var, "value"):
        val = var.value
        if val is not None:
            return val
        return safe_val(var, fallback=fallback, rows=rows, cols=cols)
    return var

def safe_array(M):
    return np.array([0.0]) if M is None or np.size(M) == 0 else M

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

def num_timesteps(z):
    if zs.ndim == 1:
        return 1
    elif zs.ndim == 2:
        return zs.shape[0]
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")

def deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def extract_attributes(obj, names):
    return {k: getattr(obj, k) for k in names if hasattr(obj, k)}

def extract_attributes_exclude(obj, exclude=()):
    excl = set(exclude)
    return {k: v for k, v in vars(obj).items() if k not in excl}