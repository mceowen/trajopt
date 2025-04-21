import numpy as np

def safe_val(var, rows=1, cols=1, fallback=0.0):
    if var is not None and var.value is not None:
        return var.value
    return fallback if (rows == 1 and cols == 1) else np.full((rows, cols), fallback)
 
def get_val(var, rows=1, cols=1, fallback=0.0):
    if hasattr(var, "value"):
        val = var.value
        if val is not None:
            return val
        return safe_val(var, fallback=fallback, rows=rows, cols=cols)
    return var

def constraint_index_selector(min_idx, max_idx, n_elem):
    M_min = -np.eye(n_elem)[min_idx, :]
    M_max = np.eye(n_elem)[max_idx, :]

    M_select = np.vstack([M_min, M_max])
    return M_select


def num_timesteps(zs):
    if zs.ndim == 1:
        return 1
    elif zs.ndim == 2:
        return zs.shape[1]
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")