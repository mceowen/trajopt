import numpy as np

def constraint_index_selector(min_idx, max_idx, n_elem):
    """
    Constraint index selector.

    Constructs a matrix M such that:
        z_out = M @ z  =>  z_out <= [-z_min; z_max]

    Args:
        min_idx (list or array-like): Indices corresponding to lower bounds (z >= z_min)
        max_idx (list or array-like): Indices corresponding to upper bounds (z <= z_max)
        n_elem (int): Total number of elements in z

    Returns:
        np.ndarray: Constraint selection matrix M of shape (len(min_idx) + len(max_idx), n_elem)
    """
    M_min = -np.eye(n_elem)[min_idx, :]
    M_max = np.eye(n_elem)[max_idx, :]

    M_select = np.vstack([M_min, M_max])
    return M_select


def num_timesteps(zs):
    """
    Returns the number of timesteps (i.e., the second dimension)
    of a state/control trajectory array.

    Parameters:
        zs : np.ndarray (shape: (n,), (n,1), or (n,N))

    Returns:
        N : int, number of time steps
    """
    if zs.ndim == 1:
        return 1
    elif zs.ndim == 2:
        return zs.shape[1]
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")