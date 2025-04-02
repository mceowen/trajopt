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