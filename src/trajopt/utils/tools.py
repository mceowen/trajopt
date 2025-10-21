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

def constraint_index_selector(min_idx, max_idx, n_elem):
    M_min = -np.eye(n_elem)[min_idx, :]
    M_max = np.eye(n_elem)[max_idx, :]

    M_select = np.vstack([M_min, M_max])
    return M_select

def load_dict(yaml_file):
    '''
    Loads dictionary from a provided yaml file and converts all lists to np arrays 
    unless the array stores tuples of strings.
    '''

    with open(yaml_name, 'r') as file:
        dict_unconverted = yaml.safe_load(file)
    
    dict_converted = convert_list(dict_unconverted)

    return dict_converted

def convert_list(dictionary):
    '''
    Converts all lists to arrays with exception of lists of tuples and strings.
    '''

    temp_dict = {}

    for key, value in dictionary.items():
        if isinstance(value, dict):
            temp_dict[key] = convert_list(value)
        elif isinstance(value, list):
            if all(isinstance(item, dict) for item in value):
                temp_dict[key] = [convert_list(item) for item in value]
            elif all(isinstance(item, (tuple, str)) for item in value):
                temp_dict[key] = value
            else:
                temp_dict[key] = np.array(value)
        else:
            temp_dict[key] = value

    return temp_dict

def num_timesteps(zs):
    if zs.ndim == 1:
        return 1
    elif zs.ndim == 2:
        return zs.shape[0]
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")