import importlib.resources
import importlib.util
import inspect
from functools import partial
from pathlib import Path

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
    if isinstance(d, (AttrDict, dict)):
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

def resolve_function_from_string(fcn_string, fcns=None):
    if isinstance(fcn_string, str):
        has_stl = any(op in fcn_string for op in ('<=', '>=', ' and ', ' or ', ' implies '))
        has_fcns_ref = 'fcns.' in fcn_string

        if fcns is not None:
            if not has_stl and not has_fcns_ref:
                raise ValueError(
                    f"Function reference '{fcn_string}' is not allowed directly in constraint/cost configs. "
                    f"Add it to the 'fcns' dict at the top of your YAML and reference it as 'fcns.{fcn_string.rsplit(':', 1)[-1]}'."
                )

            if has_stl:
                from trajopt.core.constraints.stl import parse_stl_expression
                return parse_stl_expression(fcn_string, fcns)

            if has_fcns_ref:
                ns = {}
                for name, fn in fcns.items():
                    if isinstance(fn, str):
                        fn = resolve_function_from_string(fn)
                    if 'fcns' in inspect.signature(fn).parameters:
                        fn = partial(fn, fcns=fcns)
                    ns[name] = _FcnExpr(fn)
                result = eval(fcn_string.replace('fcns.', ''), {'__builtins__': {}}, ns)
                return result._fcn if isinstance(result, _FcnExpr) else result

    file_path_str, func_name = fcn_string.rsplit(':', 1)
    if file_path_str.startswith('trajopt/') or file_path_str.startswith('/trajopt/'):
        file_path_str = file_path_str.lstrip('/')
        parts = file_path_str.split('/')
        file_path = importlib.resources.files('.'.join(parts[:-1])).joinpath(parts[-1])
    else:
        file_path = Path(file_path_str)

    try:
        spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        raise type(e)(f"could not load file '{file_path}' (from '{fcn_string}'): {e}") from None

    try:
        return getattr(module, func_name)
    except AttributeError:
        available = [n for n, obj in inspect.getmembers(module, inspect.isfunction) if obj.__module__ == module.__name__]
        raise AttributeError(
            f"function '{func_name}' not found in '{file_path}'\n"
            f"  requested: '{fcn_string}'\n"
            f"  available: {available}"
        ) from None
    
def update_problem_from_config(problem, updated_config_vals_flat, nondim):
    for path, value in updated_config_vals_flat.items():
        
        # update constraint value from config
        if path.startswith("constraints."):
            constraint_name = path.split(".")[1]
            path_to_spec = ".".join(path.split(".")[2:])

            constraint = problem.constraints.get(name=constraint_name)[0]
            set_attr_from_path(constraint, path_to_spec, value)
        
        # update cost value from config changes 
        if path.startswith("costs."):
            cost_name = path.split(".")[1]

            if cost_name in problem.costs.names:
                cost = problem.costs.get(name=cost_name)[0]
                path_to_spec = ".".join(path.split(".")[2:])
                set_attr_from_path(cost, path_to_spec, value)
        
        if path.startswith("params."):
            path_to_param = path.split("params.")[1]
            set_attr_from_path(problem.params, path_to_param, value)
        
    # Nondimensionalize constraints that were just updated 
    updated_constraint_names = set()
    for path in updated_config_vals_flat.keys():
        if path.startswith("constraints."):
            constraint_name = path.split(".")[1]
            updated_constraint_names.add(constraint_name)
    
    for constraint in problem.constraints.get():
        if constraint.name in updated_constraint_names:
            constraint.nondim_constraint(nondim)

ITER_DATA_KEYS_TO_KEEP = {
    "iter_num", "converged", "cost", "solve_time", "prop_time", "parse_time", "t_full",
    "t_opt", "z_opt", "nu_opt", "dt_opt", "T_opt",
    "t_nl", "z_nl", "nu_nl", "t_init", "z_init", "nu_init", "t_init_nl", "z_init_nl", "nu_init_nl",
    "constraint_data", "trajectory_data", "conv_data", "W", "dual", "penalty",
}

METHOD_DATA_KEYS_TO_KEEP = {
    'time_grid', 'N_dens', 'Npm', 'T_init', 'T_max', 'T_min', 'Ts_init', 'conv', 'conv_data', 'cost_init', 
    'dT_max', 'ddt_max', 'dt_max', 'dt_min', 'flags', 'line_guess_u_init',
    'name', 'n_minus', 'n_plus', 'nl_guess_u_start', 'nl_guess_u_stop', 'solver_opts',
    'nondim', 'initial_guess', 'penalty', 'z_ind'
}