import importlib.util
import inspect
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import jax
import numpy as np

class AttrDict(dict):
    """Dictionary that allows attribute access to keys.

    Example: d = AttrDict({'a':1}); d.a == 1; d['a'] == 1
    """

    def __getattr__(self, name: str) -> Any:
        """Return the dict value for name, raising AttributeError if missing."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set the dict value for name."""
        self[name] = value

# register the AttrDict class with jax to make it traceable

jax.tree_util.register_pytree_node(AttrDict,
                                   flatten_func=lambda d: (list(d.values()), list(d.keys())),
                                   unflatten_func=lambda keys, vals: AttrDict(zip(keys, vals)),
                                   )

def recursive_attrdict(d: dict | list) -> Any:
    """Recursively convert a nested dict into a nested AttrDict."""
    if isinstance(d, dict):
        return AttrDict({k: recursive_attrdict(v) for k, v in d.items()})
    if isinstance(d, list):
        return [recursive_attrdict(i) for i in d]
    return d

def recursive_to_dict(d: dict | list) -> Any:
    """Recursively convert a nested AttrDict into a plain dict."""
    if isinstance(d, (AttrDict, dict)):
        return {k: recursive_to_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [recursive_to_dict(i) for i in d]
    return d

def get_from_path(d: dict, path: str) -> Any:
    """Retrieve a value from a nested dict using a dot-separated path."""
    keys = path.split(".")
    current_dict = d

    for key in keys:
        current_dict = current_dict[key]

    return current_dict

def set_from_path(d: dict, path: str, value: Any) -> None:
    """Set a value in a nested dict using a dot-separated path."""
    keys = path.split(".")
    current_dict = d

    for key in keys[:-1]:
        current_dict = current_dict[key]

    current_dict[keys[-1]] = value

def deep_merge(base: dict, override: dict) -> "AttrDict":
    """Recursively merge override into base, returning a new AttrDict."""
    current_dict = recursive_attrdict(base.copy())
    for key, val in override.items():
        if key in current_dict and isinstance(current_dict[key], dict) and isinstance(val, dict):
            current_dict[key] = deep_merge(current_dict[key], val)
        else:
            current_dict[key] = val
    return current_dict

def expand_dot_keys(d: dict) -> dict:
    """Expand dot-notation keys in a dict into nested dicts."""
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

def flatten_dict(d: dict, parent_key: str = "") -> "AttrDict":
    """Flatten a nested dict into a single-level AttrDict with dot-separated keys."""
    items: AttrDict = AttrDict({})

    for key, value in d.items():
        new_key = f"{parent_key}.{key}" if parent_key else key

        if isinstance(value, dict) and value:
            items.update(flatten_dict(value, new_key))
        else:
            items[new_key] = value

    return items

def expand_to_array_if_scalar(x: Any, n: int) -> np.ndarray:
    """Broadcast a scalar or single-element array to length n."""
    x = np.asarray(x)

    if x.ndim == 0 or x.size == 1:
        return np.full(n, x)

    return x

def resolve_function_from_string(fcn_string: str, fcns: "AttrDict | None" = None) -> Callable:
    """Resolve a function reference string to a callable."""
    if isinstance(fcn_string, str):
        has_stl = any(op in fcn_string for op in ("<=", ">=", " and ", " or ", " implies "))
        has_fcns_ref = "fcns." in fcn_string

        if fcns is not None:
            if not has_stl and not has_fcns_ref:
                msg = (
                    f"Function reference '{fcn_string}' is not allowed directly in constraint/cost configs. "
                    f"Add it to the 'fcns' dict at the top of your YAML and reference it as 'fcns.{fcn_string.rsplit(':', 1)[-1]}'."
                )
                raise ValueError(
                    msg,
                )

            if has_stl:
                from trajopt.constraints.stl import parse_stl_expression
                return parse_stl_expression(fcn_string, fcns)

            if has_fcns_ref:
                ns = {}
                for name, fn in fcns.items():
                    if isinstance(fn, str):
                        fn = resolve_function_from_string(fn)
                    if "fcns" in inspect.signature(fn).parameters:
                        fn = partial(fn, fcns=fcns)
                    ns[name] = _FcnExpr(fn)
                result = eval(fcn_string.replace("fcns.", ""), {"__builtins__": {}}, ns)
                return result._fcn if isinstance(result, _FcnExpr) else result

    file_path_str, func_name = fcn_string.rsplit(":", 1)
    file_path = Path(file_path_str).resolve()

    parent_dir = file_path.parent
    pkg_init = parent_dir / "__init__.py"

    try:
        if pkg_init.exists():
            pkg_name = parent_dir.name
            mod_name = f"{pkg_name}.{file_path.stem}"

            if pkg_name not in sys.modules:
                pkg_spec = importlib.util.spec_from_file_location(
                    pkg_name, pkg_init, submodule_search_locations=[str(parent_dir)],
                )
                pkg_mod = importlib.util.module_from_spec(pkg_spec)
                sys.modules[pkg_name] = pkg_mod
                pkg_spec.loader.exec_module(pkg_mod)

            spec = importlib.util.spec_from_file_location(mod_name, file_path)
            module = importlib.util.module_from_spec(spec)
            module.__package__ = pkg_name
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
        else:
            spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    except Exception as e:
        raise type(e)(f"could not load file '{file_path}' (from '{fcn_string}'): {e}") from None

    try:
        fn = getattr(module, func_name)
    except AttributeError:
        available = [n for n, obj in inspect.getmembers(module, inspect.isfunction) if obj.__module__ == module.__name__]
        raise AttributeError(
            f"function '{func_name}' not found in '{file_path}'\n"
            f"  requested: '{fcn_string}'\n"
            f"  available: {available}",
        ) from None

    if fcns is not None and "fcns" in inspect.signature(fn).parameters:
        fn = partial(fn, fcns=fcns)
    return fn

class _FcnExpr:
    """Lightweight wrapper enabling arithmetic on (t, x, u, params) callables."""

    def __init__(self, fcn: Callable) -> None:
        """Wrap fcn in a _FcnExpr to enable arithmetic composition."""
        self._fcn = fcn

    def __mul__(self, other: Any) -> "_FcnExpr":
        """Return element-wise product of this callable with other."""
        f = self._fcn
        if isinstance(other, _FcnExpr):
            g = other._fcn
            return _FcnExpr(lambda t, x, u, params: f(t, x, u, params) * g(t, x, u, params))
        if isinstance(other, list):
            other = np.array(other)
        return _FcnExpr(lambda t, x, u, params: f(t, x, u, params) * other)

    def __rmul__(self, other: Any) -> "_FcnExpr":
        """Return element-wise product of other with this callable."""
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "_FcnExpr":
        """Return element-wise quotient of this callable divided by other."""
        f = self._fcn
        if isinstance(other, _FcnExpr):
            g = other._fcn
            return _FcnExpr(lambda t, x, u, params: f(t, x, u, params) / g(t, x, u, params))
        return _FcnExpr(lambda t, x, u, params: f(t, x, u, params) / other)

    def __add__(self, other: Any) -> "_FcnExpr":
        """Return element-wise sum of this callable with other."""
        f = self._fcn
        if isinstance(other, _FcnExpr):
            g = other._fcn
            return _FcnExpr(lambda t, x, u, params: f(t, x, u, params) + g(t, x, u, params))
        return _FcnExpr(lambda t, x, u, params: f(t, x, u, params) + other)

    def __radd__(self, other: Any) -> "_FcnExpr":
        """Return element-wise sum of other with this callable."""
        return self.__add__(other)

    def __sub__(self, other: Any) -> "_FcnExpr":
        """Return element-wise difference of this callable minus other."""
        f = self._fcn
        if isinstance(other, _FcnExpr):
            g = other._fcn
            return _FcnExpr(lambda t, x, u, params: f(t, x, u, params) - g(t, x, u, params))
        return _FcnExpr(lambda t, x, u, params: f(t, x, u, params) - other)

    def __neg__(self) -> "_FcnExpr":
        """Return element-wise negation of this callable."""
        f = self._fcn
        return _FcnExpr(lambda t, x, u, params: -f(t, x, u, params))