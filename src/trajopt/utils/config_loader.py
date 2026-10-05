import importlib
import importlib.resources
import importlib.util
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from trajopt.utils.tools import (
    AttrDict,
    deep_merge,
    expand_dot_keys,
    flatten_dict,
    recursive_attrdict,
)

# =============================================================================
# MAIN CONFIG LOADER
# =============================================================================

def load_trajopt_config(config_path: str) -> AttrDict:
    """Load a single config file, resolving all inheritance and expressions."""
    config = load_yaml(config_path)
    config = _resolve_inheritance(config, _source=config_path)
    try:
        return _eval_values(config, {"np": np}, phase_params={})
    except Exception as e:
        raise type(e)(f"error evaluating expressions in '{config_path}': {e}") from None

# =============================================================================
# METHOD CLASS RESOLUTION
# =============================================================================

DEFAULT_METHOD_CLASS      = "dev.scp_phases"
DEFAULT_FORMULATION_CLASS = "dev.phases"


def resolve_scp_method_class(method_config: AttrDict):
    """The ``Method`` class exported by the ``trajopt.methods.<method_class>`` package."""
    method_class = method_config.get("method_class", DEFAULT_METHOD_CLASS)
    return _import_package_attr("trajopt.methods", method_class, "Method")


def resolve_formulation_trajectory_class(method_config: AttrDict):
    """The ``Trajectory`` class exported by the ``trajopt.formulations.<formulation_class>`` package."""
    formulation_class = method_config.get("formulation_class", DEFAULT_FORMULATION_CLASS)
    return _import_package_attr("trajopt.formulations", formulation_class, "Trajectory")


def _import_package_attr(root: str, name: str, attr: str):
    target = f"{root}.{name}"
    try:
        package = importlib.import_module(target)
    except ModuleNotFoundError as e:
        # only the package itself being absent means a bad name; anything else is a real import error
        if e.name is None or not (target == e.name or target.startswith(e.name + ".")):
            raise
        raise ValueError(f"unknown class '{name}': no package {target}") from None
    if not hasattr(package, attr):
        raise ValueError(f"package {root}.{name} does not export '{attr}' from its __init__.py")
    return getattr(package, attr)

# =============================================================================
# YAML LOADING
# =============================================================================

def load_yaml(path_str: str) -> AttrDict:
    """Load a YAML file, resolving .py:func paths relative to the file's directory."""
    path     = Path(path_str)
    base_dir = path.resolve().parent

    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"config not found: '{path_str}'") from None

    config = expand_dot_keys(recursive_attrdict(raw))
    return _resolve_fcn_paths(config, base_dir)

# =============================================================================
# INHERITANCE
# =============================================================================

def _resolve_inherit_path(raw: str, _source: str) -> str:
    """A single 'inherit' entry resolved to an absolute file path."""
    if raw.startswith("trajopt/"):
        parts = raw.lstrip("/").split("/")
        return str(importlib.resources.files(".".join(parts[:-1])).joinpath(parts[-1]))
    source_dir = Path(_source).resolve().parent if _source != "unknown" else Path.cwd()
    return str((source_dir / raw).resolve())


def _resolve_inheritance(d: Any, _source: str = "unknown") -> Any:
    """Recursively resolve 'inherit' keys (a path or list of paths, later wins), merging parent configs in."""
    if not isinstance(d, dict):
        return d

    d = {k: _resolve_inheritance(v, _source=_source) for k, v in d.items()}

    if "inherit" in d:
        raw = d["inherit"]
        paths = raw if isinstance(raw, list) else [raw]
        merged: dict = {}
        for path in paths:
            parent_path = _resolve_inherit_path(path, _source)
            try:
                parent = _resolve_inheritance(load_yaml(parent_path), _source=parent_path)
                merged = deep_merge(merged, parent)
            except Exception as e:
                raise type(e)(f"error resolving 'inherit: {path}' (from '{_source}'): {e}") from None
        d = deep_merge(merged, d)
        d.pop("inherit", None)

    return recursive_attrdict(d)

# =============================================================================
# FUNCTION PATH RESOLUTION
# =============================================================================

def _resolve_fcn_paths(d: Any, base_dir: Path) -> Any:
    """Resolve file.py:func references to absolute paths relative to base_dir."""
    if isinstance(d, dict):
        return {k: _resolve_fcn_paths(v, base_dir) for k, v in d.items()}
    if isinstance(d, list):
        return [_resolve_fcn_paths(item, base_dir) for item in d]
    if isinstance(d, str) and ".py:" in d:
        file_part, func = d.rsplit(":", 1)
        fp = Path(file_part)

        # Support package-rooted references like "trajopt/.../file.py:func".
        # These should resolve from the installed/importable trajopt package root,
        # not relative to the local config directory.
        if file_part.startswith("trajopt/"):
            spec = importlib.util.find_spec("trajopt")
            if spec is None or spec.origin is None:
                raise ModuleNotFoundError("could not resolve package root for 'trajopt'")
            pkg_root = Path(spec.origin).resolve().parent
            fp = (pkg_root / file_part[len("trajopt/"):]).resolve()
        elif not fp.is_absolute():
            fp = (base_dir / fp).resolve()

        return f"{fp}:{func}"
    return d

# =============================================================================
# EXPRESSION EVALUATION
# =============================================================================

def _bind_params(ctx: dict, params: Any) -> None:
    """Add ``params.<path>`` lookups for every scalar in a params tree."""
    for key, val in flatten_dict(params).items():
        ctx[f"params.{key}"] = val


def _eval_expr(expr: str, ctx: dict) -> Any:
    """Evaluate a Python expression, resolving dotted references from ctx."""
    local = dict(ctx)
    for ref in sorted(set(re.findall(r"\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)\b", expr)), key=len, reverse=True):
        val = ctx.get(ref)
        if val is None:
            continue
        safe = ref.replace(".", "_")
        local[safe] = val
        expr = expr.replace(ref, safe)
    return eval(expr, local)


def _eval_string(expr_obj: str, ctx: dict) -> Any:
    m = re.fullmatch(r"\$\{([^}]+)\}", expr_obj.strip())
    if m:
        return _eval_expr(m.group(1), ctx)
    return re.sub(
        r"\$\{([^}]+)\}",
        lambda match: str(_eval_expr(match.group(1), ctx)),
        expr_obj,
    )


def _eval_params(params: dict, ctx: dict) -> AttrDict:
    """Evaluate a phase ``params`` block; sibling keys are visible in order."""
    result = AttrDict({})
    local  = dict(ctx)
    for key, val in params.items():
        result[key] = _eval_values(val, local, phase_params={})
        if not isinstance(result[key], dict):
            local[key] = result[key]
    return result


def _eval_phase(phase: dict, name: str, phase_params: dict) -> AttrDict:
    """Evaluate one phase: ``params`` first, then all remaining keys."""
    ctx = {"np": np}
    out = AttrDict({})

    if "params" in phase:
        out.params = _eval_params(phase.params, ctx)
        phase_params[name] = out.params
        _bind_params(ctx, out.params)
    elif name in phase_params:
        _bind_params(ctx, phase_params[name])

    for key, val in phase.items():
        if key == "params":
            continue
        out[key] = _eval_values(val, ctx, phase_params)

    return out


def _eval_values(obj: Any, ctx: dict, phase_params: dict, key: str | None = None) -> Any:
    """Recursively evaluate ``${...}`` expressions in a config tree."""
    if isinstance(obj, dict):
        if "phases" in obj and isinstance(obj.phases, dict):
            result = AttrDict(dict(obj))
            result.phases = AttrDict({
                name: _eval_phase(seg, name, phase_params)
                for name, seg in obj.phases.items()
            })
            for key, val in obj.items():
                if key == "phases":
                    continue
                result[key] = _eval_values(val, ctx, phase_params)
            return result

        if "params" in obj and isinstance(obj.params, dict):
            # A flat (single-phase) trajectory/phase-like dict: evaluate its
            # own params first so siblings can reference them as ``${params.x}``,
            # same as a named phase under a multi-phase ``phases:`` block.
            return _eval_phase(obj, key or "trajectory", phase_params)

        result = AttrDict({})
        local  = dict(ctx)

        remaining = obj
        if "trajectory" in obj and "method" in obj:
            # root config: eval trajectory first so method: can also reference ${params.x}
            result["trajectory"] = _eval_values(obj["trajectory"], local, phase_params, key="trajectory")
            merged_params: dict = {}
            for p in phase_params.values():
                merged_params = deep_merge(merged_params, p)
            if merged_params:
                _bind_params(local, merged_params)
            remaining = {k: v for k, v in obj.items() if k != "trajectory"}

        for k, v in remaining.items():
            result[k] = _eval_values(v, local, phase_params, key=k)
            bare = result[k]
            if not isinstance(bare, dict):
                local[k] = bare
        return result

    if isinstance(obj, list):
        results = [_eval_values(item, ctx, phase_params) for item in obj]
        if all(isinstance(x, (int, float, np.number, np.ndarray)) for x in results):
            arr = np.array(results)
            if key and "idx" in key and arr.dtype.kind == "f" and np.all(arr == np.round(arr)):
                return np.round(arr).astype(np.intp)
            return arr
        return results

    if isinstance(obj, str) and "${" in obj:
        return _eval_values(_eval_string(obj, ctx), ctx, phase_params)

    return obj
