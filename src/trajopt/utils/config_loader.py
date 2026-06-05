import importlib.resources
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
        return _eval_values(config, {"np": np}, segment_params={})
    except Exception as e:
        raise type(e)(f"error evaluating expressions in '{config_path}': {e}") from None

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

def _resolve_inheritance(d: Any, _source: str = "unknown") -> Any:
    """Recursively resolve 'inherit' keys, merging parent configs in."""
    if not isinstance(d, dict):
        return d

    d = {k: _resolve_inheritance(v, _source=_source) for k, v in d.items()}

    if "inherit" in d:
        raw = d["inherit"]
        if raw.startswith("trajopt/"):
            parts = raw.lstrip("/").split("/")
            parent_path = str(importlib.resources.files(".".join(parts[:-1])).joinpath(parts[-1]))
        else:
            source_dir  = Path(_source).resolve().parent if _source != "unknown" else Path.cwd()
            parent_path = str((source_dir / raw).resolve())
        try:
            parent = _resolve_inheritance(load_yaml(parent_path), _source=parent_path)
            d = deep_merge(parent, d)
        except Exception as e:
            raise type(e)(f"error resolving 'inherit: {raw}' (from '{_source}'): {e}") from None
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
        if not fp.is_absolute():
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
    """Evaluate a segment ``params`` block; sibling keys are visible in order."""
    result = AttrDict({})
    local  = dict(ctx)
    for key, val in params.items():
        result[key] = _eval_values(val, local, segment_params={})
        if not isinstance(result[key], dict):
            local[key] = result[key]
    return result


def _eval_segment(segment: dict, name: str, segment_params: dict) -> AttrDict:
    """Evaluate one segment: ``params`` first, then all remaining keys."""
    ctx = {"np": np}
    out = AttrDict({})

    if "params" in segment:
        out.params = _eval_params(segment.params, ctx)
        segment_params[name] = out.params
        _bind_params(ctx, out.params)
    elif name in segment_params:
        _bind_params(ctx, segment_params[name])

    for key, val in segment.items():
        if key == "params":
            continue
        out[key] = _eval_values(val, ctx, segment_params)

    return out


def _eval_values(obj: Any, ctx: dict, segment_params: dict, key: str | None = None) -> Any:
    """Recursively evaluate ``${...}`` expressions in a config tree."""
    if isinstance(obj, dict):
        if "segments" in obj and isinstance(obj.segments, dict):
            result = AttrDict(dict(obj))
            result.segments = AttrDict({
                name: _eval_segment(seg, name, segment_params)
                for name, seg in obj.segments.items()
            })
            for key, val in obj.items():
                if key == "segments":
                    continue
                result[key] = _eval_values(val, ctx, segment_params)
            return result

        result = AttrDict({})
        local  = dict(ctx)
        for k, v in obj.items():
            result[k] = _eval_values(v, local, segment_params, key=k)
            bare = result[k]
            if not isinstance(bare, dict):
                local[k] = bare
        return result

    if isinstance(obj, list):
        results = [_eval_values(item, ctx, segment_params) for item in obj]
        if all(isinstance(x, (int, float, np.number, np.ndarray)) for x in results):
            arr = np.array(results)
            if key and "idx" in key and arr.dtype.kind == "f" and np.all(arr == np.round(arr)):
                return np.round(arr).astype(np.intp)
            return arr
        return results

    if isinstance(obj, str) and "${" in obj:
        return _eval_values(_eval_string(obj, ctx), ctx, segment_params)

    return obj
