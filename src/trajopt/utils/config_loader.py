"""YAML config loader with inheritance, dot-notation, and expression evaluation.

Path resolution:
- inherit paths: resolved relative to the declaring file. Paths starting
  with 'trajopt/' are resolved to the installed package (src/trajopt/).
- .py:func paths: always resolved relative to the declaring YAML file.
"""

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

    problem = config.get("problem", AttrDict({}))
    method  = config.get("method", AttrDict({}))

    ctx = {**flatten_dict(problem), **flatten_dict(method), "np": np}
    try:
        problem = _eval_values(problem, ctx)
        method  = _eval_values(method, ctx)
    except Exception as e:
        raise type(e)(f"error evaluating expressions in '{config_path}': {e}") from None

    problem.setdefault("trajectories", {})

    for c in problem.constraints.values():
        c.setdefault("ct", 0)

    config.problem = problem
    config.method  = method

    return config

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
    """Recursively resolve 'inherit' keys, merging parent configs in.

    Inherit paths are resolved relative to the file that declares them.
    """
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
    """Resolve file.py:func references to absolute paths relative to base_dir.

    Called at load time so that after inheritance merging, every function
    reference carries its own absolute path regardless of which file declared it.
    """
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

def _eval_expr(expr: str, ctx: dict) -> Any:
    """Evaluate a Python expression, resolving dotted references from ctx."""
    refs = re.findall(r"\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)\b", expr)
    local = ctx
    for ref in sorted(refs, key=len, reverse=True):
        val = ctx.get(ref) or ctx.get(ref.replace(".", "_"))
        if val is not None:
            safe = ref.replace(".", "_")
            local[safe] = val
            expr = expr.replace(ref, safe)
    return eval(expr, local)


def _eval_values(obj: Any, ctx: dict, key: str | None = None, _path: str = "") -> Any:
    """Recursively evaluate ${} expressions in a config tree."""
    if isinstance(obj, dict):
        result = AttrDict({})
        for k, v in obj.items():
            local_ctx = dict(ctx)
            for sk, sv in obj.items():
                bare = result[sk] if sk in result else sv
                if not isinstance(bare, dict):
                    local_ctx[sk] = bare
            evaluated = _eval_values(v, local_ctx, key=k, _path=f"{_path}.{k}")
            result[k] = evaluated
            ctx[f"{_path}.{k}".lstrip(".")] = evaluated
        return result

    if isinstance(obj, list):
        results = [_eval_values(item, ctx, _path=f"{_path}[{i}]") for i, item in enumerate(obj)]
        if all(isinstance(x, (int, float, np.number, np.ndarray)) for x in results):
            arr = np.array(results)
            if key and "idx" in key and arr.dtype.kind == "f" and np.all(arr == np.round(arr)):
                arr = np.round(arr).astype(np.intp)
            return arr
        return results

    if isinstance(obj, str) and "${" in obj:
        m = re.fullmatch(r"\$\{([^}]+)\}", obj.strip())
        if m:
            try:
                return _eval_values(_eval_expr(m.group(1), ctx), ctx, _path=_path)
            except Exception as e:
                raise type(e)(f"error evaluating '${{{m.group(1)}}}' at '{_path}': {e}") from None

        def _sub(match):
            try:
                return str(_eval_expr(match.group(1), ctx))
            except Exception as e:
                raise type(e)(f"error evaluating '${{{match.group(1)}}}' at '{_path}': {e}") from None
        return re.sub(r"\$\{([^}]+)\}", _sub, obj)

    return obj