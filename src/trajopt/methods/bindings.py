"""Update numeric inputs without rebuilding the prepared problem.

Supported inputs are existing numeric model parameters and initial state
values. Resolved YAML expressions do not change when a model parameter changes.
Changes to time horizons, channel layouts, constraint layouts, or unbound
bounds require a new preparation. Reset the algorithm state after each update.
"""
import copy
from collections.abc import Mapping

import numpy as np

from trajopt.formulations.specs import ProblemSpec, SegmentSpec


class RebuildRequired(ValueError):
    """The requested input change requires a new prepared problem."""


def _numeric_value(old, new, path):
    before, after = np.asarray(old), np.asarray(new)
    if before.dtype.kind not in "iuf" or after.dtype.kind not in "iuf":
        raise RebuildRequired(f"{path}: only existing real numeric values are runtime inputs")
    if before.shape != after.shape:
        raise RebuildRequired(f"{path}: changing a parameter shape requires preparation again")
    if not np.isfinite(after).all():
        raise ValueError(f"{path}: runtime inputs must be finite")
    if before.dtype.kind in "iu" and np.any(after != np.floor(after)):
        raise RebuildRequired(f"{path}: changing an integer parameter's type requires preparation again")
    converted = after.astype(before.dtype)
    if not np.isfinite(converted).all():
        raise ValueError(f"{path}: runtime input exceeds the existing numeric type")
    if before.dtype.kind in "iu" and np.any(converted != after):
        raise ValueError(f"{path}: runtime input exceeds the existing integer type")
    if isinstance(old, np.ndarray):
        return converted.copy()
    if isinstance(old, list):
        return [_numeric_value(item, converted[index], f"{path}[{index}]")
                for index, item in enumerate(old)]
    if isinstance(old, tuple):
        return tuple(_numeric_value(item, converted[index], f"{path}[{index}]")
                     for index, item in enumerate(old))
    return converted.item()


def _merge_params(old, patch, path):
    if not isinstance(patch, Mapping):
        raise TypeError(f"{path} updates must be a mapping")
    result = copy.deepcopy(old)
    for key, value in patch.items():
        if key not in old:
            raise RebuildRequired(f"{path}.{key}: adding model parameters requires preparation again")
        if isinstance(old[key], Mapping):
            result[key] = _merge_params(old[key], value, f"{path}.{key}")
        else:
            result[key] = _numeric_value(old[key], value, f"{path}.{key}")
    return result


def _replace_mapping(target, source):
    """Update parameters in place to preserve references held by model functions."""
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
            _replace_mapping(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _initial_state_update(block, value):
    matches = [(name, constraint) for name, constraint in block.constraints.items()
               if constraint.type == "initial_state"]
    if len(matches) != 1:
        raise RebuildRequired(
            f"{block.name}: initial_state requires exactly one existing initial-state constraint"
        )
    name, constraint = matches[0]
    template = block.problem.config.constraints[name]["value"]
    old = np.asarray(template, dtype=object)
    new = np.asarray(value, dtype=object)
    if old.shape != new.shape or new.ndim != 1:
        raise RebuildRequired(f"{block.name}.initial_state: changing the constrained layout requires preparation again")
    old_mask = np.array([item is not None for item in old])
    new_mask = np.array([item is not None for item in new])
    if not np.array_equal(old_mask, new_mask):
        raise RebuildRequired(f"{block.name}.initial_state: changing the constrained mask requires preparation again")
    numeric = _numeric_value(np.asarray(old[old_mask], dtype=float),
                             np.asarray(new[new_mask].tolist()), f"{block.name}.initial_state")
    if not hasattr(constraint, "value_param"):
        raise RebuildRequired(f"{block.name}.initial_state: this prepared recipe has no numeric binding")
    result = [float(item) if present else None for item, present in zip(new, new_mask)]
    scaled = block.nondim.M.state.d2nd[np.ix_(constraint.constraint.idx,
                                            constraint.constraint.idx)] @ numeric
    if scaled.shape != constraint.value_param.shape:
        raise RebuildRequired(f"{block.name}.initial_state: changing the constrained layout requires preparation again")
    return name, constraint, numeric, scaled, result


def _check_param_consumers(block):
    bound_at_construction = {"convex_inequality", "initial_convex_inequality", "final_convex_inequality"}
    static_constraints = [c.name for c in block.constraints.values() if c.type in bound_at_construction]
    static_costs = [c.name for c in block.costs.values() if c.type in {"convex_terminal", "convex_running"}]
    if static_constraints or static_costs:
        names = ", ".join(static_constraints + static_costs)
        raise RebuildRequired(
            f"{block.name}.params: {names} bind model parameters during construction; prepare again"
        )


def update_inputs(prepared, updates):
    """Update numeric inputs and the saved problem specification.

    ``updates`` maps segment names to ``params``, ``initial_state``, or both.
    ``params`` contains nested updates to existing numeric parameters.
    ``initial_state`` contains dimensional values with the same constrained
    indices as before. All segments pass validation before any input changes.

    Return the same ``PreparedSolve`` instance. Preserve its backend problem,
    variables, compiled functions, and current reference trajectory.
    """
    if not isinstance(updates, Mapping):
        raise TypeError("Runtime updates must map segment names to input updates")
    unknown = set(updates) - set(prepared.segments)
    if unknown:
        raise RebuildRequired(f"Unknown segments {sorted(unknown)}; prepare a new problem")
    staged = {}
    specs = dict(prepared.problem.segments)
    for name, patch in updates.items():
        if not isinstance(patch, Mapping):
            raise TypeError(f"Runtime inputs for {name!r} must be a mapping")
        unsupported = set(patch) - {"params", "initial_state"}
        if unsupported:
            raise RebuildRequired(f"{name}: unbound inputs {sorted(unsupported)} require preparation again")
        if not patch:
            continue
        block = prepared.segments[name]
        config = specs[name].config
        params = None
        boundary = None
        if patch.get("params"):
            _check_param_consumers(block)
            params = _merge_params(block.params, patch["params"], f"{name}.params")
            config.params = copy.deepcopy(params)
        elif "params" in patch and not isinstance(patch["params"], Mapping):
            raise TypeError(f"{name}.params updates must be a mapping")
        if "initial_state" in patch:
            boundary = _initial_state_update(block, patch["initial_state"])
            config.constraints[boundary[0]]["value"] = boundary[-1]
        staged[name] = (params, boundary)
        specs[name] = SegmentSpec(name, config, specs[name].agent)
    if not staged:
        return prepared

    # Construct replacement immutable provenance before mutating any live input.
    previous_problem = prepared.problem
    new_problem = ProblemSpec(specs, previous_problem.couplings, previous_problem.metadata,
                              flat=previous_problem.flat)
    physical_configs = new_problem.config
    if new_problem.flat:
        physical_configs = {next(iter(specs)): physical_configs}
    else:
        physical_configs = physical_configs.phases
    rollback = {}
    try:
        for name, (params, boundary) in staged.items():
            block = prepared.segments[name]
            rollback[name] = (copy.deepcopy(block.params), block.problem.config,
                              None if boundary is None else boundary[1].constraint._value_dim.copy(),
                              None if boundary is None else boundary[1].value_param.value.copy())
            if params is not None:
                _replace_mapping(block.params, params)
            if boundary is not None:
                _, constraint, numeric, scaled, _ = boundary
                constraint.constraint._value_dim = numeric.copy()
                constraint.value_param.value = scaled
            block.problem.config = copy.deepcopy(physical_configs[name])
        prepared.problem = new_problem
        from trajopt.methods.prepare import _signature
        prepared.signature = _signature(new_problem, prepared.method, prepared.segments)
    except Exception:
        for name, (params, config, physical_value, bound_value) in rollback.items():
            block = prepared.segments[name]
            _replace_mapping(block.params, params)
            block.problem.config = config
            boundary = staged[name][1]
            if boundary is not None:
                boundary[1].constraint._value_dim = physical_value
                boundary[1].value_param.value = bound_value
        prepared.problem = previous_problem
        raise
    return prepared
