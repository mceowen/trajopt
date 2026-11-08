"""Repeat a flow stage until its pseudospectral mesh meets the error tolerance."""

import copy
from time import perf_counter

import jax

from trajopt.analysis.validation import detached_phases
from trajopt.methods.common.discretize import segment_ranges
from trajopt.methods.dev.ps.mesh_error import estimate_mesh_error
from trajopt.methods.dev.ps.refinement import refine_mesh
from .config import refinement_options


def _estimate_phases(output, options):
    errors, meshes, proposals = {}, {}, {}
    for name, phase in detached_phases(output).items():
        if phase.mesh.discretize != "ps":
            raise ValueError(
                "Adaptive hp requires pseudospectral discretization in every phase"
            )
        data = phase.iter_data_list[-1]
        dynamics = next(
            c.fcn_znu
            for c in phase.problem.constraints.values()
            if c.type == "dynamics"
        )
        estimate = estimate_mesh_error(
            phase.mesh,
            data.z_opt,
            data.nu_opt,
            jax.jit(dynamics),
            phase.problem.params,
            phase.problem.index_map.indices.nu.dilation_factor,
        )
        errors[name] = [r["max_relative_error"] for r in estimate["segments"]]
        ranges = segment_ranges(len(phase.mesh.tau), phase.mesh.segments)
        meshes[name] = dict(
            knots=[float(phase.mesh.tau[a]) for a, b in ranges] + [1.0],
            degrees=[b - a for a, b in ranges],
        )
        proposals[name] = refine_mesh(phase.mesh, errors[name], options)
    return errors, meshes, proposals


def _mesh_config(config, meshes):
    config = copy.deepcopy(config)
    phases = config["trajectory"].get(
        "phases", {next(iter(meshes)): config["trajectory"]}
    )
    for name, mesh in meshes.items():
        phases[name]["hp_mesh"] = mesh
        phases[name]["num_nodes"] = 1 + sum(mesh["degrees"])
    return config


def _node_count(meshes):
    return sum(1 + sum(mesh["degrees"]) for mesh in meshes.values())


def _repeat_stage(flow, spec, path, current, meshes, verbose):
    seed = {
        "from": "previous",
        "endpoint_weights": spec.get("initial_guess", {}).get(
            "endpoint_weights", False
        ),
    }
    child = dict(name=spec["name"], config=spec["config"], initial_guess=seed)
    config = _mesh_config(current.config, meshes)
    return flow._solve_stage(child, path, config, {"previous": current}, verbose)


def _mesh_record(stage, meshes, **details):
    return dict(
        meshes=meshes,
        status=stage.status,
        iterations=stage.iterations,
        timings=dict(stage.timings),
        **details,
    )


def refine_stage(flow, spec, path, initial, verbose):
    """Keep each mesh result and count the work across all solves."""
    options = refinement_options(spec["refinement"])
    current = initial
    history, outputs = [], []
    totals = dict(initial.timings, estimation_s=0.0)
    iterations = initial.iterations
    try:
        while True:
            before = perf_counter()
            errors, meshes, proposals = _estimate_phases(current.output, options)
            estimation_s = perf_counter() - before
            totals["estimation_s"] += estimation_s
            totals["total_s"] += estimation_s
            maximum = max(max(e) for e in errors.values())
            history.append(
                _mesh_record(
                    current,
                    meshes,
                    errors=errors,
                    max_relative_error=maximum,
                    rejected=getattr(current.output, "rejected_attempts", 0),
                    estimation_s=estimation_s,
                )
            )
            outputs.append(current.output)
            if verbose:
                print(
                    f"Adaptive hp mesh {len(history)}: error={maximum:.6g}, "
                    f"nodes={_node_count(meshes)}",
                    flush=True,
                )
            if maximum <= options["tolerance"]:
                current.reason = "SCP convergence and mesh-error tolerance satisfied"
                break
            if len(history) >= options["max_meshes"]:
                current.status, current.reason = (
                    "not_converged",
                    "Mesh refinement budget exhausted",
                )
                break
            if _node_count(proposals) > options["max_nodes"]:
                current.status, current.reason = (
                    "not_converged",
                    "Mesh refinement node budget exhausted",
                )
                break
            current = _repeat_stage(flow, spec, path, current, proposals, verbose)
            iterations += current.iterations
            for key, value in current.timings.items():
                totals[key] = totals.get(key, 0.0) + value
            if not current.converged:
                history.append(_mesh_record(current, proposals, reason=current.reason))
                if current.output is not None:
                    outputs.append(current.output)
                break
    except Exception as exc:
        current.status, current.reason = "failed", f"{type(exc).__name__}: {exc}"
        if current.output is not None and (
            not outputs or outputs[-1] is not current.output
        ):
            outputs.append(current.output)
    current.mesh_history, current.mesh_outputs = history, outputs
    current.iterations, current.timings = iterations, totals
    current.seed_from = initial.seed_from
    current.transferred_phases = initial.transferred_phases
    return current
