"""Run and compare experiment cases with different method settings."""
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import time

import numpy as np

from trajopt.utils._snapshot import freeze, thaw
from trajopt.utils.tools import deep_merge, get_from_path, set_from_path


@dataclass(frozen=True)
class ExperimentCase:
    """Problem inputs, method settings, and any random draws for one case."""

    name: str
    config: Mapping
    seed: int | None = None
    nominal: bool = False
    draws: Mapping = field(default_factory=dict)
    draw_index: int = 0

    def __post_init__(self):
        if not self.name:
            raise ValueError("Experiment cases need a nonempty name")
        object.__setattr__(self, "config", freeze(self.config))
        object.__setattr__(self, "draws", freeze(self.draws))


@dataclass(frozen=True)
class ExperimentVariant:
    """Method settings and an optional solver factory for each case.

    The factory receives a copy of the resolved case inputs. It returns a
    method or a flow. ``operation_id`` identifies the factory in the experiment.
    """

    name: str
    method_overrides: Mapping = field(default_factory=dict)
    operation_factory: Callable | None = None
    operation_id: str = "SCPDriver"

    def __post_init__(self):
        if not self.name:
            raise ValueError("Experiment variants need a nonempty name")
        object.__setattr__(self, "method_overrides", freeze(self.method_overrides))


@dataclass(frozen=True)
class ExperimentRun:
    case: str
    variant: str
    config: Mapping
    result: object = None
    evaluation: object = None
    error: str | None = None
    wall_s: float = 0.

    def __post_init__(self):
        object.__setattr__(self, "config", freeze(self.config))

    @property
    def solve_result(self):
        """Return the solve result, or the result from the final flow stage."""
        return _solve_output(self.result)


@dataclass(frozen=True)
class ExperimentResult:
    cases: tuple[ExperimentCase, ...]
    variants: tuple[ExperimentVariant, ...]
    runs: tuple[ExperimentRun, ...]

    def __post_init__(self):
        for name in ("cases", "variants", "runs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def run(self, case, variant):
        return next(run for run in self.runs if run.case == case and run.variant == variant)

    def primary_run(self, variant=None):
        """Return the variant's nominal run, or its first run if none is nominal."""
        if not self.cases or not self.variants:
            raise ValueError("The experiment has no cases or variants")
        variant = self.variants[0].name if variant is None else variant
        nominal = next((case for case in self.cases if case.nominal), self.cases[0])
        return self.run(nominal.name, variant)


def _build_solver(config):
    from trajopt.formulations.specs import normalize
    from trajopt.methods.scp import SCPDriver

    problem, method = normalize(config)
    return SCPDriver(problem=problem, method=method)


def _solve_output(result):
    """Return the solve result or the final flow stage's result, including failures."""
    from trajopt.flows.results import FlowResult

    from trajopt.outputs.results import SolveResult
    if isinstance(result, FlowResult):
        result = result.final.output if result.final is not None else None
    if result is not None and not isinstance(result, SolveResult):
        raise TypeError("Experiment operations must return SolveResult or FlowResult")
    return result


def run_experiment(cases, variants=None, *, solver_factory=None, verbose=None,
                   evaluate=False, analysis_config=None):
    """Solve each case with each variant, using a separate copy of the inputs.

    ``solver_factory(config)`` returns a method or a configured flow that
    implements ``solve(verbose=...)``. The factory creates a new instance for
    each run.
    """
    cases = tuple(cases)
    if not cases:
        raise ValueError("An experiment needs at least one case")
    if sum(bool(case.nominal) for case in cases) > 1:
        raise ValueError("An experiment can have at most one explicitly nominal case")
    variants = tuple([ExperimentVariant(cases[0].config["method"].get("name", "method1"))]
                     if variants is None else variants)
    if not variants:
        raise ValueError("An experiment needs at least one variant")
    for kind, items in (("case", cases), ("variant", variants)):
        names = [item.name for item in items]
        if len(set(names)) != len(names):
            raise ValueError(f"Experiment {kind} names must be unique")
    runs = []
    for variant in variants:
        factory = variant.operation_factory or solver_factory or _build_solver
        for case in cases:
            config = thaw(case.config)
            config.method = deep_merge(config.method, thaw(variant.method_overrides))
            evaluation_config = config.get("analysis", {}) if analysis_config is None else analysis_config
            if evaluate and any(evaluation_config.get(key, False)
                                for key in ("compute_iters", "show_iters", "show_weights", "show_vb")):
                config.method.history = "full"
            inputs = freeze(config)
            start = time.perf_counter()
            operation, result, evaluation, error = None, None, None, None
            try:
                operation = factory(config)
                result = operation.solve(verbose=verbose)
                if result is None:
                    raise TypeError("An experiment operation must return SolveResult or FlowResult")
                _solve_output(result)
                if evaluate:
                    from trajopt.analysis.validation import evaluate as evaluate_result
                    output = _solve_output(result)
                    if output is not None:
                        evaluation = evaluate_result(output, evaluation_config)
            except Exception as exc:
                # Preserve the failed case and any accepted result produced by
                # the operation. Other independent cases remain executable.
                candidate = result if result is not None else getattr(operation, "result", None)
                try:
                    _solve_output(candidate)
                except TypeError:
                    result = None
                else:
                    result = candidate
                error = f"{type(exc).__name__}: {exc}"
            runs.append(ExperimentRun(case.name, variant.name, inputs, result, evaluation,
                                      error, time.perf_counter() - start))
    return ExperimentResult(cases, variants, tuple(runs))


def monte_carlo_cases(config):
    """Create nominal and dispersed cases with a local random number generator.

    A seeded ``RandomState`` preserves the existing draw sequence. Each case
    stores the seed, draw index, sampled offsets, and resolved inputs.
    Parameter paths can start at the run configuration or its ``trajectory``
    block.
    """
    config = thaw(freeze(config))
    variation = config.get("variations", {})
    seed = int(variation.get("seed", 0))
    count = int(variation.get("num", 0))
    if count < 0:
        raise ValueError("Monte Carlo case count must be nonnegative")
    random = np.random.RandomState(seed)
    cases = [ExperimentCase("nominal", config, seed=seed, nominal=True)]
    for index in range(1, count + 1):
        inputs, draws = thaw(freeze(config)), {}
        for path, spec in variation.get("samples", {}).items():
            if spec["type"] == "uniform":
                delta = random.uniform(spec["lb"], spec["ub"])
            elif spec["type"] == "normal":
                delta = random.normal(spec["mu"], spec["sigma"])
            else:
                raise ValueError(f"unknown variation type: {spec['type']!r}")
            resolved_path = path if path.split(".", 1)[0] in inputs else "trajectory." + path
            current = get_from_path(inputs, resolved_path)
            value = np.asarray(current) + delta
            if np.ndim(current) == 0:
                value = value.item()
            set_from_path(inputs, resolved_path, value)
            draws[resolved_path] = delta
        cases.append(ExperimentCase(f"case_{index:04d}", inputs, seed=seed,
                                    draws=draws, draw_index=index))
    return tuple(cases)
