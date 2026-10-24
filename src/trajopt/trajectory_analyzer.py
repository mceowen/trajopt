import os
import sys

import trajopt.utils.config_loader as config_loader
import trajopt.analysis.plotting as plotting
from trajopt.trajectories import TrajectorySet
from trajopt.utils.tools import deep_merge, recursive_attrdict


class _Tee:
    """A writable stream that duplicates everything written to it across several streams."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


class TrajectoryAnalyzer():
    """Solve, evaluate, and plot a trajectory from a configuration.

    Set ``console_log=True`` to write console output to a file.
    Use a ``with`` block to restore ``stdout`` and close the log file,
    even if an error occurs.
    """

    def __init__(self, config_path, method_overrides=None, *, initial_guess=None,
                 console_log=False) -> None:

        if console_log:
            self._start_console_log()

        try:
            self.config_path = config_path
            self.config = config_loader.load_trajopt_config(config_path)
            if method_overrides:
                self.config.method = deep_merge(self.config.method, recursive_attrdict(method_overrides))
            self._build_method(initial_guess=initial_guess)
        except Exception:
            self.stop_console_log()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_console_log()
        return False

    def _build_method(self, *, initial_guess=None):
        from trajopt.formulations.specs import normalize

        analysis_config = self.config.get("analysis", {})
        full_history = any(analysis_config.get(key, False)
                           for key in ("compute_iters", "show_iters", "show_weights", "show_vb"))
        config = deep_merge(self.config, {"method": {"history": "full"}}) if full_history else self.config
        problem, method = normalize(config)
        from trajopt.methods.scp import SCPDriver
        self.method = SCPDriver(problem=problem, method=method, initial_guess=initial_guess)
        self.problem = self.method.problem

    def _start_console_log(self, path=None):
        """Tee everything printed from here on to plots/console_log.txt as well as the terminal."""
        path = path or os.path.join("plots", "console_log.txt")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._console_log_file = open(path, "w")
        self._orig_stdout = sys.stdout
        sys.stdout = _Tee(self._orig_stdout, self._console_log_file)

    def stop_console_log(self):
        """Restore stdout and close the log file."""
        if getattr(self, "_orig_stdout", None) is not None:
            sys.stdout = self._orig_stdout
            self._orig_stdout = None
        if getattr(self, "_console_log_file", None) is not None:
            self._console_log_file.close()
            self._console_log_file = None

    def solve(self, verbose=None):
        """Run the solver and return its result before analysis."""
        self.__dict__.pop("evaluation", None)
        self.__dict__.pop("experiment", None)
        try:
            self.result = self.method.solve(verbose=verbose)
        except Exception:
            self.result = getattr(self.method, "result", None)
            raise
        from trajopt.outputs.results import SolveResult
        if not isinstance(self.result, SolveResult):
            raise TypeError("A trajectory solve must return a SolveResult")
        return self.result

    def export_guess(self, *, include_aux=True):
        """Export phase guesses; omit aux for destination reconstruction."""
        return self.method.export_guess(include_aux=include_aux)

    def reset(self, *, initial_guess=None):
        """Restart SCP from a trajectory source or named-segment mapping, resetting penalties and history.

        Omitted phases use configured providers. Mesh or model changes require
        reconfigure(); reset retains the compiled problem.
        """
        self.method.reset(initial_guess=initial_guess)
        self.__dict__.pop('result', None)
        self.__dict__.pop('evaluation', None)
        self.__dict__.pop('experiment', None)

    def analyze(self):
        """Evaluate the solve result or run the configured experiment."""
        from trajopt.analysis.experiments import (
            ExperimentCase, ExperimentVariant, monte_carlo_cases, run_experiment,
        )
        options = self.config.get("analysis", {})
        kind = options.get("type", "standalone")
        if kind == "standalone":
            if getattr(self, "result", None) is None:
                self.solve()
            return self.evaluate()
        if kind == "mc":
            cases, variants = monte_carlo_cases(self.config), None
        elif kind == "method_variation":
            cases = [ExperimentCase("nominal", self.config, nominal=True)]
            variants = [ExperimentVariant(name, overrides or {})
                        for name, overrides in options.methods.items()]
        else:
            raise ValueError(f"Unknown analysis type {kind!r}")
        self.__dict__.pop("result", None)
        self.__dict__.pop("evaluation", None)
        self.experiment = run_experiment(cases, variants, evaluate=True, analysis_config=options)
        return self.experiment

    @property
    def solution(self) -> TrajectorySet | None:
        """The accepted physical trajectory, available immediately after solve()."""
        result = getattr(self, "result", None)
        if result is None and getattr(self, "experiment", None) is not None:
            variants = {variant.name for variant in self.experiment.variants}
            preferred = self.config.method.get("name", "method1")
            run = self.experiment.primary_run(preferred if preferred in variants else None)
            result = run.solve_result
            if result is None:
                raise RuntimeError("The nominal experiment run has no solve result: " + (run.error or "unavailable"))
        if result is None:
            raise RuntimeError("No solve result yet; call solve() before accessing solution.")
        return result.trajectory

    def evaluate(self, **options):
        """Evaluate the current solve result without running an experiment."""
        from trajopt.analysis.validation import evaluate

        if getattr(self, "result", None) is None:
            raise RuntimeError("No solve result yet; call solve() before evaluating it.")
        self.evaluation = evaluate(self.result, deep_merge(self.config.get("analysis", {}), options))
        return self.evaluation

    def plot(self, data=None, *, save=True, show=False, save_dir=None, format="pdf"):
        """Plot canonical solve, evaluation, or experiment data."""
        from trajopt.analysis.validation import EvaluationResult
        from trajopt.analysis.experiments import ExperimentResult
        from trajopt.outputs.results import SolveResult

        if data is None:
            data = getattr(self, "experiment", None)
            if data is None:
                data = getattr(self, "evaluation", None)
            if data is None:
                data = getattr(self, "result", None)
            if data is None:
                data = self.analyze()
        options = dict(analysis_config=self.config.get("analysis", {}), save=save,
                       show=show, save_dir=save_dir, format=format)
        if isinstance(data, ExperimentResult):
            return plotting.plot_experiment(data, **options)
        if isinstance(data, EvaluationResult):
            return plotting.plot_result(data.solve_result, data, **options)
        if isinstance(data, SolveResult):
            return plotting.plot_result(data, **options)
        raise TypeError("plot requires SolveResult, EvaluationResult, or ExperimentResult")

    def update_inputs(self, updates, *, initial_guess=None):
        """Update supported numeric inputs and reset the algorithm state."""
        self.method.update_inputs(updates, initial_guess=initial_guess)
        self.config.trajectory = self.method.problem.config
        self.problem = self.method.problem
        for name in ("result", "evaluation", "experiment"):
            self.__dict__.pop(name, None)

    def reconfigure(self):
        """Rebuild the prepared solve from this analyzer's configuration.

        Call this after modifying ``self.config`` from external code (e.g., a C
        interface) to propagate the changes into the internal problem objects.

        Notes:
        - This is the full-rebuild path: it reconstructs the CVXPY subproblem
          and discards all compiled JAX kernels, so the next solve pays
          construction and JIT compilation again. Use ``update_inputs`` for
          supported measured-boundary and parameter changes; it validates the
          bindings and records the changed physical inputs in the next result.
        - ``${...}`` expressions were evaluated once at config load; editing a
          param they referenced does not re-evaluate them. Edit the resolved
          value directly.
        """
        print("Reconfiguring trajopt with updated config...")
        self._build_method()
        for name in ("result", "evaluation", "experiment"):
            self.__dict__.pop(name, None)
