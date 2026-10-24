"""Sequential solves connected by owned physical trajectories."""
import copy
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

import numpy as np

from trajopt.flows.results import FlowResult, StageResult
from trajopt.flows.transfer import transfer_guesses
from trajopt.methods.trajectory_conversion import segment_layout
from trajopt.methods.scp import SCPDriver
from trajopt.formulations.specs import normalize
from trajopt.utils import config_loader
from trajopt.outputs.results import SolveResult


class AlgorithmFlow:
    def __init__(self, config_path):
        self.config_path = Path(config_path).expanduser().resolve()
        config = config_loader.load_trajopt_config(self.config_path)
        if not isinstance(config, Mapping):
            raise ValueError('Expected a flow configuration mapping')
        flow = config.get('flow')
        if not isinstance(flow, Mapping) or set(flow) != {'stages'}:
            raise ValueError('Expected flow.stages')
        stages = flow['stages']
        if not isinstance(stages, list) or not stages:
            raise ValueError('flow.stages must be a nonempty list')
        self._stages = []
        names = set()
        for stage in stages:
            if not isinstance(stage, Mapping) or set(stage) - {'name', 'config', 'initial_guess', 'allow_nonconverged'}:
                raise ValueError('Stage fields are name, config, initial_guess, and allow_nonconverged')
            name = stage.get('name')
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError(f'Expected a unique stage name, got {name!r}')
            path = stage.get('config')
            if not isinstance(path, str) or not path:
                raise ValueError(f"Stage '{name}' needs a config path")
            if type(stage.get('allow_nonconverged', False)) is not bool:
                raise ValueError(f"Stage '{name}': allow_nonconverged must be a boolean")
            seed = stage.get('initial_guess')
            if seed is not None:
                if not isinstance(seed, Mapping) or set(seed) - {'from', 'phases'} or 'from' not in seed:
                    raise ValueError(f"Stage '{name}': initial_guess needs from and optional phases")
                if not isinstance(seed['from'], str) or seed['from'] not in names:
                    raise ValueError(f"Stage '{name}': initial_guess.from must name an earlier stage")
                phases = seed.get('phases')
                if phases is not None and (not isinstance(phases, list) or not phases
                        or any(not isinstance(p, str) or not p for p in phases)
                        or len(set(phases)) != len(phases)):
                    raise ValueError(f"Stage '{name}': phases must be a nonempty list of unique names")
            path = (self.config_path.parent / Path(path).expanduser()).resolve()
            resolved = config_loader.load_trajopt_config(path)
            if (not isinstance(resolved, Mapping) or not {'trajectory', 'method'} <= resolved.keys()
                    or not isinstance(resolved['trajectory'], Mapping)
                    or not isinstance(resolved['method'], Mapping) or 'flow' in resolved):
                raise ValueError(f"Stage '{name}' must reference a trajectory/method configuration")
            self._stages.append((copy.deepcopy(stage), path, resolved))
            names.add(name)

    def solve(self, *, verbose=None) -> FlowResult:
        result = FlowResult()
        started = perf_counter()
        for spec, path, resolved in self._stages:
            stage = self._solve_stage(spec, path, resolved, result.stages, verbose)
            result.stages[stage.name] = stage
            if stage.status != 'failed' and stage.output is not None and stage.output.usable:
                result.selected_stage = stage.name
            permitted = stage.converged or (
                stage.status == 'not_converged' and spec.get('allow_nonconverged', False))
            if not permitted:
                result.failed_stage = stage.name
                break
        else:
            result.completed = True
        result.elapsed_s = perf_counter() - started
        return result

    def _solve_stage(self, spec, path, resolved, previous, verbose):
        config = copy.deepcopy(resolved)
        seed = spec.get('initial_guess')
        stage = StageResult(spec['name'], path, copy.deepcopy(config),
                            seed_from=None if seed is None else seed['from'])
        stage.timings = dict(preparation_s=0., transfer_s=0., solve_s=0., export_s=0.)
        started = perf_counter()
        method = None
        try:
            before = perf_counter()
            problem, method_plan = normalize(config)
            stage.layouts = {name: segment_layout(physical)
                             for name, physical in problem.build_segments().items()}
            stage.timings['preparation_s'] = perf_counter() - before

            before = perf_counter()
            guess = None if seed is None else transfer_guesses(
                previous[seed['from']], stage.layouts, seed.get('phases'))
            stage.transferred_phases = () if guess is None else tuple(guess)
            stage.timings['transfer_s'] = perf_counter() - before

            before = perf_counter()
            method = SCPDriver(problem=problem, method=method_plan, initial_guess=guess)
            stage.timings['preparation_s'] += perf_counter() - before
            before = perf_counter()
            try:
                output = method.solve(verbose=verbose)
            finally:
                stage.timings['solve_s'] = perf_counter() - before
            if not isinstance(output, SolveResult):
                raise TypeError('A solve stage must return SolveResult')
            self._record_output(stage, output)
            for name, guess in stage.trajectory.items():
                samples = guess.sample(guess.native_tau)
                if (np.any(np.diff(samples.t) <= 0) or
                        (samples.s is not None and np.any(samples.s <= 0))):
                    raise ValueError(f"Phase '{name}' has an unusable time map")
            stage.execution_status = 'completed'
        except Exception as exc:
            output = getattr(method, 'result', None)
            if isinstance(output, SolveResult):
                self._record_output(stage, output)
            stage.status = 'failed'
            stage.reason = f'{type(exc).__name__}: {exc}'
        stage.timings['total_s'] = perf_counter() - started
        return stage

    @staticmethod
    def _record_output(stage, output):
        stage.output = output
        stage.status = output.status
        stage.reason = output.reason
        stage.solver_status = output.solver_status
        stage.iterations = output.iterations
        stage.trajectory = {} if output.trajectory is None else output.trajectory
        stage.layouts = {name: data.layout for name, data in stage.trajectory.items()}
