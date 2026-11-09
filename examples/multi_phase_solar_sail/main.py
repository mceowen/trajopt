"""Solve the solar-sail example with the native first-order SCP method."""

import argparse
import json
from pathlib import Path
import time

import jax

jax.config.update("jax_enable_x64", True)

from trajopt.common.config_reader import load_trajopt_config
from trajopt.common.persistence import save_result
from trajopt.methods.stable.scp.scp_method import SCPMethod

if __package__:
    from .mission.problem import build_trajectory_config
    from .mission.validation import validate
else:
    from mission.problem import build_trajectory_config
    from mission.validation import validate


def solve(config, *, initial_guess=None, budget=None):
    if config.method.hyperparams.trust_region.second_order:
        raise ValueError("This example uses first-order methods")
    method = SCPMethod(
        problem=build_trajectory_config(config.problem),
        method=config.method,
        initial_guess=initial_guess,
    )
    return method.solve(verbose=False, budget=budget)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--revolutions", type=int)
    parser.add_argument("--output", type=Path, default=Path("plots/solar_sail"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    path = Path(args.config)
    if not path.exists():
        path = Path(__file__).parent / path
    config = load_trajopt_config(path)
    if args.revolutions is not None:
        config.problem.revolutions = args.revolutions
    started = time.perf_counter()
    result = solve(config)
    elapsed = time.perf_counter() - started
    report, replay = validate(result, config.problem)
    report["setup_warmup_solve_s"] = elapsed
    args.output.mkdir(parents=True, exist_ok=True)
    save_result(result, args.output / "solution")
    (args.output / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if args.plot and replay:
        if __package__:
            from .plot import plot_solution
        else:
            from plot import plot_solution
        plot_solution(result, replay, config.problem, args.output)
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
