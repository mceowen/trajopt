"""MSRL entry, descent, and landing: discrete to CTCS aerodynamic-load enforcement."""
from pathlib import Path

from trajopt.methods.stable.algflows import AlgorithmFlow


def main():
    result = AlgorithmFlow(Path(__file__).with_suffix(".yaml")).solve()
    for name, stage in result.stages.items():
        print(f"{name}: {stage.status}, {stage.iterations} iterations, "
              f"{stage.timings['total_s']:.2f} s — {stage.reason}")
    return 0 if result.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
