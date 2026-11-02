"""6-DoF lander: nodal to CTCS minimum-thrust enforcement."""
from pathlib import Path

import jax

from trajopt.methods.stable.algflows import AlgorithmFlow


def main():
    jax.config.update("jax_enable_x64", True)
    result = AlgorithmFlow(Path(__file__).with_suffix(".yaml")).solve()
    for name, stage in result.stages.items():
        print(f"{name}: {stage.status}, {stage.iterations} iterations, "
              f"{stage.timings['total_s']:.2f} s — {stage.reason}")
    return 0 if result.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
