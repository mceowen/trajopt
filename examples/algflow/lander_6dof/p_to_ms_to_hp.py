"""6-DoF lander: pseudospectral p-method to multiple shooting to hp-method."""
from pathlib import Path

import jax

from trajopt.flows import AlgorithmFlow
import numpy as np
import matplotlib.pyplot as plt


def main():
    jax.config.update("jax_enable_x64", True)
    result = AlgorithmFlow(Path(__file__).with_suffix(".yaml")).solve()
    for name, stage in result.stages.items():
        print(f"{name}: {stage.status}, {stage.iterations} iterations, "
              f"{stage.timings['total_s']:.2f} s — {stage.reason}")
    return result


if __name__ == "__main__":
    result = main()
    print("converged.")

    sol = result.stages["p_method"].trajectory["main"]
    nt = len(sol.native_tau)
    b = np.ones(nt)
    t = sol.native_tau
    plt.scatter(t, b)