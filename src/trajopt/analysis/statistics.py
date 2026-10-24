"""Solution quality metrics for Monte Carlo SCP runs."""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# =============================================================================
# Main function
# =============================================================================


def analyze_quality_metrics(runs: dict, config: dict | None = None, filename: str | None = None) -> dict:
    """Monte Carlo solution quality metrics computation."""
    # File to save LaTeX tables to
    filename_full = os.path.expanduser(filename) if filename else None

    # Define variables to find statistics of and associated metrics
    if config is None:
        config = load_configuration()

    # Pull out variables from each run
    extracted = extract_data(config, runs)

    # Calculated associated metrics for each variable for each method across runs
    stats = generate_statistics(config, extracted)

    # Generate tables for statistics
    table = generate_statistic_table(stats)

    # Plot tables and save LaTeX tables if filename provided
    plot_tables(table, filename_full)

    return {
        "config": config,
        "data": runs,
        "extracted": extracted,
        "stats": stats,
        "table_cell": table,
    }


# =============================================================================
# Helper functions
# =============================================================================


def load_configuration() -> dict:
    """Define standard data and metric settings when no config is provided.

    NOTE: Currently these are all implemented data and metric options.
    """
    #  All metrics available for calculations
    metrics = [
        "Max",
        "Mean",
        "Median",
        "Min",
        "Mode",
        "Std",
        "Var",
        "1-Norm",
        "2-Norm",
        "Inf-Norm",
    ]

    # Standard variables to be analyzed and their assocaited metrics
    config = {
        "variable_name": {
            "No. of Iteration": metrics[0:7],
            "No. of Attempts": metrics[0:7],
            "Rejected Attempts": metrics[0:7],
            "Solve Time/Iteration (ms/iter)": metrics[0:7],
            "Parse Time/Iteration (ms/iter)": metrics[0:7],
            "Discretization Time/Iteration (ms/iter)": metrics[0:7],
            "Total Solve Time (ms)": metrics[0:7],
            "Total Parse Time (ms)": metrics[0:7],
            "Total Discretization Time (ms)": metrics[0:7],
            "Dynamics Defect": metrics[8:10],
            "Cost": metrics[0:8],
            "Converged": metrics[0:8],
        },
    }

    warnings.filterwarnings("ignore")
    return config


def extract_data(config: dict, data) -> dict:
    """Extract metrics from an ExperimentResult or named sequences of SolveResult.

    Shared solve timings include rejected attempts and are counted once. Dynamics
    defect entries are segment stopping ratios, not physical rollout errors.
    """
    from collections.abc import Mapping
    from trajopt.analysis.experiments import ExperimentResult
    from trajopt.outputs.results import SolveResult

    if isinstance(data, ExperimentResult):
        grouped = {variant.name: [run.solve_result for run in data.runs
                                  if run.variant == variant.name] for variant in data.variants}
    elif isinstance(data, Mapping):
        grouped = dict(data)
        if any(not isinstance(result, SolveResult) for runs in grouped.values() for result in runs):
            raise TypeError("Statistics require named sequences of SolveResult")
    else:
        raise TypeError("Statistics require ExperimentResult or named sequences of SolveResult")
    print("Extracting data from runs...")
    extracted: dict = {}

    for method, results in grouped.items():
        extracted[method] = {}
        all_vars: dict = {
            "No. of Iteration": [],
            "No. of Attempts": [],
            "Rejected Attempts": [],
            "Solve Time/Iteration (ms/iter)": [],
            "Parse Time/Iteration (ms/iter)": [],
            "Discretization Time/Iteration (ms/iter)": [],
            "Total Solve Time (ms)": [],
            "Total Parse Time (ms)": [],
            "Total Discretization Time (ms)": [],
            "Dynamics Defect": [],
            "Cost": [],
            "Converged": [],
        }

        for result in results:
            if result is None:
                num_iters = num_attempts = num_rejected = 0
                time_solve = time_parse = time_disc = 0.
                dyn_defect = np.array([np.nan])
                cost, converged = np.nan, False
            else:
                num_iters = result.iterations
                num_attempts = len(result.attempts)
                num_rejected = result.rejected_attempts
                time_solve = 1000. * result.timings.get("numerical_s", 0.)
                time_parse = 1000. * result.timings.get("canonicalization_s", 0.)
                time_disc = 1000. * result.timings.get("local_model_s", 0.)
                diagnostics = [result.diagnostics[name] for name in result.segment_data
                               if name in result.diagnostics]
                dyn_defect = np.asarray([float(d.get("stopping_ratios", {}).get("dynamics", np.nan))
                                         for d in diagnostics])
                if not dyn_defect.size:
                    dyn_defect = np.array([np.nan])
                objectives = [d.get("objective") for d in diagnostics]
                cost = (sum(float(value) for value in objectives)
                        if result.usable and objectives and all(value is not None for value in objectives)
                        else np.nan)
                converged = result.converged

            divisor = num_iters if num_iters else np.nan
            all_vars["Solve Time/Iteration (ms/iter)"].append(time_solve / divisor)
            all_vars["Parse Time/Iteration (ms/iter)"].append(time_parse / divisor)
            all_vars["Discretization Time/Iteration (ms/iter)"].append(time_disc / divisor)
            all_vars["Total Solve Time (ms)"].append(time_solve)
            all_vars["Total Parse Time (ms)"].append(time_parse)
            all_vars["Total Discretization Time (ms)"].append(time_disc)
            all_vars["No. of Iteration"].append(num_iters)
            all_vars["No. of Attempts"].append(num_attempts)
            all_vars["Rejected Attempts"].append(num_rejected)
            all_vars["Cost"].append(cost)
            all_vars["Dynamics Defect"].append(np.atleast_1d(dyn_defect))
            all_vars["Converged"].append(converged)

        for var in config["variable_name"]:
            if var in all_vars:
                extracted[method][var] = all_vars[var]

    return extracted


def generate_statistics(config: dict, extracted: dict) -> dict:
    """Compute overall statistics across Monte Carlo runs per method."""
    print("Computing statistics across runs...")
    stats: dict = {}

    # Loop through each method
    for method, method_data in extracted.items():
        stats[method] = {}
        # Loop through each variable to calculate metrics
        for variable, data in method_data.items():
            stats[method][variable] = {}
            # Deal with Dynamic Defects seperatly
            if variable == "Dynamics Defect":
                for norm in ["1-Norm", "2-Norm", "Inf-Norm"]:
                    stats[method][variable][norm] = {}
                    temp = []
                    for run_data in data:
                        temp.append(calc_stat(run_data, norm))
                    for stat in ["Max", "Mean", "Median", "Min", "Mode", "Std", "Var"]:
                        stats[method][variable][norm][stat] = calc_stat(temp, stat)
            # Calculate the metrics for all other variables
            else:
                metrics_to_calc = config["variable_name"][variable]
                for metric in metrics_to_calc:
                    stats[method][variable][metric] = calc_stat(data, metric)

    return stats


def generate_statistic_table(stats: dict) -> dict[str, pd.DataFrame]:
    """Convert nested stats dict into separate DataFrames for each metric or submetric."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Statistics tables require the optional pandas dependency; "
                          "numeric extraction and statistics do not") from exc

    print("Generating statistic tables...")
    tables: dict = {}

    for method, metrics in stats.items():
        for metric_name, metric_values in metrics.items():
            if isinstance(metric_values, dict) and all(isinstance(v, dict) for v in metric_values.values()):
                # Nested submetrics (e.g. "Dynamics Defect" -> "1-Norm", "2-Norm", "Inf-Norm")
                for sub_name, sub_values in metric_values.items():
                    key = f"{metric_name} ({sub_name})"
                    tables.setdefault(key, {})[method] = sub_values
            else:
                # Regular metrics (e.g. "No. of Iteration", "Cost", etc.)
                tables.setdefault(metric_name, {})[method] = metric_values

    # Convert each collected metric dict to a DataFrame
    return {name: pd.DataFrame(data) for name, data in tables.items()}


def plot_tables(tables: dict[str, pd.DataFrame], filename: str | None = None) -> list[str]:
    """Print each metric table and optionally write all LaTeX tables to a file."""
    print("Plotting tables...")
    latex_blocks = []

    for name, df in tables.items():
        print(f"\n=== {name} ===\n")
        print(df.round(6))
        print()

        latex_code = df.to_latex(float_format="%.6g", escape=False, caption=name, label=name.replace(" ", "_"))
        latex_blocks.append(latex_code)

    combined_latex = "\n\n".join(latex_blocks)

    if filename:
        with open(filename, "w") as f:
            f.write("% Auto-generated LaTeX tables\n\n")
            f.write(combined_latex)
        print(f"\n✅ LaTeX tables successfully written to '{filename}'")

    return latex_blocks


def calc_stat(data: list | np.ndarray, metric: str) -> float | np.ndarray:
    """Compute a single statistical metric on data."""
    metric_funcs = {
        "Max": lambda x: np.max(x),
        "Mean": lambda x: np.mean(x),
        "Median": lambda x: np.median(x),
        "Min": lambda x: np.min(x),
        "Mode": lambda x: np.unique(x, return_counts=True)[0][np.argmax(np.unique(x, return_counts=True)[1])],
        "Std": lambda x: np.std(x, ddof=1),
        "Var": lambda x: np.var(x, ddof=1),
        "1-Norm": lambda x: np.linalg.norm(x, ord=1),
        "2-Norm": lambda x: np.linalg.norm(x, ord=2),
        "Inf-Norm": lambda x: np.linalg.norm(x, ord=np.inf),
    }

    if metric not in metric_funcs:
        msg = f"Unknown metric: {metric}"
        raise ValueError(msg)

    return metric_funcs[metric](data)
