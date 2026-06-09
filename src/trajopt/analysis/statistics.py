"""Solution quality metrics for Monte Carlo SCP runs."""

import os
import warnings

import numpy as np
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


def extract_data(config: dict, data: dict) -> dict:
    """Extract key variables from each Monte Carlo run.

    Supports the new TrajectoryAnalyzer format where each run contains
    ``scp_iters`` (a list of raw SCP iteration AttrDicts) alongside the
    propagated ``iter_data_list``.
    """
    print("Extracting data from runs...")
    extracted: dict = {}

    for method, method_data in data.items():
        extracted[method] = {}
        all_vars: dict = {
            "No. of Iteration": [],
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

        for run_data in method_data["runs"]:
            iters = run_data.get("scp_iters", run_data.get("iters", []))
            num_iters = max(len(iters) - 1, 1)

            time_solve = 0.0
            time_disc = 0.0
            time_parse = 0.0
            for it in iters:
                if int(it.get("iter_num", 0)) == 0:
                    continue
                time_solve += float(it.get("solve_time", 0))
                time_disc += float(it.get("discretization_time", it.get("prop_time", 0)))
                time_parse += float(it.get("parse_time", 0))

            last = iters[-1] if iters else {}

            chk = last.get("chk", last.get("conv_data", {}))
            dyn_defect = float(chk.get("dynamics", 0))

            all_vars["Solve Time/Iteration (ms/iter)"].append(time_solve / num_iters)
            all_vars["Parse Time/Iteration (ms/iter)"].append(time_parse / num_iters)
            all_vars["Discretization Time/Iteration (ms/iter)"].append(time_disc / num_iters)
            all_vars["Total Solve Time (ms)"].append(time_solve)
            all_vars["Total Parse Time (ms)"].append(time_parse)
            all_vars["Total Discretization Time (ms)"].append(time_disc)
            all_vars["No. of Iteration"].append(num_iters)
            all_vars["Cost"].append(float(last.get("cost", 0)))
            all_vars["Dynamics Defect"].append(np.atleast_1d(dyn_defect))
            all_vars["Converged"].append(bool(last.get("converged", False)))

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


# =============================================================================
# main() testing function
# =============================================================================

if __name__ == "__main__":
    # Run the full statistics generation
    analyze_quality_metrics()
