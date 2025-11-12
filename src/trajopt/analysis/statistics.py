"""
Solution quality metrics for Monte Carlo SCP runs.
"""

import numpy as np
import os
import warnings
import pandas as pd


# =============================================================================
# Main function
# =============================================================================

def analyze_quality_metrics(mc_data, config=None, filename=None):
    """
    Monte Carlo solution quality metrics computation.
    """

    # File to save LaTeX tables to
    filename_full = os.path.expanduser(filename) if filename else None

    # Define variables to find statistics of and associated metrics
    if config is None:
        config      = load_configuration()

    # Pull out variables from each run
    extracted   = extract_data(config, mc_data)

    # Calculated associated metrics for each variable for each method across runs
    stats       = generate_statistics(config, extracted)

    # Generate tables for statistics
    table = generate_statistic_table(stats)

    # Plot tables and save LaTeX tables if filename provided
    plot_tables(table, filename_full)

    return {
        "config": config,
        "data": mc_data,
        "extracted": extracted,
        "stats": stats,
        "table_cell": table,
    }


# =============================================================================
# Helper functions
# =============================================================================

def load_configuration():
    """
    Define standard data and metric settings if config not provided to function.

    NOTE: Currently these are all implemented data and metric options

    """

    #  All metrics available for calculations
    metrics = [
            "Max", "Mean", "Median", "Min", "Mode",
            "Std", "Var", "1-Norm", "2-Norm", "Inf-Norm",
        ]
    
    # Standard variables to be analyzed and their assocaited metrics
    config = {
        "variable_name": {
            "No. of Iteration": metrics[0:7],
            "Solve Time/Iteration (ms/iter)": metrics[0:7],
            "Parse Time/Iteration (ms/iter)": metrics[0:7],
            "Propagation Time/Iteration (ms/iter)": metrics[0:7],
            "Total Solve Time (ms)": metrics[0:7],
            "Total Parse Time (ms)": metrics[0:7],
            "Total Propagation Time (ms)": metrics[0:7],
            "Total Convergence Time (ms)": metrics[0:7],
            "Dynamics Defect": metrics[8:10],
            "Constraint Violation": metrics[0:7],
            "Cost": metrics[0:8],
            "Converged": metrics[0:8],
        },
    }

    warnings.filterwarnings("ignore")
    return config


def extract_data(config, data):
    """Extract key variables from each Monte Carlo run."""
    print("Extracting data from runs...")
    extracted = {}

    # Loop through each method and store extracted data
    for method, method_data in data.items():
        extracted[method] = {}
        all_vars = { 
            "No. of Iteration": [],
            "Solve Time/Iteration (ms/iter)": [],
            "Parse Time/Iteration (ms/iter)": [],
            "Propagation Time/Iteration (ms/iter)": [],
            "Total Solve Time (ms)": [],
            "Total Parse Time (ms)": [],
            "Total Propagation Time (ms)": [],
            "Total Convergence Time (ms)": [],
            "Dynamics Defect": [],
            "Constraint Violation": [],
            "Cost": [],
            "Converged": [],
        }

        # Loop through all runs and store all variables as defined within config
        for run_data in method_data['mc_data']:
            num_iters = len(run_data['iters']) - 1
            # Summation of all time series data across the iterations
            for iter_data in run_data['iters']:
                if iter_data['iter_num'] == 0:
                    time_solve = 0
                    time_prop = 0
                    time_parse = 0
                else:
                    time_solve += iter_data['solve_time']
                    time_prop += iter_data['prop_time']
                    time_parse += iter_data['parse_time']

            # Finding avg dynamic defect across time for each state
            defects = np.array(run_data['iters'][-1]['conv_data']['defect'])
            avg_state_vec = np.mean(defects, axis=0)

            # Saving all variables data in a temporary dictionary
            all_vars["Solve Time/Iteration (ms/iter)"].append(time_solve/num_iters)
            all_vars["Parse Time/Iteration (ms/iter)"].append(time_parse/num_iters)
            all_vars["Propagation Time/Iteration (ms/iter)"].append(time_prop/num_iters)
            all_vars["Total Solve Time (ms)"].append(time_solve)
            all_vars["Total Parse Time (ms)"].append(time_parse)
            all_vars["Total Propagation Time (ms)"].append(time_prop)
            all_vars["Total Convergence Time (ms)"].append((run_data['t_all'] * 1000) - time_parse)
            all_vars["No. of Iteration"].append(num_iters)
            all_vars['Cost'].append(run_data['iters'][-1]['cost'])
            all_vars["Dynamics Defect"].append(avg_state_vec)
            all_vars["Constraint Violation"].append(run_data['iters'][-1]['conv_data']['chk_feas'])
            all_vars["Converged"].append(run_data['iters'][-1]['converged'])
        
        # Only saving variables defined in the config
        for var in config['variable_name'].keys():
            extracted[method][var] = all_vars[var]
    
    return extracted


def generate_statistics(config, extracted):
    """Compute overall statistics across Monte Carlo runs per method."""
    print("Computing statistics across runs...")
    stats = {}

    # Loop through each method
    for method, method_data in extracted.items():
        stats[method] = {}
        # Loop through each variable to calculate metrics
        for variable, data in method_data.items():
            stats[method][variable] = {}
            # Deal with Dynamic Defects seperatly
            if variable == "Dynamics Defect":
                for norm in ['1-Norm', '2-Norm', 'Inf-Norm']:
                    stats[method][variable][norm] = {}
                    temp = []
                    for run_data in data:
                        temp.append(calc_stat(run_data, norm))
                    for stat in ["Max", "Mean", "Median", "Min", "Mode", "Std", "Var" ]:
                        stats[method][variable][norm][stat] = calc_stat(temp, stat)
            # Calculate the metrics for all other variables
            else:
                metrics_to_calc = config['variable_name'][variable]
                for metric in metrics_to_calc:
                    stats[method][variable][metric] = calc_stat(data, metric)

    return stats


def generate_statistic_table(stats):
    """Convert nested stats dict into separate DataFrames for each metric or submetric."""
    print("Generating statistic tables...")
    tables = {}

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
    df_tables = {name: pd.DataFrame(data) for name, data in tables.items()}
    return df_tables


def plot_tables(tables, filename=None):
    """Print each metric table and optionally write all LaTeX tables to a file."""
    print("Plotting tables...")
    latex_blocks = []

    for name, df in tables.items():
        print(f"\n=== {name} ===\n")
        print(df.round(6))
        print()

        latex_code = df.to_latex(
            float_format="%.6g",
            escape=False,
            caption=name,
            label=name.replace(" ", "_")
        )
        latex_blocks.append(latex_code)

    combined_latex = "\n\n".join(latex_blocks)

    if filename:
        with open(filename, "w") as f:
            f.write("% Auto-generated LaTeX tables\n\n")
            f.write(combined_latex)
        print(f"\n✅ LaTeX tables successfully written to '{filename}'")

    return latex_blocks

def calc_stat(data, metric):
        """Compute a single statistical metric on data."""
        metric_funcs = {
            "Max": lambda x: np.max(x),
            "Mean": lambda x: np.mean(x),
            "Median": lambda x: np.median(x),
            "Min": lambda x: np.min(x),
            "Mode": lambda x: np.unique(x, return_counts=True)[0][
                np.argmax(np.unique(x, return_counts=True)[1])
            ],
            "Std": lambda x: np.std(x, ddof=1),
            "Var": lambda x: np.var(x, ddof=1),
            "1-Norm": lambda x: np.linalg.norm(x, ord=1),
            "2-Norm": lambda x: np.linalg.norm(x, ord=2),
            "Inf-Norm": lambda x: np.linalg.norm(x, ord=np.inf),
        }

        if metric not in metric_funcs:
            raise ValueError(f"Unknown metric: {metric}")

        return metric_funcs[metric](data)

# =============================================================================
# main() testing function
# =============================================================================

if __name__ == "__main__":
    # Run the full statistics generation
    analyze_quality_metrics()
