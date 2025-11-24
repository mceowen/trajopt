import trajopt.analysis.default_analysis as default_analysis

def run_standalone_analysis(problem, method_name):
    # populate scenario_data dict for plotting
    scenario_data = {
        method_name: {
            "mc_data": [default_analysis.perform_default_analysis(problem)]
        }
    }

    return scenario_data