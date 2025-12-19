import trajopt.analysis.default_analysis as default_analysis

def run_standalone_analysis(trajopt_obj, method_name):
    # populate scenario_data dict for plotting
    scenario_data = {
        method_name: {
            "mc_data": [default_analysis.perform_default_analysis(trajopt_obj)]
        }
    }

    return scenario_data