import trajopt.core.analysis.default_analysis as default_analysis

def run_standalone_analysis(trajopt_obj):
    
    # populate scenario_data dict for plotting
    scenario_data = {
        "autotune": {
            "mc_data": [default_analysis.perform_default_analysis(trajopt_obj)]
        }
    }

    return scenario_data