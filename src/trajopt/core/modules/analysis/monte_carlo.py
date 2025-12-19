import trajopt.utils.config_loader as cfg
import numpy as np
import copy
import trajopt.core.trajopt_obj as traj
import trajopt.core.modules.method.scp as scp
import trajopt.analysis.default_analysis as default_analysis
import trajopt.utils.tools as tools

def add_monte_carlo_dispersions(mission_dict, realization):
        for mc_var, mc_disp in realization.items():
            mission_dict[mc_var] = mission_dict[mc_var] + mc_disp

def run_mc_analysis(example_name, nominal_config, gen_mc_variations=1, save_mc_variations=0, save_scenario_data=0, mc_name="mc1", local=False):


    mv_variations = cfg.load_mv_variations(example_name, local=local)

    if gen_mc_variations:
        mc_variations = cfg.gen_mc_variations(example_name, local=local)

        if save_mc_variations:
            np.save(f"data/mc_variations/{mc_name}", mc_variations)
    else:
        mc_variations = np.load(f"data/mc_variations/{mc_name}.npy", allow_pickle=True).item()

    variations = {
        "method": mv_variations,
        "mission": mc_variations
    }

    scenario_data = {}

    # loop through method variations
    for name, method_variation in variations["method"].items():
        
        # initialize method sub-dictionary for scenario_data dict
        scenario_data[name] = {"method_params": {},
                                    'mc_data': [None] * (variations["mission"]["num_variations"] + 1),
                                    }

        cached_subprob = None
        
        # loop through monte-carlo mission parameter realizations (number of runs)
        for run_idx, realization in enumerate(variations["mission"]["realizations"]):
            
            # take in nominal configs
            run_config = copy.deepcopy(nominal_config)

            # set method variations
            run_config["method"] = tools.deep_update(run_config["method"], method_variation)

            # set monte carlo mission variations
            add_monte_carlo_dispersions(run_config["mission"], realization)

            # create trajopt_obj instance
            trajopt_obj = traj.Problem(run_config, cached_subprob)
            
            # run SCP
            trajopt_obj = scp.run_scp(trajopt_obj)

            # perform default analysis on this mc run and store related params
            scenario_data[name]["mc_data"][run_idx] = default_analysis.perform_default_analysis(trajopt_obj)

            # store total time for scp (used to calculate time to converge)
            scenario_data[name]['mc_data'][run_idx]['t_full'] = trajopt_obj.solution['t_full']
            
            # cache subproblem graph to speed up solves
            cached_subprob = None # trajopt_obj.method.subprob

    if save_scenario_data:
        np.save(f"data/scenario_data/{example_name}_{mc_name}", scenario_data)

    return scenario_data, trajopt_obj