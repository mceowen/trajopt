import copy
from trajopt.core.problem import Problem
from trajopt.core.solution_method import SolutionMethod
import trajopt.utils.config_loader as cfg
import trajopt.library.methods.scp as scp
import trajopt.core.analysis.analysis as analysis
import trajopt.core.analysis.plotting as plotting
import trajopt.core.analysis.temporary_hardcoded_plotting as temp_plots
from trajopt.core.analysis.trajplots import *
import yaml

class TrajectoryAnalyzer:

    def __init__(self, mission_path, model_path, method_path, variations=None):
        config = cfg.load_trajopt_config(mission_path, model_path, method_path)
        problem_config = config['problem']
        method_config = config['method']

        self.problem = Problem(copy.deepcopy(problem_config))
        self.method = SolutionMethod(self.problem, copy.deepcopy(method_config))
        self.problem_config = copy.deepcopy(problem_config)
        self.method_config = copy.deepcopy(method_config)

        self.solution = None
        self.scenario_data = None
        self.variation_config = None
        
        if variations is not None:
            with open(variations, 'r') as f:
                self.variation_config = yaml.safe_load(f)

    def solve(self):
        self.solution = scp.run_scp(self)

    def analyze(self, analysis_type="standalone", animate=False):

        # run standalone anaylsis by default or method/parameter variations if specified
        if analysis_type == "standalone":
            self.scenario_data = analysis.run_standalone_analysis(self)
        elif analysis_type == "mc":
            self.scenario_data = analysis.run_mc_analysis(self)

        # plot the results
        plotting.plot_default(self)

        # animate the results if specified
        if animate:
            plotting.plot_animated(self)

        return self.scenario_data