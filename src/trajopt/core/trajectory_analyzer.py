from trajopt.core.problem import Problem
from trajopt.core.solution_method import SolutionMethod
import trajopt.utils.config_loader as cfg
import trajopt.library.methods.scp as scp
import trajopt.core.analysis.analysis as analysis
import trajopt.core.analysis.plotting as plotting
import trajopt.core.analysis.temporary_hardcoded_plotting as temp_plots
from trajopt.core.analysis.trajplots import *

class TrajectoryAnalyzer:

    def __init__(self, mission_path, model_path, method_path, variation_config_path=None):
        config = cfg.load_trajopt_config(mission_path, model_path, method_path)
        problem_config = config['problem']
        method_config = config['method']

        self.problem = Problem(problem_config)
        self.method = SolutionMethod(self.problem, method_config)

        self.solution = None
        self.scenario_data = None
        self.variation_config_path = variation_config_path

    def solve(self):
        self.solution = scp.run_scp(self)

    def analyze(self, temporary_plotting_name=None, run_type="standalone", animate=False):
        if run_type == "standalone":
            self.scenario_data = analysis.run_standalone_analysis(self)
        elif run_type == "mc":
            self.scenario_data = run_mc_analysis(self)

        plotting.plot_default(self)

        if animate:
            plotting.plot_animated(self)

        return self.scenario_data
