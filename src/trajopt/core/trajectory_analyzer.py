from trajopt.core.problem import Problem
from trajopt.core.solution_method import SolutionMethod
import trajopt.utils.config_loader as cfg
import trajopt.library.methods.scp as scp
import trajopt.core.analysis.analysis as analysis
import trajopt.core.analysis.plotting as plotting
import trajopt.core.analysis.temporary_hardcoded_plotting as temp_plots
from trajopt.core.analysis.trajplots import *

class TrajectoryAnalyzer:
    def __init__(self, trajopt_config_path):

        # load configs
        trajopt_config = cfg.load_trajopt_config(trajopt_config_path)
        problem_config = trajopt_config['problem']
        method_config = trajopt_config['method']

        # build optimal control problem and solution method from configs
        self.problem = Problem(problem_config)
        self.method  = SolutionMethod(self.problem, method_config)

        self.solution = None
        self.scenario_data = None

    def solve(self):
        """
        Solve the optimal control problem using configured method.
        """
        self.solution = scp.run_scp(self)

    def analyze(self, temporary_plotting_name=None, run_type="standalone"):
        """
        Perform analysis of the solution
        """

        # run analysis
        if run_type == "standalone":
            self.scenario_data = analysis.run_standalone_analysis(self)
        
        elif run_type == "mc":
            self.scenario_data = run_mc_analysis(self)

        # plot the results
        plotting.plot_default(self)

        return self.scenario_data
