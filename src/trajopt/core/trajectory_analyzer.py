from trajopt.core.indexing.index_map import IndexMap
from trajopt.core.problem import Problem
from trajopt.core.solution_method import SolutionMethod
import trajopt.utils.config_loader as cfg
import trajopt.library.methods.scp as scp
import trajopt.core.analysis.analysis as analysis
import trajopt.core.analysis.plotting as plotting
from trajopt.core.analysis.trajplots import *

class TrajectoryAnalyzer:

    def __init__(self, mission, model, method, variations=None):
        
        self.config = cfg.load_trajopt_config(mission, model, method, variations)

        index_map    = IndexMap(self.config)
        self.problem = Problem(self.config, index_map=index_map)
        self.method  = SolutionMethod(self.problem, self.config, index_map=index_map)

        self.solution         = None
        self.results          = None
        self.variation_config = None

    def solve(self):
        scp.run_scp(self)

    def analyze(self, analysis_type="standalone", compute_iters=False):
        self.analysis_type = analysis_type

        # run standalone anaylsis by default or method/parameter variations if specified
        if analysis_type == "standalone":
            self.results = analysis.run_standalone_analysis(self, show_iters=compute_iters)
        
        elif analysis_type == "mc":
            self.results = analysis.run_mc_analysis(self)

        return self.results

    def plot(self, data, analysis_type="standalone", show_iters = False, animate=False):

        # plot the results (use MC-style plots when analysis_type is "mc")
        plotting.plot_default(self, data, analysis_type, show_iters=show_iters)

        # animate the results if specified
        if animate:
            plotting.plot_animated(self)