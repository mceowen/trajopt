from trajopt.problem import Problem
import trajopt.methods.scp.scp as scp
import trajopt.utils.config_loader as config_loader
from trajopt.indexing.index_map import IndexMap
from trajopt.scaling.nondim import Nondim
import trajopt.analysis.analysis as analysis
import trajopt.analysis.plotting as plotting

class TrajectoryAnalyzer:
    def __init__(self, config_path):

        self.config    = config_loader.load_trajopt_config(config_path)
        self.index_map = IndexMap(self.config)
        self.nondim    = Nondim(self.config, self.index_map)
        
        self.problem = Problem(self.config, self.index_map, self.nondim)

        # TODO (Carlos): directly pointing to GeneralSCvx method, need to generalize path
        SolutionMethod = getattr(scp, "SCP")
        self.method = SolutionMethod(self.config, self.index_map, self.problem)

        self.solution         = None
        self.results          = None
        self.variation_config = None

    def solve(self):
        self.method.solve()

    def analyze(self):
        analysis_cfg = self.config.get("analysis", {})
        analysis_type = analysis_cfg.get("type", "standalone")
        self.analysis_type = analysis_type

        if analysis_type == "standalone":
            self.results = analysis.run_standalone_analysis(self)
        elif analysis_type == "mc":
            self.results = analysis.run_mc_analysis(self)

        return self.results

    def plot(self, data):
        plotting.plot(self, data)