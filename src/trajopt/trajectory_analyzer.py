import trajopt.utils.config_loader as config_loader
import trajopt.analysis.analysis as analysis
import trajopt.analysis.plotting as plotting
from trajopt.trajectory import Trajectory
from trajopt.methods.scp.scp_method import SCPMethod
from trajopt.utils.tools import deep_merge, recursive_attrdict


class TrajectoryAnalyzer():

    def __init__(self, config_path, method_overrides=None) -> None:

        self.config_path = config_path
        self.config = config_loader.load_trajopt_config(config_path)

        if method_overrides:
            self.config.method = deep_merge(self.config.method, recursive_attrdict(method_overrides))

        self.trajectory = Trajectory(self.config.trajectory)
        self.method = SCPMethod(self.config.method, self.trajectory)
        self._solved = False

    def solve(self):
        self.method.solve()
        self._solved = True

    def analyze(self):
        analysis_cfg = self.config.get("analysis", {})
        analysis_type = analysis_cfg.get("type", "standalone")
        self.analysis_type = analysis_type

        if analysis_type == "standalone":
            self.results = analysis.run_standalone_analysis(self)

        elif analysis_type == "mc":
            self.results = analysis.run_mc_analysis(self)

        elif analysis_type == "method_variation":
            self.results = analysis.run_method_variation(self)

        return self.results

    def plot(self, data):
        analysis_cfg = self.config.get("analysis", {})
        analysis_type = analysis_cfg.get("type", "standalone")

        if analysis_type == "method_variation":
            plotting.plot_method_variation(self, data)
        else:
            plotting.plot(self, data)
