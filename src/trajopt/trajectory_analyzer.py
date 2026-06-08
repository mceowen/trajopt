import trajopt.utils.config_loader as config_loader
import trajopt.analysis.analysis as analysis
import trajopt.analysis.plotting as plotting
from trajopt.trajectory import Trajectory
from trajopt.methods.scp.scp_method import SCPMethod


class TrajectoryAnalyzer():

    def __init__(self, config_path) -> None:
        
        # load config
        self.config = config_loader.load_trajopt_config(config_path)

        # create trajectory
        trajectory_config = self.config.trajectory
        self.trajectory = Trajectory(trajectory_config)

        # create method
        method_config = self.config.method
        self.method = SCPMethod(method_config, self.trajectory)

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
