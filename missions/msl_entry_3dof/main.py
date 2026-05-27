from trajopt.trajectory_analyzer import TrajectoryAnalyzer

config_path = "config.yaml"
trajopt = TrajectoryAnalyzer(config_path="config.yaml")

trajopt.solve()

data = trajopt.analyze()

trajopt.plot(data)
