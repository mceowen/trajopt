from trajopt.trajectory_analyzer import TrajectoryAnalyzer

trajopt = TrajectoryAnalyzer("config.yaml")
trajopt.solve()

data = trajopt.analyze()
trajopt.plot(data)