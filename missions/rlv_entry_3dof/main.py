from trajopt.trajectory_analyzer import TrajectoryAnalyzer

config_path = "./config.yaml"

traj = TrajectoryAnalyzer(config_path)
traj.solve()

data = traj.analyze()
traj.plot(data)
