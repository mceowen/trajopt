from trajopt.trajectory_analyzer import TrajectoryAnalyzer

traj = TrajectoryAnalyzer("config.yaml")

data = traj.analyze()
traj.plot(data)
