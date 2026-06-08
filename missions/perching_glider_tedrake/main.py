from trajopt.trajectory_analyzer import TrajectoryAnalyzer

traj = TrajectoryAnalyzer("config.yaml")
traj.solve()

data = traj.analyze()
traj.plot(data)
