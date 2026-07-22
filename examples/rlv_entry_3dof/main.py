from trajopt.trajectory_analyzer import TrajectoryAnalyzer

traj = TrajectoryAnalyzer("config.yaml")

traj.solve()
traj.analyze()
traj.plot()
