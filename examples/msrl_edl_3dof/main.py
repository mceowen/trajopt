from pathlib import Path

from trajopt.trajectory_analyzer import TrajectoryAnalyzer

traj = TrajectoryAnalyzer(Path(__file__).with_name("config.yaml"))

traj.solve()
traj.analyze()
traj.plot()
