from trajopt.trajectory import Trajectory

config_path = "config.yaml"
traj = Trajectory(config_path)

traj.solve()

data = traj.analyze()

traj.plot(data)
