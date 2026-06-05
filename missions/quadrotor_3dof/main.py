from trajopt.trajectory import Trajectory

traj = Trajectory("config.yaml")

traj.solve()

data = traj.analyze()

traj.plot(data)
