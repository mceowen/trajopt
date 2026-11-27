import jax
jax.config.update("jax_enable_x64", True)
from trajopt.trajectory_analyzer import TrajectoryAnalyzer

traj = TrajectoryAnalyzer("config.yaml")
traj.solve()
traj.analyze()
traj.plot(save_dir="plots/standalone")
r = traj.result
print(f"\nRESULT converged={r.converged} iterations={r.iterations} status={r.status}")
