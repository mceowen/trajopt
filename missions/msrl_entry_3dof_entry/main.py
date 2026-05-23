from trajopt.trajectory_analyzer import TrajectoryAnalyzer

mission = "mission.yaml"
model   = "trajopt/models/configs/reentry_3dof.yaml"
method  = "method.yaml"

trajopt = TrajectoryAnalyzer(mission, model, method)
trajopt.solve()
