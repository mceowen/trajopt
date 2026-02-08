import importlib
import numpy as np

# Trajopt imports --> pip install -e ~/ACL/trajopt
import trajopt; importlib.reload(trajopt)
from trajopt.core.trajectory_analyzer import TrajectoryAnalyzer
np.random.seed(0)  # for reproducibility

# create trajectroy analyzer object
mission = "mission.yaml"
model = "trajopt/library/models/reentry_3dof.yaml"
method = "method.yaml"
variations = "variations.yaml"

trajopt = TrajectoryAnalyzer(mission, model, method, variations=variations)

trajopt.solve()