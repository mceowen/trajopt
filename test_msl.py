import sys
sys.path.insert(0, 'src')
from trajopt import TrajOpt
import yaml

# Load configs
with open('examples/msl_entry_3dof/method.yaml') as f:
    method_cfg = yaml.safe_load(f)
with open('examples/msl_entry_3dof/mission.yaml') as f:
    mission_cfg = yaml.safe_load(f)

# Create trajopt instance
trajopt = TrajOpt(
    method_config=method_cfg,
    mission_config=mission_cfg
)

# Try to solve
try:
    trajopt.solve()
except Exception as e:
    import traceback
    print(f'Error: {e}')
    traceback.print_exc()
