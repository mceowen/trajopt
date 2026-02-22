"""Quick syntax check - just import the modules"""
import sys
sys.path.insert(0, 'src')

try:
    from trajopt.library.methods import hyperparameters
    from trajopt.library.methods import subproblem
    from trajopt.library.methods import subproblem_constraints
    print("✓ All modules imported successfully")
    print("✓ No syntax errors detected")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
