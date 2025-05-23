import os
import json
import numpy as np

# Allowed base types for internal traversal
ALLOWED_TYPES = (int, float, str, bool, list, tuple, dict, set, np.ndarray, type(None))

def is_allowed(obj):
    return isinstance(obj, ALLOWED_TYPES)

def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items() if is_allowed(v)}
    elif isinstance(obj, list):
        return [clean(v) for v in obj if is_allowed(v)]
    elif isinstance(obj, tuple):
        return [clean(v) for v in obj if is_allowed(v)]  # convert to list
    elif isinstance(obj, set):
        return [clean(v) for v in obj if is_allowed(v)]  # convert to list
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # convert to list
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return None  # skip unsupported types

def dump_filtered_dict(data, filename):
    # Expand ~ to full path
    filename = os.path.expanduser(filename)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    cleaned_data = clean(data)
    with open(filename, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
