import numpy as np
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj.__dict__

def dumps(obj):
    return json.dumps(obj, cls=NpEncoder)

def loads(str):
    return json.loads(str)