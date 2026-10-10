"""Read manually supplied raw NN outputs without loading Isaac Sim or a policy."""
import json
import math
from pathlib import Path


def load_manual_inputs(path, controller):
    counts = {'force': 2, 'authority': 12}
    if controller not in counts:
        raise ValueError('Manual NN inputs are supported for force and authority only')
    data = json.loads(Path(path).read_text())
    values = data.get(controller) if isinstance(data, dict) else None
    count = counts[controller]
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f'{controller}: provide a list of {count} raw NN outputs in {path}')
    if any(value is None for value in values):
        raise ValueError(f'{controller}: manual NN outputs are not set yet; fill all {count} entries in {path}')
    if any(type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 3.4e38
           for value in values):
        raise ValueError(f'{controller}: raw NN outputs must be finite float32 numbers')
    return values
