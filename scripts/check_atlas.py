"""Write a numerical residual report for the USD-derived T1 (not a PhysX rollout)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source/whole_body_tracking/whole_body_tracking/utils'))
from atlas_model import AtlasModel, FEET
from atlas_qp import AtlasQP, Contact
from atlas_control import force_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('eval_data/atlas/numerical_residuals.json'))
    args = parser.parse_args()
    model = AtlasModel()
    q = pin.neutral(model.model); q[2] = .72
    state = model.state(q, np.zeros(29))
    cases = []
    for feet in [(), FEET[:1], FEET[1:], FEET]:
        limits = np.full(23, 60.)
        result = AtlasQP(limits).solve(state, np.zeros(6), [Contact(f) for f in feet])
        diag = force_diagnostics(result, state, limits)
        cases.append(dict(active_feet=list(feet), variables=result['variable_count'], **result['metrics'],
                          minimum_point_normal_force_N=float(diag['normal_forces'].min()),
                          maximum_torque_utilization=float(diag['torque_utilization'].max())))
    report = dict(description='Analytical T1 numerical residuals; not simulator validation',
                  model_mass_kg=model.mass, acceptance_tolerance=2e-5, cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
