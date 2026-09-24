"""Reference feedback and explicit contact planning for the WBCACC-style QP."""
import numpy as np
import pinocchio as pin

try:  # Also permit direct offline imports without the package's Isaac registration.
    from .wbc_acc_model import FEET
    from .wbc_acc_qp import Contact, MotionTask
except ImportError:
    from wbc_acc_model import FEET
    from wbc_acc_qp import Contact, MotionTask


def reference_contact_schedule(reference_states, height_tolerance=.025):
    """Offline planned FULL-SOLE contact; no sensor or policy gating.

    Normalize the motion's floor once from the tenth percentile of sole-center
    heights. A low sole center schedules load bearing. Using the lowest corner
    of a tilted retargeted foot can otherwise classify every frame as flight.
    Users can replace this schedule with a supplied boolean (frames,2) array.
    """
    heights = []
    for state in reference_states:
        heights.append([((Contact(foot).points@state['frames'][foot]['rotation'].T)
                         +state['frames'][foot]['position'])[:, 2] for foot in FEET])
    heights = np.asarray(heights)
    centers = heights.mean(axis=-1)
    floor = np.quantile(centers, .1)
    return centers <= floor+height_tolerance


def contact_weights(action, gains=None):
    """Bounded per-foot weights in left/right order; stable even for large logits."""
    from scipy.special import expit
    action = np.asarray(action, dtype=float)
    if action.shape != (2,) or not np.isfinite(action).all():
        raise ValueError('Expected two finite contact-weight logits (left, right)')
    low, high = getattr(gains, 'contact_weight_min', .1), getattr(gains, 'contact_weight_max', 100.)
    if not np.isfinite([low, high]).all() or not 0 < low < high:
        raise ValueError('Contact weight bounds must be finite and satisfy 0 < min < max')
    return low+(high-low)*expit(action)


def build_reference_tasks(state, reference, action, gains=None):
    """Fixed reference feedback; the two policy logits only set contact weights."""
    contact_weights(action, gains)  # Validate the two-output interface.
    nv = state['M'].shape[0]
    value = lambda name, default: getattr(gains, name, default)
    rate = np.r_[state['mass']*value('com_position_gain', 40)*(reference['com']-state['com'])
                  +value('linear_momentum_gain', 10)*(reference['momentum'][:3]-state['momentum'][:3]),
                  value('angular_momentum_gain', 15)*(reference['momentum'][3:]-state['momentum'][3:])]
    posture_target = value('posture_position_gain', 40)*(reference['q'][7:]-state['q'][7:])+value('posture_velocity_gain', 8)*(reference['v'][6:]-state['v'][6:])
    tasks = [MotionTask(np.eye(nv)[6:], posture_target, value('posture_weight', .1), 'posture')]
    pelvis, target = state['frames']['Trunk'], reference['frames']['Trunk']
    angular = value('pelvis_position_gain', 60)*pin.log3(target['rotation']@pelvis['rotation'].T)+value('pelvis_velocity_gain', 12)*(target['velocity'][3:]-pelvis['velocity'][3:])
    tasks.append(MotionTask(pelvis['J'][3:], angular-pelvis['bias'][3:], value('pelvis_weight', 5), 'pelvis_orientation'))
    return rate, tasks


def swing_tasks(state, reference, active, gains=None):
    weight = getattr(gains, 'swing_weight', 0.)
    if weight == 0:
        return []
    tasks = []
    for foot in FEET:
        if foot in active:
            continue
        current, target = state['frames'][foot], reference['frames'][foot]
        error = np.r_[target['position']-current['position'], pin.log3(target['rotation']@current['rotation'].T)]
        acceleration = getattr(gains, 'swing_position_gain', 80)*error+getattr(gains, 'swing_velocity_gain', 16)*(target['velocity']-current['velocity'])
        tasks.append(MotionTask(current['J'], acceleration-current['bias'], weight, f'swing_{foot}'))
    return tasks


def force_diagnostics(result, state, torque_limits):
    """Fixed-size output; absent feet have exactly zero forces, undefined CoP NaN."""
    forces = np.zeros((2, 4, 3)); normal = np.zeros((2, 4)); tangent = np.zeros((2, 4))
    utilization = np.zeros((2, 4)); cop = np.full((2, 2), np.nan)
    active = np.zeros(2, dtype=bool)
    for contact in result['contacts']:
        index = FEET.index(contact.body)
        active[index] = True
        normal_axis = np.asarray(contact.normal)/np.linalg.norm(contact.normal)
        points = result['contact_forces'][contact.body]
        f = np.array([force for _, force in points])
        forces[index] = f
        fn = f@normal_axis
        ft = np.linalg.norm(f-fn[:, None]*normal_axis, axis=-1)
        normal[index], tangent[index] = fn, ft
        # Ratios at unloaded points amplify solver roundoff; report them undefined.
        utilization[index] = np.divide(ft, contact.friction*fn, out=np.full_like(ft, np.nan), where=fn > 1.)
        if fn.sum() > 1e-6:
            # CoP projected into the contact body's local XY plane.
            local = (np.array([p for p, _ in points])-state['frames'][contact.body]['position'])@state['frames'][contact.body]['rotation']
            cop[index] = np.sum(local[:, :2]*fn[:, None], axis=0)/fn.sum()
    return dict(point_forces=forces, normal_forces=normal, tangential_forces=tangent,
                friction_utilization=utilization, cop=cop, active_contact=active,
                torque_utilization=np.abs(result['torque'])/torque_limits)
