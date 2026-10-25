"""Live-state FORCE controller with independently sampled manual crawl outputs."""
import math
import torch

try:
    from .wbc_crawl_contact import WBCCrawlForceQP
except ImportError:
    from wbc_crawl_contact import WBCCrawlForceQP


class CrawlForceController:
    def __init__(self, model, cfg, geometry, reference_q, reference_v, actions, fps, limits,
                 backend='batched', solver_tolerance=1e-6, friction=.6, hold_frame=None,
                 reference_contact_labels=None, action_fps=None):
        self.model, self.cfg = model, cfg
        self.q, self.v = reference_q, reference_v
        self.labels, self.actions, self.fps, self.limits = reference_contact_labels, actions, float(fps), limits
        if hold_frame is not None and (not isinstance(hold_frame, int) or not 0 <= hold_frame < len(reference_q)):
            raise ValueError('Held frame must index the supplied reference')
        self.hold_frame = hold_frame
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('Invalid crawl FPS')
        self.action_fps = self.fps if action_fps is None else float(action_fps)
        multiple = self.action_fps/self.fps
        if (not math.isfinite(multiple) or multiple < 1 or not math.isclose(multiple, round(multiple))
                or actions.shape != (len(reference_q)*round(multiple), 8)
                or not bool(torch.isfinite(actions).all())):
            raise ValueError('Live crawl schedule requires eight finite outputs at an integer multiple of reference FPS')
        if self.labels is not None and (self.labels.dtype != torch.bool or self.labels.shape != (len(reference_q), 8)):
            raise ValueError('Optional reference annotations must be boolean (frames,8)')
        # Match WBCACCActionCfg's live stopping criterion. Physical force,
        # torque and inverse-dynamics checks retain their 2e-5 tolerance;
        # the tighter offline optimum comparison uses WBCCrawlForceQP directly.
        if not math.isfinite(friction) or friction <= 0:
            raise ValueError('Friction must be finite and positive')
        self.solver = WBCCrawlForceQP(cfg, geometry, backend=backend, solver_tolerance=solver_tolerance,
                                     friction=friction)

    def reference(self, time_s):
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError('Reference time must be finite and nonnegative')
        position = min(time_s*self.fps, len(self.q)-1) if self.hold_frame is None else self.hold_frame
        index = min(int(math.floor(position+1e-10)), len(self.q)-1)
        following = min(index+1, len(self.q)-1)
        fraction = max(0., position-index)
        q = self.q[index:index+1]*(1-fraction)+self.q[following:following+1]*fraction
        # Short-arc quaternion interpolation; binary labels never interpolate.
        first, second = self.q[index, 3:7], self.q[following, 3:7]
        if torch.dot(first, second) < 0:
            second = -second
        quat = first*(1-fraction)+second*fraction
        q[:, 3:7] = quat/quat.norm()
        v = self.v[index:index+1]*(1-fraction)+self.v[following:following+1]*fraction
        if self.hold_frame is not None:
            v = torch.zeros_like(v)
        return self.model.state(q, v), index

    def compute(self, actual_state, time_s):
        reference, index = self.reference(time_s)
        sample = min(int(math.floor(time_s*self.action_fps+1e-10)), len(self.actions)-1)
        if self.hold_frame is not None:
            sample = round(self.hold_frame*self.action_fps/self.fps)
        result = self.solver.solve(actual_state, reference, self.actions[sample:sample+1], self.limits)
        if self.labels is not None:
            # Kept only for comparing the recorded trajectory with reference
            # geometry. Changing these annotations cannot change the torque.
            result['contact_labels'] = self.labels[index:index+1]
        result.update(reference_frame=index, reference_q=reference['q'], reference_v=reference['v'],
                      policy_sample=sample, raw_actions=self.actions[sample:sample+1],
                      joint_rmse=(actual_state['q'][:, 7:]-reference['q'][:, 7:]).square().mean(-1).sqrt(),
                      root_error=(actual_state['q'][:, :3]-reference['q'][:, :3]).norm(dim=-1))
        return result


def classify_contact_points(local_points, entries, surface_names):
    """Assign measured contact locations to the nearest authored convex surface.

    A link may have both hand and forearm, or knee and shin colliders. Classify
    with the actual hull planes, not the link name or the scheduled label.
    """
    if not entries:
        raise ValueError('Missing collision surfaces for contact-bearing body')
    distances = []
    for entry in entries:
        planes = local_points.new_tensor(entry['planes_body'])
        distances.append((local_points@planes[:, :3].T+planes[:, 3]).amax(-1).abs())
    selected = torch.stack(distances, -1).argmin(-1)
    indices = torch.tensor([surface_names.index(entry['region']) for entry in entries], device=local_points.device)
    return indices[selected]
