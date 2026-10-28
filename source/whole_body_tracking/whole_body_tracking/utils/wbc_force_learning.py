"""FORCE contact phase timing and weighted statistics, without simulator imports."""
import math
import torch


def swing_weights(clearance, low=.001, high=.005):
    if not 0 <= low < high:
        raise ValueError('Invalid swing clearance thresholds')
    phase = ((clearance-low)/(high-low)).clamp(0., 1.)
    return phase.square()*(3.-2.*phase)


def unloading_weights(swing, lead_time, dt):
    """Ramp unloading before swing; never extend it past reference touchdown.

    A future swing weight is discounted linearly with its time ahead. The
    endpoint is held, never wrapped to the start of another motion.
    """
    if not math.isfinite(lead_time) or lead_time < 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError('Unloading lead time must be nonnegative and dt positive')
    steps = int(round(lead_time/dt))
    result = swing.clone()
    frames = torch.arange(len(swing), device=swing.device)
    for offset in range(1, min(steps, len(swing)-1)+1):
        future = swing[(frames+offset).clamp_max(len(swing)-1)]
        result = torch.maximum(result, future*(1.-offset/(steps+1.)))
    return result


class ForceMetrics:
    """Accumulate numerators/denominators on device; drain once per PPO rollout.

    Conditional means omit missing data instead of reporting a false zero.
    Counts and episode weighting are independent of the number of resets in a
    simulator step. Episode state itself lives in MotionCompletionReward.
    """
    def __init__(self):
        self.sums = {}
        self.counts = {}
        self.totals = {}

    def mean(self, name, values, mask=None):
        values = values.detach().double()
        if mask is None:
            total, count = values.sum(), values.new_tensor(values.numel())
        else:
            mask = torch.broadcast_to(mask, values.shape)
            total = torch.where(mask, values, 0.).sum()
            count = mask.sum().to(values.dtype)
        if name not in self.sums:
            self.sums[name], self.counts[name] = total.clone(), count.clone()
        else:
            self.sums[name] += total
            self.counts[name] += count

    def count(self, name, values):
        value = values.detach().double().sum()
        if name not in self.totals:
            self.totals[name] = value.clone()
        else:
            self.totals[name] += value

    def drain(self):
        names = list(self.sums)
        totals = list(self.totals)
        if not names and not totals:
            return {}
        packed = torch.stack([self.sums[n] for n in names] + [self.counts[n] for n in names]
                             + [self.totals[n] for n in totals]).cpu().tolist()
        size = len(names)
        result = {f'WBCForce/{n}': packed[i]/packed[size+i] for i, n in enumerate(names)
                  if packed[size+i] > 0}
        result.update({f'WBCForce/{n}': packed[2*size+i] for i, n in enumerate(totals)})
        self.sums.clear()
        self.counts.clear()
        self.totals.clear()
        return result
