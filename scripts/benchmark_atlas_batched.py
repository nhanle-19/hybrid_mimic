"""Measure device-resident Atlas dynamics and constrained QP throughput."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source/whole_body_tracking/whole_body_tracking/utils'))
from atlas_torch_model import AtlasTorchModel
from atlas_batched_control import BatchedAtlasQP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batches', nargs='+', type=int, default=[2, 64, 1024])
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if torch.device(args.device).type != 'cuda':
        parser.error('This benchmark requires CUDA')
    cfg = SimpleNamespace(friction=.6, momentum_weights=(1.,1.,1.,10.,10.,10.), force_weight=1e-5,
        acceleration_weight=1e-5, force_objective_weight=.001, force_logit_clip=10., angular_force_scale=20.,
        base_acceleration_weight=200., enforce_stance=False, batched_max_iterations=60,
        batched_tolerance=1e-7, residual_tolerance=2e-5)
    model = AtlasTorchModel(args.device)
    solver = BatchedAtlasQP(['left_foot_link','right_foot_link','left_hand_link','right_hand_link'],cfg,args.device)
    torch.manual_seed(42)
    print(json.dumps({'gpu': torch.cuda.get_device_name(args.device), 'dtype': 'float64'}), flush=True)
    with torch.inference_mode():
        for batch in args.batches:
            q = torch.zeros((batch,30), device=args.device, dtype=torch.float64)
            q[:,6] = 1.
            q[:,7:] = .1*torch.randn_like(q[:,7:])
            v = .1*torch.randn((batch,29), device=args.device, dtype=torch.float64)
            pd = 20*torch.randn((batch,23), device=args.device, dtype=torch.float64)
            limits = torch.full_like(pd,60.)
            acc = torch.randn((batch,6), device=args.device, dtype=torch.float64)
            logits = torch.randn((batch,5), device=args.device, dtype=torch.float64)
            active = torch.rand((batch,4), device=args.device)>.5
            def step():
                return solver.solve(model.state(q,v),pd,limits,acc,logits,torch.zeros_like(pd),torch.ones_like(pd),active)
            result = step()
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            start = perf_counter()
            for _ in range(args.repeats):
                result = step()
            torch.cuda.synchronize(args.device)
            seconds = (perf_counter()-start)/args.repeats
            print(json.dumps({'batch': batch, 'seconds_per_batch': seconds, 'env_solves_per_second': batch/seconds,
                'valid': int(result['valid'].sum()), 'iterations': result['solver_info']['iterations'],
                'peak_torch_memory_mib': torch.cuda.max_memory_allocated(args.device)/2**20}), flush=True)
            if not bool(result['valid'].all()):
                raise RuntimeError('Benchmark QP validation failed')


if __name__ == '__main__':
    main()
