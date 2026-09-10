"""Plot PD and momentum tracking NPZs without launching Isaac Sim."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BODY_NAMES = ["Trunk", "Left hand", "Right hand", "Left foot", "Right foot"]
KEYS = ["sim_pos", "ref_pos", "sim_vel", "ref_vel", "joint_torque", "joint_vel"]


def load_data(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in KEYS}
    shape = data["sim_pos"].shape
    if len(shape) != 4 or shape[2:] != (5, 3):
        raise ValueError(f"{path}: expected body arrays shaped (steps, envs, 5, 3)")
    for key, value in data.items():
        expected = shape if key in KEYS[:4] else (*shape[:2], 23)
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(f"{path}: invalid shape or nonfinite values in {key}")
    # The legacy evaluator can leave zero-filled frames after recording stops.
    valid = np.any(data["ref_pos"] != 0, axis=(1, 2, 3))
    indices = np.flatnonzero(valid)
    if not len(indices) or not valid[:indices[-1] + 1].all():
        raise ValueError(f"{path}: empty data or unrecorded interior frames")
    end = indices[-1] + 1
    if end != shape[0]:
        print(f"Trimmed {shape[0] - end} trailing zero frames from {path}")
    return {key: value[:end] for key, value in data.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pd", type=Path, default=Path("eval_data/comparison/pd.npz"))
    parser.add_argument("--momentum", type=Path, default=Path("eval_data/comparison/momentum.npz"))
    parser.add_argument("--output_dir", type=Path, default=Path("eval_data/comparison/plots"))
    parser.add_argument("--dt", type=float, default=0.02, help="Control interval in seconds.")
    args = parser.parse_args()
    if args.dt <= 0:
        parser.error("--dt must be positive")
    runs = {"PD": load_data(args.pd), "Momentum": load_data(args.momentum)}
    if runs["PD"]["ref_pos"].shape != runs["Momentum"]["ref_pos"].shape:
        raise ValueError("Evaluations must have matching frame and environment counts")
    for key in ["ref_pos", "ref_vel"]:
        if not np.allclose(runs["PD"][key], runs["Momentum"][key], atol=1e-5, rtol=1e-5):
            raise ValueError(f"Reference trajectories differ ({key}); use matching motion and resets")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps, envs = runs["PD"]["sim_pos"].shape[:2]
    time = (np.arange(steps) + 1) * args.dt
    colors = {"PD": "#2563eb", "Momentum": "#e07818"}
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{name}.png", dpi=180)
        plt.close(fig)

    def trace(ax, values, label):
        ax.plot(time, values.mean(axis=1), label=label, color=colors[label])
        ax.fill_between(time, values.min(axis=1), values.max(axis=1), color=colors[label], alpha=0.15)

    tracking, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    effort, effort_axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    bars, bar_ax = plt.subplots(figsize=(10, 4))
    kick, kick_ax = plt.subplots(figsize=(10, 4))
    kick_ax.plot(time, runs["PD"]["ref_pos"][:, :, 4, 2].mean(axis=1), "k--", label="Reference")
    rows = []
    for index, (label, data) in enumerate(runs.items()):
        pos_sq = np.sum((data["sim_pos"] - data["ref_pos"]) ** 2, axis=-1)
        vel_sq = np.sum((data["sim_vel"] - data["ref_vel"]) ** 2, axis=-1)
        body_rmse = np.sqrt(pos_sq.mean(axis=(0, 1)))
        torque = data["joint_torque"]
        power = np.sum(np.abs(torque * data["joint_vel"]), axis=-1)
        trace(axes[0], np.sqrt(pos_sq.mean(axis=2)) * 100, label)
        trace(axes[1], np.sqrt(vel_sq.mean(axis=2)), label)
        trace(effort_axes[0], np.sqrt((torque ** 2).mean(axis=2)), label)
        trace(effort_axes[1], power, label)
        trace(kick_ax, data["sim_pos"][:, :, 4, 2], label)
        bar_ax.bar(np.arange(5) + (index - 0.5) * 0.36, body_rmse * 100, width=0.36, label=label, color=colors[label])
        row = {"controller": label, "steps": steps, "environments": envs,
               "duration_s": steps * args.dt,
               "position_rmse_m": np.sqrt(pos_sq.mean()),
               "velocity_rmse_m_s": np.sqrt(vel_sq.mean()),
               "torque_rms_Nm": np.sqrt((torque ** 2).mean()),
               "peak_abs_torque_Nm": np.abs(torque).max(),
               "mean_abs_joint_power_W": power.mean(),
               "absolute_joint_work_J": power.sum(axis=0).mean() * args.dt}
        row.update({f"{name.lower().replace(' ', '_')}_rmse_m": value for name, value in zip(BODY_NAMES, body_rmse)})
        rows.append(row)
    for ax, ylabel in zip(axes, ["Body position RMSE (cm)", "Body velocity RMSE (m/s)"]):
        ax.set_ylabel(ylabel)
    for ax, ylabel in zip(effort_axes, ["RMS across joint torques (Nm)", "Total absolute joint power (W)"]):
        ax.set_ylabel(ylabel)
    for ax in [*axes, *effort_axes, kick_ax]:
        ax.grid(alpha=0.2)
        ax.legend()
    axes[-1].set_xlabel("Time after stepping (s)")
    effort_axes[-1].set_xlabel("Time after stepping (s)")
    axes[0].set_title("Tracking — environment mean; shading shows environment range")
    effort_axes[0].set_title("Control effort — environment mean; shading shows environment range")
    bar_ax.set(xticks=np.arange(5), xticklabels=BODY_NAMES, ylabel="Position RMSE (cm)", title="Position RMSE over time and environments")
    bar_ax.legend()
    kick_ax.set(xlabel="Time after stepping (s)", ylabel="Right-foot height (m)", title="Kick height — environment mean; shading shows environment range")
    for fig, name in [(tracking, "tracking_errors"), (bars, "body_position_rmse"), (kick, "kick_height"), (effort, "control_effort")]:
        save(fig, name)
    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved four plots and metrics.csv to {args.output_dir.resolve()}")
    for row in rows:
        print(f"{row['controller']}: position RMSE {row['position_rmse_m'] * 100:.2f} cm; torque RMS {row['torque_rms_Nm']:.2f} Nm")


if __name__ == "__main__":
    main()
