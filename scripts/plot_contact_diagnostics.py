"""Plot foot motion and, when recorded, contact sensor/model diagnostics."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from compare_tracking import body_order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("eval_data/comparison/momentum.npz"))
    parser.add_argument("--output_dir", type=Path, default=Path("eval_data/comparison/contact_plots"))
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--contact_threshold", type=float, default=10.0, help="Net force magnitude threshold in N.")
    args = parser.parse_args()
    if args.dt <= 0 or args.contact_threshold <= 0:
        parser.error("dt and contact_threshold must be positive")
    with np.load(args.input, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    order = body_order(data)
    for key in ["sim_pos", "ref_pos", "sim_vel", "ref_vel", "sim_angvel", "ref_angvel"]:
        data[key] = data[key][:, :, order]
    valid = np.any(data["ref_pos"] != 0, axis=(1, 2, 3))
    indices = np.flatnonzero(valid)
    if not len(indices) or not valid[:indices[-1] + 1].all():
        raise ValueError("Empty data or unrecorded interior frames")
    steps = indices[-1] + 1
    dt = float(data.get("control_dt", args.dt))
    time = (np.arange(steps) + 1) * dt
    args.output_dir.mkdir(parents=True, exist_ok=True)
    has_contacts = "contact_net_forces_w" in data
    for env_id in range(data["sim_pos"].shape[1]):
        rows = 5 if has_contacts else 4
        fig, axes = plt.subplots(rows, 2, figsize=(13, 3 * rows), sharex=True)
        for col, (body_id, name, body_name) in enumerate([(3, "Left foot", "left_foot_link"), (4, "Right foot", "right_foot_link")]):
            for prefix, label, style in [("sim", "Measured motion", "-"), ("ref", "Reference", "--")]:
                pos = data[f"{prefix}_pos"][:steps, env_id, body_id]
                vel = data[f"{prefix}_vel"][:steps, env_id, body_id]
                ang = data[f"{prefix}_angvel"][:steps, env_id, body_id]
                for key, value in [("pos", pos), ("vel", vel), ("angvel", ang)]:
                    if not np.isfinite(value).all():
                        raise ValueError(f"Nonfinite {prefix}_{key}")
                axes[0, col].plot(time, pos[:, 2], style, label=label)
                axes[1, col].plot(time, np.linalg.norm(vel[:, :2], axis=-1), style, label=label)
                axes[2, col].plot(time, vel[:, 2], style, label=label)
                axes[3, col].plot(time, np.linalg.norm(ang, axis=-1), style, label=label)
            axes[0, col].set_title(name)
            if has_contacts:
                sensor_id = list(data["contact_body_names"]).index(body_name)
                force = data["contact_net_forces_w"][:steps, env_id, sensor_id]
                contact = np.linalg.norm(force, axis=-1) > args.contact_threshold
                axes[4, col].plot(time, force[:, 2], label="Sensor normal-force z component")
                if "model_contact_wrench_w" in data:
                    model_id = list(data["model_contact_body_names"]).index(body_name)
                    model = data["model_contact_wrench_w"][:steps, env_id, model_id]
                    axes[4, col].plot(time, model[:, 2], "--", label="QP Fz (last substep)")
                for ax in axes[:, col]:
                    ax.fill_between(time, 0, 1, where=contact, transform=ax.get_xaxis_transform(), color="green", alpha=0.08, label="Force-detected contact")
                    if "contact_sample_reset" in data:
                        for t in time[data["contact_sample_reset"][:steps, env_id]]:
                            ax.axvline(t, color="red", alpha=0.4)
        labels = ["Foot-link height (m)", "Horizontal body speed (m/s)", "Vertical body velocity (m/s)", "Angular speed (rad/s)", "Force (N)"]
        for row in range(rows):
            for col in range(2):
                axes[row, col].set_ylabel(labels[row])
                axes[row, col].grid(alpha=0.2)
                axes[row, col].legend(fontsize=7)
        for ax in axes[-1]:
            ax.set_xlabel("Time after stepping (s)")
        note = "Green: net force > threshold; red: reset. Forces sampled at control rate." if has_contacts else "Kinematic proxies only: contact state and sole-point slip are NOT measured."
        fig.suptitle(f"{args.input.stem} — environment {env_id}\n{note}")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = args.output_dir / f"{args.input.stem}_env{env_id}_contact.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
