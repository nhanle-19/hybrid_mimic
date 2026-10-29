"""Collect existing FORCE references and labels without changing their source files.

Only motions/ is a MotionLibrary input. Labels remain archival annotations:
the new run keeps its height-based penalty and does not consume manual labels.
Labels are never actor observations or runtime QP contact masks.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "retargeted_motion/labeled_force"
CASES = (
    ("01_walk", "retargeted_motion/hdm_bk_walk_in_place_settled_t1_training.npz",
     "eval_data/force/walk_in_place/walk_force_labels.npz", 261,
     "9e7117448732115882a63d3e7575f30d091ad7ac144f523aac67fdab59158b79",
     "Historical CPU replay completed all frames with a geometry-gated controller."),
    ("02_tilted_sway", "eval_data/toy_standing_sway_wide02/reference.npz",
     "eval_data/toy_standing_sway_wide02/force_reference_schedule.npz", 500,
     "35cececc299976aced21b09a766242bfca8ceb1ba6883ae4deb7b1f727dddeea",
     "Historical manual schedule completed two GPU trials with a geometry-gated controller."),
    ("03_standing", "eval_data/toy_standing/reference.npz", "configs/toy_wbc_inputs.json", 500,
     "738a0330786ccc38162d16ac0543c060ece49e792af3988d511cb62ecbbd85b2",
     "Constant fractions 0.2/0.6. Historical completed replay used plain-sigmoid logits; current JSON preserves those fractions under the dead-zone mapping."),
    ("04_narrow_sway", "eval_data/toy_standing_sway/reference.npz", "configs/toy_sway_wbc_inputs.json", 500,
     "62c763c8495e67b4ec8dc9251b2b246e9843ce71c7726e171aed858615b91f19",
     "Constant fractions 0.12/0.8. Historical geometry-gated FORCE replay completed 500 frames."),
    ("05_slide", "eval_data/toy_left_foot_slide/reference.npz",
     "eval_data/toy_left_foot_slide/force/force_refined_schedule.npz", 500,
     "ef53d4b3569419d76bad77c8f221d15b7e5a11bec40d64ad839ea830abeaf177",
     "Historical refined GPU replay completed; starts in shifted support pose, without testing the transfer from centered standing."),
    ("06_sole_rotation", "eval_data/toy_left_foot_rotation/reference.npz",
     "eval_data/toy_left_foot_rotation/force/labels.npz", 500,
     "d6cd5b0985ec6543680bab5f29a12ee82369d4a2e54c64065056aa6786580812",
     "Historical GPU replay completed; supported initial pose and fixed sole-center pivot."),
    ("07_toe_rotation", "eval_data/toy_left_foot_toe_rotation/reference.npz",
     "eval_data/toy_left_foot_toe_rotation/force/labels.npz", 500,
     "c7c5055bfecfafdca0c0bacc9fef831ccc40a800eb3aefc3399aca5c3fac7e80",
     "Historical GPU replay completed; begins with raised heel and fixed toe-edge pivot, without testing initial heel lift."),
    ("08_turn_right", "eval_data/reference_e1_turn_right/short_reanchor/reference.npz",
     "eval_data/reference_e1_turn_right/short_reanchor/force/labels.npz", 661,
     "b03922821a748e16bb6bdc4546c9dd5047b2f4d8ed4e5248cf2ab1da66c13a3d",
     "Historical CPU replay completed and settled but failed stricter tracking criteria; touchdown was delayed 0.82 s. Historical friction 0.3 and controller gains differ from current FORCE defaults."),
    ("09_front_kick", "eval_data/reference_planted_kick_20261016/g3_front_kick_t1_training.npz",
     "eval_data/g3_manual_20261016/force_replay/force_best_replay.npz", 115,
     "71149b98123642cfe54dd05ac539de70e744302e7d4e931fad381d2ee141d489",
     "Complete 115-row label table. Historical replay terminated after 52 steps; rows 52:115 are a reference-derived unvalidated tail, not a demonstrated solution."),
)

ALTERNATES = (
    ("05_slide", "eval_data/toy_left_foot_slide/force/force_reference_schedule.npz", "Unvalidated initial schedule; same reference as selected refined schedule."),
    ("05_slide", "eval_data/toy_left_foot_slide/force/force_slide_schedule.npz", "Earlier successful schedule; selected refined replay supersedes it."),
    ("08_turn_right", "eval_data/reference_e1_turn_right/short_reanchor/force/planned_labels.npz", "Planned input for the selected executed labels; same motion, not an additional training clip."),
    ("09_front_kick", "eval_data/g3_manual_20261016/force_replay/force_prestep_replay.npz", "Alternate same-motion schedule; historical 29-step trial."),
    ("09_front_kick", "eval_data/g3_manual_20261016/force_debug/force_left_step_first.npz", "Alternate same-motion schedule; historical 32-step trial."),
    ("09_front_kick", "eval_data/g3_manual_20261016/force_debug/replay/force_left_step_first_replay.npz", "Frozen replay of the alternate left-step-first schedule; historical 32-step trial."),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_identical(path: Path, payload: bytes) -> None:
    """Idempotently create an artifact; never overwrite differing existing data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to overwrite differing existing file: {path}")
    else:
        path.write_bytes(payload)


def write_json(path: Path, value: object) -> None:
    write_identical(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def copy_source(source: str, destination: Path, output: Path) -> dict:
    src = ROOT / source
    write_identical(destination, src.read_bytes())
    if digest(src) != digest(destination):
        raise ValueError(f"Copied artifact hash differs: {destination}")
    return {"source": source, "destination": str(destination.relative_to(output)), "sha256": digest(src)}


def decode(actions: np.ndarray) -> np.ndarray:
    return np.clip((1.0 / (1.0 + np.exp(-actions.astype(np.float64))) - 0.1) / 0.8, 0.0, 1.0)


def load_labels(source: str, frames: int, motion_hash: str) -> tuple:
    path = ROOT / source
    if path.suffix == ".json":
        actions = np.broadcast_to(np.asarray(json.loads(path.read_text())["force"], dtype=np.float64), (frames, 2)).copy()
        fractions = decode(actions)
        metadata = {"status": "constant_manual_inputs", "motion_hash_embedded_in_source": False}
        diagnostic = ("eval_data/toy_standing/force/diagnostics.npz" if "toy_wbc_inputs" in source
                      else "eval_data/toy_standing_sway/force_recheck/diagnostics.npz")
        with np.load(ROOT / diagnostic, allow_pickle=False) as archive:
            config = json.loads(str(archive["controller_config_json"]))
            np.testing.assert_allclose(fractions, archive["support_activation"], atol=1e-6, rtol=0)
        metadata.update(controller_config=config, association_evidence=diagnostic)
    else:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"]))
            if metadata["motion_sha256"] != motion_hash:
                raise ValueError(f"Motion-label hash mismatch: {source}")
            if metadata["controller"] != "force" or metadata["schema_version"] != 2:
                raise ValueError(f"Unsupported label schema: {source}")
            if metadata["action_layout"] != ["left_fz_limit", "right_fz_limit"]:
                raise ValueError(f"Unexpected label order: {source}")
            transform = metadata["action_transform"]
            if transform["type"] != "sigmoid_dead_zones" or transform["epsilon"] != 0.1:
                raise ValueError(f"Unexpected action transform: {source}")
            if float(archive["fps"].item()) != 50.0:
                raise ValueError(f"Unexpected label FPS: {source}")
            np.testing.assert_array_equal(archive["reference_frame"], np.arange(frames))
            actions, fractions = archive["actions"].copy(), archive["fractions"].copy()
    if actions.shape != (frames, 2) or fractions.shape != actions.shape:
        raise ValueError(f"Wrong label shape: {source}")
    if not np.isfinite(actions).all() or not np.isfinite(fractions).all():
        raise ValueError(f"Nonfinite labels: {source}")
    if np.any((fractions < 0) | (fractions > 1)):
        raise ValueError(f"Out-of-range fractions: {source}")
    np.testing.assert_allclose(decode(actions), fractions, atol=2e-7, rtol=0)
    return actions, fractions, metadata


def validate_library(output: Path) -> dict:
    location = ROOT / "source/whole_body_tracking/whole_body_tracking/utils/motion_library.py"
    spec = importlib.util.spec_from_file_location("force_library_validation", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    library = module.MotionLibrary(output / "motions", device="cpu")
    expected_names = [row[0] for row in CASES]
    if library.clip_names != expected_names or library.time_step_total != 4037:
        raise ValueError("Unexpected clips or total frame count in assembled MotionLibrary")
    if float(library.fps) != 50.0 or library.clip_lengths.tolist() != [row[3] for row in CASES]:
        raise ValueError("Unexpected frame rates or clip lengths")
    return {"loader": "MotionLibrary", "num_clips": library.num_clips, "total_frames": library.time_step_total,
            "fps": float(library.fps), "duration_s": library.time_step_total / float(library.fps),
            "joint_count": library.joint_pos.shape[1], "body_count": library._body_pos_w.shape[1],
            "finite_arrays_and_identical_joint_body_order": True}


def assemble(output: Path = DEFAULT_OUTPUT) -> dict:
    output = output.resolve()
    expected_motion_names = {row[0] + ".npz" for row in CASES}
    unexpected = {p.name for p in (output / "motions").glob("*.npz")} - expected_motion_names
    if unexpected:
        raise ValueError(f"Unexpected motion files; refusing to silently include or remove them: {sorted(unexpected)}")
    manifest = {
        "schema_version": 1,
        "purpose": "Nine existing labeled two-foot FORCE references, with original annotations preserved separately.",
        "motion_directory": "motions",
        "sampling": "One distinct clip per entry; the controller's motion sampler determines reset probabilities.",
        "label_use": {
            "actor_input": False, "qp_contact_mask": False, "cap_imitation_target_enabled_by_collection": False,
            "active_training_reward": "unchanged height-based non-contact penalty",
            "manual_annotations_consumed_by_active_training": False,
            "manual_noncontact_weight": "float32 1 where the selected manual cap fraction is exactly zero, otherwise 0",
            "scope": "Archival and inspection annotations only. The user selected the existing height-based penalty for this training run; manual zero-cap weights are not consumed by the reward.",
            "caution": "Manual zero caps can denote unloading during grounded sliding or rotation, not literal geometric separation. Positive caps are not target physical forces.",
        },
        "historical_validation": "Recorded manual trials used differing gains, force maxima, friction, and historical contact geometry gates. They do not establish performance of the current ungated controller.",
        "motions": [], "alternate_schedules": [], "separate": [],
    }
    for stem, source, labels, frames, expected_hash, note in CASES:
        if digest(ROOT / source) != expected_hash:
            raise ValueError(f"Canonical reference changed: {source}")
        motion_record = copy_source(source, output / "motions" / (stem + ".npz"), output)
        label_dir = output / "labels" / stem
        originals = [copy_source(labels, label_dir / Path(labels).name, output)]
        csv_source = str(Path(labels).with_suffix(".csv"))
        if Path(labels).suffix == ".npz" and (ROOT / csv_source).is_file():
            originals.append(copy_source(csv_source, label_dir / Path(csv_source).name, output))
        actions, fractions, metadata = load_labels(labels, frames, expected_hash)
        weight = (fractions == 0).astype(np.float32)
        annotation_metadata = {
            "schema_version": 1, "motion_source": source, "motion_sha256": expected_hash,
            "label_source": labels, "label_source_sha256": digest(ROOT / labels),
            "foot_order": ["left", "right"], "historical_label_status": metadata.get("status"),
            "manual_noncontact_weight_definition": "(manual_cap_fractions == 0).astype(float32)",
            "actor_input": False, "qp_contact_mask": False, "active_training_reward_input": False,
        }
        buffer = io.BytesIO()
        np.savez_compressed(buffer, fps=np.asarray(50.0), reference_frame=np.arange(frames, dtype=np.int64),
                            manual_raw_actions=actions.astype(np.float32), manual_cap_fractions=fractions.astype(np.float32),
                            manual_noncontact_weight=weight,
                            metadata_json=np.asarray(json.dumps(annotation_metadata, sort_keys=True)))
        annotation = label_dir / "annotations.npz"
        write_identical(annotation, buffer.getvalue())
        record = {
            "name": stem, **motion_record, "frames": frames, "fps": 50.0, "action_dim": 2,
            "original_labels": originals,
            "annotations": {"destination": str(annotation.relative_to(output)), "sha256": digest(annotation)},
            "manual_noncontact_frames_left_right": weight.sum(axis=0).astype(int).tolist(),
            "manual_force_max_n": metadata.get("controller_config", {}).get("contact_force_max"),
            "historical_label_status": metadata.get("status"), "historical_note": note,
        }
        if stem == "09_front_kick":
            record["historical_executed_prefix_frames"] = 52
            record["unvalidated_label_tail_frames_half_open"] = [52, 115]
        manifest["motions"].append(record)
    for stem, source, note in ALTERNATES:
        manifest["alternate_schedules"].append({"motion_name": stem, "source": source,
                                                "sha256": digest(ROOT / source), "note": note})

    crawl_dir = output / "separate/crawl"
    crawl_sources = (
        ("eval_data/reference_crawl_forward/reference.npz", "reference.npz"),
        ("eval_data/reference_crawl_forward/force/labels.npz", "labels.npz"),
        ("eval_data/reference_crawl_forward/force/planned_labels.npz", "planned_labels.npz"),
        ("configs/crawl_force_schedule.json", "configs/crawl_force_schedule.json"),
        ("configs/crawl_manual_nn_outputs.json", "configs/crawl_manual_nn_outputs.json"),
        ("configs/crawl_contact_surfaces.json", "configs/crawl_contact_surfaces.json"),
    )
    crawl = {"name": "crawl", "included_in_two_foot_training": False, "reference_frames": 599,
             "reference_fps": 29.96248436848687, "label_samples": 2396, "action_dim": 8,
             "reason": "Separate eight-region controller, contact model and collision asset; raw reference schema differs.",
             "artifacts": [copy_source(src, crawl_dir / dst, output) for src, dst in crawl_sources],
             "related_source_paths": ["eval_data/reference_crawl_forward/contact_geometry.npz",
                                      "eval_data/reference_crawl_forward/contact_labels.csv",
                                      "eval_data/reference_crawl_forward/physics_asset/t1_crawl.usda",
                                      "eval_data/reference_crawl_forward/physics_asset/collision_manifest.json",
                                      "eval_data/reference_crawl_forward/force/controller_profile.json",
                                      "eval_data/reference_crawl_forward/force/validation.json",
                                      "eval_data/reference_crawl_forward/force/replay_command.json"]}
    write_json(crawl_dir / "collection.json", crawl)
    manifest["separate"].append(crawl)
    g18_dir = output / "separate/g18_missing_labels"
    g18 = {"name": "g18_kick", "included_in_two_foot_training": False, "reference_frames": 142, "fps": 50.0,
           "reason": "Manual NPZ/CSV labels were deleted at the user's request. Retained summaries are not runnable label tables.",
           "artifacts": [copy_source("retargeted_motion/g18_push_kick_right_t1_training.npz", g18_dir / "reference.npz", output),
                         copy_source("eval_data/g18_kick_summary/README.txt", g18_dir / "original_summary.txt", output),
                         copy_source("eval_data/g18_kick_summary/verified_results.json", g18_dir / "verified_results.json", output)]}
    write_json(g18_dir / "collection.json", g18)
    manifest["separate"].append(g18)
    manifest["validation"] = validate_library(output)
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = assemble(args.output)
    print(json.dumps({"output": str(args.output.resolve()), **result["validation"]}, indent=2))
