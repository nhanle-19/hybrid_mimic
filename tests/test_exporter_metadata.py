"""Exercise ONNX metadata serialization without starting Isaac Sim."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import onnx
import pytest
import torch


@pytest.fixture
def exporter(monkeypatch):
    stubs = {
        "isaaclab.envs": {"ManagerBasedRLEnv": object},
        "isaaclab_rl.rsl_rl.exporter": {"_OnnxPolicyExporter": object},
        "whole_body_tracking.tasks.tracking.mdp": {"MotionCommand": object},
    }
    for name, attrs in stubs.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[1] / "source/whole_body_tracking/whole_body_tracking/utils/exporter.py"
    spec = importlib.util.spec_from_file_location("metadata_exporter_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("nominal", [False, True])
def test_metadata_export(exporter, tmp_path, hybrid, nominal):
    joints = [f"joint_{i}" for i in range(23)]
    data = NS(joint_names=joints, joint_stiffness=torch.zeros(2, 23),
              joint_damping=torch.zeros(2, 23), default_joint_pos=torch.ones(2, 23))
    if nominal:
        data.default_joint_pos_nominal = torch.full((23,), 2.)
    terms = {"joint_pos": NS(_scale=torch.full((2, 23), .25))}
    motion = NS(seq_len=1, cfg=NS(anchor_body_name="pelvis", body_names=["pelvis"]))
    env = NS(scene={"robot": NS(data=data)},
             action_manager=NS(active_terms=list(terms), get_term=terms.__getitem__),
             command_manager=NS(active_terms=["motion"], get_term=lambda _: motion),
             observation_manager=NS(active_terms={"policy": ["joint_pos"]},
                                    cfg=NS(policy=NS(history_length=1))))
    if hybrid:
        env.hybrid_controller = NS(action_dim=57, end_effector_names=["left_foot", "right_foot", "left_hand", "right_hand"],
            desired_linear_velocity_scale=.25, desired_angular_velocity_scale=.5, torque_action_scale=.1)
    model = onnx.helper.make_model(onnx.helper.make_graph([], "metadata_test", [], []))
    onnx.save(model, tmp_path / "policy.onnx")
    exporter.attach_onnx_metadata(env, "test_run", str(tmp_path))
    saved = onnx.load(tmp_path / "policy.onnx")
    metadata = {entry.key: entry.value for entry in saved.metadata_props}
    assert metadata["run_path"] == "test_run"
    assert metadata["joint_names"].split(",") == joints
    assert [float(x) for x in metadata["default_joint_pos"].split(",")] == [2. if nominal else 1.] * 23
    scales = [float(x) for x in metadata["action_scale"].split(",")]
    assert scales == [.25] * 23
    if hybrid:
        assert metadata["action_type"] == "hybrid_mimic"
        assert metadata["policy_action_dim"] == "57"
        assert metadata["action_joint_names"].split(",") == joints
        assert metadata["action_layout"] == "joint_position,linear_velocity_body,wrench_logits,torque_reference,angular_velocity_body"
        assert metadata["desired_linear_velocity_scale"] == "0.25"
        assert metadata["desired_angular_velocity_scale"] == "0.5"
        assert metadata["torque_action_scale"] == "0.1"
    else:
        assert "action_type" not in metadata
