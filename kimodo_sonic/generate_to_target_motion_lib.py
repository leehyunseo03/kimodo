#!/usr/bin/env python3
"""Generate or convert a Kimodo G1 walk into a GEAR-Sonic motion_lib target run.

This mirrors the MotionBricks target-export layout used by GEAR-Sonic:

  output_dir/
    qpos/<motion_name>.npy
    target_reference/<forward_target_name>.npz
    robot_filtered/kimodo_target/<motion_name>.pkl
    visualization/<motion_name>_trajectory_markers.json

The generated PKL can be passed to:
``manager_env.commands.motion.motion_lib_cfg.motion_file`` with
``smpl_motion_file=dummy``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.spatial.transform import Rotation


KIMODO_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_WS = KIMODO_ROOT.parent
GROOT_ROOT = ISAACLAB_WS / "GR00T-WholeBodyControl"

for path in (KIMODO_ROOT, GROOT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


MUJOCO_DOF_AXES = np.asarray(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def repo_path(path_text: str | Path, root: Path = GROOT_ROOT) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ISAACLAB_WS))
    except ValueError:
        return str(path)


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    quat_xyzw = Rotation.from_euler("z", yaw).as_quat()
    return quat_xyzw[[3, 0, 1, 2]].astype(np.float32)


def quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
    quat_xyzw = np.asarray(quat_wxyz, dtype=np.float64)[[1, 2, 3, 0]]
    return float(Rotation.from_quat(quat_xyzw).as_euler("xyz")[2])


def qpos_to_pose_aa(qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float32)
    pose_aa = np.zeros((qpos.shape[0], 30, 3), dtype=np.float32)
    quat_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]]
    pose_aa[:, 0, :] = Rotation.from_quat(quat_xyzw).as_rotvec().astype(np.float32)
    pose_aa[:, 1:, :] = qpos[:, 7:36, None] * MUJOCO_DOF_AXES[None, :, :]
    return pose_aa


def make_target_qpos(forward_meters: float, target_height: float, yaw: float = 0.0) -> np.ndarray:
    qpos = np.zeros(36, dtype=np.float32)
    qpos[:3] = [forward_meters, 0.0, target_height]
    qpos[3:7] = yaw_to_quat_wxyz(yaw)
    qpos[7:36] = np.asarray(
        [
            0.0,   # left_hip_pitch
            0.0,   # left_hip_roll
            0.0,   # left_hip_yaw
            0.0,   # left_knee
            0.0,   # left_ankle_pitch
            0.0,   # left_ankle_roll
            0.0,   # right_hip_pitch
            0.0,   # right_hip_roll
            0.0,   # right_hip_yaw
            0.0,   # right_knee
            0.0,   # right_ankle_pitch
            0.0,   # right_ankle_roll
            0.0,   # waist_yaw
            0.0,   # waist_roll
            0.0,   # waist_pitch
            -1.55, # left_shoulder_pitch
            0.35,  # left_shoulder_roll
            0.0,   # left_shoulder_yaw
            0.45,  # left_elbow
            0.0,   # left_wrist_roll
            0.0,   # left_wrist_pitch
            0.0,   # left_wrist_yaw
            -1.55, # right_shoulder_pitch
            -0.35, # right_shoulder_roll
            0.0,   # right_shoulder_yaw
            0.45,  # right_elbow
            0.0,   # right_wrist_roll
            0.0,   # right_wrist_pitch
            0.0,   # right_wrist_yaw
        ],
        dtype=np.float32,
    )
    return qpos


def load_qpos_csv(path: Path) -> np.ndarray:
    qpos = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"Expected Kimodo G1 MuJoCo qpos CSV shape (T, 36), got {qpos.shape}")
    return qpos.astype(np.float32)


def load_qpos_any(path: Path, qpos_key: str) -> np.ndarray:
    if path.suffix == ".csv":
        return load_qpos_csv(path)
    if path.suffix == ".npy":
        qpos = np.load(path, allow_pickle=True)
    elif path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if qpos_key not in data:
            raise KeyError(f"{qpos_key!r} not found in {path}; keys={data.files}")
        qpos = data[qpos_key]
    else:
        raise ValueError(f"Unsupported qpos input {path}; expected .csv, .npy, or .npz")
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"Expected qpos shape (T, 36), got {qpos.shape}")
    return qpos


def final_zero_dof_constraint(num_frames: int, forward_meters: float, target_height: float, fps: int) -> dict[str, Any]:
    """Build a Kimodo full-body constraint whose exported MuJoCo target has 29 zero DOFs."""
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.geometry import matrix_to_axis_angle
    from kimodo.skeleton.registry import build_skeleton

    skeleton = build_skeleton(34)
    converter = MujocoQposConverter(skeleton)

    # qpos_to_motion_dict derives smooth_root_pos internally. Kimodo's root
    # smoother is singular for one-frame clips, so use a short valid path and
    # keep only the terminal frame as the full-body keyframe constraint.
    start_qpos = make_target_qpos(0.0, target_height)
    target_qpos = make_target_qpos(forward_meters, target_height)
    alpha = np.linspace(0.0, 1.0, 5, dtype=np.float32)[:, None]
    qpos_seq = (1.0 - alpha) * start_qpos[None, :] + alpha * target_qpos[None, :]
    qpos_seq[:, 3:7] = target_qpos[3:7]

    motion_dict = converter.qpos_to_motion_dict(qpos_seq, source_fps=float(fps))
    local_aa = matrix_to_axis_angle(motion_dict["local_rot_mats"][-1]).detach().cpu().numpy()
    root_position = motion_dict["root_positions"][-1].detach().cpu().numpy()

    return {
        "type": "fullbody",
        "frame_indices": [int(num_frames - 1)],
        "root_positions": [root_position.astype(float).round(6).tolist()],
        "smooth_root_2d": [[float(root_position[0]), float(root_position[2])]],
        "local_joints_rot": [local_aa.astype(float).round(6).tolist()],
    }


def write_constraints(args: argparse.Namespace, constraints_path: Path) -> Path:
    num_frames = max(2, int(round(float(args.duration) * args.fps)))
    waypoint_count = max(2, int(args.root_waypoints))
    frame_indices = np.linspace(0, num_frames - 1, waypoint_count).round().astype(int)
    progress = np.linspace(0.0, 1.0, waypoint_count, dtype=np.float32)

    # Kimodo is Y-up and uses XZ as the ground plane. MuJoCo forward +X maps to Kimodo +Z.
    root2d = np.stack([np.zeros_like(progress), progress * float(args.forward_meters)], axis=1)
    constraints: list[dict[str, Any]] = [
        {
            "type": "root2d",
            "frame_indices": frame_indices.tolist(),
            "smooth_root_2d": root2d.astype(float).round(6).tolist(),
        },
        final_zero_dof_constraint(num_frames, args.forward_meters, args.target_height, args.fps),
    ]
    constraints_path.parent.mkdir(parents=True, exist_ok=True)
    constraints_path.write_text(json.dumps(constraints, indent=2) + "\n", encoding="utf-8")
    return constraints_path


def run_kimodo_generation(args: argparse.Namespace, work_dir: Path) -> Path:
    constraints_path = write_constraints(args, work_dir / "constraints" / f"{args.motion_name}_constraints.json")
    output_stem = work_dir / "kimodo_raw" / args.motion_name
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "kimodo.scripts.generate",
        args.prompt,
        "--model",
        args.model,
        "--duration",
        str(args.duration),
        "--num_samples",
        "1",
        "--diffusion_steps",
        str(args.diffusion_steps),
        "--constraints",
        str(constraints_path),
        "--output",
        str(output_stem),
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.text_encoder_device:
        os.environ["TEXT_ENCODER_DEVICE"] = args.text_encoder_device
    print("running Kimodo generation:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=KIMODO_ROOT, check=True)
    csv_path = output_stem.with_suffix(".csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Kimodo generation finished but did not write expected G1 CSV: {csv_path}")
    return csv_path


def normalize_root_to_target(qpos: np.ndarray, target_qpos: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return qpos.astype(np.float32)

    out = qpos.astype(np.float32).copy()
    start_xy = out[0, :2].copy()
    end_xy = out[-1, :2].copy()
    delta = end_xy - start_xy
    denom = float(np.dot(delta, delta))
    if denom < 1e-8:
        progress = np.linspace(0.0, 1.0, out.shape[0], dtype=np.float32)
    else:
        progress = ((out[:, :2] - start_xy[None, :]) @ delta) / denom
        progress = np.clip(progress.astype(np.float32), 0.0, 1.0)
        progress = np.maximum.accumulate(progress)

    residual = out[:, :2] - (start_xy[None, :] + progress[:, None] * delta[None, :])
    residual *= (1.0 - progress[:, None])
    out[:, 0] = progress * target_qpos[0] + residual[:, 0]
    out[:, 1] = progress * target_qpos[1] + residual[:, 1]

    out[:, 2] += target_qpos[2] - out[-1, 2]
    out[:, 3:7] /= np.maximum(np.linalg.norm(out[:, 3:7], axis=1, keepdims=True), 1e-8)
    return out


def append_target_settle(qpos: np.ndarray, target_qpos: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    pieces = [qpos.astype(np.float32)]
    if args.snap_to_target_frames > 0:
        start = qpos[-1].astype(np.float32)
        frames = []
        for idx in range(1, args.snap_to_target_frames + 1):
            alpha = idx / float(args.snap_to_target_frames)
            frame = (1.0 - alpha) * start + alpha * target_qpos
            frame[3:7] = target_qpos[3:7]
            frames.append(frame.astype(np.float32))
        pieces.append(np.asarray(frames, dtype=np.float32))
    if args.append_target_hold > 0:
        pieces.append(np.repeat(target_qpos[None, :], args.append_target_hold, axis=0).astype(np.float32))
    return np.concatenate(pieces, axis=0).astype(np.float32)


def write_markers(json_path: Path, xml_path: Path, root_xyz: np.ndarray, target_xyz: np.ndarray, stride: int) -> None:
    stride = max(1, int(stride))
    sampled = root_xyz[::stride]
    payload = {
        "motion_name": json_path.stem.replace("_trajectory_markers", ""),
        "coordinate_frame": "MuJoCo qpos root translation: x forward, y left, z up",
        "intermediate_marker_color": "yellow",
        "target_marker_color": "blue",
        "marker_stride": stride,
        "intermediate_xyz": sampled.astype(float).round(6).tolist(),
        "target_xyz": np.asarray(target_xyz, dtype=float).round(6).tolist(),
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    geoms = [
        f'    <geom name="kimodo_path_{idx:04d}" type="sphere" size="0.035" '
        f'pos="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" rgba="1 0.72 0.05 0.85"/>'
        for idx, xyz in enumerate(sampled)
    ]
    geoms.append(
        f'    <geom name="kimodo_target" type="sphere" size="0.07" '
        f'pos="{target_xyz[0]:.6f} {target_xyz[1]:.6f} {target_xyz[2]:.6f}" rgba="0.1 0.35 1 1"/>'
    )
    xml_path.write_text(
        '<mujoco model="kimodo trajectory markers">\n  <worldbody>\n'
        + "\n".join(geoms)
        + "\n  </worldbody>\n</mujoco>\n",
        encoding="utf-8",
    )


def export_outputs(args: argparse.Namespace, qpos: np.ndarray, target_qpos: np.ndarray, source_path: Path) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    qpos_dir = output_dir / "qpos"
    target_dir = output_dir / "target_reference"
    robot_dir = output_dir / "robot_filtered" / "kimodo_target"
    viz_dir = output_dir / "visualization"
    for directory in (qpos_dir, target_dir, robot_dir, viz_dir):
        directory.mkdir(parents=True, exist_ok=True)

    motion_name = args.motion_name
    target_name = args.forward_target_name
    qpos_npy = qpos_dir / f"{motion_name}.npy"
    qpos_npz = qpos_dir / f"{motion_name}.npz"
    target_npz = target_dir / f"{target_name}.npz"
    target_pkl = target_dir / f"{target_name}.pkl"
    robot_pkl = robot_dir / f"{motion_name}.pkl"
    markers_json = viz_dir / f"{motion_name}_trajectory_markers.json"
    markers_xml = viz_dir / f"{motion_name}_trajectory_markers.xml"

    np.save(qpos_npy, qpos)
    np.savez_compressed(
        qpos_npz,
        qpos=qpos,
        target_qpos=target_qpos,
        root_xyz=qpos[:, :3],
        dof=qpos[:, 7:36],
        fps=np.asarray(args.fps, dtype=np.int32),
    )
    np.savez_compressed(
        target_npz,
        target_qpos=target_qpos,
        qpos=target_qpos[None, :],
        dof_mujoco=target_qpos[7:36],
        fps=np.asarray(args.fps, dtype=np.int32),
    )
    joblib.dump({target_name: {"target_qpos": target_qpos, "qpos": target_qpos[None, :], "fps": args.fps}}, target_pkl)

    entry = {
        "root_trans_offset": qpos[:, :3].astype(np.float32),
        "pose_aa": qpos_to_pose_aa(qpos),
        "dof": qpos[:, 7:36].astype(np.float32),
        "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]].astype(np.float32),
        "smpl_joints": np.zeros((qpos.shape[0], 24, 3), dtype=np.float32),
        "fps": int(args.fps),
    }
    joblib.dump({motion_name: entry}, robot_pkl, compress=True)
    write_markers(markers_json, markers_xml, qpos[:, :3], target_qpos[:3], args.marker_stride)

    final_dof_rmse = float(np.linalg.norm(qpos[-1, 7:36] - target_qpos[7:36]) / math.sqrt(29))
    manifest = {
        "created_unix": time.time(),
        "source": display_path(source_path),
        "fps": args.fps,
        "motion_name": motion_name,
        "forward_meters": args.forward_meters,
        "target_height": args.target_height,
        "target_root_xyz": target_qpos[:3].round(6).tolist(),
        "target_yaw": quat_wxyz_to_yaw(target_qpos[3:7]),
        "target_dof": "both-hand raise pose in MuJoCo 29DOF order",
        "frames": int(qpos.shape[0]),
        "append_target_hold": args.append_target_hold,
        "snap_to_target_frames": args.snap_to_target_frames,
        "final_root_xyz": qpos[-1, :3].round(6).tolist(),
        "final_root_xy_error": float(np.linalg.norm(qpos[-1, :2] - target_qpos[:2])),
        "final_dof_rmse": final_dof_rmse,
        "robot_pkl": display_path(robot_pkl),
        "qpos_npy": display_path(qpos_npy),
        "qpos_npz": display_path(qpos_npz),
        "target_npz": display_path(target_npz),
        "target_pkl": display_path(target_pkl),
        "visualization": {
            "trajectory_markers_json": display_path(markers_json),
            "trajectory_markers_xml": display_path(markers_xml),
            "marker_stride": args.marker_stride,
        },
        "gearsonic_motion_file": f"/workspace/GR00T-WholeBodyControl/{robot_pkl.relative_to(GROOT_ROOT)}",
        "smpl_motion_file": "dummy",
        "viewer_command": (
            "LIVESTREAM=2 /workspace/isaaclab/isaaclab.sh -p "
            "kimodo_sonic/kimodo_metrics.py --livestream 2"
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote manifest: {manifest_path}", flush=True)
    print(f"wrote GEAR-Sonic motion: {robot_pkl}", flush=True)
    print(f"final_root_xy_error={manifest['final_root_xy_error']:.6f} m final_dof_rmse={final_dof_rmse:.6f} rad")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kimodo G1 5 m target exporter for GEAR-Sonic.")
    parser.add_argument("--input_csv", type=str, default=None, help="Existing Kimodo G1 MuJoCo qpos CSV.")
    parser.add_argument("--input_qpos", type=str, default=None, help="Existing qpos .npy/.npz with shape (T, 36).")
    parser.add_argument("--qpos_key", type=str, default="qpos")
    parser.add_argument("--output_dir", type=str, default=str(GROOT_ROOT / "kimodo_sonic" / "motion"))
    parser.add_argument("--motion_name", type=str, default="kimodo_to_target_forward_5m_hand_raise")
    parser.add_argument("--forward_meters", type=float, default=5.0)
    parser.add_argument("--forward_target_name", type=str, default="forward_5m_hand_raise_target")
    parser.add_argument("--target_height", type=float, default=0.78)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--root_waypoints", type=int, default=7)
    parser.add_argument("--prompt", type=str, default="A humanoid robot walks forward and comes to a stable stop.")
    parser.add_argument("--model", type=str, default="Kimodo-G1-RP-v1")
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--text_encoder_device", type=str, default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--normalize_root_to_target", type=int, default=1)
    parser.add_argument("--snap_to_target_frames", type=int, default=30)
    parser.add_argument("--append_target_hold", type=int, default=100)
    parser.add_argument("--marker_stride", type=int, default=10)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    work_dir = repo_path(args.output_dir)
    target_qpos = make_target_qpos(args.forward_meters, args.target_height)

    if args.input_csv:
        source_path = repo_path(args.input_csv, root=KIMODO_ROOT)
        qpos = load_qpos_csv(source_path)
    elif args.input_qpos:
        source_path = repo_path(args.input_qpos, root=KIMODO_ROOT)
        qpos = load_qpos_any(source_path, args.qpos_key)
    else:
        source_path = run_kimodo_generation(args, work_dir)
        qpos = load_qpos_csv(source_path)

    qpos = normalize_root_to_target(qpos, target_qpos, bool(args.normalize_root_to_target))
    qpos = append_target_settle(qpos, target_qpos, args)
    export_outputs(args, qpos, target_qpos, source_path)


if __name__ == "__main__":
    main()
