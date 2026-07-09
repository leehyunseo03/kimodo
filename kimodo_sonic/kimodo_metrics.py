#!/usr/bin/env python3
"""Plot Kimodo marker-path vs GEAR-Sonic robot tracking.

This compares the yellow trajectory markers written by
``generate_to_target_motion_lib.py`` against the actual robot body path recorded
by ``BodyTrackingCallback`` during ``run_gearsonic_kimodo_eval.sh``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kimodo_metrics_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


KIMODO_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_WS = KIMODO_ROOT.parent
GROOT_ROOT = ISAACLAB_WS / "GR00T-WholeBodyControl"

DEFAULT_MOTION_NAME = "kimodo_to_target_forward_5m_hand_raise"
DEFAULT_TARGET_NAME = "forward_5m_hand_raise_target"
DEFAULT_RECORDING = GROOT_ROOT / "metrics" / "kimodosonic" / "motion"
DEFAULT_MARKER_JSON = (
    GROOT_ROOT
    / "kimodo_sonic"
    / "motion"
    / "visualization"
    / f"{DEFAULT_MOTION_NAME}_trajectory_markers.json"
)
DEFAULT_QPOS = GROOT_ROOT / "kimodo_sonic" / "motion" / "qpos" / f"{DEFAULT_MOTION_NAME}.npz"
DEFAULT_TARGET = (
    GROOT_ROOT / "kimodo_sonic" / "motion" / "target_reference" / f"{DEFAULT_TARGET_NAME}.npz"
)
DEFAULT_OUT_DIR = KIMODO_ROOT / "kimodo_sonic" / "metrics"

UNIT_SCALES = {
    "m": (1.0, "m"),
    "cm": (100.0, "cm"),
    "mm": (1000.0, "mm"),
}


def repo_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute() or path.exists():
        return path

    text = str(path)
    mappings = {
        "/workspace/GR00T-WholeBodyControl": GROOT_ROOT,
        "/workspace/kimodo": KIMODO_ROOT,
    }
    for prefix, root in mappings.items():
        if text.startswith(prefix):
            return root / text[len(prefix) + 1 :]

    for root in (Path.cwd(), KIMODO_ROOT, GROOT_ROOT, ISAACLAB_WS):
        candidate = root / path
        if candidate.exists():
            return candidate
    return path


def load_npz(path: Path) -> dict[str, np.ndarray]:
    # Compatibility for object arrays written by newer NumPy builds.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def resolve_recording(path_text: str | Path) -> Path:
    path = repo_path(path_text)
    if path.is_dir():
        path = path / "first_episode_body_tracking.npz"
    if not path.exists():
        raise FileNotFoundError(f"Recording not found: {path}")
    return path


def load_marker_json(path_text: str | Path) -> dict[str, Any]:
    path = repo_path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Marker JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_qpos_meta(path_text: str | Path | None) -> dict[str, np.ndarray]:
    if path_text is None:
        return {}
    path = repo_path(path_text)
    if not path.exists():
        return {}
    if path.suffix == ".npz":
        return load_npz(path)
    if path.suffix == ".npy":
        return {"qpos": np.load(path, allow_pickle=True)}
    return {}


def load_target_xyz(path_text: str | Path | None) -> np.ndarray | None:
    if path_text is None:
        return None
    path = repo_path(path_text)
    if not path.exists():
        return None
    if path.suffix == ".npz":
        data = load_npz(path)
        for key in ("target_qpos", "qpos"):
            if key in data:
                arr = np.asarray(data[key], dtype=float)
                return arr.reshape(-1)[:3]
    return None


def object_array_to_list(value: np.ndarray | None) -> list[str]:
    if value is None:
        return []
    return [str(item) for item in np.asarray(value, dtype=object).reshape(-1).tolist()]


def find_root_body_index(body_names: list[str], preferred: str | None) -> int:
    if preferred:
        if preferred not in body_names:
            raise ValueError(f"Root body '{preferred}' not found. Available: {body_names}")
        return body_names.index(preferred)
    for name in ("pelvis", "pelvis_link", "torso_link", "base_link", "root"):
        if name in body_names:
            return body_names.index(name)
    return 0


def marker_eval_steps(
    marker_count: int,
    marker_stride: int,
    source_fps: float,
    eval_fps: float,
) -> np.ndarray:
    source_frames = np.arange(marker_count, dtype=float) * float(marker_stride)
    return source_frames / float(source_fps) * float(eval_fps)


def interp_path(path_xyz: np.ndarray, src_steps: np.ndarray, dst_steps: np.ndarray) -> np.ndarray:
    out = np.empty((len(dst_steps), 3), dtype=float)
    for axis in range(3):
        out[:, axis] = np.interp(
            dst_steps,
            src_steps,
            path_xyz[:, axis],
            left=path_xyz[0, axis],
            right=path_xyz[-1, axis],
        )
    return out


def infer_source_fps(qpos_meta: dict[str, np.ndarray], fallback: float) -> float:
    if "fps" in qpos_meta:
        return float(np.asarray(qpos_meta["fps"]).reshape(-1)[0])
    return float(fallback)


def align_marker_path(
    marker_xyz: np.ndarray,
    marker_steps: np.ndarray,
    robot_steps: np.ndarray,
    robot_root: np.ndarray,
    ref_root: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    marker_at_robot = interp_path(marker_xyz, marker_steps, robot_steps)
    if ref_root is not None and len(ref_root):
        offset = ref_root[0] - marker_at_robot[0]
    else:
        offset = robot_root[0] - marker_at_robot[0]
    return marker_xyz + offset[None, :], marker_at_robot + offset[None, :]


def plot_xy_path(
    marker_world: np.ndarray,
    robot_root: np.ndarray,
    target_world: np.ndarray | None,
    out_path: Path,
    unit: str,
    scale: float,
    root_name: str,
) -> None:
    display_origin = marker_world[0]
    marker_plot = marker_world - display_origin[None, :]
    robot_plot = robot_root - display_origin[None, :]
    target_plot = target_world - display_origin if target_world is not None else None

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    ax.plot(
        marker_plot[:, 0] * scale,
        marker_plot[:, 1] * scale,
        color="#d6a700",
        linewidth=2.0,
        label="target reference",
    )
    ax.scatter(
        marker_plot[:, 0] * scale,
        marker_plot[:, 1] * scale,
        color="#ffd21a",
        edgecolor="black",
        linewidth=0.25,
        s=34,
        label="target reference samples",
    )
    ax.plot(
        robot_plot[:, 0] * scale,
        robot_plot[:, 1] * scale,
        color="#1f77b4",
        linewidth=1.8,
        label="actual robot",
    )
    ax.scatter(robot_plot[0, 0] * scale, robot_plot[0, 1] * scale, marker="o", s=70, label="robot start")
    ax.scatter(robot_plot[-1, 0] * scale, robot_plot[-1, 1] * scale, marker="x", s=90, label="robot end")
    if target_plot is not None:
        ax.scatter(
            target_plot[0] * scale,
            target_plot[1] * scale,
            marker="*",
            s=190,
            color="#2457ff",
            label="target",
        )
    ax.set_title("Kimodo yellow marker path vs GEAR-Sonic robot path")
    ax.set_xlabel(f"relative x ({unit})")
    ax.set_ylabel(f"relative y ({unit})")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_axis_by_step(
    marker_steps: np.ndarray,
    marker_world: np.ndarray,
    robot_steps: np.ndarray,
    robot_root: np.ndarray,
    out_path: Path,
    unit: str,
    scale: float,
    root_name: str,
) -> None:
    display_origin = marker_world[0]
    marker_plot = marker_world - display_origin[None, :]
    robot_plot = robot_root - display_origin[None, :]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for axis, name in enumerate(("x", "y", "z")):
        ax = axes[axis]
        ax.plot(
            robot_steps,
            robot_plot[:, axis] * scale,
            color="#1f77b4",
            linewidth=1.7,
            label="actual robot",
        )
        ax.scatter(
            marker_steps,
            marker_plot[:, axis] * scale,
            color="#ffd21a",
            edgecolor="black",
            linewidth=0.25,
            s=22,
            label="target reference",
        )
        ax.plot(
            marker_steps,
            marker_plot[:, axis] * scale,
            color="#d6a700",
            linewidth=1.2,
            alpha=0.8,
        )
        ax.set_ylabel(f"relative {name} ({unit})")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("GEAR-Sonic motion step")
    fig.suptitle("Coordinate trace: yellow markers vs actual robot")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_error(
    robot_steps: np.ndarray,
    delta: np.ndarray,
    out_path: Path,
    unit: str,
    scale: float,
) -> None:
    xy_error = np.linalg.norm(delta[:, :2], axis=1)
    xyz_error = np.linalg.norm(delta, axis=1)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(robot_steps, xy_error * scale, label="XY error", linewidth=2.0)
    ax.plot(robot_steps, xyz_error * scale, label="XYZ error", linewidth=1.5, alpha=0.8)
    ax.set_title("GEAR-Sonic actual path error from Kimodo yellow marker path")
    ax.set_xlabel("GEAR-Sonic motion step")
    ax.set_ylabel(unit)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_29dof_joint_graphs(
    recording: dict[str, np.ndarray],
    trace_path: Path,
    error_path: Path,
) -> dict[str, Any]:
    ref_joint = recording.get("ref_joint_pos")
    robot_joint = recording.get("robot_joint_pos")
    if ref_joint is None or robot_joint is None:
        return {"has_29dof_joint_graphs": False}

    ref_joint = np.asarray(ref_joint, dtype=float)
    robot_joint = np.asarray(robot_joint, dtype=float)
    if ref_joint.ndim != 2 or robot_joint.ndim != 2 or ref_joint.size == 0 or robot_joint.size == 0:
        return {"has_29dof_joint_graphs": False}

    frame_count = min(ref_joint.shape[0], robot_joint.shape[0])
    joint_count = min(ref_joint.shape[1], robot_joint.shape[1], 29)
    ref_joint = ref_joint[:frame_count, :joint_count]
    robot_joint = robot_joint[:frame_count, :joint_count]
    time_step = (
        np.asarray(recording["time_step"], dtype=float)[:frame_count]
        if "time_step" in recording
        else np.arange(frame_count, dtype=float)
    )

    joint_names = object_array_to_list(recording.get("joint_names"))
    if not joint_names:
        joint_names = [f"joint_{idx:02d}" for idx in range(joint_count)]
    joint_names = joint_names[:joint_count]

    abs_rad = np.abs(robot_joint - ref_joint)
    abs_deg = np.rad2deg(abs_rad)

    cols = 3
    rows = int(np.ceil(joint_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, max(8, rows * 2.45)), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for idx, ax in enumerate(axes[:joint_count]):
        ax.plot(time_step, ref_joint[:, idx], color="#d6a700", linewidth=1.2, label="target reference")
        ax.plot(time_step, robot_joint[:, idx], color="#1f77b4", linewidth=1.0, alpha=0.85, label="actual robot")
        ax.set_title(
            f"{joint_names[idx]}\nmean {np.nanmean(abs_deg[:, idx]):.2f} deg, max {np.nanmax(abs_deg[:, idx]):.2f} deg",
            fontsize=8,
        )
        ax.grid(alpha=0.25)
        if idx % cols == 0:
            ax.set_ylabel("qpos (rad)")
        if idx == 0:
            ax.legend(fontsize=8)
    for ax in axes[joint_count:]:
        ax.axis("off")
    for ax in axes[-cols:]:
        if ax.has_data():
            ax.set_xlabel("GEAR-Sonic motion step")
    fig.suptitle("29DOF qpos: target reference vs actual robot", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.975])
    fig.savefig(trace_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(rows, cols, figsize=(16, max(8, rows * 2.25)), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for idx, ax in enumerate(axes[:joint_count]):
        ax.plot(time_step, abs_deg[:, idx], color="#d62728", linewidth=1.0)
        ax.fill_between(time_step, 0.0, abs_deg[:, idx], color="#d62728", alpha=0.16)
        ax.set_title(joint_names[idx], fontsize=8)
        ax.grid(alpha=0.25)
        if idx % cols == 0:
            ax.set_ylabel("abs error (deg)")
    for ax in axes[joint_count:]:
        ax.axis("off")
    for ax in axes[-cols:]:
        if ax.has_data():
            ax.set_xlabel("GEAR-Sonic motion step")
    fig.suptitle("29DOF abs qpos error over time", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.975])
    fig.savefig(error_path, dpi=180)
    plt.close(fig)

    return {
        "has_29dof_joint_graphs": True,
        "num_joint_frames": int(frame_count),
        "num_joints": int(joint_count),
        "joint_mean_abs_error_rad": float(np.nanmean(abs_rad)),
        "joint_p95_abs_error_rad": float(np.nanpercentile(abs_rad, 95)),
        "joint_max_abs_error_rad": float(np.nanmax(abs_rad)),
        "joint_final_mean_abs_error_rad": float(np.nanmean(abs_rad[-1])),
        "joint_mean_abs_error_deg": float(np.nanmean(abs_deg)),
        "joint_p95_abs_error_deg": float(np.nanpercentile(abs_deg, 95)),
        "joint_max_abs_error_deg": float(np.nanmax(abs_deg)),
        "joint_final_mean_abs_error_deg": float(np.nanmean(abs_deg[-1])),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kimodo yellow marker path vs GEAR-Sonic robot plotter.")
    parser.add_argument("--recording", type=str, default=str(DEFAULT_RECORDING))
    parser.add_argument("--marker-json", type=str, default=str(DEFAULT_MARKER_JSON))
    parser.add_argument("--qpos", type=str, default=str(DEFAULT_QPOS), help="Optional qpos npz/npy used to read source fps.")
    parser.add_argument("--target", type=str, default=str(DEFAULT_TARGET))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--unit", choices=sorted(UNIT_SCALES), default="mm")
    parser.add_argument("--root-body", type=str, default=None)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--eval-fps", type=float, default=50.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scale, unit = UNIT_SCALES[args.unit]

    recording_path = resolve_recording(args.recording)
    recording = load_npz(recording_path)
    marker_payload = load_marker_json(args.marker_json)
    qpos_meta = load_qpos_meta(args.qpos)

    marker_xyz = np.asarray(marker_payload["intermediate_xyz"], dtype=float)
    if marker_xyz.ndim != 2 or marker_xyz.shape[1] != 3:
        raise ValueError(f"Expected marker intermediate_xyz shape (N, 3), got {marker_xyz.shape}")

    robot_body = np.asarray(recording["robot_body_pos_w"], dtype=float)
    ref_body = np.asarray(recording["ref_body_pos_w"], dtype=float) if "ref_body_pos_w" in recording else None
    body_names = object_array_to_list(recording.get("body_names"))
    root_idx = find_root_body_index(body_names, args.root_body)
    root_name = body_names[root_idx] if body_names else f"body_{root_idx}"

    robot_root = robot_body[:, root_idx, :]
    ref_root = ref_body[:, root_idx, :] if ref_body is not None and ref_body.ndim == 3 else None
    robot_steps = (
        np.asarray(recording["time_step"], dtype=float)
        if "time_step" in recording
        else np.arange(len(robot_root), dtype=float)
    )

    marker_stride = int(marker_payload.get("marker_stride", 1))
    source_fps = infer_source_fps(qpos_meta, args.source_fps)
    marker_steps = marker_eval_steps(len(marker_xyz), marker_stride, source_fps, args.eval_fps)
    marker_world, marker_at_robot = align_marker_path(
        marker_xyz, marker_steps, robot_steps, robot_root, ref_root
    )
    target_xyz = load_target_xyz(args.target)
    target_world = None
    if target_xyz is not None:
        target_world = target_xyz + (marker_world[0] - marker_xyz[0])

    out_dir = repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_xy_path(
        marker_world,
        robot_root,
        target_world,
        out_dir / "kimodo_yellow_markers_vs_robot_xy.png",
        unit,
        scale,
        root_name,
    )
    plot_axis_by_step(
        marker_steps,
        marker_world,
        robot_steps,
        robot_root,
        out_dir / "kimodo_yellow_markers_vs_robot_by_step.png",
        unit,
        scale,
        root_name,
    )

    delta = robot_root - marker_at_robot
    plot_error(
        robot_steps,
        delta,
        out_dir / "kimodo_yellow_markers_vs_robot_error.png",
        unit,
        scale,
    )

    xy_error = np.linalg.norm(delta[:, :2], axis=1)
    xyz_error = np.linalg.norm(delta, axis=1)
    display_origin = marker_world[0]
    robot_final_relative = robot_root[-1] - display_origin
    joint_summary = plot_29dof_joint_graphs(
        recording,
        out_dir / "kimodo_29dof_reference_vs_robot_qpos.png",
        out_dir / "kimodo_29dof_abs_error_timeseries.png",
    )
    summary = {
        "recording": str(recording_path),
        "marker_json": str(repo_path(args.marker_json)),
        "qpos": str(repo_path(args.qpos)),
        "target": str(repo_path(args.target)),
        "root_body": root_name,
        "num_robot_frames": int(len(robot_root)),
        "num_marker_points": int(len(marker_world)),
        "marker_stride_source_frames": marker_stride,
        "source_fps": float(source_fps),
        "eval_fps": float(args.eval_fps),
        "unit": unit,
        "robot_final_x_m": float(robot_root[-1, 0]),
        "robot_final_y_m": float(robot_root[-1, 1]),
        "robot_final_z_m": float(robot_root[-1, 2]),
        "robot_final_relative_x_m": float(robot_final_relative[0]),
        "robot_final_relative_y_m": float(robot_final_relative[1]),
        "robot_final_relative_z_m": float(robot_final_relative[2]),
        "mean_xy_error_m": float(np.nanmean(xy_error)),
        "max_xy_error_m": float(np.nanmax(xy_error)),
        "final_xy_error_m": float(xy_error[-1]),
        "mean_xyz_error_m": float(np.nanmean(xyz_error)),
        "max_xyz_error_m": float(np.nanmax(xyz_error)),
        "final_xyz_error_m": float(xyz_error[-1]),
        "plots": [
            "kimodo_yellow_markers_vs_robot_xy.png",
            "kimodo_yellow_markers_vs_robot_by_step.png",
            "kimodo_yellow_markers_vs_robot_error.png",
        ],
    }
    summary.update(joint_summary)
    if summary["has_29dof_joint_graphs"]:
        summary["plots"].append("kimodo_29dof_reference_vs_robot_qpos.png")
        summary["plots"].append("kimodo_29dof_abs_error_timeseries.png")
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"wrote plots to: {out_dir}")
    print(
        "yellow_marker_vs_robot: "
        f"mean_xy={summary['mean_xy_error_m'] * scale:.3f} {unit}, "
        f"final_xy={summary['final_xy_error_m'] * scale:.3f} {unit}, "
        f"max_xy={summary['max_xy_error_m'] * scale:.3f} {unit}"
    )
    if summary["has_29dof_joint_graphs"]:
        print(
            "29dof_joint_error: "
            f"mean={summary['joint_mean_abs_error_deg']:.3f} deg, "
            f"p95={summary['joint_p95_abs_error_deg']:.3f} deg, "
            f"max={summary['joint_max_abs_error_deg']:.3f} deg"
        )


if __name__ == "__main__":
    main()
