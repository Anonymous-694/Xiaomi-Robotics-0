# Copyright (C) 2026 Xiaomi Corporation.
"""Convert GenieSim G2 (task family ``g2op_gs_if``) lerobot **v2.1** datasets into the XR-0 training
format (one JSON annotation per episode + multi-view videos).

The emitted JSON is version-agnostic, so the downstream dataloader
(``mibot/data/datasets/json_dataset.py``) and the norm tool (``tools/compute_norm_stats.py``) are
unchanged. See ``xr0/geniesim_deploy/DESIGN_GENIESIM_SFT.md`` §4 and ``xr0/docs/data_format.md``.

Index maps are read **by name** from ``meta/info.json`` → ``features.<feature>.field_descriptions``
(no hardcoded slices): each named channel carries an explicit ``indices`` list, so the layout is
taken straight from the data. The channels this converter consumes:

  observation.state (183-D)         action (40-D)
    state/left_effector/position      action/left_effector/position     gripper L (1)
    state/right_effector/position     action/right_effector/position    gripper R (1)
    state/end/position       (6)      action/end/position       (6)      EEF xyz  [Lxyz, Rxyz]
    state/end/orientation    (8)      action/end/orientation    (8)      EEF quat [Lxyzw, Rxyzw]
    state/waist/position     (5)      action/waist/position     (5)      waist; joint5 = last idx

Source layout on disk (paths come from ``info.json``'s templates):
  data/chunk-{c:03d}/episode_{ep:06d}.parquet                     columns action, observation.state
  videos/chunk-{c:03d}/{video_key}/episode_{ep:06d}.mp4
  meta/annotations.json    per-episode action_steps[].action_text (instruction) + episode_status

Frame convention: the training **action** EEF pose and the proprio **state** EEF pose are both taken
from the ``end/position`` / ``end/orientation`` channels — verified to be the **same frame** (state
``end/*`` equals ``action end/*`` frame-for-frame, mean-abs-diff 0.0; the separate ``end/arm_*``
channels are a *different* frame and are not used). XR-0 actions are frame-invariant deltas and the
state EEF is the delta reference, so state and action sharing one frame is what matters. The state
EEF is stored as ``[x, y, z, roll, pitch, yaw]`` in a dedicated ``{side}_state_eef`` proprio field;
the ``{side}_arm_joint`` fields stay zeros so the masked joint-action delta remains clean. The
quaternion→Euler conversion uses the ``xyz`` (extrinsic, radians) sequence and MUST match
``geniesim_deploy/xr0_corobot_adapter.py`` — the inference adapter must feed the EEF in this same
``end/*`` frame.

Quaternions are **xyzw** (scalar-last), empirically verified (unit-norm). Joint channels are written
as zeros: they are masked during training (design decision 2; G2's 7-DOF arms do not fit the 6-DOF
slots). Waist joint5 (torso-twist DOF, ``idx05_body_joint5`` = last of the 5-D waist block) is emitted
when present; only the manip family supervises it (config ``active_parts += waist``), so the
instruction-following family — which does not twist the torso — leaves it masked downstream.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

# --- source channel names (resolved to indices via meta/info.json field_descriptions) ----------
S_GRIP_L = "state/left_effector/position"
S_GRIP_R = "state/right_effector/position"
S_EEF_POS = "state/end/position"       # 6: [Lx,Ly,Lz, Rx,Ry,Rz]
S_EEF_ORI = "state/end/orientation"    # 8: [Lqx,Lqy,Lqz,Lqw, Rqx,Rqy,Rqz,Rqw] (xyzw)
S_WAIST = "state/waist/position"       # 5: waist joints; joint5 (torso-twist) = last index

A_GRIP_L = "action/left_effector/position"
A_GRIP_R = "action/right_effector/position"
A_EEF_POS = "action/end/position"      # 6
A_EEF_ORI = "action/end/orientation"   # 8 (xyzw)
A_WAIST = "action/waist/position"      # 5; joint5 = last index


@dataclass(frozen=True)
class Fields:
    """Resolved source indices (read by name from ``field_descriptions``), not hardcoded.

    ``*_grip`` = [L_idx, R_idx]; ``*_eef_pos`` = 6 indices [Lxyz, Rxyz]; ``*_eef_ori`` = 8 indices
    [Lxyzw, Rxyzw]; ``*_waist`` = the single waist joint5 index (last of the waist block) or ``None``
    when the dataset has no waist channel (then the waist output is omitted → masked downstream).
    """

    s_grip: list[int]
    s_eef_pos: list[int]
    s_eef_ori: list[int]
    s_waist: int | None
    a_grip: list[int]
    a_eef_pos: list[int]
    a_eef_ori: list[int]
    a_waist: int | None


def _field_descriptions(info: dict, feature: str) -> dict:
    fd = info.get("features", {}).get(feature, {}).get("field_descriptions")
    if not fd:
        raise ValueError(
            f"meta/info.json {feature!r} has no 'field_descriptions'. This converter targets lerobot "
            f"v2.1 GenieSim exports, which carry per-channel indices there."
        )
    return fd


def _indices(fd: dict, name: str) -> list[int]:
    entry = fd.get(name)
    idx = (entry or {}).get("indices")
    if not idx:
        raise ValueError(f"field_descriptions missing/empty channel {name!r}")
    return list(idx)


def resolve_fields(info: dict) -> Fields:
    """Read the state/action index maps by channel name from ``info.json`` field_descriptions."""
    sfd = _field_descriptions(info, "observation.state")
    afd = _field_descriptions(info, "action")

    def waist(fd: dict, name: str) -> int | None:
        entry = fd.get(name)
        idx = (entry or {}).get("indices")
        return int(idx[-1]) if idx else None  # joint5 = last of the waist block

    return Fields(
        s_grip=_indices(sfd, S_GRIP_L) + _indices(sfd, S_GRIP_R),
        s_eef_pos=_indices(sfd, S_EEF_POS),
        s_eef_ori=_indices(sfd, S_EEF_ORI),
        s_waist=waist(sfd, S_WAIST),
        a_grip=_indices(afd, A_GRIP_L) + _indices(afd, A_GRIP_R),
        a_eef_pos=_indices(afd, A_EEF_POS),
        a_eef_ori=_indices(afd, A_EEF_ORI),
        a_waist=waist(afd, A_WAIST),
    )


# Euler sequence for the state EEF (roll,pitch,yaw). MUST stay identical to the inference adapter
# (geniesim_deploy/xr0_corobot_adapter.py) or the state fed at train/infer time will disagree.
EULER_SEQ = "xyz"  # extrinsic, radians

# lerobot camera key -> XR-0 view name
CAM_MAP = {
    "observation.images.top_head": "ego",
    "observation.images.hand_left": "wrist_left",
    "observation.images.hand_right": "wrist_right",
}

# XR-0 three-view human prompt (must match json_dataset._prompt / the corobot adapter). The
# training pipeline appends " /no_cot" to the human turn and sets the gpt turn to "<cot></cot>".
PROMPT_TEMPLATE = (
    "The following observations are captured from multiple views.\n"
    "# Ego View\n<image>\n"
    "# Left-Wrist View\n<image>\n"
    "# Right-Wrist View\n<image>\n"
    "Generate robot actions for the task:\n{instruction}"
)


def quat_xyzw_to_rotm_flat(quats: np.ndarray) -> list[list[float]]:
    """(N,4) xyzw quaternions -> (N,9) row-major flattened rotation matrices."""
    rotms = Rotation.from_quat(np.asarray(quats, dtype=np.float64)).as_matrix()  # (N,3,3)
    return rotms.reshape(len(rotms), 9).astype(np.float32).tolist()


def quat_xyzw_to_rpy(quats: np.ndarray) -> np.ndarray:
    """(N,4) xyzw quaternions -> (N,3) [roll,pitch,yaw] Euler angles (EULER_SEQ, radians)."""
    return Rotation.from_quat(np.asarray(quats, dtype=np.float64)).as_euler(EULER_SEQ).astype(np.float32)


def _col_to_2d(table, name: str) -> np.ndarray:
    """Read a parquet list-column into a dense (N, D) float array."""
    return np.asarray(table.column(name).to_pylist(), dtype=np.float32)


def load_episode_meta(dataset_dir: Path) -> dict[int, dict]:
    """episode_index -> {instruction, status} from meta/annotations.json.

    The natural-language instruction lives in ``action_steps[*].action_text`` (the top-level
    ``instruction`` field is empty); we join the per-step texts. ``episode_status == "approved"``
    marks a usable trajectory.
    """
    ann = json.load(open(dataset_dir / "meta" / "annotations.json"))
    meta: dict[int, dict] = {}
    for key, rec in ann.items():
        ep = int(rec.get("episode_index", key))
        steps = rec.get("action_steps") or []
        texts = [str(s.get("action_text", "")).strip() for s in steps]
        instruction = " ".join(t for t in texts if t)
        meta[ep] = {"instruction": instruction, "status": rec.get("episode_status")}
    return meta


def build_arrays(table, info: dict) -> tuple[dict, dict, int]:
    """Extract XR-0 proprios/actions dicts from one episode's parquet table.

    Indices are resolved by name from ``info``'s field_descriptions (see ``resolve_fields``); no
    hardcoded slices. Shared by the converter and ``tools/compute_norm_stats.py``.
    """
    f = resolve_fields(info)
    action = _col_to_2d(table, "action")            # (N, action_dim)
    state = _col_to_2d(table, "observation.state")  # (N, state_dim)
    n = len(action)
    zeros_joint = np.zeros((n, 6), dtype=np.float32).tolist()

    # State EEF (end/* frame, same as the action delta reference) as [x,y,z,roll,pitch,yaw], fed
    # into the model's proprio state slot.
    s_pos = state[:, f.s_eef_pos]  # (N,6) [Lx,Ly,Lz, Rx,Ry,Rz]
    s_ori = state[:, f.s_eef_ori]  # (N,8) [Lxyzw, Rxyzw]
    left_state_eef = np.concatenate([s_pos[:, 0:3], quat_xyzw_to_rpy(s_ori[:, 0:4])], axis=1).tolist()
    right_state_eef = np.concatenate([s_pos[:, 3:6], quat_xyzw_to_rpy(s_ori[:, 4:8])], axis=1).tolist()
    s_grip = state[:, f.s_grip]  # (N,2) [L, R]

    proprios = {
        "left_ee_pos": s_pos[:, 0:3].tolist(),
        "left_ee_rotm": quat_xyzw_to_rotm_flat(s_ori[:, 0:4]),
        "left_arm_joint": zeros_joint,
        "left_state_eef": left_state_eef,
        "left_gripper_pos": s_grip[:, 0:1].tolist(),
        "right_ee_pos": s_pos[:, 3:6].tolist(),
        "right_ee_rotm": quat_xyzw_to_rotm_flat(s_ori[:, 4:8]),
        "right_arm_joint": zeros_joint,
        "right_state_eef": right_state_eef,
        "right_gripper_pos": s_grip[:, 1:2].tolist(),
    }
    a_pos = action[:, f.a_eef_pos]  # (N,6)
    a_ori = action[:, f.a_eef_ori]  # (N,8)
    a_grip = action[:, f.a_grip]    # (N,2)
    actions = {
        "left_ee_pos": a_pos[:, 0:3].tolist(),
        "left_ee_rotm": quat_xyzw_to_rotm_flat(a_ori[:, 0:4]),
        "left_arm_joint": zeros_joint,
        "left_gripper_pos": a_grip[:, 0:1].tolist(),
        "right_ee_pos": a_pos[:, 3:6].tolist(),
        "right_ee_rotm": quat_xyzw_to_rotm_flat(a_ori[:, 4:8]),
        "right_arm_joint": zeros_joint,
        "right_gripper_pos": a_grip[:, 1:2].tolist(),
    }
    # Waist joint5 is emitted only when both proprio and action expose it. The dataloader treats a
    # missing waist_joint as "unsupervised" (dim 13 stays zero + masked), so the instruction-following
    # family — which does not use the waist — is unaffected.
    if f.s_waist is not None and f.a_waist is not None:
        proprios["waist_joint"] = state[:, f.s_waist : f.s_waist + 1].tolist()   # (N,1) current
        actions["waist_joint"] = action[:, f.a_waist : f.a_waist + 1].tolist()   # (N,1) target
    return proprios, actions, n


def _chunk(ep: int, info: dict) -> int:
    return ep // int(info.get("chunks_size", 1000) or 1000)


def resolve_data_path(dataset_dir: Path, info: dict, ep: int) -> Path:
    """Locate one episode's parquet using info.json's ``data_path`` template."""
    rel = info["data_path"].format(episode_chunk=_chunk(ep, info), episode_index=ep)
    return dataset_dir / rel


def _video_src(dataset_dir: Path, info: dict, cam_key: str, ep: int) -> Path:
    rel = info["video_path"].format(episode_chunk=_chunk(ep, info), video_key=cam_key, episode_index=ep)
    return dataset_dir / rel


def resolve_video_path(dataset_dir: Path, out_dir: Path, info: dict, cam_key: str, ep: int, mode: str) -> str:
    """Return the video path to store in the JSON, materializing it per `mode`."""
    src = _video_src(dataset_dir, info, cam_key, ep)
    if not src.exists():
        raise FileNotFoundError(f"missing video: {src}")
    if mode == "reference":
        return str(src.resolve())
    view = CAM_MAP[cam_key]
    dst_dir = out_dir / "videos"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"episode_{ep:04d}_{view}.mp4"
    if mode == "symlink":
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
    elif mode == "copy":
        import shutil

        shutil.copy2(src, dst)
    else:
        raise ValueError(f"unknown --video-mode {mode}")
    return str(dst.resolve())


def convert_episode(dataset_dir: Path, out_dir: Path, info: dict, ep: int, meta: dict, video_mode: str, fps: int) -> dict | None:
    data_path = resolve_data_path(dataset_dir, info, ep)
    if not data_path.exists():
        return None
    table = pq.read_table(data_path, columns=["action", "observation.state"])
    proprios, actions, n = build_arrays(table, info)

    ep_meta = meta.get(ep, {})
    instruction = (ep_meta.get("instruction") or "").strip()
    if not instruction:
        print(f"  [warn] episode {ep}: empty instruction, skipping", file=sys.stderr)
        return None
    # AgiBot review status: "approved" == success. Anything else is treated as invalid (masked).
    traj_type = "success" if ep_meta.get("status") == "approved" else "invalid"

    observations = {}
    for cam_key, view in CAM_MAP.items():
        path = resolve_video_path(dataset_dir, out_dir, info, cam_key, ep, video_mode)
        observations[view] = [{"path": path, "start": 0, "end": n, "fps": fps, "crop_bbox": None}]

    human = PROMPT_TEMPLATE.format(instruction=instruction)
    return {
        "trajectory_type": traj_type,
        "time": f"{dataset_dir.name}_ep{ep:04d}",
        "num_frames": n,
        "instruction": {
            "general": [
                {
                    "images": ["observations.ego", "observations.wrist_left", "observations.wrist_right"],
                    "conversations": [
                        {"from": "human", "value": human},
                        {"from": "gpt", "value": "<bot></bot>"},
                    ],
                }
            ]
        },
        "observations": observations,
        "proprios": proprios,
        "actions": actions,
    }


def convert_dataset(dataset_dir: Path, out_root: Path, video_mode: str, limit: int | None) -> int:
    info = json.load(open(dataset_dir / "meta" / "info.json"))
    resolve_fields(info)  # validate field_descriptions up front (raises with a clear message)
    fps = int(info.get("fps", 30))
    total = int(info.get("total_episodes", 0))
    meta = load_episode_meta(dataset_dir)

    out_dir = out_root / dataset_dir.name
    json_dir = out_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    episodes = range(total if limit is None else min(limit, total))
    written = 0
    for ep in episodes:
        try:
            traj = convert_episode(dataset_dir, out_dir, info, ep, meta, video_mode, fps)
        except Exception as error:  # noqa: BLE001 - keep converting the rest
            print(f"  [err] episode {ep}: {error}", file=sys.stderr)
            continue
        if traj is None:
            continue
        with open(json_dir / f"episode_{ep:04d}.json", "w") as file:
            json.dump(traj, file)
        written += 1
        if written % 50 == 0:
            print(f"  {dataset_dir.name}: {written} episodes written")
    print(f"[done] {dataset_dir.name}: {written}/{total} episodes -> {json_dir}")
    return written


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+", help="One or more lerobot v2.1 dataset dirs (e.g. .../pick_block_color_500)")
    parser.add_argument("--out", required=True, help="Output root; each dataset -> <out>/<dataset_name>/{json,videos}")
    parser.add_argument(
        "--video-mode",
        choices=["reference", "symlink", "copy"],
        default="reference",
        help="reference=store absolute path to source mp4 (default; needs data reachable at train time); "
        "symlink=symlink into <out>/videos; copy=copy the mp4.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Convert at most N episodes per dataset (debug).")
    return parser.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out).resolve()
    grand = 0
    for d in args.datasets:
        dataset_dir = Path(d).resolve()
        if not (dataset_dir / "meta" / "info.json").exists():
            print(f"[skip] {dataset_dir}: not a lerobot dataset (no meta/info.json)", file=sys.stderr)
            continue
        print(f"=== converting {dataset_dir.name} (video-mode={args.video_mode}) ===")
        grand += convert_dataset(dataset_dir, out_root, args.video_mode, args.limit)
    print(f"\nTOTAL episodes written: {grand}")
    print(f"Point configs/data/g2op_if.yaml train_path at: {out_root}/<dataset>/json")


if __name__ == "__main__":
    main()
