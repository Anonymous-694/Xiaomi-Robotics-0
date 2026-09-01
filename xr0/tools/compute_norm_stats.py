# Copyright (C) 2026 Xiaomi Corporation.
"""Compute per-timestep ``(action_length, 32)`` action mean/std over converted XR-0 JSON episodes,
for the GenieSim G2 EEF route. Emits YAML-ready blocks to paste into ``configs/data/g2op_if.yaml``.

It replicates ``mibot/data/datasets/json_dataset._arm_action`` exactly (the delta the model sees):
  pos delta = rotm.T @ (target_pos - pos)
  aa  delta = rotm2aa(rotm.T @ target_rotm)
  gripper/joint delta = target - current

Only the active EEF/gripper channels (see ``io.EEF_ACTIVE_PARTS``) get real statistics. Every masked
or reserved channel (joints 7-12/21-26, dims 13/27-31) is forced to ``mean=0, std=1`` so normalization
is a no-op and never divides by zero (design §5). Stats are accumulated over valid (non-padded) steps
only, so later timesteps are estimated from the episodes long enough to reach them.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mibot.utils.io import ACTION_DIM, ACTION_PARTS, EEF_ACTIVE_PARTS, WAIST_SLICE, rotm2aa_batch  # noqa: E402

ACTIVE_SLICES = [slc for name, slc in ACTION_PARTS if name in set(EEF_ACTIVE_PARTS)]


def _arm_delta(traj, arm, frame, steps):
    prop, act = traj["proprios"], traj["actions"]
    rotm = np.asarray(prop[f"{arm}_ee_rotm"][frame], dtype=np.float32).reshape(3, 3)
    pos = np.asarray(prop[f"{arm}_ee_pos"][frame], dtype=np.float32)
    target_pos = np.asarray(act[f"{arm}_ee_pos"][frame : frame + steps], dtype=np.float32)
    target_rotm = np.asarray(act[f"{arm}_ee_rotm"][frame : frame + steps], dtype=np.float32).reshape(-1, 3, 3)
    grip_c = np.asarray(prop[f"{arm}_gripper_pos"][frame], dtype=np.float32)
    grip_t = np.asarray(act[f"{arm}_gripper_pos"][frame : frame + steps], dtype=np.float32)
    return (
        (rotm.T @ (target_pos - pos).T).T,          # (steps,3) ee_pos delta
        rotm2aa_batch(rotm.T @ target_rotm),         # (steps,3) ee_aa delta
        grip_t - grip_c,                             # (steps,1) gripper delta
    )


def episode_chunks(traj, action_length, waist=False):
    """Yield (chunk (steps,32), steps) for every valid start frame, deltas only in active dims."""
    n = int(traj["num_frames"])
    count = n - action_length + 1
    if count <= 0:
        return
    prop, act = traj["proprios"], traj["actions"]
    has_waist = waist and prop.get("waist_joint") is not None and act.get("waist_joint") is not None
    for frame in range(count):
        steps = action_length  # full chunks only (count guarantees frame+action_length <= n)
        chunk = np.zeros((steps, ACTION_DIM), dtype=np.float32)
        for arm, base in (("left", 0), ("right", 14)):
            dpos, daa, dgrip = _arm_delta(traj, arm, frame, steps)
            chunk[:, base : base + 3] = dpos
            chunk[:, base + 3 : base + 6] = daa
            chunk[:, base + 6 : base + 7] = dgrip
        if has_waist:  # waist joint5 delta (target - current), same convention as json_dataset
            cur = np.asarray(prop["waist_joint"][frame], dtype=np.float32)
            tgt = np.asarray(act["waist_joint"][frame : frame + steps], dtype=np.float32)
            chunk[:, WAIST_SLICE] = tgt - cur
        yield chunk, steps


def iter_json_trajs(paths):
    """Yield (label, traj) from XR-0 JSON dirs/files (the converted training format)."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True)))
        elif p.endswith(".json"):
            files.append(p)
    if not files:
        raise SystemExit("no JSON files found")
    print(f"found {len(files)} episodes", file=sys.stderr)
    for path in files:
        with open(path) as f:
            yield path, json.load(f)


def iter_lerobot_trajs(paths):
    """Yield (label, traj) straight from lerobot v2.1 dataset dirs — no JSON written to disk.

    Reuses ``lerobot_to_xr0.build_arrays`` (indices resolved by name from field_descriptions) so the
    proprios/actions are byte-for-byte what conversion would emit; ``episode_chunks`` then produces the
    exact deltas the model trains on. Norm is computed over ALL episodes present (matching the converted
    JSON set); trajectory_type only reweights loss, it does not change the action distribution we
    normalize against.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415 - only needed in this mode

    from lerobot_to_xr0 import build_arrays, resolve_data_path  # noqa: PLC0415

    for p in paths:
        dataset_dir = Path(p)
        if not (dataset_dir / "meta" / "info.json").exists():
            print(f"[skip] {dataset_dir}: not a lerobot dataset (no meta/info.json)", file=sys.stderr)
            continue
        info = json.load(open(dataset_dir / "meta" / "info.json"))
        total = int(info.get("total_episodes", 0))
        print(f"[{dataset_dir.name}] {total} episodes", file=sys.stderr)
        for ep in range(total):
            data_path = resolve_data_path(dataset_dir, info, ep)
            if not data_path.exists():
                continue
            table = pq.read_table(data_path, columns=["action", "observation.state"])
            proprios, actions, n = build_arrays(table, info)
            yield f"{dataset_dir.name}:ep{ep}", {"num_frames": n, "proprios": proprios, "actions": actions}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="XR-0 JSON dirs/files (default), or lerobot dataset dirs with --lerobot")
    parser.add_argument("--lerobot", action="store_true", help="Read lerobot dataset dirs directly (no JSON on disk).")
    parser.add_argument("--waist", action="store_true", help="Also compute stats for the waist joint dim (waist route).")
    parser.add_argument("--action-length", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1, help="Sub-sample start frames by this stride (speed).")
    parser.add_argument("--out", default=None, help="Optional path to write the mean/std YAML block.")
    args = parser.parse_args()

    L = args.action_length
    total = np.zeros((L, ACTION_DIM), dtype=np.float64)
    total_sq = np.zeros((L, ACTION_DIM), dtype=np.float64)
    count = np.zeros((L, 1), dtype=np.float64)

    trajs = iter_lerobot_trajs(args.paths) if args.lerobot else iter_json_trajs(args.paths)
    for i, (label, traj) in enumerate(trajs):
        for j, (chunk, steps) in enumerate(episode_chunks(traj, L, waist=args.waist)):
            if j % args.stride:
                continue
            total[:steps] += chunk[:steps]
            total_sq[:steps] += chunk[:steps] ** 2
            count[:steps] += 1
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1} episodes (last: {label})", file=sys.stderr)

    if not count.any():
        raise SystemExit("no valid chunks (episodes shorter than action_length?)")
    count = np.maximum(count, 1.0)
    mean = total / count
    var = np.maximum(total_sq / count - mean ** 2, 0.0)
    std = np.sqrt(var)

    # Force every non-active channel to mean=0, std=1 (masked / reserved dims).
    active = np.zeros(ACTION_DIM, dtype=bool)
    for slc in ACTIVE_SLICES:
        active[slc] = True
    if args.waist:
        active[WAIST_SLICE] = True
    mean[:, ~active] = 0.0
    std[:, ~active] = 1.0
    # Guard active dims against a degenerate zero-variance timestep.
    std[:, active] = np.maximum(std[:, active], 1e-6)

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    def fmt(block):
        lines = []
        for row in block:
            lines.append("        - [" + ", ".join(repr(float(x)) for x in row) + "]")
        return "\n".join(lines)

    out = f"      mean:\n{fmt(mean)}\n      std:\n{fmt(std)}\n"
    active_names = [n for n, _ in ACTION_PARTS if n in set(EEF_ACTIVE_PARTS)]
    print(f"# active dims: {active_names}", file=sys.stderr)
    print(f"# valid-step counts per timestep: min={int(count.min())} max={int(count.max())}", file=sys.stderr)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()
