# XR-0 → GenieSim G2 SFT: Usage Guide

End-to-end recipe for fine-tuning XR-0 on the [GenieSim](https://github.com/AgibotTech/genie_sim)
G2 omnipicker (task family `g2op_gs_if`) and serving it back into the simulator.

Pipeline:

```
lerobot v2.1 datasets ──(1) convert──▶ XR-0 JSON + videos
                                          │
                                          ├─(2) compute norm stats ─┐
                                          │                         ▼
                                          │              (3) edit configs/data/g2op_if.yaml
                                          │                         │
                                          └─────────────────────────┤
                                                                     ▼
                                              (4) train ─▶ checkpoint dir (config.py + *.ckpt)
                                                                     │
                                                                     ▼
                              (5) serve ─▶ xr0-server :10086 (GPU) + xr0-adapter :8007 (corobot)
                                                                     │
                                                                     ▼
                                          (6) run GenieSim task ──▶ evaluation
```

The `xr0` conda/venv from the top-level [`README.md`](../README.md) covers steps 1–5. Step 6 runs
inside the GenieSim docker container. Replace every `/path/to/...` below with your own locations.

Design rationale (frames, masking, the two-tier server) lives in
[`DESIGN_GENIESIM_SFT.md`](DESIGN_GENIESIM_SFT.md); this file is the how-to.

---

## 0. What the data looks like

The source is one or more [lerobot **v2.1**](https://github.com/huggingface/lerobot) dataset dirs
(30 fps, dual-arm), each laid out as:

```
<dataset>/
  meta/info.json           # feature schema + field_descriptions (the index map) + path templates
  meta/annotations.json    # per-episode instruction text + approval status
  data/chunk-000/episode_000000.parquet ...     # observation.state (183-D), action (40-D)
  videos/chunk-000/observation.images.top_head/episode_000000.mp4   # + hand_left / hand_right
```

The converter reads channel indices **by name** from `meta/info.json →
features.<feature>.field_descriptions` — nothing is hardcoded, so a schema with the same channel
names but different offsets still works. Channels consumed: `end/position` + `end/orientation` (EEF
pose, both arms), `{left,right}_effector/position` (grippers), and `waist/position` (joint5 = last
of the 5-D block, optional). See the converter's module docstring for the full map.

---

## 1. Convert lerobot v2.1 → XR-0 JSON

```bash
python tools/lerobot_to_xr0.py \
    /path/to/geniesim_data/instruction_and_robust/v21/pick_block_color_500 \
    /path/to/geniesim_data/instruction_and_robust/v21/pick_block_shape_500 \
    ... \
    --out /path/to/xr0_data/g2op_if
```

- Each `<dataset>` becomes `/path/to/xr0_data/g2op_if/<dataset>/{json,videos}`.
- `--video-mode reference` (default) writes the **absolute path** of each source mp4 into the JSON —
  fastest, but the videos must stay reachable at train time. Use `symlink` or `copy` if you need the
  videos beside the JSON (e.g. to move the dataset to another host).
- `--limit N` converts only N episodes per dataset — handy for a smoke test.

Instructions come from `meta/annotations.json` (the joined `action_steps[*].action_text`); episodes
with `episode_status == "approved"` are marked `success`, the rest `invalid` (masked in the loss).

---

## 2. Compute normalization statistics

XR-0 normalizes actions with a per-timestep `(action_length, 32)` mean/std that lives **inside the
data config** and travels with the trained checkpoint. Regenerate it for your data — the placeholder
values shipped in `g2op_if.yaml` are identity (no-op) and must not be used for a real run.

Read straight from the lerobot dirs (no JSON needed on disk):

```bash
python tools/compute_norm_stats.py --lerobot \
    /path/to/geniesim_data/instruction_and_robust/v21/pick_block_color_500 \
    /path/to/geniesim_data/instruction_and_robust/v21/pick_block_shape_500 \
    ... \
    --out /tmp/g2_stats.yaml
```

Or, if you already converted, point it at the JSON dirs (drop `--lerobot`):

```bash
python tools/compute_norm_stats.py /path/to/xr0_data/g2op_if/*/json --out /tmp/g2_stats.yaml
```

Then paste the `mean:` / `std:` blocks from `/tmp/g2_stats.yaml` into
`configs/data/g2op_if.yaml`, replacing the placeholder blocks. Only the active EEF/gripper dims
(0–6, 14–20) carry real statistics; every masked/reserved dim is forced to `mean=0, std=1`.

> **Consistency rule:** train and inference must use the *same* mean/std. `deploy.py` reads them
> from the checkpoint's `config.py`, which is written from this data config — so as long as you
> compute stats on the same data you train on, they stay aligned. Recompute whenever the dataset mix
> changes.

Add `--waist` only if you intend to supervise the torso-twist joint (see §3).

---

## 3. Edit the data config

`configs/data/g2op_if.yaml` is the v2.1 example. Set:

- **`train_path`** — the list of converter-output `json` dirs from step 1.
- **`mean` / `std`** — the blocks from step 2.
- **`batch_size`** / **`action_length`** (default 30) as needed.
- **`active_parts`** — which action channels are supervised. The default is EEF-only:
  `left_ee_pos, left_ee_aa, left_gripper, right_ee_pos, right_ee_aa, right_gripper`. The 6-DOF joint
  channels are masked out because G2's 7-DOF arms don't fit XR-0's joint slots — GenieSim handles IK.
  To also supervise the waist joint, add `waist` here **and** recompute norm stats with `--waist`.

---

## 4. Train

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 RESOURCE_GPU=1 \
bash scripts/train.sh \
    data=g2op_if \
    model=XR0 \
    trainer.project="xr0_geniesim" \
    trainer.exp_name="g2op_if" \
    trainer.default_root_dir="train/" \
    model.params.model.pretrained="pretrained_ckpt/xr0_pretrained.pt"
```

Multi-GPU (single node, 8 cards):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RESOURCE_GPU=8 \
bash scripts/train.sh data=g2op_if model=XR0 \
    trainer.project="xr0_geniesim" trainer.exp_name="g2op_if" \
    trainer.default_root_dir="train/" \
    model.params.model.pretrained="pretrained_ckpt/xr0_pretrained.pt"
```

Checkpoints land under `{trainer.default_root_dir}/{trainer.project}/{trainer.exp_name}/` — with the
overrides above that is `train/xr0_geniesim/g2op_if/`. That directory (holding `config.py` and the
`*.ckpt` dirs) is the `<model_dir>` the server loads in step 5. GenieSim consumes a full action chunk
before re-querying, so keep the synchronous `XR0.yaml` default (`async_train: false`).

---

## 5. Serve the model

Two-tier stack, launched in a tmux session by `serve.sh`:

| window        | port    | device | role                                                      |
|---------------|---------|--------|-----------------------------------------------------------|
| `xr0-server`  | `10086` | GPU    | `mibot/server/deploy.py` — the DiT rectified-flow policy  |
| `xr0-adapter` | `8007`  | CPU    | `xr0_corobot_adapter.py` — corobot msgpack/WS ↔ pickle/TCP |

Trained checkpoints (skip step 4 and download directly):
- ModelScope: https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset/tree/master/checkpoints/xiaomi-robotics-0
- Hugging Face: https://huggingface.co/datasets/agibot-world/GenieSim3.0-Dataset/tree/main/checkpoints/xiaomi-robotics-0

```bash
bash geniesim_deploy/serve.sh <model_dir> [gpu_id] [model_port] [adapter_port] [ckpt]
# e.g.
bash geniesim_deploy/serve.sh train/xr0_geniesim/g2op_if 0 10086 8007
# pick a specific step:
bash geniesim_deploy/serve.sh train/xr0_geniesim/g2op_if 0 10086 8007 'epoch=0-step=20000.ckpt'
```

- `<model_dir>` is the training output dir from step 4. Its `config.py` carries `active_parts` +
  `mean`/`std`, so the server masks and normalizes exactly as trained.
- The adapter waits for the model port to open, then serves corobot on `:8007`. Point GenieSim at
  `<host>:8007`.
- **Waist:** if the model was trained with `waist` in `active_parts`, launch with `XR0_WAIST=1` so
  the adapter drives the torso-twist joint; otherwise leave it unset.
- On sm_89 GPUs (e.g. 4090) that lack FlashAttention-2, set `XR0_ATTN_IMPL=sdpa`.

Attach with `tmux attach -t xr0_geniesim`; stop with `tmux kill-session -t xr0_geniesim`. To run two
models side by side, give each its own `XR0_SESSION`, `model_port`, and `adapter_port`.

---

## 6. Run a GenieSim task

From the host, `run_task.sh` drives the simulator inside the `genie_sim_benchmark` docker container
and points it at the adapter:

```bash
bash geniesim_deploy/run_task.sh <sub_task> [infer_host=127.0.0.1:8007] [num_episode=1] [seed=1]
# e.g.
bash geniesim_deploy/run_task.sh pick_block_color
```

Available `<sub_task>` values (each maps to `source/geniesim/config/arxone_if_<sub_task>.yaml`):

```
pick_billiards_color   pick_block_color    pick_block_number   pick_block_shape
pick_block_size        pick_common_sense   pick_follow_logic_or pick_object_type
pick_specific_object   straighten_object
```

The task runs with `robot_cfg=G2_omnipicker` and `--model_arc corobot`. If the server runs on
another machine, pass its address as the second argument (e.g. `10.0.0.5:8007`).

---

## Troubleshooting

- **Robot flails / ignores the instruction** — almost always an OOD mismatch: the converter layout,
  the training mask/norm, and the adapter's recover/frame must all agree. Confirm `config.py` in the
  served `<model_dir>` matches the data config you trained with, and that `XR0_WAIST` matches
  `active_parts`.
- **`validate_stats` shape error at train start** — `mean`/`std` in the config must be exactly
  `(action_length, 32)`. Recompute with the matching `--action-length`.
- **Adapter can't reach the model** — the server takes a while to load the VLM; the adapter polls
  `:10086` and prints `waiting for xr0 server...` until it's up.
- **Wrong EEF frame at inference** — the adapter must feed the EEF pose in the same `end/*` frame the
  converter used, with `xyzw` quaternions and the `xyz` Euler sequence. See `DESIGN_GENIESIM_SFT.md`
  §2.3 and §7.2.
