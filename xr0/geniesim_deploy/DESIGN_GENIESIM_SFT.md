# Xiaomi-Robotics-0 (XR-0) Adaptation to GenieSim G2 — SFT Training & Inference Design Document

> Status: **Design draft v2 (decisions confirmed)** · Date 2026-07-22
> Reference: LingBot-VLA v2's GenieSim adaptation (`geniesim_deploy/GENIESIM_SFT_ADAPTER_GUIDE.html`)
> Goal: post-train (SFT) the pretrained XR-0 (Qwen3-VL-4B + DiT rectified-flow) onto GenieSim's **G2 (omnipicker, dual 7-DOF arms)** embodiment, and close the loop for inference in the GenieSim benchmark.

> **Four confirmed decisions (v2)**
> 1. **Control space = EEF_ABS + GenieSim's built-in IK** (confirmed).
> 2. **All joint dims of state are masked (set to 0)**: state keeps only the dual grippers + vision, no joint proprioception is fed in.
> 3. **SFT task = dual-arm, multiple lerobot datasets** (not GenieSim webm recordings).
> 4. **The EEF pose used for training comes from lerobot's `end/position`/`end/orientation` channels** (verified on v2.1: state's `end/*` and action's `end/*` are frame-identical per-frame, mean-abs-diff=0.0 — i.e. state (the delta reference) and action share the same frame; there is a separate `end/arm_*` channel group that is a **different frame** and is not used). Since XR-0's delta is independent of the base frame (see §2.3), it's enough that state and action share the same frame — the training frame and the inference base_link frame don't need to match, so no coordinate conversion is needed. (The earlier design called this frame `arm_base_link`; under v2.1 we go by the `end/*` channel name — which named base frame it actually is doesn't affect correctness.)

---

## 0. Background & Key Constraints

XR-0's architecture differs greatly from LingBot/openpi — LingBot's approach **cannot simply be copied**; a redesign is needed. Key differences:

| Dimension | LingBot-VLA v2 | **XR-0 (this repo)** |
|------|----------------|-------------------|
| Data format | lerobot parquet | **One JSON + multi-view videos per episode** (`docs/data_format.md`) |
| Normalization | per-feature openpi json (q01/q99) | **per-timestep `(30,32)` mean/std, hardcoded in the data yaml** |
| Action space | driven by robot_config, configurable slicing | **hardcoded 32-D dual-arm layout** (`mibot/utils/io.py:ACTION_PARTS`) |
| Inference server protocol | flat WebSocket | **pickle-over-raw-TCP** (`mibot/server/runtime/server.py`) |
| Model output | chunk×N | **`(30, 32)` action chunk, rectified flow 5-step Euler integration** |
| Config system | in-house | **Hydra + OmegaConf** |

GenieSim-side protocol (`benchmark/policy/corobotpolicy.py`, confirmed):

- Transport: **JSON-RPC over msgpack over WebSocket**
- Request `params`: `images{head,hand_left,hand_right}` (JPEG, RGB→BGR encoded), `states{arm_joint_states=[L,R], gripper_states, waist_joint_states, head_joint_states, end_pose}`, `prompt`, `robot_type`, `eef_type`
- Response `result`: `left_arm{kind,values}` / `right_arm{...}` / `left_effector` / `right_effector` / (optional) `waist`/`head`
- **`kind` supports `JOINT_ABS` and `EEF_ABS`**; for `EEF_ABS`, arm values = `[x,y,z,qx,qy,qz,qw]` (base_link frame) — GenieSim has a **built-in IK solver** (`ikfk_solver.eef_actions_to_joint`) that solves the EEF target into joint values.

G2_omnipicker embodiment (`utils/name_utils.py`, confirmed): **7 joints per arm**, 5 waist joints, 2 head joints, `robot_type=g2a_sim`, `label_state_crsb` + `process_gripper_action_crsb`.

---

## 1. Overall Architecture (Three-Tier)

```
┌──────────────────────────────┐   corobot: JSON-RPC / msgpack (WebSocket)
│ GenieSim (Isaac Sim, Docker) │──────────────────────────────────────────┐
│  --model_arc corobot          │  {images(JPEG), states, prompt,          │
│  --infer_host <ip>:8007       │◀── result{left_arm(EEF_ABS), right_arm,   │
└──────────────────────────────┘     left_effector, right_effector} ───────┤
                                                                            ▼
                                   ┌────────────────────────────────────────────┐
                                   │ xr0_corobot_adapter.py   :8007   (no GPU)    │
                                   │  ① JPEG BGR→RGB→PIL                          │
                                   │  ② states + end_pose → XR-0 robot_state     │
                                   │     (ee_pos/ee_rotm/arm_joint[6]/gripper)   │
                                   │  ③ prompt → same Qwen3-VL template as       │
                                   │     training + tokenize                     │
                                   │  ④ receive (30,32) → recover_action → EEF   │
                                   │     pose → rotm→quat → EEF_ABS(base_link)   │
                                   │     response                                │
                                   └────────────────────────────────────────────┘
                                                                            │ pickle / raw TCP
                                                                            ▼
                                   ┌────────────────────────────────────────────┐
                                   │ xr0 deploy server        :10086  (GPU)      │
                                   │  mibot/server/deploy.py + runtime/server.py │
                                   │  load SFT ckpt + mean/std from data yaml    │
                                   │  normalize state / denormalize action       │
                                   │  DiT rectified-flow outputs (30,32)         │
                                   └────────────────────────────────────────────┘
```

- **The adapter uses no GPU** — it only translates protocol + layout.
- The adapter takes over the responsibilities of the original `mibot/server/runtime/client.py` (Qwen3-VL tokenization, state assembly, action recovery), but swaps the upstream from a "robot controller" to "GenieSim corobot".
- Port convention: `:8007` adapter (corobot), `:10086` XR-0 model service (pickle-TCP, configurable).

---

## 2. ⭐ Core Design Decision: Use **EEF-Space Control** (not Joint Space)

This is the crux of the whole adaptation — it directly determines every layout choice that follows.

### The Problem
- G2 has **7 DOF per arm**; but in XR-0's 32-D action layout, each arm's joint slot only has **6 dims** (`left_joint` = dim 7–12, `right_joint` = dim 21–26, see `io.py:ACTION_PARTS`).
- Direct joint-space control → 7 doesn't fit into 6. Changing `ACTION_DIM`/`ACTION_PARTS` to 7-DOF would change the DiT's state/action head dimensions → **the pretrained weights could no longer be loaded**, losing the benefit of pretraining.

### Solution: use the EEF delta XR-0 natively predicts, for Cartesian control
XR-0's 32-D layout already contains an **end-effector pose delta**:
- `left_ee_pos`(0–2) + `left_ee_aa`(3–5, axis-angle) + `left_gripper`(6)
- `right_ee_pos`(14–16) + `right_ee_aa`(17–19) + `right_gripper`(20)

EEF is a **6-DOF Cartesian quantity, independent of whether the arm has 6 or 7 DOF**. Therefore:
1. **Training**: supervise EEF pos/aa + gripper (8 active channels per arm), and **zero out the joint dims (7–12, 21–26) in `action_mask`** (unsupervised).
2. **Inference**: take only the model's EEF delta output → `io.recover_action()` restores it to an absolute EEF pose → converted into an `EEF_ABS` response → **GenieSim's built-in IK solves the EEF target into 7-DOF joints**. The 7-vs-6 conflict is completely sidestepped.

This is philosophically aligned with LingBot's approach of "keeping `end.position` but masking it out", but XR-0 goes through a **genuine EEF channel** (LingBot's g2op goes through the joint channel).

> **Alternative (not recommended)**: joint space, taking 6 of G2's 7 joints. This loses one degree of freedom, requires manually picking which joint to drop, and `recover` has no IK fallback. Only fall back to this if the EEF/IK route doesn't reach acceptable accuracy on G2.

### Frame Convention (must be self-consistent) — ⭐ delta is base-frame-independent

**Key property**: XR-0's action is "a relative quantity in the current EEF's local frame" (`json_dataset._arm_action`):
- position delta = `rotm.T @ (target_pos - pos)`
- orientation delta = `rotm2aa(rotm.T @ target_rotm)`

For any rigid-body base transform `(R, t)` (`pos'=R·pos+t`, `rotm'=R·rotm`), one can show `delta' = delta` (`R.T·R=I` cancels). **In other words, the delta does not depend on whether absolute poses are expressed in arm_base_link or base_link.** Consequently:

- **Training**: the EEF poses in the data come from the `end/position`/`end/orientation` channels (state and action use the same channel group, verified to share the same frame), **used as-is, with no conversion**. The resulting delta / normalization stats are frame-independent.
- **Inference**: corobot's `states.end_pose.base_link.{left,right}_arm` is in the **base_link** frame (`corobotpolicy._build_end_pose`, verified). The adapter **feeds it directly as the current pose** into `recover_action` → yielding a target pose **in the base_link frame** → returned as `EEF_ABS`.
- On the GenieSim side, `_post_process_action` (verified) expects `EEF_ABS` arm values = `[x,y,z,qx,qy,qz,qw]` (**base_link frame**); internally `_transform_pose_to_frame` converts to arm_base_link → xyzrpy → `eef_actions_to_joint` for IK.

**Conclusion: no coordinate-frame conversion is needed inside the adapter.** The training pipeline and the inference pipeline (base_link) each close their own loop; delta invariance guarantees the model generalizes across them (as long as, within each pipeline, "the frame of state/current pose == the frame of action/response EEF_ABS"). On the inference side, "the frame of the current pose == the frame of the returned EEF_ABS" = base_link (holds naturally).

---

## 3. State / Action Layout Mapping (32-D ↔ G2)

### 3.1 How XR-0's `(30,32)` output action is used (EEF route)

| dim | XR-0 channel | G2 usage | training mask | inference |
|-----|-----------|---------|-----------|------|
| 0–2 | left_ee_pos Δ | left-arm EEF position | **1** | recover→EEF_ABS left |
| 3–5 | left_ee_aa Δ | left-arm EEF orientation | **1** | recover→quat |
| 6 | left_gripper Δ | left gripper | **1** | → left_effector |
| 7–12 | left_joint Δ | (unused) | **0** | ignored |
| 13 | reserved | — | 0 | — |
| 14–16 | right_ee_pos Δ | right-arm EEF position | **1** | recover→EEF_ABS right |
| 17–19 | right_ee_aa Δ | right-arm EEF orientation | **1** | recover→quat |
| 20 | right_gripper Δ | right gripper | **1** | → right_effector |
| 21–26 | right_joint Δ | (unused) | **0** | ignored |
| 27–31 | reserved | — | 0 | — |

### 3.2 XR-0 input state `(1,32)` — all joints masked (Decision 2)

The stock `compose_state` fills dim 6/20 (grippers) + 7–12/21–26 (6 joints each). **Decision 2: zero out all joint dims — state keeps only the dual grippers** (dim 6, dim 20) + vision.

- Implementation: `compose_state` gains an `active_state_parts` (or, for G2 specifically, fills only the gripper), joint slots stay 0.
- **Norm pitfall**: state is also normalized by the server. If a dim is constantly 0, the computed `std=0` → division-by-zero/NaN. → `compute_norm_stats` **forces `std=1` (`mean=0`) for constant/masked state dims**. Gripper dims are statisticized normally.
- Implication: proprioception is reduced to just gripper open/close, everything else relies on vision — this is a deliberate choice that sidesteps the problem of G2's 7 joints not fitting into 6 slots, and fully decouples state from joint DOF.

### 3.3 GenieSim corobot ↔ XR-0 Data-Flow Mapping

| Direction | corobot field | XR-0 side |
|------|-------------|---------|
| Image | `images.head/hand_left/hand_right` (JPEG, BGR) | `ego/wrist_left/wrist_right` (PIL, RGB) |
| Current joints | `states.arm_joint_states`=[L7,R7] | **unused** (state joints all masked) |
| Current gripper | `states.gripper_states`=[Lg,Rg] | state dim 6/20 |
| Current EEF | `states.end_pose.base_link.{left,right}_arm` | used directly as `recover`'s `ee_pos/ee_rotm` (base_link) |
| Instruction | `prompt` | wrapped with the training template (§6.3) |
| Response | `left_arm/right_arm{kind:EEF_ABS,values:[x,y,z,qx,qy,qz,qw]}` | `recover_action` output converted to quat |
| Response | `left_effector/right_effector` | model's gripper target (absolute value after recover) |

Waist/head: under the EEF route XR-0 does not predict waist or head → **the `waist`/`head` fields are not returned**, G2 holds its observed pose (consistent with LingBot's convention of not returning `waist` for embodiments without a waist).

---

## 4. Data Preparation & Conversion (lerobot → XR-0 JSON)

Data source (Decision 3): **multiple lerobot datasets, dual-arm**, each dataset containing parquet (per-frame state/action) + mp4 (multi-camera) + `meta/info.json`. Write a conversion script `tools/lerobot_to_xr0.py` that, per dataset → per episode, produces XR-0 JSON+mp4 (`docs/data_format.md`).

### 4.1 Conversion Notes

> Indices are not hardcoded: every channel's column indices are read **by name** from `meta/info.json` → `features.<feature>.field_descriptions` (each named channel carries its own `indices` list) — the layout is derived directly from the data. `resolve_fields(info)`/`build_arrays(table, info)` handle this.

1. **Video**: map each lerobot camera key (check `meta/info.json` for the naming, e.g. `observation.images.{top_head, hand_left, hand_right}`) to XR-0's `ego/wrist_left/wrist_right`. lerobot videos are typically already mp4 (h264), readable by `decord`; transcode if the codec is incompatible, aligning frame-by-frame at 30fps.
2. **proprios (current, per-frame absolute values `[N,D]`)**:
   - `left/right_ee_pos [N,3]` + `left/right_ee_rotm [N,9]`: taken from lerobot's `end/position`/`end/orientation` channels (quaternion xyzw → 3×3 rotation matrix, row-major flattened). **Used as-is, no conversion** (§2.3 delta invariance; state and action share the same `end/*` group, verified to be the same frame).
   - `left/right_arm_joint`: still written per schema (`docs/data_format.md` requires `[N,6]`); but since state joints are fully masked (Decision 2), this field **does not go into state and is not supervised** — written as all-zero placeholders purely to satisfy the JSON schema.
   - `left/right_gripper_pos [N,1]`: lerobot gripper state (`{left,right}_effector/position`), used as-is (norm is self-consistent).
   - `waist_joint [N,1]` (optional): the **last dim** of `waist/position` (joint5, torso-twist DOF); supervised only on the manip route, not written/masked on the IF route.
3. **actions (future targets, per-frame absolute values)**: for each timestep t, take the **absolute EEF pose + gripper** for the next 30 steps (likewise taken from the action's `end/*` channels; padded/truncated per XR-0 convention near the end of the episode). The pipeline's internal `_arm_action` computes the delta automatically — no manual computation needed. Joint action entries are likewise just schema placeholders (masked during training).
4. **instruction**: wrap in XR-0's three-view template (§6.3), filled with that episode's natural-language instruction, taken from `meta/annotations.json` (keyed by episode index, text = concatenation of `action_steps[*].action_text`; the top-level `instruction` is usually empty).
5. **trajectory_type**: `episode_status == "approved"` in `meta/annotations.json` → `success`, otherwise `invalid` (masked).
6. `num_frames` == length of each proprio/action == frame count of each video.

### 4.2 Merging Multiple Datasets
- Each lerobot dataset is converted into its own JSON directory; `configs/data/g2op_if.yaml`'s `train_path` lists **all directories** (this field is already a list).
- **Consistency check**: each dataset's camera naming, gripper units/open-close direction, and EEF frame must first be aligned to the same 32-D layout before merging to compute the norm stats — otherwise the norm gets skewed.
- Dual arms naturally fit the 32-D dual-arm layout; when one arm is stationary its delta ≈ 0, which is expected.

---

## 5. Normalization Statistics (one set per task)

XR-0 uses **per-timestep `(30,32)` mean/std**, hardcoded in the data yaml. Need to write `tools/compute_norm_stats.py`:

- Walk the converted JSON, using the same delta-computation logic as `json_dataset`, and compute mean/std per timestep × channel.
- **action**: masked joint dims (7–12, 21–26) and reserved dims → `mean=0, std=1`; active dims (EEF pos/aa, gripper) get real statistics.
- **state**: joint dims are all masked (Decision 2) and constantly 0 → **force `mean=0, std=1`** (otherwise std=0 causes division by zero); gripper dims get real statistics.
- Output is pasted directly into `configs/data/g2op_if.yaml`'s `mean`/`std` (shape `(30,32)`). With multiple datasets, compute stats over the **merged full set**.

> Consistency rule (carried over from LingBot): **the mean/std used for training must equal the mean/std the deploy service loads**. `deploy.py:load_stats` reads it from the ckpt directory's `config.py`, which is generated from the training artifact, so they're naturally consistent — when switching task/ckpt, manually verify they haven't gotten mixed up.

---

## 6. Training

### 6.1 Data config `configs/data/g2op_if.yaml`
Adapted from `earphone.yaml`:
- `train_path`: points to the converted JSON directories.
- `batch_size` / `action_length` (30).
- `mean`/`std`: generated in §5.

### 6.2 action_mask: zero out joint dims (key change)
Currently `io.build_action_mask` sets 1 for all `ACTION_PARTS` (including joints). The EEF route needs **joint dims unsupervised**. Two options:
- **(A) Config flag (recommended)**: add an `active_parts` parameter to `build_action_mask`; for G2 only activate `left_ee_pos/aa/gripper` + the corresponding right-side channels, passed in from the data yaml. Change is centralized and reusable.
- (B) robot_type branch: in `io.py`, select the active channel set based on embodiment name.

Both places that generate the mask — `json_dataset.__getitem__` and `server` — must go through the same logic.

### 6.3 Prompt template alignment (easy to get wrong)
The training pipeline (`json_dataset._prompt`) appends ` /no_cot` at the end of the human turn, and sets the assistant turn to `<cot></cot>`.
**But the repo's own `client.py._messages` uses `<bot></bot>` with no `/no_cot`** — the two are inconsistent!
→ **The adapter must replicate the training template** (`/no_cot` + `<cot></cot>`), otherwise the distribution won't match after SFT. The data-conversion script's instruction written into the JSON must also use the same template.

### 6.4 Synchronous vs. Asynchronous
The GenieSim benchmark consumes actions by "running the whole chunk, then re-querying" → **start with synchronous `async_train: false`** (the default in `configs/model/XR0.yaml`). Asynchronous (prefix-conditioned) is left as a later latency optimization.

### 6.5 Launch
```bash
CUDA_VISIBLE_DEVICES=0 RESOURCE_GPU=1 \
bash scripts/train.sh \
  data=g2op_if \
  model=XR0 \
  trainer.project="xr0_geniesim" \
  trainer.exp_name="g2op_if" \
  trainer.default_root_dir="train/" \
  model.params.model.pretrained="pretrained_ckpt/xr0_pretrained.pt"
# Multi-GPU: CUDA_VISIBLE_DEVICES=0..7 RESOURCE_GPU=8
```

---

## 7. Inference Deployment

### 7.1 XR-0 model service (:10086, largely unchanged)
```bash
python mibot/server/deploy.py --model train/xr0_geniesim/g2op_if/<ckpt_dir> --port 10086
```
`server.py` already handles normalization/denormalization, outputting `(1,30,32)` denormalized absolute-delta-domain actions. **Almost no changes needed** (optional: zero out the joint-dim output, as a safety measure).

> Trained checkpoints are published at:
> - ModelScope: https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset/tree/master/checkpoints/xiaomi-robotics-0
> - Hugging Face: https://huggingface.co/datasets/agibot-world/GenieSim3.0-Dataset/tree/main/checkpoints/xiaomi-robotics-0

### 7.2 corobot adapter layer `geniesim_deploy/xr0_corobot_adapter.py` (new)
Responsibilities (merging the original `client.py` logic):
```
async handler(msgpack infer request):
  params = req["params"]
  # ① Images: JPEG(BGR) → RGB → PIL → resize_image(factor=32, max_pixels=90000)
  ego   = jpeg_to_pil(params.images.head)
  wl    = jpeg_to_pil(params.images.hand_left)
  wr    = jpeg_to_pil(params.images.hand_right)
  # ② Assemble robot_state: EEF taken directly from base_link end_pose (no conversion), joints all masked
  robot_state = {
    left_ee_pos  = end_pose.base_link.left_arm.position,
    left_ee_rotm = quat2rotm(end_pose.base_link.left_arm.orientation),  # xyzw
    right_ee_pos/right_ee_rotm likewise,
    # joints left unfilled (state fully masked); gripper used for state
    left_gripper_pos, right_gripper_pos,
  }
  # ③ prompt wrapped with the training template (/no_cot, <cot></cot>), Qwen3-VL tokenization
  payload = processor.apply_chat_template(_messages(prompt, ego,wl,wr))
  payload["state"] = compose_state(grippers only, joints=0)        # (1,1,32)
  # ④ Send to XR-0 :10086 (pickle/TCP), receive (30,32)
  action = tcp_client.infer(payload)                        # reuses client._send/_recv
  # ⑤ recover: EEF delta → absolute base_link pose (no coordinate conversion) → quat(xyzw)
  out30 = [recover_action(action[i], robot_state) for i in range(30)]
  # ⑥ Build the corobot result (EEF_ABS, base_link frame)
  result = {
    "left_arm":  {"kind":"EEF_ABS", "values":[[x,y,z,qx,qy,qz,qw],...30]},
    "right_arm": {"kind":"EEF_ABS", "values":[...]},
    "left_effector":  [[g],...30],
    "right_effector": [[g],...30],
  }
  ws.send(msgpack.packb({"result": result}))
```
Notes:
- **No coordinate-frame conversion**: both the current pose and the response now use base_link; GenieSim converts to arm_base_link itself for IK (§2.3).
- `recover_action` already restores the EEF delta into an absolute `ee_pos`+`ee_rotm`; rotm→quat uses scipy (xyzw).
- Reuses `mibot/server/runtime/client.py`'s `_send/_recv` (pickle + 4-byte length header) and `resize_image`.
- Dependencies: `cv2`, `msgpack`, `websockets`, `transformers` (AutoProcessor), `scipy`. The adapter must run in a venv with these installed.

### 7.3 Launching a task from GenieSim
GenieSim uses `robot_cfg=G2_omnipicker` (robot_type=g2a_sim), `--model_arc corobot`, `--infer_host <ip>:8007`, headless. See this repo's `run-geniesim-local` skill / LingBot's `run_geniesim_task.sh` for reference.

---

## 8. Change List (quick reference)

| # | Add/Change | File | Key points |
|---|---------|------|------|
| 1 | data conversion script | `xr0/tools/lerobot_to_xr0.py` | multiple lerobot datasets → XR-0 JSON+mp4; EEF taken as-is from the `end/*` channels (state/action share the same frame); indices read by field_descriptions name; joints are placeholders only |
| 2 | norm stats script | `xr0/tools/compute_norm_stats.py` | computes `(30,32)` mean/std over the merged full set; masked/constant dims get std=1 |
| 3 | data config | `xr0/configs/data/g2op_if.yaml` | train_path lists all JSON directories + mean/std + active_parts |
| 4 | action_mask supports joint masking | `xr0/mibot/utils/io.py:build_action_mask` | adds `active_parts`, zeroes out joint dims (§6.2) |
| 5 | state joint masking | `xr0/mibot/utils/io.py:compose_state` | G2 fills only the dual grippers, joint slots = 0 (Decision 2) |
| 6 | dataset & server use the new mask | `json_dataset.py` / `server.py` | both consistently go through active_parts |
| 7 | corobot adapter | `xr0/geniesim_deploy/xr0_corobot_adapter.py` | msgpack/WS ↔ pickle/TCP; base_link EEF used directly; recover→EEF_ABS; prompt template |
| 8 | launch scripts | `xr0/geniesim_deploy/{serve.sh,run_task.sh}` | brings up :10086 + :8007; launches tasks |

**Self-consistency is the only goal**: the data-conversion layout/frame == the training mask/norm == the adapter's recover/response frame. A mismatch in any link → OOD → erratic behavior.

---

## 9. Key Differences from the LingBot Approach (notes for anyone porting this)

1. **Control space**: LingBot's g2op uses **joint space** (16-D continuous L7R7); XR-0 uses **EEF_ABS + GenieSim IK**, because XR-0's joint slots only have 6-DOF.
2. **Protocol**: LingBot's upstream is WebSocket; XR-0's upstream is **pickle/TCP** — the adapter has to bridge the two protocols.
3. **Norm**: LingBot uses per-feature json; XR-0 uses **per-timestep (30,32)**, hardcoded in the data yaml and carried with the ckpt.
4. **Data**: LingBot uses lerobot; XR-0 uses **one JSON + videos per episode**, requiring a dedicated conversion script.
5. **Embodiment selection**: LingBot's three experiments share a robot_type and distinguish norm via `ROBO_NAME`; XR-0 has one norm per ckpt, so the adapter can run **one embodiment per process** — no `ROBO_NAME`-based routing authority is needed yet (to be introduced when running multiple experiments).

---

## 10. Risks & Validation

| Risk | Mitigation |
|------|------|
| EEF→IK accuracy / G2 IK convergence | first replay IK on a small set of GT EEF trajectories inside GenieSim to confirm `eef_actions_to_joint` is stable; fall back to the joint route if it isn't |
| base_link/arm_base_link frame | **already resolved** (§2.3 delta invariance, zero conversion in the adapter); still write a unit test comparing one frame's recover round-trip |
| Inconsistent lerobot layout/units (across datasets) | the conversion script explicitly declares the cam/EEF/gripper mapping per dataset; verify alignment to a unified 32-D layout before merging |
| Inconsistent prompt template | data conversion and the adapter reuse the same `_messages`, both carrying `/no_cot`+`<cot></cot>` |
| Joint mask not taking effect → model learns the wrong thing | print the batch's action_mask to confirm 7–12/21–26 are all 0; check that loss only decreases on the EEF dims |
| Gripper scale/open-close direction | norm is computed on real G2 data and passed through self-consistently; check gripper delta is sane during probing |
| webm decoding | uniformly transcode to mp4 during conversion |

**Probe (mandatory before going live)**: following LingBot, step0's EEF target should ≈ the current EEF (small delta, closely tracking the pose); under a single-arm instruction, the other arm's EEF drift should be noticeably smaller. **A layout/frame mismatch is invisible as a round-trip in delta mode — you must manually diff the adapter-assembled `robot_state` against the data-conversion's stored values; don't rely on the probe alone.**

---

## 11. Milestones

1. **M1 Data pipeline working**: conversion script + a small sample of 1 task → visually verify JSON/video/EEF FK are correct.
2. **M2 Norm + training running**: compute_norm_stats → data yaml → single-GPU few-step training runs without errors, loss only decreases on the EEF dims.
3. **M3 Inference pipeline working**: adapter + server come up, GenieSim sends one frame, the response schema is valid, IK doesn't crash, probe looks sane.
4. **M4 Closed-loop SFT**: train on the full dataset with multiple GPUs → run the GenieSim benchmark for success rate.
5. **M5 (optional) Asynchronous execution**: `async_train:true` + prefix conditioning to reduce latency.

---

## Confirmed Decisions (v2)

1. ✅ **Control space = EEF_ABS + GenieSim IK**.
2. ✅ **State joints fully masked** (only dual grippers + vision remain).
3. ✅ **SFT = dual-arm, multiple lerobot datasets**.
4. ✅ **EEF frame = arm_base_link**; thanks to delta invariance, training/inference are self-consistent across frames, with zero conversion in the adapter.

## Information Still Needed From You (doesn't block the design, needed before implementation)

1. **lerobot camera key naming**: what the three camera channels are called in each dataset's `meta/info.json` (for the → ego/wrist_left/wrist_right mapping).
2. **lerobot EEF field layout**: which feature holds the EEF pose, whether orientation uses quaternion (wxyz/xyzw) or rotation vector/rpy, and the units.
3. **Gripper units and open-close direction**: omnipicker's gripper value range (determines whether norm and passing `left_effector` through directly need scaling).
4. **Dataset inventory and scale**: how many datasets, total episode/frame counts, source of instruction text (`meta/annotations.json`).
5. **Pretrained weights path**: the XR-0 ckpt that `model.params.model.pretrained` should point to.
