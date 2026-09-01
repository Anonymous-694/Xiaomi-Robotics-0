#!/usr/bin/env python3
# Copyright (C) 2026 Xiaomi Corporation.
"""corobot-protocol adapter in front of the XR-0 deploy server, for GenieSim G2.

GenieSim's ``CoRobotPolicy`` talks JSON-RPC-over-msgpack over a WebSocket:
    <- {"method":"infer","params":{images, states, prompt, robot_type, ...}}
    -> {"result":{left_arm{kind,values}, right_arm{...}, left_effector, right_effector, base_link}}

The XR-0 deploy server (``mibot/server/deploy.py``, :10086) speaks pickle-over-raw-TCP and outputs a
denormalized ``(30,32)`` action chunk. This adapter is a thin translation LAYER (no GPU, no model):

  1. decode the three JPEG views (BGR -> RGB -> PIL);
  2. read ``states.end_pose.arm_base_link.{left,right}_arm`` (arm_base_link frame, xyzw quaternion) as
     BOTH the delta reference EEF pose and the proprio ``state`` EEF (``[x,y,z,roll,pitch,yaw]``,
     ``{side}_state_eef`` — matching training's converter, arm_base_link + EULER_SEQ), and
     ``states.gripper_states`` as the current grippers. Joint channels stay zeros (masked during SFT);
  3. run the SAME tokenize + compose_state + recover_action pipeline as ``mibot .../client.py`` (we
     reuse its ``Client``), but with the TRAINING prompt template (``/no_cot`` + ``<cot></cot>``);
  4. for each of the 30 chunk steps, ``recover_action`` gives absolute EEF targets in the arm_base_link
     frame (delta invariance, design §2.3 — NO coordinate conversion here); convert rotm -> xyzw quat
     and emit ``EEF_ABS`` values ``[x,y,z,qx,qy,qz,qw]``. GenieSim's built-in IK turns them into the
     7-DOF joint command.

Frames: reference pose and returned targets are both arm_base_link. The result declares
``base_link: "arm_base_link"`` so GenieSim skips its base_link->arm_base_link transform and uses the
poses directly (sim commit "Let the policy server declare its EEF_ABS frame"). Waist/head are not
predicted, so those fields are omitted and the robot holds them.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
import traceback

import cv2
import msgpack
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

# XR-0 repo root (…/xr0) is the parent of geniesim_deploy/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mibot.server.runtime.client import Client  # noqa: E402

import websockets.asyncio.server as ws_server  # noqa: E402

# Euler sequence for the state EEF (roll,pitch,yaw). MUST match tools/lerobot_to_xr0.py EULER_SEQ.
EULER_SEQ = "xyz"  # extrinsic, radians

# Waist route: feed the current waist joint5 into the model state (dim 13) AND command the predicted
# waist target back to the sim. Enable ONLY for a model trained with the waist joint in active_parts;
# EEF-only models (waist masked, dim 13 = 0) must leave this off.
_WAIST_ENABLED = os.environ.get("XR0_WAIST", "") not in ("", "0", "false", "False", "no")


class Xr0Client(Client):
    """Same TCP pipeline as the stock client, but with the SFT training prompt template.

    ``json_dataset._prompt`` appends ``" /no_cot"`` to the human turn and sets the gpt turn to
    ``"<cot></cot>"``. The stock ``Client._messages`` uses ``"<bot></bot>"`` with no ``/no_cot`` — that
    is the *pretrained* convention and would be out-of-distribution after SFT. We must match training.
    """

    @staticmethod
    def _messages(instruction, ego_obs, left_wrist_obs, right_wrist_obs):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "The following observations are captured from multiple views.\n# Ego View\n"},
                    {"type": "image", "image": ego_obs},
                    {"type": "text", "text": "\n# Left-Wrist View\n"},
                    {"type": "image", "image": left_wrist_obs},
                    {"type": "text", "text": "\n# Right-Wrist View\n"},
                    {"type": "image", "image": right_wrist_obs},
                    {"type": "text", "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]


def _decode_jpeg_rgb(d) -> Image.Image:
    raw = d.get("image_data") if isinstance(d, dict) else d
    if raw is None and isinstance(d, dict):
        raw = d.get(b"image_data")
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _quat_xyzw_to_rotm(quat_xyzw) -> np.ndarray:
    return Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_matrix().astype(np.float32)


def _build_robot_state(states):
    """corobot states -> XR-0 recover_action robot_state.

    Everything EEF comes from ``end_pose.arm_base_link`` — the single frame training uses throughout:
      * delta base (``{side}_ee_pos``/``ee_rotm``): recover_action then emits arm_base_link absolute
        targets, and _to_corobot_result declares ``base_link: arm_base_link`` so GenieSim skips its
        base_link->arm_base_link transform (delta invariance, design §2.3).
      * proprio state EEF (``{side}_state_eef`` = [x,y,z,roll,pitch,yaw]): same arm_base_link pose,
        matching how training writes the state EEF (converter, arm_base_link + EULER_SEQ).
    Joint channels stay zeros (masked in SFT).
    """
    end_pose = states.get("end_pose")
    if not end_pose or "arm_base_link" not in end_pose:
        raise ValueError("params.states.end_pose.arm_base_link missing — XR-0 is trained in arm_base_link; the sim must send it")
    arm_base = end_pose["arm_base_link"]
    grip = np.asarray(states.get("gripper_states", [0.0, 0.0]), dtype=np.float32)
    zeros6 = np.zeros(6, dtype=np.float32)

    robot_state = {}
    for side, key in (("left", "left_arm"), ("right", "right_arm")):
        ab = arm_base[key]
        ab_pos = np.asarray(ab["position"], dtype=np.float32).reshape(3)
        robot_state[f"{side}_ee_pos"] = ab_pos
        robot_state[f"{side}_ee_rotm"] = _quat_xyzw_to_rotm(ab["orientation"]).reshape(9)
        robot_state[f"{side}_arm_joint"] = zeros6
        ab_rpy = Rotation.from_quat(np.asarray(ab["orientation"], dtype=np.float64)).as_euler(EULER_SEQ).astype(np.float32)
        robot_state[f"{side}_state_eef"] = np.concatenate([ab_pos, ab_rpy]).astype(np.float32)
    robot_state["left_gripper_pos"] = np.asarray([grip[0]], dtype=np.float32)
    robot_state["right_gripper_pos"] = np.asarray([grip[1] if grip.size > 1 else grip[0]], dtype=np.float32)

    if _WAIST_ENABLED:
        wj = states.get("waist_joint_states")
        if wj is None:
            raise ValueError("XR0_WAIST is on but params.states.waist_joint_states missing — the sim must send it")
        wj = np.asarray(wj, dtype=np.float32).reshape(-1)
        robot_state["waist_joint_full"] = wj       # all body joints; held except joint5
        robot_state["waist_joint"] = wj[-1:]        # joint5 = last (idx05_body_joint5) -> state dim 13
    return robot_state


def _to_corobot_result(targets, waist_delta=None, robot_state=None):
    """recover_action targets (arm_base_link, per-step) -> corobot EEF_ABS result dict.

    The top-level ``base_link: "arm_base_link"`` field declares the frame of the EEF_ABS values so
    GenieSim uses them directly and skips its own base_link->arm_base_link transform (sim commit
    "Let the policy server declare its EEF_ABS frame").

    When ``waist_delta`` (per-step joint5 delta, from action dim 13) and the current waist are given
    (manip route), also emit ``waist`` as a JOINT_ABS command: joint5 = current + delta, the other
    body joints held at their current value.
    """
    chunk = len(targets["left_ee_pos"])
    left_arm, right_arm, left_eff, right_eff = [], [], [], []
    for i in range(chunk):
        for side, arm_out, eff_out in (("left", left_arm, left_eff), ("right", right_arm, right_eff)):
            pos = np.asarray(targets[f"{side}_ee_pos"][i], dtype=np.float64).reshape(3)
            rotm = np.asarray(targets[f"{side}_ee_rotm"][i], dtype=np.float64).reshape(3, 3)
            quat_xyzw = Rotation.from_matrix(rotm).as_quat(scalar_first=False)  # [qx,qy,qz,qw]
            arm_out.append([*pos.tolist(), *quat_xyzw.tolist()])
            eff_out.append([float(np.asarray(targets[f"{side}_gripper_pos"][i]).reshape(-1)[0])])
    result = {
        "left_arm": {"kind": "EEF_ABS", "values": left_arm},
        "right_arm": {"kind": "EEF_ABS", "values": right_arm},
        "left_effector": left_eff,
        "right_effector": right_eff,
        "base_link": "arm_base_link",  # frame declaration for the EEF_ABS values (see docstring)
    }
    if waist_delta is not None and robot_state is not None and "waist_joint_full" in robot_state:
        base5 = np.asarray(robot_state["waist_joint_full"], dtype=np.float64).reshape(-1)
        wd = np.asarray(waist_delta, dtype=np.float64).reshape(-1)  # per-step joint5 delta
        waist_values = []
        for i in range(chunk):
            row = base5.copy()
            row[-1] = base5[-1] + (wd[i] if i < wd.size else wd[-1])  # joint5 = current + predicted delta
            waist_values.append(row.tolist())
        result["waist"] = {"kind": "JOINT_ABS", "values": waist_values}
    return {"result": result}


CLIENT: Xr0Client = None
_LOCK = threading.Lock()  # Client owns one TCP socket -> serialize infers


def _infer(params):
    imgs = params["images"]
    ego = _decode_jpeg_rgb(imgs["head"])
    wl = _decode_jpeg_rgb(imgs["hand_left"])
    wr = _decode_jpeg_rgb(imgs["hand_right"])

    robot_state = _build_robot_state(params["states"])

    prompt = params.get("prompt", "")
    if isinstance(prompt, (list, tuple)):
        prompt = " ".join(str(p) for p in prompt)

    with _LOCK:
        out = CLIENT(robot_state, ego, wl, wr, str(prompt))
    if os.environ.get("XR0_LOG_ACTION"):
        ra = np.asarray(out["raw_action"])  # (30,32) denormalized per-step deltas
        tg = out["action_targets"]
        lp0 = np.asarray(tg["left_ee_pos"][0]).reshape(-1)
        print(
            f"[infer] |ΔLee_pos|max={np.abs(ra[:, 0:3]).max():.3f} |ΔLee_aa|max={np.abs(ra[:, 3:6]).max():.3f} "
            f"|ΔRee_pos|max={np.abs(ra[:, 14:17]).max():.3f} Lgrip={ra[0,6]:+.2f} "
            f"| L_target[0]=({lp0[0]:.3f},{lp0[1]:.3f},{lp0[2]:.3f}) prompt='{str(prompt)[:50]}'",
            flush=True,
        )
    waist_delta = out["action_components"].get("waist") if _WAIST_ENABLED else None
    return _to_corobot_result(out["action_targets"], waist_delta, robot_state)


async def _handler(ws):
    async for msg in ws:
        try:
            req = msgpack.unpackb(msg, raw=False)
            if req.get("method") != "infer":
                await ws.send(msgpack.packb({"error": f"unknown method: {req.get('method')}"}))
                continue
            result = await asyncio.to_thread(_infer, req["params"])
            await ws.send(msgpack.packb(result))
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            await ws.send(msgpack.packb({"error": f"{type(error).__name__}: {error}"}))


async def _main(host, port, up_host, up_port):
    global CLIENT
    CLIENT = Xr0Client(host=up_host, port=up_port)
    print(f"[adapter] connected upstream XR-0 at {up_host}:{up_port}", flush=True)
    async with ws_server.serve(_handler, host, port, compression=None, max_size=None):
        print(f"[adapter] corobot server on ws://{host}:{port} -> XR-0 {up_host}:{up_port}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0", help="corobot WS bind host")
    ap.add_argument("--port", type=int, default=8007, help="corobot WS port (GenieSim --infer_host)")
    ap.add_argument("--upstream-host", default="127.0.0.1", help="XR-0 deploy server host")
    ap.add_argument("--upstream-port", type=int, default=10086, help="XR-0 deploy server port")
    args = ap.parse_args()
    asyncio.run(_main(args.host, args.port, args.upstream_host, args.upstream_port))
