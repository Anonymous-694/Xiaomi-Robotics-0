#!/usr/bin/env bash
# Launch a GenieSim G2 (platform arxone) instruction-following pick task against the XR-0 corobot
# adapter. This drives the SIM side; the model side must already be up via serve.sh.
#
# Two-machine picture:
#   [this host] .venv XR-0 server :10086 (GPU)  <-  adapter :8007  <-  GenieSim (docker, host net)
# GenieSim runs inside the `genie_sim_benchmark` container with the repo mounted at /geniesim/main;
# it POSTs corobot infer requests to --infer_host, which is the adapter started by serve.sh.
#
# Usage:  bash geniesim_deploy/run_task.sh <sub_task> [infer_host] [num_episode] [seed]
#   <sub_task> one of the 10 g2op_if tasks (arxone_if_<sub_task>.yaml must exist in the container):
#       pick_billiards_color pick_block_color pick_block_number pick_block_shape pick_block_size
#       pick_common_sense pick_follow_logic_or pick_object_type pick_specific_object straighten_object
#   [infer_host] default 127.0.0.1:8007   [num_episode] default 1   [seed] default 1
#   e.g.  bash geniesim_deploy/run_task.sh pick_block_color 127.0.0.1:8007 5
set -euo pipefail

SUB_TASK="${1:?usage: run_task.sh <sub_task> [infer_host] [num_episode] [seed]}"
INFER_HOST="${2:-127.0.0.1:8007}"
NUM_EPISODE="${3:-1}"
SEED="${4:-1}"

CTN="${GENIESIM_CTN:-genie_sim_benchmark}"
CFG="source/geniesim/config/arxone_if_${SUB_TASK}.yaml"   # arxone == G2 platform
ROSENV='export ROS_DISTRO=jazzy; export RMW_IMPLEMENTATION=rmw_fastrtps_cpp; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib; export PYTHONPATH=$PYTHONPATH:/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/rclpy'

docker ps --format '{{.Names}}' | grep -qx "$CTN" || { echo "container '$CTN' not running"; exit 1; }
docker exec "$CTN" test -f "/geniesim/main/$CFG" || { echo "task config not found in container: $CFG"; exit 1; }

# Concurrency guard: never let two Isaac Sims run at once (GPU-OOM protection).
if docker exec "$CTN" pgrep -f app.py >/dev/null 2>&1; then
  echo "ABORT: an app.py is already running in $CTN — refusing to start a second sim."
  exit 1
fi

# Sanity: the adapter should be reachable before we spin up the (expensive) sim.
host="${INFER_HOST%%:*}"; port="${INFER_HOST##*:}"
if ! (echo > "/dev/tcp/$host/$port") 2>/dev/null; then
  echo "WARN: adapter $INFER_HOST not reachable from this host — start it with serve.sh first." >&2
fi

echo "===== RUN $SUB_TASK  infer_host=$INFER_HOST  episodes=$NUM_EPISODE seed=$SEED ====="
# CLI overrides (mmengine-style dotted keys) layer on top of the task yaml — no file edits in the
# container. model_arc=corobot routes infer to --infer_host; headless for a GPU-only host.
docker exec "$CTN" bash -lc "$ROSENV; cd /geniesim/main && /isaac-sim/python.sh source/geniesim/app/app.py \
  --config $CFG \
  --benchmark.model_arc=corobot \
  --benchmark.infer_host=$INFER_HOST \
  --benchmark.num_episode=$NUM_EPISODE \
  --benchmark.seed=$SEED \
  --app.headless=true"
echo "===== DONE $SUB_TASK ====="
