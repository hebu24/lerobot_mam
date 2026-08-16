#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[01]$ ]]; then
  echo "Usage: $0 NODE_RANK  # NODE_RANK is 0 or 1" >&2
  exit 2
fi

NODE_RANK="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOSTFILE="${HOSTFILE:-${REPO_ROOT}/configs/deepspeed/franka_dp_12gpu.hostfile}"
DEEPSPEED_PYTHON="${DEEPSPEED_PYTHON:-/cephfs/shared/Yanbang/envs/starvla/bin/python}"
MASTER_ADDR="${MASTER_ADDR:-10.233.66.236}"
MASTER_PORT="${MASTER_PORT:-29531}"
WORKER="${REPO_ROOT}/scripts/run_dp_franka_real_deepspeed_worker.sh"

if [[ ! -f "${HOSTFILE}" ]]; then
  echo "Hostfile not found: ${HOSTFILE}" >&2
  exit 1
fi
if [[ ! -x "${DEEPSPEED_PYTHON}" ]]; then
  echo "DeepSpeed Python not found: ${DEEPSPEED_PYTHON}" >&2
  exit 1
fi

# --no_ssh lets the controller start one launcher per VM. Both launchers read
# the same hostfile, so DeepSpeed assigns global ranks 0..11 consistently even
# though the nodes expose different GPU counts.
# The DeepSpeed environment has no CUDA toolkit (CUDA_HOME). CPU mode is enough
# for its launcher and avoids importing optional CUDA extension builders; the
# spawned worker still uses CUDA through the separate ManiSkill environment.
export DS_ACCELERATOR=cpu
exec "${DEEPSPEED_PYTHON}" -u -m deepspeed.launcher.runner \
  --hostfile "${HOSTFILE}" \
  --no_ssh \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  --no_python \
  --no_local_rank \
  "${WORKER}"
