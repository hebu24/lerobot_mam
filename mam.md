# LIBERO-10 STPM and MAM training

Run these commands on the 6-GPU VM:

```bash
ssh root@10.233.75.162
cd /cephfs/shared/Yanbang/lerobot/mam_lerobot0.5.1/lerobot_mam
```

## Action representation

- The dataset stores OSC pose absolute-goal actions.
- The MAM preprocessor converts each absolute action chunk to chunk-relative
  SE(3), anchored at the latest observation state.
- The MAM model is trained to predict normalized relative actions.
- The postprocessor unnormalizes the prediction and converts it back to an
  absolute-goal action chunk.
- LIBERO must therefore use `ENV_CONTROL_MODE=absolute`.

Do not change `--policy.use_relative_actions=true` or
`ENV_CONTROL_MODE=absolute`; the launcher validates this contract.

## STPM baseline: 10 task-specific models, 6 epochs

The following command uses all 6 GPUs by assigning independent LIBERO-10
tasks to single-GPU workers. STPM itself does not implement DDP or DeepSpeed.

```bash
cd /cephfs/shared/Yanbang/lerobot/mam_lerobot0.5.1/lerobot_mam
export PATH=/cephfs/shared/Yanbang/envs/lerobot0.5.1/bin:$PATH

RUN_ID="stpm_libero10_6epoch_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="outputs/logs/${RUN_ID}"
mkdir -p "${LOG_DIR}"

assignments=("8" "9" "0,5" "7,3" "2,6" "1,4")
for gpu in 0 1 2 3 4 5; do
  tasks="${assignments[$gpu]}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" \
    TASK_IDS="${tasks}" \
    EPOCHS=6 \
    STEPS= \
    VISION_CKPT=/cephfs/shared/Yanbang/maniskill/pretrained/clip-vit-base-patch32 \
    bash scripts/train_stpm_libero10_v3_all.sh
  ) >"${LOG_DIR}/gpu${gpu}_tasks_${tasks//,/_}.log" 2>&1 &
done
wait
```

Expected outputs:

```text
outputs/train/stpm_libero10_v2_task0
...
outputs/train/stpm_libero10_v2_task9
```

Verify all final checkpoints:

```bash
for task_id in {0..9}; do
  test -f "outputs/train/stpm_libero10_v2_task${task_id}/config.yaml"
  test -f "outputs/train/stpm_libero10_v2_task${task_id}/checkpoints/reward_best.pt"
  test -f "outputs/train/stpm_libero10_v2_task${task_id}/checkpoints/reward_final.pt"
done
echo "All STPM checkpoints are present."
```

These 10 models were completed on 2026-07-26. Re-running the command with the
default `SKIP_EXISTING=true` skips directories containing `config.yaml` and
`reward_best.pt`.

## Larger STPM comparison

The larger STPM changes only model capacity. It uses the same dataset,
per-task train/validation split, batch size, learning rate, and six epochs:

```text
                         Baseline                 Larger
d_model                  256                      512
Transformer layers       2                        4
Attention heads          4                        8
Trainable parameters     2,111,745                14,198,273
reward_best.pt           about 8.1 MiB            about 54.2 MiB
Output prefix            stpm_libero10_v2_task    stpm_libero10_v3_large_d512_l4_task
```

Best validation MSE comparison:

| LIBERO-10 task | Baseline | Larger | Relative change |
|---:|---:|---:|---:|
| 0 | 0.00194768 | 0.00218154 | +12.01% |
| 1 | 0.00220672 | 0.00172670 | -21.75% |
| 2 | 0.00235228 | 0.00229707 | -2.35% |
| 3 | 0.00285695 | 0.00300759 | +5.27% |
| 4 | 0.00386689 | 0.00420679 | +8.79% |
| 5 | 0.00293919 | 0.00231362 | -21.28% |
| 6 | 0.00559830 | 0.00520008 | -7.11% |
| 7 | 0.00234376 | 0.00224598 | -4.17% |
| 8 | 0.00104366 | 0.00107031 | +2.55% |
| 9 | 0.00111280 | 0.00142699 | +28.23% |
| **Mean** | **0.00262682** | **0.00256767** | **-2.25%** |
| **Median** | **0.00234802** | **0.00227152** | **-3.26%** |

The larger model improves five tasks and regresses five tasks. Its mean best
validation MSE is 2.25% lower. The MAM run below intentionally uses all ten
larger STPM models, including the tasks that regressed.

The larger checkpoints were completed and verified on 2026-07-26:

```bash
for task_id in {0..9}; do
  root="outputs/train/stpm_libero10_v3_large_d512_l4_task${task_id}"
  test -f "${root}/config.yaml"
  test -f "${root}/checkpoints/reward_best.pt"
  test -f "${root}/checkpoints/reward_final.pt"
done
```

## Previous MAM run: stopped

The random-mask-only run started on 2026-07-26 and was stopped on
2026-07-27 at approximately step 117,600 so it could be replaced by the
reference mixed-mask experiment:

```text
VM:         root@10.233.75.162
tmux:       mam_large_stpm
job:        mam_libero10_v3_relative_150k_6gpu_large_stpm_multirankeval_20260726_161112
output:     outputs/train/mam_libero10_v3_relative_150k_6gpu_large_stpm_multirankeval_20260726_161112
log:        outputs/logs/mam_libero10_v3_relative_150k_6gpu_large_stpm_multirankeval_20260726_161112.log
STPM:       stpm_libero10_v3_large_d512_l4_task0 through task9
```

Attach with `tmux attach -t mam_large_stpm`. The first logged training metric
at step 200 had loss 0.53186, gradient norm 2.847, and warmup learning rate
2.01e-5.

Reviewed settings:

- 6 A10 GPUs, batch size 16 per GPU, effective batch size 96.
- BF16 mixed precision.
- 293M trainable parameters with U-Net dimensions 512/1024/2048.
- 150,000 optimizer steps.
- Save every 10,000 steps and run fixed 50-episode LIBERO-10 evaluation every
  5,000 steps. This gives 30 evaluations through the final 150,000-step
  checkpoint.
- Evaluation also uses all 6 GPUs. The ten tasks are assigned to ranks as
  `[0,6]`, `[1,7]`, `[2,8]`, `[3,9]`, `[4]`, and `[5]`; with five episodes per
  task, the per-rank loads are `10/10/10/10/5/5`. A full evaluation is expected
  to take roughly 10 minutes when rollouts reach their time limit.
- Weighted mask loss with known-region weight 0.2.
- 32-step prediction horizon and 15 executed actions per inference call.
- Local CLIP tokenizer path is required because the launcher runs offline and
  the Hub tokenizer is not present in the configured HF cache.
- The target VM's complete LIBERO simulator assets are in
  `/root/.cache/libero/assets`; the repository-local default path is absent.

Start training:

```bash
cd /cephfs/shared/Yanbang/lerobot/mam_lerobot0.5.1/lerobot_mam

export CONDA_PREFIX=/root/miniconda3
export CONDA_ENV_PATH=/cephfs/shared/Yanbang/envs/lerobot0.5.1
export LIBERO_ASSETS_PATH=/root/.cache/libero/assets
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NUM_GPUS=6

export DATASET_REPO_ID=local/libero10_mam_v3_unfiltered_train
export DATASET_ROOT=outputs/datasets/libero10_mam_v3_unfiltered_train
export MAM_EVAL_DATASET_REPO_ID=local/libero10_mam_v3_unfiltered_eval
export MAM_EVAL_DATASET_ROOT=outputs/datasets/libero10_mam_v3_unfiltered_eval
export STPM_BASE_DIR=outputs/train
export STPM_NAME_PREFIX=stpm_libero10_v3_large_d512_l4_task

export STEPS=150000
export BATCH_SIZE=16
export NUM_WORKERS=8
export PREFETCH_FACTOR=4
export PERSISTENT_WORKERS=true
export MIXED_PRECISION=bf16

export ENABLE_EVAL=true
export EVAL_FREQ=5000
export SAVE_FREQ=10000
export LOG_FREQ=200
export EVAL_N_EPISODES=50
export EVAL_BATCH_SIZE=1
export EVAL_USE_ASYNC_ENVS=false
export ENV_TASK=libero_10
export ENV_TASK_IDS='[0,1,2,3,4,5,6,7,8,9]'
export ENV_CONTROL_MODE=absolute
export ENV_OBSERVATION_HEIGHT=128
export ENV_OBSERVATION_WIDTH=128
export ENV_MAX_PARALLEL_TASKS=1

export LEARNING_RATE=1e-4
export WEIGHT_DECAY=1e-6
export WARMUP_STEPS=500
export GRAD_CLIP_NORM=10.0
export MASK_TYPE=random_mask
export MASK_LOSS_MODE=weighted
export MASK_KNOWN_REGION_WEIGHT=0.2
export MASK_INPAINTING=false
export MASK_PADDING_LOSS=true
export DO_MASK_LOSS_FOR_PADDING=true
export PRETRAINED_BACKBONE_WEIGHTS=null
export PUSH_TO_HUB=false
export WANDB_ENABLE=false

RUN_ID="$(date +%Y%m%d_%H%M%S)"
export JOB_NAME="mam_libero10_v3_relative_150k_6gpu_large_stpm_multirankeval_${RUN_ID}"
export OUTPUT_DIR="outputs/train/${JOB_NAME}"

bash scripts/run_mam_libero10_conda.sh \
  --policy.language_tokenizer_name=/cephfs/shared/Yanbang/maniskill/pretrained/clip-vit-base-patch32
```

The launcher maps all ten STPM roots from
`outputs/train/stpm_libero10_v3_large_d512_l4_task0` through `task9`. Its
preflight checks the train/eval manifests, relative-action statistics,
source-episode separation, mask type, controller mode, and STPM artifacts
before starting.

Run the final command in a detached tmux session. Keep the log outside
`OUTPUT_DIR` because the training config requires `OUTPUT_DIR` not to exist at
startup:

```bash
mkdir -p outputs/logs
export MAM_LOG="outputs/logs/${JOB_NAME}.log"
tmux new-session -d -s mam_large_stpm \
  "bash scripts/run_mam_libero10_conda.sh \
    --policy.language_tokenizer_name=/cephfs/shared/Yanbang/maniskill/pretrained/clip-vit-base-patch32 \
    >'${MAM_LOG}' 2>&1"
tmux attach -t mam_large_stpm
```

## MAM reference mixed-mask restart: 150,000 steps on 6 GPUs

Only the mask dataset configuration changes. Model size, larger STPM models,
optimizer, learning-rate schedule, batch size, six-GPU DDP, BF16, seed,
training steps, save/eval frequency, loss settings, inpainting setting,
relative-action representation, and LIBERO evaluation settings remain exactly
the same as the previous run.

The mask configuration matches
`rgb_newnew_LiftPegUpright_500demo_points1(25%)_3D_points1(25%)_3D_points0.2(25%)_pose0.2(25%)_long32-128_evalAll`.

Training uses four slots in `composition` mode:

| Slot | Mask type | Retain ratio | Episode composition |
|---:|---|---:|---:|
| 0 | `points` | 1.0 | 25% |
| 1 | `3D_points` | 1.0 | 25% |
| 2 | `3D_points` | 0.2 | 25% |
| 3 | `pose_motion_planning` | 0.2 | 25% |

Evaluation uses five slots, each assigned to 20% of the fixed 50 episodes:

| Slot | Mask type | Retain ratio | Episodes |
|---:|---|---:|---:|
| 0 | `points` | 1.0 | 10 |
| 1 | `3D_points` | 1.0 | 10 |
| 2 | `3D_points` | 0.2 | 10 |
| 3 | `pose_motion_planning` | 0.2 | 10 |
| 4 | `mix0` | n/a | 10 |

Because the two `3D_points` settings share a mask type name, use
`per_mask_slot_success` to read all five evaluation results independently.

The old MAM datasets preserve the complete absolute actions, so remasking
reuses the original 450/50 source split and changes only
`mam.mas_action_mask`. The new datasets and old checkpoints are kept in
separate directories.

The reproducible launcher is:

```bash
scripts/run_mam_libero10_refmask_6gpu.sh
```

It first creates:

```text
data/libero10_mam/libero10_mam_v3_refmix_train
data/libero10_mam/libero10_mam_v3_refmix_eval
```

and then launches the unchanged six-GPU training configuration.

Active restart:

```text
VM:         root@10.233.75.162
tmux:       mam_refmix
job:        mam_libero10_v3_refmix_150k_6gpu_large_stpm_multirankeval_20260727_183201
output:     outputs/train/mam_libero10_v3_refmix_150k_6gpu_large_stpm_multirankeval_20260727_183201
log:        outputs/logs/mam_libero10_v3_refmix_150k_6gpu_large_stpm_multirankeval_20260727_183201.log
```

Start a new run manually:

```bash
cd /cephfs/shared/Yanbang/lerobot/mam_lerobot0.5.1/lerobot_mam
RUN_ID="$(date +%Y%m%d_%H%M%S)"
JOB_NAME="mam_libero10_v3_refmix_150k_6gpu_large_stpm_multirankeval_${RUN_ID}"
LOG_PATH="outputs/logs/${JOB_NAME}.log"

tmux new-session -d -s mam_refmix \
  "cd '$PWD' && exec env \
    JOB_NAME='${JOB_NAME}' \
    OUTPUT_DIR='outputs/train/${JOB_NAME}' \
    PYTHONUNBUFFERED=1 \
    bash scripts/run_mam_libero10_refmask_6gpu.sh \
    >'${LOG_PATH}' 2>&1"
```

Monitor it with:

```bash
tmux attach -t mam_refmix
tail -f outputs/logs/mam_libero10_v3_refmix_150k_6gpu_large_stpm_multirankeval_20260727_183201.log
nvidia-smi
```
