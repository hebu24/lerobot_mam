# MAM training commands

This file records the runnable commands for training MAM on LIBERO
`put_bowl_on_plate` in this workspace.

Workspace:

```bash
cd /cephfs/shared/Yanbang/lerobot/mam_lerobot0.5.1/lerobot_mam
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /cephfs/shared/Yanbang/envs/lerobot0.5.1
export MUJOCO_GL=egl
export LIBERO_ASSETS_PATH=/root/.cache/libero/assets
```

## 1. Prepare MAM dataset

Current generated datasets:

```text
libero_put_bowl_on_plate_mam_train
libero_put_bowl_on_plate_mam_eval
```

Regenerate them from the absolute LIBERO dataset if needed:

```bash
python scripts/convert_libero_absolute_to_mam.py \
  --input-root=/cephfs/shared/Yanbang/lerobot/mam_lerobot0.5.1/lerobot_mam/libero_put_bowl_on_plate_absolute \
  --input-repo-id=local/libero_put_bowl_on_plate_absolute \
  --output-root=/cephfs/shared/Yanbang/lerobot/mam_lerobot0.5.1/lerobot_mam/libero_put_bowl_on_plate_mam \
  --output-repo-id=local/libero_put_bowl_on_plate_mam \
  --mask-type=random_mask \
  --retain-ratio=0.2 \
  --n-obs-steps=2 \
  --horizon=16 \
  --overwrite
```

## 2. Train STPM

For this dataset, STPM uses one optimization step per batch. With
`total_frames=4067`, default `val_ratio=0.1`, and `batch_size=64`, the training
split has `3661` samples, so one epoch is `ceil(3661 / 64) = 58` steps. Two
epochs are `116` steps.

```bash
CUDA_VISIBLE_DEVICES=0 lerobot-train-stpm \
  --dataset.repo_id=local/libero_put_bowl_on_plate_mam_train \
  --dataset.root=/cephfs/shared/Yanbang/lerobot/mam_lerobot0.5.1/lerobot_mam/libero_put_bowl_on_plate_mam_train \
  --output_dir=outputs/train/stpm_libero_put_bowl_on_plate_mam \
  --n_obs_steps=6 \
  --frame_gap=2 \
  --batch_size=64 \
  --num_workers=4 \
  --prefetch_factor=4 \
  --steps=116 \
  --device=cuda \
  --require_cuda \
  --vision_ckpt=/cephfs/shared/Yanbang/maniskill/pretrained/clip-vit-base-patch32 \
  --task_description="put the bowl on the plate"
```

## 3. Train MAM

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl lerobot-train-mam \
  --policy.type=mam \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/libero_put_bowl_on_plate_mam_train \
  --dataset.root=/cephfs/shared/Yanbang/lerobot/mam_lerobot0.5.1/lerobot_mam/libero_put_bowl_on_plate_mam_train \
  --env.type=libero \
  --env.task=libero_goal \
  --env.task_ids='[8]' \
  --env.control_mode=absolute \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --output_dir=outputs/train/mam_libero_put_bowl_on_plate \
  --batch_size=16 \
  --num_workers=8 \
  --prefetch_factor=4 \
  --persistent_workers=true \
  --steps=50000 \
  --save_freq=5000 \
  --eval_freq=5000 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --env.max_parallel_tasks=1 \
  --policy.mam_eval_dataset_repo_id=local/libero_put_bowl_on_plate_mam_eval \
  --policy.mam_eval_dataset_root=/cephfs/shared/Yanbang/lerobot/mam_lerobot0.5.1/lerobot_mam/libero_put_bowl_on_plate_mam_eval \
  --policy.stpm_path=outputs/train/stpm_libero_put_bowl_on_plate_mam \
  --policy.device=cuda
```

Enable MAM inpainting by appending:

```bash
--policy.inpainting=true
```

## 4. Smoke test

Use this before long training if the environment or data changed:

```bash
MUJOCO_GL=egl lerobot-train-mam \
  --policy.type=mam \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/libero_put_bowl_on_plate_mam_train \
  --dataset.root=/cephfs/shared/Yanbang/lerobot/mam_lerobot0.5.1/lerobot_mam/libero_put_bowl_on_plate_mam_train \
  --output_dir=outputs/train/mam_smoke \
  --batch_size=2 \
  --num_workers=0 \
  --steps=1 \
  --save_freq=1 \
  --eval_freq=0 \
  --policy.device=cpu
```

## 5. Local wrapper script

The local convenience wrapper creates the MAM dataset if missing and launches
multi-GPU MAM training:

```bash
bash scripts/run_mam_libero_put_bowl_on_plate_conda.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 BATCH_SIZE=32 STEPS=50000 \
  bash scripts/run_mam_libero_put_bowl_on_plate_conda.sh
```
