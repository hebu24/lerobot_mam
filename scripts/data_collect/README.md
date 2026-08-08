# LIBERO-10 固定 MAM 评测集采集

完整流程：使用随机环境/策略种子加载 LIBERO 环境，以指定 Diffusion Policy checkpoint
做闭环 absolute-controller rollout；仅保留成功且不超过相应任务训练集最长轨迹的 demo；
随后生成与 `libero10_mam_v3_unfiltered_eval` 一致的 MAM v3 eval 数据。

```bash
bash scripts/data_collect/build_libero10_100_eval.sh
```

默认产物：

- 暂存轨迹：`outputs/datasets/libero10_100_rollout_absolute_staging`（可断点续采）
- 原始成功 rollout：`outputs/datasets/libero10_100_rollout_absolute`
- 最终固定评测集：`outputs/datasets/libero10_100_eval`

采集完成并校验后上传：

```bash
UPLOAD=true bash scripts/data_collect/build_libero10_100_eval.sh
```

若原始数据已经生成，只需重新预处理：

```bash
uv run python scripts/convert_libero_absolute_to_mam.py \
  --input-root outputs/datasets/libero10_100_rollout_absolute \
  --input-repo-id local/libero10_100_rollout_absolute \
  --output-root outputs/datasets/libero10_100 \
  --output-repo-id local/libero10_100 \
  --eval-per-task 10 --only-split eval \
  --eval-mask-types random_mask --retain-ratio 0.2
```

上传目标为 dataset repo `hebu2024/libero10_mam` 下的 `libero10_100_eval/`；脚本仅使用
本机 Hugging Face 登录态，不读取或打印 token。
