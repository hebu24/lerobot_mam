# LIBERO-10 固定 MAM 评测集采集

完整流程：每个 task 从 LPB 的 `test_start_seed=100000` 开始连续使用原生随机环境 seed，
以指定 Diffusion Policy checkpoint
做闭环 absolute-controller rollout；仅保留成功且不超过相应任务训练集最长轨迹的 demo；
保存 reset 后的真实 MuJoCo state，随后生成与原 eval 集格式一致的 MAM v3 fixed eval 数据。

```bash
bash scripts/data_collect/build_libero10_100_eval.sh
```

默认产物：

- 暂存轨迹：`outputs/datasets/libero10_100_rollout_absolute_lpb_staging`（可断点续采）
- 原始成功 rollout：`outputs/datasets/libero10_100_rollout_absolute_lpb`
- 最终固定评测集：`outputs/datasets/libero10_100_eval_lpb`

采集完成并校验后上传：

```bash
UPLOAD=true bash scripts/data_collect/build_libero10_100_eval.sh
```

若原始数据已经生成，只需重新预处理：

```bash
uv run python scripts/libero/data/convert_libero_absolute_to_mam.py \
  --input-root outputs/datasets/libero10_100_rollout_absolute_lpb \
  --input-repo-id local/libero10_100_rollout_absolute_lpb \
  --output-root outputs/datasets/libero10_100_lpb \
  --output-repo-id local/libero10_100_lpb \
  --eval-output-root outputs/datasets/libero10_100_eval_lpb \
  --eval-output-repo-id local/libero10_100_eval_lpb \
  --eval-per-task 10 --only-split eval \
  --eval-mask-types random_mask --retain-ratio 0.2
```

上传目标为 dataset repo `hebu2024/libero10_mam` 下的 `libero10_100_eval_lpb/`；脚本仅使用
本机 Hugging Face 登录态，不读取或打印 token。

## 扩充训练集至 1000 条

以下脚本从每个 task 的原生随机环境 seed 50 开始，用 DP checkpoint 采集 50 条成功且不超过
对应官方轨迹长度上限的 rollout；失败或过长的尝试仅写入日志并继续使用下一个 seed。随后将新数据
物化为 MAM 格式、与 `libero10_500_train` 合并为 `libero10_1000_train`，并在指定时上传到
`hebu2024/libero10_mam/libero10_1000_train/`。

```bash
UPLOAD=true bash scripts/data_collect/build_libero10_1000_train.sh
```

中断后直接用相同命令重跑即可从 staging 目录续采。默认需要可用 CUDA；可通过 `CHECKPOINT`、
`BATCH_SIZE`、`MAX_ATTEMPTS_PER_TASK` 和各输出目录环境变量覆盖配置。上传默认使用宿主机
Hugging Face 登录目录；自定义认证目录可通过 `UPLOAD_HF_HOME` 指定。
