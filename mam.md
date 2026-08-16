# MAM / STPM LIBERO-10 实验记录

更新：2026-08-16。

## 公共口径

- 数据动作：原始数据是 OSC pose absolute-goal；MAM 训练目标是 chunk-relative SE(3)。
- 执行口径：`policy.use_relative_actions=true`，`ENV_CONTROL_MODE=absolute`。
- LIBERO-10 默认时序：`n_obs_steps=2`，`horizon=32`，`n_action_steps=15`。
- TopN 口径：按 eval checkpoint 的 overall SR 降序；同分取更早 step；`avg` 是 TopN 平均 SR。

## STPM 版本

| 名称              | 输出前缀                                                                  | 变量                             |
| ----------------- | ------------------------------------------------------------------------- | -------------------------------- |
| STPM v2 baseline  | `stpm_libero10_v2_task{0..9}`                                             | d256 / l2 / h4，6 epochs         |
| STPM v3 large     | `stpm_libero10_v3_large_d512_l4_task{0..9}`                               | d512 / l4 / h8，6 epochs         |
| STPM v4 Maniskill | `stpm_libero10_v4_maniskill_d768_l8_obs6_gap2_seed42_20260729_task{0..9}` | d768 / l8，obs=6，gap=2，seed=42 |
| STPM v5           | `stpm_libero10_v5_d768_l8_obs6_gap2_seed0_6epoch_task{0..9}`              | d768 / l8，obs=6，gap=2，seed=0  |
| STPM v6           | `stpm_libero10_v6_d544_l5_obs6_gap2_seed0_6epoch_task{0..9}`              | d544 / l5，obs=6，gap=2，seed=0  |

## Mask 口径

| 名称                  | Train mask                                                 | Eval mask       |
| --------------------- | ---------------------------------------------------------- | --------------- |
| `random`              | `random_mask`                                              | `random_mask`   |
| `refmix train4/eval5` | `points`, `3D_points`, `3D_points`, `pose_motion_planning` | train4 + `mix0` |

## LIBERO-10 MAM 主实验

| 实验名                                                                                                                                | Train / Eval data                 | Mask                          | STPM         | MAS                  |      BS |           Steps | Top3 SR                              | Top5 SR                                                |
| ------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------- | ------------ | -------------------- | ------: | --------------: | ------------------------------------ | ------------------------------------------------------ |
| `mam_libero10_v3_relative_150k_6gpu_large_stpm_multirankeval_20260726_161112`                                                         | 450 unfiltered / 50 eval          | random                        | v3 large     | short15, long32, d64 | 6×16=96 | stopped @117.6k | 60@100k, 58@75k, 56@115k; avg 58.0   | 60@100k, 58@75k, 56@115k, 54@85k, 54@90k; avg 56.4     |
| `mam_libero10_v3_refmix_150k_6gpu_large_stpm_multirankeval_20260727_183201`                                                           | 450 refmix / 50 refmix            | refmix train4/eval5           | v3 large     | short15, long32, d64 | 6×16=96 |  stopped @85.2k | 52@55k, 44@70k, 42@35k; avg 46.0     | 52@55k, 44@70k, 42@35k, 40@45k, 40@50k; avg 43.6       |
| `mam_libero10_v3_refmix_150k_6gpu_large_stpm_short0_long64_dim128_avgmse_seed1000_cudnnbench_multirankeval_20260728_142023`           | 450 refmix / 50 refmix            | refmix train4/eval5           | v3 large     | short0, long64, d128 | 6×16=96 |            200k | 78@100k, 76@115k, 76@165k; avg 76.7  | 78@100k, 76@115k, 76@165k, 76@185k, 76@190k; avg 76.4  |
| `mam_libero10_v4_refmix_150k_4gpu_maniskill_stpm_d768_l8_obs6gap2_short0_long64_dim128_avgmse_seed1000_multirankeval_20260729_213200` | 450 refmix / 50 refmix            | refmix train4/eval5           | v4 Maniskill | short0, long64, d128 |     4×? |   stopped @1.2k | —                                    | —                                                      |
| `mam_libero10_500train_100eval5ptask_150k_4gpu_maniskill_short0_long64_dim128_avgmse_seed1000_20260803_173414`                        | 500 train / first 50 of 100 eval  | random                        | v4 Maniskill | short0, long64, d128 | 4×12=48 |    stopped @95k | 56@90k, 50@70k, 48@55k; avg 51.3     | 56@90k, 50@70k, 48@55k, 48@80k, 44@60k; avg 49.2       |
| `mam_libero10_refmix_train4_eval5_long64_avgmse_scratch_8gpu_20260805_125120`                                                         | 450 refmix / 50 refmix            | refmix train4/eval5           | v4 Maniskill | short0, long64, d128 | 8×12=96 | stopped @135.2k | 68@125k, 68@130k, 64@110k; avg 66.7  | 68@125k, 68@130k, 64@110k, 64@120k, 64@135k; avg 65.6  |
| `mam_libero10_refmix_train4_eval5_long64_avgmse_resume090000_4gpu_20260805_125120`                                                    | 450 refmix / 50 refmix            | refmix train4/eval5           | v4 Maniskill | short0, long64, d128 | 4×24=96 |            150k | 78@145k, 74@110k, 74@140k; avg 75.3  | 78@145k, 74@110k, 74@140k, 70@105k, 70@125k; avg 73.2  |
| `mam_libero10_500train_100first50eval_refmix_train4_eval5_long64_avgmse_scratch_8gpu_20260806_140235`                                 | 500 refmix / 50 refmix            | refmix train4/eval5           | v4 Maniskill | short0, long64, d128 | 8×12=96 |            150k | 60@120k, 58@150k, 54@125k; avg 57.3  | 60@120k, 58@150k, 54@125k, 52@115k, 52@130k; avg 55.2  |
| `mam_libero10_500train_100first50eval_refmix_long64_avgmse_resume090000to180000_4gpu_20260806_140856`                                 | 500 refmix / 50 refmix            | refmix train4/eval5           | v4 Maniskill | short0, long64, d128 | 4×24=96 |            180k | 66@125k, 64@140k, 64@150k; avg 64.7  | 66@125k, 64@140k, 64@150k, 64@175k, 62@110k; avg 64.0  |
| `mam_libero10_500train_100first50eval_refmix_train4_eval5_long64_avgmse_stpmv3_scratch_8gpu_200k_keep100k_20260807_222122`            | 500 refmix / 50 refmix            | refmix train4/eval5           | v3 large     | short0, long64, d128 | 8×12=96 |            200k | 56@55k, 56@120k, 56@135k; avg 56.0   | 56@55k, 56@120k, 56@135k, 56@155k, 56@170k; avg 56.0   |
| `mam_libero10_500train_100first50eval_refmix_train4_eval5_long64_avgmse_stpmv3_scratch_8gpu_150k_keep100k_20260809_2049_b64_h32_a15`  | 500 refmix / 50 refmix            | refmix train4/eval5           | v3 large     | short0, long64, d128 |  8×8=64 | stopped @144.7k | 56@95k, 56@120k, 50@130k; avg 54.0   | 56@95k, 56@120k, 50@130k, 48@110k, 48@115k; avg 51.6   |
| `mam_libero10_1k_random090k_refmix180k_6a10_20260813_133350`                                                                          | 1000 train / fixed 50 of 100 eval | random 0–90k → refmix 90–180k | v4 Maniskill | short0, long64, d128 | 6×16=96 |            180k | 100@150k, 98@170k, 94@130k; avg 97.3 | 100@150k, 98@170k, 94@130k, 94@135k, 94@145k; avg 96.0 |

## 1k random → refmix 两阶段实验（2026-08-13）

- 实验：`mam_libero10_1k_random090k_refmix180k_6a10_20260813_133350`。
- 硬件：单机 6×NVIDIA A10；每卡 batch 16，全局 batch 96。
- 数据：每个 LIBERO-10 task 100 个训练 episode；固定评测每个 task 5 个 episode，共 50 个。
- Phase 1：从头训练至 90k，train/eval 均使用 `random_mask`。
- Phase 2：从 90k 恢复训练至 180k；训练使用 4-slot refmix，评测使用 5-slot refmix。
- 模型：STPM v4 Maniskill；MAM `horizon=32`、`n_action_steps=15`、short0/long64/d128、average MSE。
- 评测频率 5k，checkpoint 频率 10k；训练正常完成，无 traceback、OOM 或 NCCL 错误。

总体 SR 曲线：

| Phase  | Step (k)                     | SR (%)                  |
| ------ | ---------------------------- | ----------------------- |
| random | 5, 10, 15, 20, 25, 30        | 0, 10, 30, 42, 32, 38   |
| random | 35, 40, 45, 50, 55, 60       | 70, 60, 78, 78, 90, 74  |
| random | 65, 70, 75, 80, 85, 90       | 86, 94, 84, 82, 82, 84  |
| refmix | 95, 100, 105, 110, 115, 120  | 64, 84, 86, 88, 80, 88  |
| refmix | 125, 130, 135, 140, 145, 150 | 86, 94, 94, 90, 94, 100 |
| refmix | 155, 160, 165, 170, 175, 180 | 90, 94, 94, 98, 94, 92  |

Phase 2 Top5 checkpoint 及 5-slot SR（同分取更早 step）：

| Rank | Step | Overall | points | 3D_points r1.0 | 3D_points r0.2 | pose r0.2 | mix0 |
| ---: | ---: | ------: | -----: | -------------: | -------------: | --------: | ---: |
|    1 | 150k |     100 |    100 |            100 |            100 |       100 |  100 |
|    2 | 170k |      98 |    100 |            100 |             90 |       100 |  100 |
|    3 | 130k |      94 |     90 |            100 |             80 |       100 |  100 |
|    4 | 135k |      94 |     90 |            100 |            100 |        90 |   90 |
|    5 | 145k |      94 |     90 |            100 |            100 |        90 |   90 |

- 原始结果位于 `outputs/train/mam_libero10_1k_random090k_refmix180k_6a10_20260813_133350/`。

## 短跑 / 失败 / 无有效 eval 的 MAM run

| 实验名                                                                                                                              | 变量                                                | 状态                             |
| ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | -------------------------------- |
| `mam_libero10_v3_relative_150k_6gpu_multirankeval_20260726_151842`                                                                  | likely random mask, baseline STPM                   | stopped @2.8k，无 eval           |
| `mam_libero10_v3_refmix_150k_6gpu_large_stpm_short0_long64_dim128_multirankeval_20260728_135202`                                    | refmix, v3 large, weighted loss, short0/long64/d128 | stopped @600，无 eval            |
| `mam_libero10_v3_refmix_150k_6gpu_large_stpm_short0_long64_dim128_avgmse_multirankeval_20260728_141120`                             | refmix, v3 large, avg MSE, short0/long64/d128       | stopped @200，无 eval            |
| `mam_libero10_refmix_train4_eval5_long64_avgmse_resume090000_8gpu_20260805_123100`                                                  | refmix, v4 Maniskill, 8GPU resume                   | stopped @92k，无 eval            |
| `mam_libero10_500train_100first50eval_refmix_long64_avgmse_resume090000to200000_8gpu_20260807_191320`                               | 500 refmix, v4 Maniskill, 8GPU resume               | stopped @97.2k，只有 95k eval=52 |
| `mam_libero10_500train_100first50eval_refmix_train4_eval5_long64_avgmse_stpmv3_scratch_8gpu_200k_20260807_220649`                   | 500 refmix, v3 large                                | stopped @1.2k，无 eval           |
| `mam_libero10_500train_100first50eval_refmix_train4_eval5_long64_avgmse_stpmv3_scratch_8gpu_150k_keep100k_20260809_2021`            | 500 refmix, v3 large                                | stopped @600，无 eval            |
| `mam_libero10_500train_100first50eval_refmix_train4_eval5_long64_avgmse_stpmv3_scratch_8gpu_150k_keep100k_20260809_2034_b64_h16_a8` | 500 refmix, v3 large, h16/a8                        | stopped @2k，无 eval             |

## 单任务 MAM 输出

| 实验名                                                                        |              Data | Eval                 |       BS | MAS / action              | Top SR                  |
| ----------------------------------------------------------------------------- | ----------------: | -------------------- | -------: | ------------------------- | ----------------------- |
| `mam_libero_put_bowl_on_plate_80m_multigpu`                                   | 44 train / 5 eval | `libero_goal` task 8 |  4×16=64 | h16/a8, short8/long16/d64 | 80@5k                   |
| `mam_libero_put_bowl_on_plate_80m_multigpu_20260628_004227`                   | 44 train / 5 eval | `libero_goal` task 8 | 4×32=128 | h16/a8, short8/long16/d64 | 100@10k/15k/20k/25k/40k |
| `mam_libero_put_bowl_on_plate_80m_multigpu_20260628_000250` and log-only dirs |           unknown | none                 |  unknown | unknown                   | —                       |
| `mam_long_window_comparison`                                                  |           unknown | none                 |  unknown | comparison artifact       | —                       |

## 当前结论

- 当前固定 MAM 条件评测的最好结果：`mam_libero10_1k_random090k_refmix180k_6a10_20260813_133350`，Top1 `100%@150k`，Top5 avg `96.0`；这不是独立 autonomous test SR。
- 该 run 先用 random mask 预训练 90k，再用 refmix 续训；150k 的五个 eval mask slot 均达到 100% SR。
- 审计未发现 train/eval 完整轨迹直接重合，但当前评测使用同一 eval episode 的 masked expert plan，并在固定 50 条数据上反复选择 checkpoint；100% 存在明显的口径和选择偏差。
- 旧的 450 refmix 最好结果为 Top1 `78%@100k`、Top5 avg `76.4`；1k 两阶段方案显著超过此前结果。
- 旧的 500 refmix 最好 Top1 为 `66%@125k`；此前在 500 refmix 上 v3 large STPM 明显弱于 v4 Maniskill STPM。
- short0 + long64 + average MSE 仍是目前表现最稳定的 MAM 配置。

## Metrics paths

- MAM eval：`outputs/train/<实验名>/logs/eval_metrics.jsonl`
- MAM train：`outputs/train/<实验名>/logs/train_metrics.jsonl`
- checkpoint：`outputs/train/<实验名>/checkpoints/{best,last}`
