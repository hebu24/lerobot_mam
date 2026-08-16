# DP 实验记录

更新：2026-08-10。

## 公共口径

- DP 指 Diffusion Policy；不使用 MAM mask loss，也不使用 STPM。
- LIBERO-10 主配置：`policy.use_relative_actions=true`，`env.control_mode=absolute`，`n_obs_steps=2`，`horizon=32`，`n_action_steps=15`。
- fixed eval：50 条，总计 5/task，使用 eval dataset init states。
- random-env eval：seed 从 100000 起，50 条，总计 5/task，不使用 eval dataset init states。
- TopN：按 checkpoint SR 降序；同分取更早 step；`avg` 是 TopN 平均 SR。

## LIBERO-10 DP 实验

| 实验名                                                                                |           Train data | Eval 方法                        |            BS | Steps / 状态                             | Top3 SR                            | Top5 SR                                              |
| ------------------------------------------------------------------------------------- | -------------------: | -------------------------------- | ------------: | ---------------------------------------- | ---------------------------------- | ---------------------------------------------------- |
| `diffusion_libero10_v3_dp_102`                                                        |   435 条 filtered v3 | fixed eval，50=5/task            |         1×4=4 | target 50k；failed/stopped @1k eval      | —                                  | —                                                    |
| `diffusion_libero10_v3_dp_102_100k_bs64_evalbs2`                                      |   435 条 filtered v3 | fixed eval，50=5/task，eval_bs=2 |       1×64=64 | target 100k；train 到 79.8k，eval 到 75k | 98@75k, 96@45k, 92@50k; avg 95.3   | 98@75k, 96@45k, 92@50k, 92@55k, 90@65k; avg 93.6     |
| `diffusion_libero10_v3_dp_102_100k_bs64_evalbs1`                                      |   435 条 filtered v3 | fixed eval，50=5/task，eval_bs=1 |       1×64=64 | 100k                                     | 100@65k, 96@90k, 96@95k; avg 97.3  | 100@65k, 96@90k, 96@95k, 94@60k, 94@70k; avg 96.0    |
| `diffusion_libero10_v3_hf_100k_bs64_evalbs1_20260721_193304`                          |   435 条 filtered v3 | fixed eval，50=5/task            |       1×64=64 | target 100k；stopped @4.2k               | —                                  | —                                                    |
| `diffusion_libero10_v3_hf_100k_4gpu_effbs64_evalbs1_20260721_201155`                  |   435 条 filtered v3 | rank0-only fixed eval，50=5/task |       4×16=64 | failed @5k eval，NCCL timeout            | —                                  | —                                                    |
| `diffusion_libero10_v3_hf_100k_4gpu_effbs64_noeval_20260721_212245`                   |   435 条 filtered v3 | no eval                          |       4×16=64 | stopped @2.2k                            | —                                  | —                                                    |
| `diffusion_libero10_v3_hf_100k_4gpu_effbs64_autoeval_20260721_213315`                 |   435 条 filtered v3 | auto eval prototype              |       4×16=64 | stopped @1.6k                            | —                                  | —                                                    |
| `diffusion_libero10_v3_hf_100k_4gpu_effbs64_multirankeval_20260721_214055`            |   435 条 filtered v3 | multi-rank fixed eval，50=5/task |       4×16=64 | failed @5k eval，processor init bug      | —                                  | —                                                    |
| `diffusion_libero10_v3_hf_100k_4gpu_effbs64_multirankeval_20260721_220743`            |   435 条 filtered v3 | multi-rank fixed eval，50=5/task |       4×16=64 | 100k                                     | 94@95k, 92@55k, 90@65k; avg 92.0   | 94@95k, 92@55k, 90@65k, 90@80k, 88@75k; avg 90.8     |
| `diffusion_libero10_v3_unfiltered_hf_100k_4gpu_effbs64_multirankeval_20260722_113500` | 450 条 unfiltered v3 | multi-rank fixed eval，50=5/task |       4×16=64 | 100k                                     | 76@70k, 72@75k, 70@85k; avg 72.7   | 76@70k, 72@75k, 70@85k, 68@80k, 68@95k; avg 70.8     |
| `diffusion_libero10_500_randomenv_20k_4gpu_effbs64_20260731_185655`                   |         500 条 train | random-env eval planned          |       4×16=64 | target 20k；stopped @200                 | —                                  | —                                                    |
| `diffusion_libero10_500_randomenv_20k_6gpu_effbs96_20260731_190220`                   |         500 条 train | random-env eval planned          |       6×16=96 | target 20k；log only，stopped ~86        | —                                  | —                                                    |
| `diffusion_libero10_500_randomenv_15k_6gpu_effbs120_20260731_190356`                  |         500 条 train | random-env eval planned          |      6×20=120 | target 15k；stopped @4k                  | —                                  | —                                                    |
| `diffusion_libero10_500_randomenv_150k_6gpu_effbs120_20260731_192539`                 |         500 条 train | random-env eval，50=5/task       | 6×20→4×30=120 | 150k                                     | 64@100k, 62@105k, 58@95k; avg 61.3 | 64@100k, 62@105k, 58@95k, 58@125k, 56@135k; avg 59.6 |

## 单任务 / 调试 DP

| 实验名                                                                            |                                Train data | Eval 方法                                |       BS | Steps / 状态                      | Top3 SR                          | Top5 SR                                          |
| --------------------------------------------------------------------------------- | ----------------------------------------: | ---------------------------------------- | -------: | --------------------------------- | -------------------------------- | ------------------------------------------------ |
| `diffusion_relative_libero_put_bowl_on_plate_multigpu`                            | 49 条 `libero_put_bowl_on_plate_absolute` | `libero_goal` task 8，50 条 eval         | 4×32=128 | train metrics 到 24k / target 50k | 100@2k, 98@16k, 98@22k; avg 98.7 | 100@2k, 98@16k, 98@22k, 96@12k, 96@14k; avg 97.6 |
| `diffusion_libero_put_bowl_on_plate_small`                                        |          49 条 `libero_put_bowl_on_plate` | offline smoke；无有效 sim eval           |  1×64=64 | ckpt 到 7k / target 100k          | —                                | —                                                |
| `diffusion_relative_libero_put_bowl_on_plate_multigpu_conda_assets_20260609_2238` | 49 条 `libero_put_bowl_on_plate_absolute` | task 8 eval videos only；无 eval_metrics | 4×32=128 | ckpt 10；videos 到 30             | —                                | —                                                |
| `diffusion_relative_libero_put_bowl_on_plate_multigpu_v2`                         |                                         — | —                                        |        — | empty / no metrics                | —                                | —                                                |
| `diffusion_relative_libero_put_bowl_on_plate_multigpu.empty_20260609_225744`      |                                         — | —                                        |        — | empty / no metrics                | —                                | —                                                |
| `diffusion_relative_libero_put_bowl_on_plate_multigpu_conda_20260609_2228`        |                                         — | —                                        |        — | empty / no metrics                | —                                | —                                                |
| `diffusion_relative_libero_put_bowl_on_plate_multigpu_conda_live_20260609_2231`   |                                         — | —                                        |        — | empty / no metrics                | —                                | —                                                |

## 简短结论

- 最高 LIBERO-10 DP：`diffusion_libero10_v3_dp_102_100k_bs64_evalbs1`，Top1 `100%@65k`，Top5 avg `96.0`。
- `evalbs2` 旧 run Top1 `98%@75k`，但 eval batch 口径不如 `evalbs1` 严格，结论优先看 `evalbs1`。
- 当前仓库内最高完整 LIBERO-10：filtered v3 4GPU multi-rank，Top1 `94%@95k`，Top5 avg `90.8`。
- unfiltered v3 明显低于 filtered v3：Top1 `76%` vs `94%`。
- 500 条 train + random-env eval 更难：Top1 `64%`，Top5 avg `59.6`，不和 fixed eval 直接横比。

## Metrics 路径

- 当前仓库：`outputs/train/<实验名>/logs/eval_metrics.jsonl`
- 旧 102 VM：`/cephfs/shared/Yanbang/lerobot_mam/outputs/train/<实验名>/logs/eval_metrics.jsonl`
- 仅 log 的 run：`outputs/logs/<实验名>.log`
