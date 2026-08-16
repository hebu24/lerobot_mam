# Franka Real 数据与训练

## 数据集

### 下载来源与本地位置

- Hugging Face 数据集：[`hebu2024/franka_mam`](https://huggingface.co/datasets/hebu2024/franka_mam)
- 下载时间：2026-07-28；数据集目录于 14:33 创建，H5 于 14:55 写入完成。
- 本地目录：`/cephfs/shared/Yanbang/maniskill/franka_train/data/franka_mam`
- 原始 H5：`/cephfs/shared/Yanbang/maniskill/franka_train/data/franka_mam/data/pick_up_front_raw.h5`
- 文件大小：`1,318,253,966` bytes，约 1.23 GiB（`du -h` 显示约 1.3G）。
- SHA256：`6fa373bce92a0a68e098d2dc7ffeb16ed54f5f3a096f2d90ed84274546b91c0d`

已在本地重新计算 SHA256，结果与数据集 README 声明的校验值完全一致，
因此 H5 文件完整可用：

```bash
sha256sum \
  /cephfs/shared/Yanbang/maniskill/franka_train/data/franka_mam/data/pick_up_front_raw.h5
```

Hugging Face 下载缓存中残留了一个 0 字节 `.incomplete` 文件：

```text
/cephfs/shared/Yanbang/maniskill/franka_train/data/franka_mam/.cache/huggingface/download/data/s4qq-JyTZB82U9LlvRnkCoGDM3w=.6fa373bce92a0a68e098d2dc7ffeb16ed54f5f3a096f2d90ed84274546b91c0d.incomplete
```

这是残留的缓存标记，不代表实际 H5 损坏；正式 H5 的大小和 SHA256 均已
验证正确。训练应使用 `data/pick_up_front_raw.h5`，不要使用缓存目录中的
文件。

### 数据概要

| 项目              | 内容                                        |
| ----------------- | ------------------------------------------- |
| 机器人            | Franka Research 3（FR3）                    |
| 任务              | `pick_up_front`                             |
| 轨迹数            | 50                                          |
| Observation 数    | 10,386                                      |
| Action 数         | 10,336                                      |
| 频率              | 约 15 Hz                                    |
| 控制模式          | `pd_ee_pose`                                |
| Observation state | `qpos(9) + qvel(9) + tcp_pose(7)`，共 25 维 |
| 图像              | `base_camera`、`hand_camera` 两路 RGB       |
| 图像规格          | `uint8`，256×256                            |
| Action            | 7 维绝对末端位姿与夹爪命令                  |
| Action 坐标系     | FR3 base frame                              |

两路原始相机分别是外部相机 `third_rs` 和腕部相机 `wrist_zed`，处理后
对应 `base_camera` 和 `hand_camera`。所有轨迹采集开始阶段约 1 秒的抖动
已统一删除。数据中没有 `success`、`terminated` 或 `truncated` 标记。

Action 的 7 个维度依次表示：

```text
[x, y, z, roll, pitch, yaw, gripper]
```

其中位置和姿态是 FR3 base frame 下的绝对目标；姿态采用 XYZ 欧拉角，
单位为弧度；夹爪命令使用 `+1`（打开）和 `-1`（关闭）。

### H5 主要字段

每条轨迹命名为 `traj_N`，主要字段如下：

```text
traj_N/actions
traj_N/obs/agent/qpos
traj_N/obs/agent/qvel
traj_N/obs/extra/tcp_pose
traj_N/obs/sensor_data/base_camera/rgb
traj_N/obs/sensor_data/hand_camera/rgb
traj_N/obs_timestamps
traj_N/action_timestamps
traj_N/base_camera_timestamps
traj_N/hand_camera_timestamps
```

读取示例：

```python
import h5py

path = (
    "/cephfs/shared/Yanbang/maniskill/franka_train/data/"
    "franka_mam/data/pick_up_front_raw.h5"
)

with h5py.File(path, "r") as dataset:
    actions = dataset["traj_0/actions"][:]
    qpos = dataset["traj_0/obs/agent/qpos"][:]
    base_rgb = dataset["traj_0/obs/sensor_data/base_camera/rgb"][:]
```

数据目录内还包含以下说明：

- `README.md`：数据集概览和读取示例。
- `docs/franka_real_data_format.md`：完整字段、坐标系和角度定义。
- `docs/franka_pick_up_front_preprocess_plan.md`：预处理方案。
- `docs/franka_pick_up_front_preprocess_report.md`：实际处理结果和校验报告。

## 4 卡训练命令

训练固定在 91（`root@10.233.66.222`）的 4 张 A10 上运行，命令必须从
ManiSkill 仓库执行。下面将每卡 batch 设为 32，因此 4 卡的 global batch
为 128。本实验同时使用 `base_camera` 和 `hand_camera`：每路相机分别
使用一个独立的 DINO2 vision encoder，两路 256 维特征拼接后投影为策略
使用的 256 维视觉特征，两个 encoder 不共享参数。训练共运行 100,000
step，从 30,000 step 开始保存 checkpoint，之后每 10,000 step 保存一次。

```bash
ssh root@10.233.66.222
cd /cephfs/shared/Yanbang/maniskill

MASTER_PORT=29511 \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHON_BIN="/cephfs/shared/Yanbang/envs/maniskill_py311/bin/python" \
EXP_NAME="FrankaReal_pick_up_front_dualcam_dualdino_AUG_unet_dino2_obs2" \
SEED=1 \
TORCH_DETERMINISTIC=true \
VISION_ENCODER="dino2" \
DINO_MODEL_PATH="Dino/dinov2-small" \
DINO_DATA_AUG=true \
RAW_DEMO_H5="franka_train/data/franka_mam/data/pick_up_front_raw.h5" \
PREPROCESSED_H5="franka_train/data/franka_mam/pick_up_front_baseline_train.h5" \
RUN_PREPROCESS=false \
OVERWRITE_PREPROCESS=false \
ENV_ID="FrankaReal-v1" \
OBS_MODE="rgb" \
CONTROL_MODE="pd_ee_pose" \
NOISE_MODEL="Unet" \
TOTAL_ITERS=100000 \
BATCH_SIZE=32 \
LR=5e-5 \
OBS_HORIZON=2 \
ACT_HORIZON=8 \
PRED_HORIZON=16 \
NUM_DATALOAD_WORKERS=2 \
EVAL_FREQ=0 \
NUM_EVAL_DEMOS=0 \
NUM_EVAL_EPISODES=0 \
NUM_EVAL_ENVS=0 \
CAPTURE_VIDEO=false \
TRACK=false \
ACTION_ROBUST_MARGIN=0 \
STATE_ROBUST_MARGIN=0 \
SAVE_START_ITER=30000 \
SAVE_FREQ=10000 \
UNET_DIMS="256 512 1024" \
bash franka_train/run_train_baseline_franka_multigpu.sh
```

训练输出位于：

```text
/cephfs/shared/Yanbang/maniskill/runs/FrankaReal_pick_up_front_dualcam_dualdino_AUG_unet_dino2_obs2/
```

checkpoint 位于该目录的 `checkpoints/` 中，最终模型为 `latest.pt`。

## 多卡参数说明

- `BATCH_SIZE` 是每张 GPU 的 batch，不是所有 GPU 的总 batch。
- 4 卡时：`NPROC_PER_NODE=4`、`BATCH_SIZE=32`，global batch = 128。
- `CUDA_VISIBLE_DEVICES=0,1,2,3` 明确限定只使用 91 上的 4 张 A10。
- 显式指定 `PYTHON_BIN`，避免脚本回退到系统 Python 或其他机器上的旧环境。
- `TOTAL_ITERS=100000`；从 `SAVE_START_ITER=30000` 开始保存，并按
  `SAVE_FREQ=10000` 的间隔继续保存。
- 双相机输入按 `base_camera`、`hand_camera` 顺序拼接为 6 通道；模型内部
  按每 3 通道拆分，每路分别经过一个独立 DINO2 encoder，再融合两路特征。
- 双相机预处理文件为
  `franka_train/data/franka_mam/pick_up_front_baseline_train.h5`。
- 6 卡时若希望近似保持 global batch 128，可使用
  `NPROC_PER_NODE=6`、`BATCH_SIZE=21`，global batch = 126。
- 如果按参考命令使用 `BATCH_SIZE=128`，4 卡 global batch 会变成 512，
  显存占用也会明显增大。
- 该脚本使用单机 `torchrun --standalone`，不会读取 DeepSpeed hostfile，
  因此一条命令只使用当前 VM 的 GPU，不能自动跨两台 VM。
- 当前训练脚本没有启用 AMP/BF16，训练精度为 FP32。
- `EXP_NAME` 使用 `obs2`，与实际的 `OBS_HORIZON=2` 保持一致。

## 当前训练状态

- 启动时间：2026-07-30 14:18（Asia/Shanghai）
- 训练机器：91（`root@10.233.66.222`），4×A10
- 模型：双相机、两个独立 DINO2 vision encoder、Unet
- tmux：`franka_real_dualcam_100k`
- 日志：`/cephfs/shared/Yanbang/maniskill/logs/franka_real_dualcam_100k_20260730_141857.log`
- 输出：`/cephfs/shared/Yanbang/maniskill/runs/FrankaReal_pick_up_front_dualcam_dualdino_AUG_unet_dino2_obs2`

## 4 卡单机 DeepSpeed hostfile 训练（2026-08-07）

- 节点：`root@10.233.75.170`，4×A10，global rank 0–3。
- hostfile：`configs/deepspeed/franka_dp_4gpu.hostfile`。
- 模型仍使用已有 PyTorch DDP；DeepSpeed 仅负责启动 4 个本地进程。
- 训练与动作控制时间基准：15 Hz（每步约 66.67 ms）。
- 参数：`obs_horizon=2`、`act_horizon=8`、`pred_horizon=16`。
- 8 步动作块覆盖约 0.533 秒，模型重规划频率约 1.875 Hz；16 步预测覆盖约 1.067 秒。
- 每卡 batch 32，global batch 128；学习率 `5e-5`；总计 100,000 step。
- checkpoint 从 30,000 step 开始，每 10,000 step 保存一次。
- 数据时间戳和 15 Hz metadata 会在每个训练 worker 启动前强制校验。

启动入口：

```bash
bash scripts/run_dp_franka_real_4gpu_after_idle.sh
```

原 12 卡等待队列已取消。用户于 2026-08-07 20:37 要求停止原 4 卡 MAM
任务并立即开始 DP；原任务已正常退出，DP 于 20:38 启动：

```text
tmux: franka_dp_15hz_4gpu_20260807_203034
controller log: outputs/logs/franka_dp_15hz_4gpu_20260807_203034.controller.log
node log: outputs/logs/FrankaReal_pick_up_front_dualcam_dualdino_aug_unet_dp_15hz_obs2_act8_pred16_4gpu_20260807_203034.node0.log
experiment: FrankaReal_pick_up_front_dualcam_dualdino_aug_unet_dp_15hz_obs2_act8_pred16_4gpu_20260807_203034
output: /cephfs/shared/Yanbang/maniskill/runs/FrankaReal_pick_up_front_dualcam_dualdino_aug_unet_dp_15hz_obs2_act8_pred16_4gpu_20260807_203034
```

启动校验：4 个 rank 均正常，15 Hz 时间戳校验通过；初始吞吐约 2.65 step/s，
每张 A10 使用约 4.3 GiB 显存。

## StarVLA VM4A Diffusion Policy 配置要求（2026-08-08）

官方说明：

- [VM4A 文档](https://github.com/starVLA/starVLA/blob/starVLA_dev/docs/VM4A.md)
- [Realman Diffusion Policy 参考配置](https://github.com/starVLA/starVLA/blob/starVLA_dev/examples/realRobots/Realman/train_files/train_realman_dp.yaml)
- [StarVLA 数据与训练指南](https://github.com/starVLA/starVLA/blob/starVLA_dev/docs/starVLA_guideline.md)

StarVLA 的 VM4A Diffusion Policy 是不含 VLM/world model 的轻量视觉运动策略，
输入为 RGB 相机、proprioceptive state 和训练时的未来 action，模型由独立的
ResNet-18 visual encoder 与 conditional 1D U-Net 组成。该路线不需要下载 Qwen
等 VLM checkpoint；`pretrained_backbone=true` 时只需要 ImageNet 预训练的
ResNet-18 权重。

### 当前 Franka 数据不能直接启动 StarVLA 正式训练

当前下载的数据位于：

```text
/cephfs/shared/Yanbang/maniskill/franka_train/data/franka_mam
```

该目录目前只有原始 H5 数据，不是 StarVLA dataloader 所需的 LeRobot 数据集。
正式训练前需要解决以下四个问题：

1. 将原始 H5 转换成 LeRobot 格式，并生成 parquet、视频和完整 metadata。
2. 原始 action 是绝对 EEF 目标，但现有 StarVLA 配置将其声明为
   `delta_eef`，action 语义不一致。
3. 原始 state 是 25 维 `qpos(9)+qvel(9)+tcp_pose(7)`，现有配置却按
   6 维读取；直接读取 `observation.state[0:6]` 得到的是前六个 qpos，
   不是 EEF pose。
4. 现有 `observation_indices=[0]` 只加载当前帧。即使模型设置
   `n_obs_steps=2`，VM4A adapter 也只会复制这一帧，不是真正的两帧观测。
   此外，当前 gripper 使用 `binary` normalization，会把原始 `-1/+1`
   语义变成 `0/1`。

### 1. LeRobot 数据目录要求

转换后的数据集建议使用兼容性更稳妥的 LeRobot v2.1 目录结构：

```text
franka_mam_lerobot/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       ├── observation.images.base_view/
│       └── observation.images.ego_view/
└── meta/
    ├── info.json
    ├── modality.json
    ├── tasks.jsonl
    ├── episodes.jsonl
    └── episodes_stats.jsonl
```

转换时必须保持：

```text
obs[t] --执行 action[t]--> obs[t+1]
```

- 控制与采样频率为 15 Hz。
- 两路相机顺序固定为 `base_camera -> base_view`、
  `hand_camera -> ego_view`。
- 每个 episode 必须有稳定的任务文本或 task index。
- episode 边界、parquet 行范围和视频时间范围必须一致。
- 图像可由原始 256×256 RGB 在 dataloader 中 resize 到 224×224。

### 2. Action 语义

当前 H5 的 7 维 action 为：

```text
[absolute_x, absolute_y, absolute_z,
 absolute_rx, absolute_ry, absolute_rz,
 gripper]
```

前 3 维是 FR3 base frame 下的绝对 TCP 目标位置；中间 3 维是绝对 XYZ
Euler 目标姿态，单位为弧度，约定为 `R=Rx@Ry@Rz`；gripper 为
`-1=闭合、+1=张开`。因此推荐保留绝对 action：

```yaml
framework:
  action_encoding: absolute_eef
  action_dim: 7

datasets:
  vla_data:
    action_mode: abs
```

`action_encoding` 在当前 VM4A DP 实现中主要用于描述/记录 action 类型；
真正决定 dataloader 是否做差分的是 `datasets.vla_data.action_mode`。

如果要改为 delta EEF，建议在 H5 -> LeRobot 转换阶段正确计算平移和旋转增量，
并把已转换的 action 继续设置为 `action_mode: abs`，避免 dataloader 再做一次差分。
不能对绝对 XYZ Euler 或四元数直接做普通逐元素相减后当作旋转增量。

### 3. State 与 `modality.json`

为了与已有 ManiSkill DP 实验公平对比，推荐保留完整 25 维 state：

```text
state[0:9]   = qpos
state[9:18]  = qvel
state[18:25] = tcp_pose = [x,y,z,qw,qx,qy,qz]
```

推荐的 `meta/modality.json` 内容为：

```json
{
  "state": {
    "qpos": { "start": 0, "end": 9, "original_key": "observation.state" },
    "qvel": { "start": 9, "end": 18, "original_key": "observation.state" },
    "tcp_position": {
      "start": 18,
      "end": 21,
      "original_key": "observation.state"
    },
    "tcp_quaternion": {
      "start": 21,
      "end": 25,
      "original_key": "observation.state"
    }
  },
  "action": {
    "eef_position": { "start": 0, "end": 3, "original_key": "action" },
    "eef_euler": { "start": 3, "end": 6, "original_key": "action" },
    "gripper": { "start": 6, "end": 7, "original_key": "action" }
  },
  "video": {
    "base_view": { "original_key": "observation.images.base_view" },
    "ego_view": { "original_key": "observation.images.ego_view" }
  },
  "annotation": {
    "human.action.task_description": { "original_key": "task_index" }
  }
}
```

也可以只使用 6 维 TCP state，但必须在数据转换时显式生成
`[tcp_x,tcp_y,tcp_z,tcp_rx,tcp_ry,tcp_rz]`，并在真机推理端生成完全相同
的表示；不能直接切原 25 维 state 的前 6 维。

### 4. DataConfig

推荐在 Franka `data_registry/data_config.py` 中注册一套专用的 absolute EEF
DP config：

```python
video_keys = ["video.base_view", "video.ego_view"]

state_keys = [
    "state.qpos",
    "state.qvel",
    "state.tcp_position",
    "state.tcp_quaternion",
]
state_key_dims = {
    "state.qpos": 9,
    "state.qvel": 9,
    "state.tcp_position": 3,
    "state.tcp_quaternion": 4,
}

action_keys = [
    "action.eef_position",
    "action.eef_euler",
    "action.gripper",
]
action_key_dims = {
    "action.eef_position": 3,
    "action.eef_euler": 3,
    "action.gripper": 1,
}

language_keys = ["annotation.human.action.task_description"]

# 15 Hz 下的真实连续两帧观测；顺序为前一帧、当前帧。
observation_indices = [-1, 0]
state_indices = [-1, 0]
language_indices = [0]

# action[t] ... action[t+15]，必须等于 framework.horizon。
action_indices = list(range(16))
```

`modality_config()` 应分别给 video/state/action/language 使用上述 indices，
其中 language 只读取当前时间点。连续的 state 和 action 可以使用
`min_max` 或 `mean_std` normalization。为了与当前绝对动作预处理一致，
推荐全部使用 `min_max`，包括 gripper；不要对 `-1/+1` gripper 使用
`binary`。

数据集还需要写入三项注册表：

```python
ROBOT_TYPE_CONFIG_MAP = {
    "franka_abs_eef_dp": FrankaAbsoluteEefDPDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {}

DATASET_NAMED_MIXTURES = {
    "franka_dp": [
        ("franka_mam_lerobot", 1.0, "franka_abs_eef_dp"),
    ],
}
```

`data_root_dir` 必须是 `franka_mam_lerobot` 的父目录，`data_mix` 必须与
`DATASET_NAMED_MIXTURES` 中的 key 一致。

### 5. 推荐的 VM4A DP YAML

与现有 15 Hz、obs 2、action 8、prediction 16 实验对齐的核心设置为：

```yaml
run_id: franka_vm4a_dp_abs_eef_15hz_obs2_act8_pred16
run_root_dir: results/Checkpoints
seed: 42
wandb_entity: disabled
wandb_project: starVLA_franka_dp
is_debug: false
version_id: "0.21"

framework:
  name: DiffusionPolicy
  action_encoding: absolute_eef
  action_dim: 7
  state_dim: 25
  horizon: 16
  n_obs_steps: 2
  n_action_steps: 8
  image_size: [224, 224]
  image_keys: [base_view, ego_view]
  pretrained_backbone: true
  num_train_timesteps: 100
  # num_inference_steps: 20  # 可选；正式部署前实测时延和效果

datasets:
  # VM4A train_starvla.py 只消费 vla_data；保留 vlm_data 仅用于配置结构兼容。
  vlm_data:
    dataset_py: vlm_datasets
    dataformat: llava_json
    dataset_use: unused
    eval_dataset: unused
    data_flatten: false
    base_interval: 2
    max_pixels: 307200
    min_pixels: 784
    model_max_length: 2048
    model_type: qwen2.5vl
    per_device_batch_size: 1

  vla_data:
    dataset_py: lerobot_datasets
    include_state: true
    data_root_dir: /path/to/lerobot_parent
    data_mix: franka_dp
    action_mode: abs
    default_prompt: "pick up the target object"
    per_device_batch_size: 8
    load_all_data_for_training: true
    obs_image_size: [224, 224]
    video_backend: torchvision_av
    num_workers: 4
    pin_memory: true
    persistent_workers: true

trainer:
  max_train_steps: 100000
  num_warmup_steps: 500
  save_interval: 10000
  eval_interval: 1000
  logging_frequency: 50

  learning_rate:
    base: 2.5e-05
    action_model: 1.0e-04

  lr_scheduler_type: cosine_with_min_lr
  scheduler_specific_kwargs:
    min_lr: 1.0e-06

  freeze_modules: ""
  loss_scale:
    vla: 1.0
    vlm: 0.1
  max_grad_norm: 1.0
  weight_decay: 0.0
  gradient_clipping: 1.0
  gradient_accumulation_steps: 1
  gradient_checkpointing: true

  optimizer:
    name: AdamW
    betas: [0.9, 0.95]
    eps: 1.0e-08
    weight_decay: 1.0e-08
```

注意：官方 Realman 配置中的 `max_train_steps=5000`、`save_interval=2500`
只是 smoke-training 默认值，不是正式 Franka 训练的固定要求。`eval_interval`
执行的是当前 batch 上的 action MSE 检查，不是真机 success-rate 评估；真实效果仍需
固定初始条件和统一真机协议进行测试。

### 6. 训练与推理时间设置

- `horizon=16`：在 15 Hz 下覆盖约 1.067 秒。
- `n_action_steps=8`：每次返回 8 步，在 15 Hz 下覆盖约 0.533 秒。
- `n_obs_steps=2`：真实读取 `[-1,0]` 两个连续观测点，覆盖约 66.7 ms
  的历史。
- `num_train_timesteps=100`：DDPM 训练噪声调度长度。
- `num_inference_steps`：可选的推理去噪步数。为严格比较可先使用 100；如果
  15 Hz 部署时延不足，可从 20 开始测试，但必须同时比较真机成功率。
- 模型每次预测 8 步动作，因此不是每个环境控制 tick 都重新运行一次网络；
  规划周期约为 0.533 秒，约 1.875 Hz。

### 7. 单机多卡启动

在数据转换、`modality.json`、DataConfig 和 YAML 全部验证后，可从 StarVLA
仓库启动：

```bash
conda activate /cephfs/shared/Yanbang/envs/starvla
cd /cephfs/shared/Yanbang/starvla

WANDB_MODE=disabled \
DATA_ROOT_DIR=/path/to/lerobot_parent \
DATASET_NAME=franka_mam_lerobot \
NUM_PROCESSES=4 \
BATCH_SIZE=8 \
MAX_STEPS=100000 \
RUN_ID=franka_vm4a_dp_abs_eef_15hz_obs2_act8_pred16_4gpu \
bash examples/realRobots/Franka/train_files/run_franka_train_dp.sh
```

该脚本使用 `accelerate launch` 和 DeepSpeed ZeRO-2。单机 4 卡时需要确认：

```text
--num_processes = 4
num_machines = 1
CUDA_VISIBLE_DEVICES = 实际使用的 4 张卡
global batch = per_device_batch_size × 4 × gradient_accumulation_steps
```

checkpoint 默认位于：

```text
/cephfs/shared/Yanbang/starvla/results/Checkpoints/<RUN_ID>/checkpoints/
```

训练结束后的最终模型位于：

```text
/cephfs/shared/Yanbang/starvla/results/Checkpoints/<RUN_ID>/final_model/pytorch_model.pt
```

正式启动前应先做 100～500 step smoke test，并检查：

1. dataloader 输出 image 为两路、每路两个连续时间点；
2. state shape 为 `[2,25]`；
3. action shape 为 `[16,7]`；
4. action 前六维仍是绝对 EEF 目标，gripper 反归一化后仍为 `-1/+1`；
5. `dataset_statistics.json` 正常生成；
6. checkpoint 同时包含 raw action model 和 `ema_averaged.*` 权重；
7. 真机推理端使用完全相同的 state 顺序、相机顺序、resize、action 语义、
   控制频率和 action chunk 执行规则。
