# Deploy MAM in lerobot

**这是我和codex的交互文档，我在里面写的指令不要动，请你根据我的指示在这个文档的指定位置增改内容，保持语言简洁清晰**

## 核心目标

我需要把之前在home/hebu/code/MASKED-ACTION-MODEL里的模型（原本部署在home/hebu/code/ManiSkill里），部署到lerobot里来。

## 实现1：跑通diffusion policy


仿真环境： `libero` 后端 LIBERO + robosuite + MuJoCo
训练数据：官方数据集 `HuggingFaceVLA/libero`

1. 安装依赖

```bash
uv sync --locked --extra training --extra diffusion --extra libero
```

2. 完整训练命令

```bash
uv run lerobot-train \
  --policy.type=diffusion \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/libero_put_bowl_on_plate \
  --dataset.root=outputs/datasets/libero_put_bowl_on_plate \
  --output_dir=outputs/train/diffusion_libero_put_bowl_on_plate \
  --batch_size=4 \
  --num_workers=4 \
  --prefetch_factor=4 \
  --persistent_workers=true \
  --steps=50000 \
  --save_freq=10000 \
  --eval_freq=0 \
  --policy.device=cuda
```

3. DP_baseline基本信息：

- 数据集action：ee_delta [-1,1]
- 仿真环境：control_mode="relative"
当前真实末端位姿 + 预测 delta → 目标末端位姿 → controller 执行

## 实现2：过拟合测试

新增参数：

- `--overfit_test=true`
- `--num_overfit=5`

训练集会被限制为 `dataset.episodes=[0,1,2,3,4]`
eval 初始环境从 episode metadata 读取 `libero/init_state_id` 并传给 LIBERO env

过拟合测试命令：

```bash
MUJOCO_GL=egl uv run lerobot-train \
  --policy.type=diffusion \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/libero_put_bowl_on_plate \
  --dataset.root=outputs/datasets/libero_put_bowl_on_plate \
  --env.type=libero \
  --env.task=libero_goal \
  --env.task_ids='[8]' \
  --env.control_mode=relative \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --output_dir=outputs/train/diffusion_delta_overfit \
  --batch_size=16 \
  --num_workers=8 \
  --prefetch_factor=4 \
  --persistent_workers=true \
  --steps=10000 \
  --save_freq=2000 \
  --eval_freq=2000 \
  --overfit_test=true \
  --num_overfit=5 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --env.max_parallel_tasks=1 \
  --policy.device=cuda
```

## DP的relative模式

接下来为了对齐后续工作，需把delta模式的diffusion_policy改为relative版

先区分一下重要概念我这里说的relative，指的是**每一个action chunk中absolute目标与起始时间的真实末端pose之差**，而你之前说的relative作为控制模式，**本质上就是我说的delta**

具体实现：

- 数据集使用absolute ee目标（目前想的是用delta数据集在仿真中replay转化）
- 数据预处理阶段，对每个actionchunk，先用absolute action值减actionchunk初始的state中的tcppose，得到relative action chunk，再进行归一化
- 训练模型预测relative action chunk
- 后处理阶段，进行反归一化，并在rollout阶段，读取执行某个action chunk前一刻的真实tcppose，并用此将relative转化为absolute action chunk
- 用libero中的‘absolute’这一control mode进行控制


### 可行性分析

1. LIBERO数据集是delta，需replay成absolute ee target
2. relative action 的归一化 stats 须按 relative chunk 重新计算
3. Diffusion Policy 内部会一次预测 `n_action_steps` 并缓存 action queue，relative→absolute 必须在“整段 chunk 入队前”完成

LIBERO absolute control 的动作格式是：

- `action[:3]`：世界坐标系 absolute eef position，不乘 `0.05`
- `action[3:6]`：世界坐标系 absolute eef orientation 的 axis-angle，robosuite 内部会 `axisangle2quat`
**注意计算relative时，先转成旋转矩阵，变为relative，再转回axis-angle存储**
  - relative：`R_rel = R_abs @ R_start.T`
  - absolute：`R_abs = R_rel @ R_start`
- `action[6]`：gripper，二值化后的连续值（-1open）


### 代码改动方案

1. 新增数据转换脚本

文件：`scripts/convert_libero_delta_to_absolute.py`

- 输入：`outputs/datasets/libero_put_bowl_on_plate`
- 输出：`outputs/datasets/libero_put_bowl_on_plate_absolute`
- 对每个 episode 用 `libero/init_state_id` reset 环境
- 用原 delta action 在 `control_mode=relative` 下 replay
- 每一步在 step 前读取真实 eef pose，按 robosuite 规则算 absolute target：
  - `target_pos = current_pos + delta[:3] * 0.05`
  - `target_rot = axisangle(delta[3:6] * 0.5) @ current_rot`
  - `target_axisangle = mat_to_axisangle(target_rot)`
  - `target_gripper = delta[6]`
- 将 dataset 的 `action` 替换为 `[target_pos, target_axisangle, target_gripper]`
- 保留 `libero/init_state_id`，并验证 replay 的 state 和数据集 state 误差

2. 新增 LIBERO pose action 工具函数

文件建议：`src/lerobot/processor/libero_relative_action_processor.py`

核心函数：

- `delta_to_absolute_action(delta, eef_pos, eef_mat)`
- `absolute_to_chunk_relative(abs_actions, anchor_state)`
- `chunk_relative_to_absolute(rel_actions, anchor_state)`

只转换 action 的前 6 维，gripper 不做相对化

3. 给 DiffusionConfig 加开关

文件：`src/lerobot/policies/diffusion/configuration_diffusion.py`

新增字段：

```python
use_relative_actions: bool = False
```
不设置relative_action_rotation参数，直接用旋转矩阵做差法
不设置relative_action_anchor_index，直接取-1

4. 修改 Diffusion preprocessor

文件：`src/lerobot/policies/diffusion/processor_diffusion.py`

顺序改为：

```text
rename -> batch -> device -> chunk_relative_action -> normalize
```

其中 `chunk_relative_action` 用 `observation.state[:, -1, :6]` 作为整段 action chunk 的 anchor，将 absolute action chunk 转成 relative chunk。

5. 重新计算 relative action stats

文件：`src/lerobot/datasets/compute_stats.py`

当前 `compute_relative_action_stats()` 只是简单 `action - state`，需要新增 LIBERO pose 版本：

- anchor 用 action chunk 对应样本的 `observation.state`
- 前 3 维做位置差
- 3:6 维做旋转相对组合
- 第 6 维 gripper 保持原样

训练 `use_relative_actions=true` 时，`dataset.meta.stats["action"]` 必须替换为这个 relative chunk stats。

6. 修改 Diffusion rollout 的 chunk 后处理

文件：`src/lerobot/scripts/lerobot_eval.py` 或新增 DP 专用 chunk 推理路径

正确流程应为：

```text
当前 obs -> preprocessor
如果 action_queue 为空：
    policy 预测 normalized relative chunk
    postprocessor 对整个 chunk 反归一化
    用当前真实 pose 把整个 relative chunk 转 absolute chunk
    absolute chunk 入队
每步从 absolute action_queue pop 一个 action -> env.step
```

不要用现有 per-step `AbsoluteActionsProcessorStep` 直接接 diffusion

### 正式改动记录

1. 新增 `scripts/convert_libero_delta_to_absolute.py`
   - 默认用 LIBERO replay 将 delta action 转为 absolute eef target
   - 支持 `--no-replay` 用数据集 state 直接转换
   - 转换后会重算 dataset stats

2. 新增 `src/lerobot/processor/libero_relative_action_processor.py`
   - `delta_to_absolute_action`
   - `absolute_to_chunk_relative`
   - `chunk_relative_to_absolute`
   - 位置做相减/相加，旋转用矩阵组合，gripper 保持原值

3. Diffusion 新增：
   - `--policy.use_relative_actions=true`
   - 训练 preprocessor 在 normalize 前把 absolute action chunk 转成 relative action chunk
   - 训练时自动计算 LIBERO chunk-relative action stats，并替换 `dataset.meta.stats["action"]`
   - eval rollout 在整段 chunk 反归一化后，用当前真实 pose 转回 absolute chunk，再入队执行

### 新bash order

先生成 absolute action 数据集：

```bash
uv run python scripts/convert_libero_delta_to_absolute.py \
  --input-root=outputs/datasets/libero_put_bowl_on_plate \
  --output-root=outputs/datasets/libero_put_bowl_on_plate_absolute \
  --output-repo-id=local/libero_put_bowl_on_plate_absolute \
  --task=libero_goal \
  --task-id=8 \
  --observation-height=256 \
  --observation-width=256 \
  --overwrite
```

过拟合测试：

```bash
MUJOCO_GL=egl uv run lerobot-train \
  --policy.type=diffusion \
  --policy.push_to_hub=false \
  --policy.use_relative_actions=true \
  --dataset.repo_id=local/libero_put_bowl_on_plate_absolute \
  --dataset.root=outputs/datasets/libero_put_bowl_on_plate_absolute \
  --env.type=libero \
  --env.task=libero_goal \
  --env.task_ids='[8]' \
  --env.control_mode=absolute \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --output_dir=outputs/train/diffusion_relative_overfit \
  --batch_size=18 \
  --num_workers=8 \
  --prefetch_factor=4 \
  --persistent_workers=true \
  --steps=10000 \
  --save_freq=1000 \
  --eval_freq=1000 \
  --overfit_test=true \
  --num_overfit=5 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --env.max_parallel_tasks=1 \
  --policy.device=cuda
```
