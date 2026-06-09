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

### 总结

dp的relative模式基本完成修改和验证，duffusion_relative_overfit测试没有问题。

## MAM model 部署思路

注意原始的MAM模型(maniskill版本)细节可以参考home/hebu/code/ManiSkill/examples/baseline/diffusion_policy以及home/hebu/code/ManiSkill/STPM

核心目标是，尽量复用现有的lerobot统一训练模板，在policies下新增mam_sc和mam两个文件夹，另外，STPM由于是预训练模块，可以放lerobot根目录下的单独文件夹里，有自己专门的训练脚本等。

核心需要解决的问题有一下几个（以mam这条pipeline为例，mam_sc可以看作简化版）：

1. STPM模块迁移

- 相关包依赖的安装
- 数据集格式不同造成的接入训练问题
- 并且需要使用absolute的数据集

2. 数据预处理与MAS的生成

- 原始数据需要使用absolute数据集，增加progress行，并根据mask type生成MAS_absolute
- MAS_absolute通过与首帧前的‘state’中的末端位姿做差获得MAS_relative(注意使用的是state值而非action值，并且角度做差时做合理的转化)
- 最后，在obs(rgb,state,mas)和action输入模型前，对state和action做与dp一样的归一化

3. mask type的相关设计

- single type mask 和 multi type mask等等的设计

4. eval集合设置 

- mam中由于需要额外控制信号，eval时不能随意选择环境种子，而是需要固定的测试集合，我考虑的是，不使用现有的lerobot_train脚本，而是在尽量复用其代码和功能的基础上新建lerobot_train_mam，实现使用固定的训练集和测试集

5. inference和在线eval

- 新增inpainting的可选项
- 在eval时实时加在mas，并在模型输出actionchunk(relative)之后，通过与首帧前末端位姿加和的到actionchunk(absolute)，用绝对的controlmode输入libero进行rollout

请你仔细阅读maniskill库里的源码，思考如何部署到这个库里来，重点关注我提出的这些问题，然后把其他我没有考虑到的部署难道也列出来，写在下面：

### 源码阅读补充：其他部署难点

已重点阅读：

- `examples/baselines/diffusion_policy/train_mam.py`
- `examples/baselines/diffusion_policy/evaluate/evaluate_mam.py`
- `examples/baselines/diffusion_policy/data_preprocess/`
- `examples/baselines/diffusion_policy/utils/{mask,progress,stpm,inpainting}_utils.py`
- `examples/baselines/diffusion_policy/models/{modeling_ditdp,mas_conv1d,mas_conv2d}.py`
- `STPM/{train_STPM.py,maniskill_dataset.py,models/stpm_encoder.py}`

额外需要注意：

1. **数据结构差异比 action 转换更大**
   - ManiSkill MAM 依赖 H5 里的 `mas`、`mask`、`source_episode_id`、`mask_type`、`mask_type_slot`、`state_schema_json` 等字段。
   - LeRobot 数据集目前主要围绕 `observation.*`、`action`、episode metadata 和 stats，需要给 MAS/mask/test split 增加稳定存储和读取路径。

2. **MAS 的归一化语义必须重新定清**
   - ManiSkill 预处理是先把 action 归一化，再对归一化 action 生成 masked MAS。
   - 迁移到 LIBERO relative 方案后，明确为：absolute action -> chunk-relative -> 用 relative action stats 归一化 -> 生成/存储 MAS；否则 MAS、action target、inpainting known action 不在同一空间。

3. **MAM loss 需要显式 action mask**
   - 原实现训练时传 `action_mask`，用 mask-weighted noise MSE。
   - LeRobot diffusion 目前主要只有 padding mask `action_is_pad`，MAM 需要额外保留 MAS mask 对 loss 的影响。

4. **模型不只是 DP 加一个输入**
   - MAM 额外有 long/short MAS window、MAS mask、MasConv1D/2D、可选 DiT/Unet、可选 DINO encoder。
   但为了简化，先使用固定的Unet（设置和之前dp一样），DINOencoder先不加，MAS的设计需要保留
   - `mam` 和 `mam_sc` 独立成 policy config/model/processor，避免把 diffusion policy 分支写得过重。

5. **在线 eval 需要专用 rollout loop**
   - MAM eval 每个 chunk 前要用 STPM 从历史 rgb/state 预测 progress，再按 progress 对齐 MAS window。
   - 这和通用 `lerobot_eval` 的 `select_action` 队列模式不同；还要支持固定 test demo、per-mask 统计、视频/控制误差记录。

6. **STPM 和 LIBERO observation 对齐风险很高**
   - STPM checkpoint 绑定 camera_names、camera order、state_paths、state_norm_json、task_description。
   - LIBERO 的图像 key、HWC/CHW、相机命名、相机位姿、state 展平顺序都要和 STPM 训练时完全一致，否则 progress 会失真。

7. **reset seed 不能直接照搬 ManiSkill**
   - ManiSkill eval 用 reset seed 对齐 demo。
   - LIBERO 应优先用 `libero/init_state_id` 或等价初始状态 metadata；需要保存 train/eval 固定列表，避免随机环境初态破坏 MAS 对齐。

8. **inpainting 必须在归一化 relative chunk 空间内做**
   - 原 inpainting 在 diffusion reverse loop 中覆盖 known action，再最后 denormalize。
   - 新方案要保证 known MAS、模型输出、relative->absolute 转换顺序一致：inpaint normalized relative chunk -> unnormalize -> relative 转 absolute -> 入队执行。

9. **action/state pose 表达要统一**
   - LIBERO absolute action 用 position + axis-angle + gripper。
   - STPM state 里可能是 tcp pose quaternion 或展平后的其他顺序；MAM relative MAS 和 action relative 都必须用同一套 pose 抽取和旋转差计算。

10. **mask 扩展会影响 episode 语义**
    - `one_demo_multi_mask` 会把同一条 source episode 扩成多条不同 mask demo。
    - LeRobot sampler、overfit、eval 子集、stats 计算都要区分 `episode_index` 和 `source_episode_id`，否则可能训练/测试泄漏或统计重复。

11. **checkpoint/Hub 保存需要包含外部依赖**
    - MAM policy checkpoint 不够，还需要 STPM config、STPM ckpt 路径或权重、state norm、mask config、action denorm/relative stats。
    - 如果以后推 Hub，需要定义这些文件的保存和加载规则。

12. **性能开销需要提前控制**
    - 在线 eval 每次 chunk 都跑 STPM/CLIP，会明显慢于普通 DP。
    - 需要批量化 STPM、缓存历史窗口、控制视频保存频率，并确认多进程 dataloader 不会重复打开大 H5/视频资源。

## 部署落盘记录

### MAM V1 代码改动记录

1. 新增 MAM 数据转换脚本

文件：`scripts/convert_libero_absolute_to_mam.py`

- 输入：LIBERO absolute action LeRobot 数据集
- 输出：`*_train`、`*_eval` 两个 MAM LeRobot 数据集
- 新增逐帧字段：
  - `mam.mas_action_absolute`
  - `mam.mas_action_mask`
  - `mam.progress`
- 新增 episode metadata：
  - `source_episode_id`
  - `mask_type`
  - `mask_type_slot`
  - `libero/init_state_id`
- 默认 mask：`random_mask`
- 默认 `retain_ratio=0.2`
- stats 中 `action` 会重算为 LIBERO chunk-relative action stats
- 支持 `--n-obs-steps`、`--horizon` 对齐 MAM policy 的 action chunk 统计

2. 新增 STPM 模块

目录：`src/lerobot/stpm/`

- `modeling.py`
  - `FrozenCLIPEncoder`
  - `RewardTransformer`
- `dataset.py`
  - `FrameLeRobotDataset`
  - 输出 `image_frames`、`state`、`targets`、`lengths`、`task`
- `normalizer.py`
  - `StateNormalizer`
  - `save_state_norm`
  - `load_state_normalizer`
- `encoder.py`
  - `STPMEncoder`
  - 加载 `config.yaml`、`state_norm.json`、`checkpoints/reward_best.pt`
  - eval 时校验 camera names、image shape、state dim、task description

3. 新增 STPM 训练入口

文件：`src/lerobot/scripts/lerobot_train_stpm.py`

入口：

```bash
lerobot-train-stpm
```

训练产物：

```text
config.yaml
state_norm.json
checkpoints/reward_best.pt
checkpoints/reward_final.pt
```

4. 新增 MAM policy

目录：`src/lerobot/policies/mam/`

- `configuration_mam.py`
  - 注册 `@PreTrainedConfig.register_subclass("mam")`
  - 默认使用 U-Net diffusion backbone
  - 新增 MAS/STPM/eval/inpainting 相关配置
- `processor_mam.py`
  - 将 absolute action chunk 转成 chunk-relative action
  - 用 action stats 归一化 MAS window
  - 构造：
    - `mam.mas_long_window`
    - `mam.mas_long_window_mask`
    - `mam.mas_short_window`
    - `mam.mas_short_window_mask`
    - `mam.action_mask`
- `modeling_mam.py`
  - `MamPolicy`
  - `MamDiffusionModel`
  - `MamLongWindowEncoder`
  - 支持 mask-aware diffusion noise MSE
- `eval_mam.py`
  - 专用 MAM online eval
  - 从 eval dataset 读取固定 `libero/init_state_id`
  - 用 STPM 预测 progress
  - 按 progress 取 MAS long/short window
  - policy 输出 normalized relative chunk
  - 可选 inpainting
  - unnormalize 后 relative -> absolute，再输入 LIBERO absolute control
  - 输出 overall success、per-mask-type success、per-mask-slot success

5. 接入 LeRobot 统一入口

改动文件：

- `src/lerobot/policies/factory.py`
  - 支持 `policy.type=mam`
  - 注册 `MamConfig`、`MamPolicy`
  - 注册 MAM pre/post processor
  - 加载已保存 MAM processor 时保留 `mam.*` 字段
- `src/lerobot/policies/__init__.py`
  - 导出 `MamConfig`
- `src/lerobot/datasets/factory.py`
  - 对 `mam.*` 字段支持 `mam_delta_indices`
- `src/lerobot/scripts/lerobot_train.py`
  - 若配置 `policy.mam_eval_dataset_repo_id`，eval 时走 MAM 专用 rollout
  - 否则仍走原通用 eval
- `src/lerobot/scripts/lerobot_train_mam.py`
  - 新入口，复用 `lerobot_train`
- `pyproject.toml`
  - 新增 extra：`mam`
  - 新增入口：
    - `lerobot-train-mam`
    - `lerobot-train-stpm`
- `uv.lock`
  - 已更新

### MAM V1 使用方法

1. 安装依赖

```bash
uv sync --locked --extra mam
```

如果只想继续用已有环境运行，也可以：

```bash
UV_CACHE_DIR=/home/hebu/code/lerobot_mam/.uv-cache uv run ...
```

2. 从 absolute 数据集生成 MAM 数据集

输入数据集应先由 `convert_libero_delta_to_absolute.py` 生成：

```bash
uv run python scripts/convert_libero_absolute_to_mam.py \
  --input-root=outputs/datasets/libero_put_bowl_on_plate_absolute \
  --input-repo-id=local/libero_put_bowl_on_plate_absolute \
  --output-root=outputs/datasets/libero_put_bowl_on_plate_mam \
  --output-repo-id=local/libero_put_bowl_on_plate_mam \
  --mask-type=random_mask \
  --retain-ratio=0.2 \
  --n-obs-steps=2 \
  --horizon=16 \
  --overwrite
```

输出：

```text
outputs/datasets/libero_put_bowl_on_plate_mam_train
outputs/datasets/libero_put_bowl_on_plate_mam_eval
```

对应 repo id：

```text
local/libero_put_bowl_on_plate_mam_train
local/libero_put_bowl_on_plate_mam_eval
```

3. 训练 STPM

强制 CUDA + 真实 CLIP 权重：

```bash
CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/home/hebu/code/lerobot_mam/.uv-cache \
uv run lerobot-train-stpm \
  --dataset.repo_id=local/libero_put_bowl_on_plate_mam_train \
  --dataset.root=outputs/datasets/libero_put_bowl_on_plate_mam_train \
  --output_dir=outputs/train/stpm_libero_put_bowl_on_plate_mam \
  --n_obs_steps=6 \
  --frame_gap=2 \
  --batch_size=64 \
  --num_workers=4 \
  --prefetch_factor=4 \
  --steps=10000 \
  --device=cuda \
  --require_cuda \
  --vision_ckpt=/home/hebu/code/ManiSkill/pretrained/clip-vit-base-patch32 \
  --task_description="put the bowl on the plate"
```

如需用已有 STPM reward ckpt 初始化，在上面追加：

```bash
  --reward_ckpt=/path/to/reward_best.pt
```

如果 ckpt 来自 ManiSkill 等不同 state_dim/task 的模型，只能做部分加载：

```bash
  --reward_ckpt=/home/hebu/code/ManiSkill/STPM_PickCube/checkpoints/reward_best.pt \
  --allow_partial_reward_ckpt \
  --d_model=768 \
  --n_layers=8 \
  --n_heads=12 \
  --lr=5e-5
```

smoke 版本：

```bash
uv run lerobot-train-stpm \
  --dataset.repo_id=local/libero_put_bowl_on_plate_mam_train \
  --dataset.root=outputs/datasets/libero_put_bowl_on_plate_mam_train \
  --output_dir=outputs/train/stpm_smoke \
  --n_obs_steps=2 \
  --frame_gap=1 \
  --batch_size=2 \
  --num_workers=0 \
  --steps=1 \
  --device=cpu \
  --task_description="put the bowl on the plate"
```

4. 训练 MAM

```bash
MUJOCO_GL=egl uv run lerobot-train-mam \
  --policy.type=mam \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/libero_put_bowl_on_plate_mam_train \
  --dataset.root=outputs/datasets/libero_put_bowl_on_plate_mam_train \
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
  --policy.mam_eval_dataset_root=outputs/datasets/libero_put_bowl_on_plate_mam_eval \
  --policy.stpm_path=outputs/train/stpm_libero_put_bowl_on_plate_mam \
  --policy.device=cuda
```

5. MAM 1-step smoke

```bash
MUJOCO_GL=egl uv run lerobot-train-mam \
  --policy.type=mam \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/libero_put_bowl_on_plate_mam_train \
  --dataset.root=outputs/datasets/libero_put_bowl_on_plate_mam_train \
  --output_dir=outputs/train/mam_smoke \
  --batch_size=2 \
  --num_workers=0 \
  --steps=1 \
  --save_freq=1 \
  --eval_freq=0 \
  --policy.device=cpu
```

6. 打开 inpainting

```bash
--policy.inpainting=true
```

inpainting 发生在 normalized relative action chunk 空间内：

```text
policy output normalized relative chunk
-> known MAS 覆盖
-> unnormalize
-> relative 转 absolute
-> env.step
```

7. 不使用 STPM 的 eval fallback

如果没有配置 STPM：

```text
policy.stpm_path=None
policy.stpm_checkpoint_path=None
policy.stpm_config_path=None
```

MAM eval 会用 rollout step ratio 作为 progress fallback，仅用于调试，不建议作为正式结果。
