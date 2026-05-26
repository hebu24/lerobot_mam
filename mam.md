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
  --num_workers=0 \
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
uv run lerobot-train \
  --policy.type=diffusion \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/libero_put_bowl_on_plate \
  --dataset.root=outputs/datasets/libero_put_bowl_on_plate \
  --env.type=libero \
  --env.task=libero_goal \
  --env.task_ids='[8]' \
  --env.control_mode=relative \
  --output_dir=outputs/train/diffusion_libero_put_bowl_on_plate_overfit \
  --batch_size=4 \
  --num_workers=0 \
  --steps=5000 \
  --save_freq=1000 \
  --eval_freq=1000 \
  --overfit_test=true \
  --num_overfit=5 \
  --policy.device=cuda
```
