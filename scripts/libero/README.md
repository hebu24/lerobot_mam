# LIBERO 脚本

- `data/`：LIBERO 数据集转换、物化和过拟合数据准备。
- `audit/`：数据集一致性审计、动作 oracle 验证与回放成功率检查。
- `train/`：Diffusion、MAM 与 STPM 的训练启动脚本。

从仓库根目录执行脚本，例如：

```bash
bash scripts/libero/train/run_diffusion_libero10.sh
```
