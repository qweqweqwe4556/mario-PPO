# Mario RL

用 `gym-super-mario-bros + Stable-Baselines3 PPO` 训练 AI 玩《Super Mario Bros.》。当前项目聚焦在 `SuperMarioBros-1-1-v0`，已经完成环境封装、训练、评估、图形化观察和断点续训脚本。

> 当前状态：模型还没有稳定通关。现有最好版本在随机策略评估下可以推进到约 `x_pos=1800~1950`，但尚未拿到终点旗帜。

## 项目功能

- 基于 `gym-super-mario-bros` 创建 Mario 环境
- 使用 PPO 进行强化学习训练
- 支持灰度化、缩放、跳帧、帧堆叠等常见图像预处理
- 支持进度奖励、卡住惩罚、跑动奖励等 reward shaping
- 支持断点续训、最佳 `x_pos` 模型保存、TensorBoard 日志
- 支持命令行评估和图形化窗口观看 AI 游玩

## 目录结构

```text
.
├── environment.yml
├── requirements.txt
├── pyproject.toml
├── scripts
│   ├── setup_env.ps1
│   ├── train.ps1
│   ├── play.ps1
│   ├── evaluate.ps1
│   ├── train_background.ps1
│   └── check_training.ps1
├── src
│   └── mario_rl
│       ├── env.py
│       ├── train.py
│       ├── evaluate.py
│       ├── play.py
│       ├── smoke_test.py
│       └── transfer_actions.py
├── models
└── runs
```

`models/`、`runs/`、`videos/` 等训练产物通常较大，默认不建议提交到 GitHub。需要分享模型时，可以单独上传到 GitHub Release 或其他文件存储服务。

## 环境要求

- Windows
- Conda
- Python 3.10
- CPU 可训练，但速度较慢；有 CUDA GPU 会更适合长时间训练

本机当前使用的 Conda 路径示例：

```powershell
D:\conda3
```

如果 Conda 安装在其他位置，需要相应修改命令里的路径。

## 安装

在项目目录下执行：

```powershell
cd D:\mario-RL
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

如果 PowerShell 禁止运行脚本，也可以直接用 Conda 手动安装：

```powershell
D:\conda3\condabin\conda.bat env create -f environment.yml
D:\conda3\condabin\conda.bat run -n mario-rl python -m pip install -r requirements.txt
D:\conda3\condabin\conda.bat run -n mario-rl python -m pip install -e .
```

安装完成后，可以做一次 smoke test：

```powershell
D:\conda3\envs\mario-rl\python.exe -m mario_rl.smoke_test
```

## 观看 AI 玩马里奥

推荐先使用当前表现最好的模型：

```powershell
D:\conda3\envs\mario-rl\python.exe -m mario_rl.play --model-path models\stochastic_from_898\best_xpos_model.zip --movement right --episodes 20 --stochastic --stuck-limit 0 --delay 0.05
```

运行后会弹出 NES 图形窗口，可以直接看到 AI 控制马里奥。

注意：这个模型是用 `right` 动作空间训练的，因此观看或评估时也要传入：

```powershell
--movement right
```

如果动作空间不一致，模型表现会明显异常。

## 评估模型

```powershell
D:\conda3\envs\mario-rl\python.exe -m mario_rl.evaluate --model-path models\stochastic_from_898\best_xpos_model.zip --movement right --episodes 30 --stochastic --stuck-limit 0
```

评估输出会包含平均奖励、平均 `x_pos`、最大 `x_pos`、是否到达终点旗帜等信息。

## 继续训练

从当前目录已有 checkpoint 继续训练：

```powershell
D:\conda3\envs\mario-rl\python.exe -m mario_rl.train --total-timesteps 1000000 --movement right --device cpu --n-envs 1 --model-dir models\stochastic_from_898 --tb-log-name ppo_mario_stochastic_from_898 --checkpoint-every 25000 --xpos-eval-every 25000 --xpos-eval-episodes 10 --xpos-eval-stochastic --xpos-eval-metric max --learning-rate 0.00005 --clip-range 0.1 --ent-coef 0.08 --stuck-limit 180 --stuck-penalty 50 --progress-reward-scale 0.4 --step-penalty 0.04 --resume
```

从某个最佳模型重新开一个训练目录：

```powershell
D:\conda3\envs\mario-rl\python.exe -m mario_rl.train --total-timesteps 1000000 --movement right --device cpu --n-envs 1 --model-dir models\stage_next --load-model models\stochastic_from_898\best_xpos_model.zip --reset-optimizer --checkpoint-every 25000 --xpos-eval-every 25000 --xpos-eval-episodes 10 --xpos-eval-stochastic --xpos-eval-metric max --learning-rate 0.00005 --clip-range 0.1 --ent-coef 0.08 --stuck-limit 180 --stuck-penalty 50 --progress-reward-scale 0.4 --step-penalty 0.04
```

## 查看训练曲线

```powershell
D:\conda3\envs\mario-rl\python.exe -m tensorboard.main --logdir runs\tensorboard
```

然后在浏览器打开 TensorBoard 提示的本地地址，一般是：

```text
http://localhost:6006
```

## 当前实验记录

| 模型目录 | 说明 | 当前表现 |
| --- | --- | --- |
| `models\anti_stuck_strong\best_xpos_model.zip` | 较稳定的确定性模型 | 约 `x_pos=898` |
| `models\stochastic_from_898\best_xpos_model.zip` | 当前推荐观看/评估模型 | 随机评估最大约 `x_pos=1800~1950` |
| `models\run_right_pass` | `run-right` 动作空间实验 | 暂未超过当前最佳 |

目前还没有生成 `passed_model.zip`，说明还没有在评估回合中检测到 `flag_get=True`。

## 常见问题

### PowerShell 提示禁止运行脚本

可以使用：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\play.ps1
```

也可以绕过脚本，直接运行 `python -m mario_rl.play`、`python -m mario_rl.train` 等命令。

### 出现 Gym 兼容性警告

`gym-super-mario-bros` 依赖旧版 Gym API，本项目通过 `shimmy` 做兼容。只要程序能正常训练、评估或弹出窗口，相关 warning 可以暂时忽略。

### 为什么观看时窗口一闪而过

常见原因是模型路径错误、环境没有安装好、或者动作空间参数和训练时不一致。优先确认：

```powershell
--model-path models\stochastic_from_898\best_xpos_model.zip
--movement right
```

## 后续计划

- 继续训练直到稳定通过 `1-1`
- 调整 reward shaping，减少局部最优和卡住行为
- 尝试更多随机种子和并行环境
- 保存 AI 游玩视频，方便展示项目结果
- 整理实验曲线和阶段性模型表现

## 版权说明

本项目仅用于课程实践、学习和研究。请不要在仓库中上传或分发任何商业游戏 ROM、Nintendo 原始素材或未经授权的资源。代码部分可以自行选择许可证后再公开发布。
