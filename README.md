# Mario PPO

用 `gym-super-mario-bros` 和 Stable-Baselines3 PPO 训练智能体完成《Super Mario Bros.》1-1。项目包含环境封装、奖励塑形、断点续训、双模式评估、前沿状态课程训练和 Windows PowerShell 脚本。

## v0.2.0 状态

这是一个可复现实验版本，尚未达到稳定通关：

- 完整关卡模型从起点确定性运行可到达 `x_pos=2226`。
- 前沿技能模型从 `x_pos=2095` 的保存状态随机评估 10 局均越过 `x_pos=2300`，最大到达 `x_pos=2994`。
- 当前发布模型均未检测到 `flag_get=True`，请勿将前沿结果理解为完整关卡通关率。

模型文件较大，不提交到 Git；发布包通过独立模型资产提供。详细参数和限制见 [MODEL_CARD.md](MODEL_CARD.md)。

## 功能

- 84×84 灰度观测、跳帧和 4 帧堆叠
- 限制最短/最长连续按住跳跃键
- 只奖励首次到达的新位置，并支持死亡、卡住、里程碑和跑动奖励
- 保存定期 checkpoint、最终模型和确定性/随机最佳模型
- 使用动作前缀恢复到关卡中段，混合完整起点与前沿起点并行训练
- 无窗口评估、图形窗口播放、TensorBoard 和后台训练脚本

## 环境要求

- Windows 10/11
- Conda（默认查找 `%USERPROFILE%\anaconda3`）
- Python 3.10
- CPU 可运行；CUDA GPU 适合训练，但并行模拟环境仍主要占用 CPU

仓库不包含 ROM、Nintendo 素材或游戏二进制文件。`gym-super-mario-bros` 自带其运行所需的兼容环境。

## 安装

在 PowerShell 中进入仓库并运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
& "$env:USERPROFILE\anaconda3\shell\condabin\conda-hook.ps1"
conda activate mario-rl
python -m mario_rl.smoke_test
```

Conda 不在默认位置时：

```powershell
.\scripts\setup_env.ps1 -CondaBat "D:\Miniconda3\condabin\conda.bat"
```

## 使用发布模型

将 `mario-ppo-v0.2.0-models.zip` 解压到仓库的 `models\release`。完整关卡模型评估：

```powershell
python -m mario_rl.evaluate `
  --model-path models\release\full_level_best.zip `
  --movement right --episodes 10 --device auto `
  --max-jump-hold 7 --stuck-limit 180 --stuck-penalty 25 `
  --death-penalty 150 --milestone-x 2050 --milestone-bonus 250
```

打开窗口观看：

```powershell
python -m mario_rl.play `
  --model-path models\release\full_level_best.zip `
  --movement right --episodes 3 --device auto `
  --max-jump-hold 7 --stuck-limit 180 --death-penalty 150 --delay 0.03
```

评估前沿技能模型时必须同时提供动作前缀，并使用随机动作采样：

```powershell
python -m mario_rl.evaluate `
  --model-path models\release\frontier_x2095_best.zip `
  --frontier-actions configs\frontier_x2095.json `
  --movement right --episodes 10 --stochastic --device auto `
  --max-jump-hold 7 --stuck-limit 60 --stuck-penalty 150 `
  --death-penalty 150 --milestone-x 2300 --milestone-bonus 250
```

## 训练

普通训练或从输出目录的最新 checkpoint 续训：

```powershell
.\scripts\train.ps1 -TotalTimesteps 100000 -Device cuda -ModelDir models\baseline
```

`TotalTimesteps` 是本次调用新增的环境步数。默认每 25,000 步生成一个 checkpoint；第一次调用并不要求一次训练 100,000 步，可以先用 10,000–25,000 步验证配置。

复现下一轮混合课程训练配置：

```powershell
.\scripts\train_curriculum.ps1 -TotalTimesteps 50000 -Device cuda
```

该脚本从完整关卡模型暖启动，使用 8 个并行环境，其中 4 个从 `x=2095` 前沿状态开始，同时分别评估完整起点和前沿起点。若模型文件位于别处，可传入 `-LoadModel`。

后台运行并把输出写入 `logs`：

```powershell
.\scripts\train_background.ps1 -TotalTimesteps 100000 -Device cuda -ModelDir models\next_run
.\scripts\check_training.ps1
```

查看训练曲线：

```powershell
python -m tensorboard.main --logdir runs\tensorboard
```

## 目录

```text
configs/                 前沿动作前缀和可复现配置
docs/                    实验历史
scripts/                 环境、训练、评估和运行脚本
src/mario_rl/            Python 包
tests/                   环境封装与配置测试
models/                  本地模型，不提交 Git
runs/                    TensorBoard 日志，不提交 Git
```

## 开发验证

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m compileall -q src tests
python -m build --no-isolation
```

版本变化见 [CHANGELOG.md](CHANGELOG.md)。历史实验结论见 [docs/TRAINING_HISTORY.md](docs/TRAINING_HISTORY.md)。

## 版权

代码仓库当前没有附带开源许可证。游戏名称、角色和素材的权利归其各自权利人所有；本项目仅用于学习和研究。
