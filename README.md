# Mario PPO

用 `gym-super-mario-bros` 和 Stable-Baselines3 PPO 训练智能体完成《Super Mario Bros.》1-1。项目包含环境封装、奖励塑形、断点续训、双模式评估、前沿状态课程训练和 Windows PowerShell 脚本。

## v0.2.0 状态

这是一个可复现实验版本，尚未达到稳定通关：

- 完整关卡模型从起点确定性运行可到达 `x_pos=2226`。
- 前沿技能模型从 `x_pos=2095` 的保存状态随机评估 10 局均越过 `x_pos=2300`，最大到达 `x_pos=2994`。
- 当前发布模型均未检测到 `flag_get=True`，请勿将前沿结果理解为完整关卡通关率。

模型文件较大，不提交到 Git；发布包通过独立模型资产提供。详细参数和限制见 [MODEL_CARD.md](MODEL_CARD.md)。

## v1.0.0 状态

在 v0.2.0 训练与发布模型能力之上，本分支增加课程演示用前端控制台：

- Web 页面内嵌马里奥游戏画面（WebSocket 推送 NES 帧，无需再单独弹窗）。
- 支持页内切换完整关卡 / 前沿技能模型，并同步显示 `x_pos`、奖励与进度轨道。
- 仍可使用原有命令行 `evaluate` / `play`；演示推荐走下方「课程演示控制台」。

前端源码在 `web/`，演示 API 为 `python -m mario_rl.demo_api`。模型文件仍不提交到 Git，需自行放到 `models\release`。

## 功能

- 84×84 灰度观测、跳帧和 4 帧堆叠
- 限制最短/最长连续按住跳跃键
- 只奖励首次到达的新位置，并支持死亡、卡住、里程碑和跑动奖励
- 保存定期 checkpoint、最终模型和确定性/随机最佳模型
- 使用动作前缀恢复到关卡中段，混合完整起点与前沿起点并行训练
- 无窗口评估、图形窗口播放、TensorBoard 和后台训练脚本
- 课程演示 Web 控制台：页内播放、模型切换与实时指标

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

## 课程演示控制台

`v1.0.0` 附带 Web 演示前端（`web/`，Vite + React）与 FastAPI 后端（`mario_rl.demo_api`）：

- 品牌化演示页：模型切换、发布指标、训练故事
- 页内嵌入游戏画面：WebSocket 推送 NES RGB 帧到浏览器 Canvas
- 无画面快速评估：SSE 实时推送 `x_pos` / reward，关卡进度条可视化

依赖：已按上文创建并激活 `mario-rl`；演示 API 额外需要 `fastapi` / `uvicorn`（可用 `pip install -e ".[demo]"`）。前端需本机已安装 Node.js。

```powershell
# 安装演示 API 依赖（首次）
python -m pip install -e ".[demo]"

# 安装前端依赖（首次）
cd web
npm install
cd ..

# 终端 1：API
python -m mario_rl.demo_api

# 终端 2：前端
cd web
npm run dev
```

浏览器打开 `http://127.0.0.1:5173`，点击「页内播放」。也可一键启动：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\demo.ps1
```

## 目录

```text
configs/                 前沿动作前缀和可复现配置
docs/                    实验历史
scripts/                 环境、训练、评估和运行脚本
src/mario_rl/            Python 包（含 demo_api）
web/                     课程演示前端（Vite + React）
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
