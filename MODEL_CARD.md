# Mario PPO v2.0.0 模型卡

## 模型用途

三个 PPO `CnnPolicy` 模型用于 `SuperMarioBros-1-1-v0` 的研究和演示。观测为 84×84 灰度图像并堆叠 4 帧，动作空间为 `gym_super_mario_bros.actions.RIGHT_ONLY`，环境每个决策重复动作 4 帧。

## 发布资产

| 文件 | 用途 | 推理方式 | 已知表现 |
| --- | --- | --- | --- |
| `full_level_best.zip` | 从关卡起点运行 | 随机 | 30 局通关 1 次，成功率 3.33%，最大 3161 |
| `full_level_deterministic_baseline.zip` | 从关卡起点运行 | 确定性 | 10/10 到达 `x_pos=2226`，未通关 |
| `frontier_x2095_best.zip` | 从 `x=2095` 前沿状态运行 | 随机 | 10/10 越过 2300，均值 2634.5，最大 2994，未通关 |

模型 SHA-256、文件大小和来源见模型资产中的 `manifest.json`。

## 必须匹配的环境参数

三个模型都需要 `movement=right`、`skip=4`、4 帧堆叠和 `max_jump_hold=7`。前沿模型还必须传入 `configs/frontier_x2095.json` 并使用随机预测；缺少动作前缀时，其输入分布和报告结果不一致。

奖励参数不改变模型网络的动作维度，但会改变回合提前结束的位置。发布命令为完整模型使用 `stuck_limit=180`、`stuck_penalty=25`、`death_penalty=150`、`milestone_x=2050` 和 `milestone_bonus=250`；前沿模型使用 `stuck_limit=60`。

## 通关证据

`full_level_best.zip` 从确定性 baseline 暖启动，冻结 `policy.features_extractor`，使用 `seed=43`、8192 steps、`n_epochs=3`、`learning_rate=5e-6`、`clip_range=0.1`、`target_kl=0.025` 和 `frontier_envs=0` 微调。

固定完整起点环境 seed 10042、策略 seed 20260910，并在 CUDA 上随机采样 4 局时，前三局最大位置依次为 1660、2762、702；第 4 局在 616 个决策步后达到 `x_pos=3161` 并返回 `flag_get=True`。发布包中的 `full_level_success_seed20260910.mp4` 是该局完整录像，配套 JSON 保存逐步动作和状态。

## 限制

- 已经观测并录制一次 `flag_get=True`，但 30 局评估只有 1 次成功，不能视为稳定通关。
- 前沿结果依赖 NES 私有状态备份接口和固定动作前缀。
- 只在 Windows、Python 3.10、NumPy 1.26.4 和 `gym-super-mario-bros==7.4.0` 的本地环境验证。
- 随机策略序列依赖 PyTorch、CUDA 和运行环境；跨设备或依赖版本不保证逐动作复现。
- 发布资产不包含 ROM 或 Nintendo 原始素材。

## v2.0.0 发布验证

2026-09-10 在 RTX 4060 Laptop GPU 上复测。通关候选随机评估 30 局平均 `x_pos=1728.7`、最大 3161、`flag_get=1/30`；确定性评估 10 局均为 `x_pos=2029`。原 baseline 确定性评估 10 局均为 `x_pos=2226`。前沿模型数据沿用 2026-09-09 的发布验证结果。
