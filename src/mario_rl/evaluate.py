from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from mario_rl.env import make_env_factory
from mario_rl.play import resolve_model_path
from mario_rl.utils import load_frontier_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO Mario model without rendering.")
    parser.add_argument("--env-id", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--movement", choices=["right", "run-right", "simple", "complex"], default="right")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--skip", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible stochastic policy sampling.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-reward-scale", type=float, default=0.0)
    parser.add_argument("--step-penalty", type=float, default=0.03)
    parser.add_argument("--stuck-limit", type=int, default=120)
    parser.add_argument("--stuck-penalty", type=float, default=25.0)
    parser.add_argument("--death-penalty", type=float, default=100.0)
    parser.add_argument("--flag-reward", type=float, default=1000.0)
    parser.add_argument("--min-jump-hold", type=int, default=0)
    parser.add_argument("--max-jump-hold", type=int, default=0)
    parser.add_argument(
        "--frontier-actions",
        type=Path,
        default=None,
        help="Replay a JSON action prefix and evaluate from the saved NES frontier state.",
    )
    parser.add_argument("--milestone-x", type=int, nargs="*", default=[])
    parser.add_argument("--milestone-bonus", type=float, default=0.0)
    parser.add_argument("--run-bonus", type=float, default=0.0)
    parser.add_argument("--no-run-penalty", type=float, default=0.0)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.policy_seed is not None:
        set_random_seed(args.policy_seed, using_cuda=args.device != "cpu")
    model_path = resolve_model_path(args.model_path)
    frontier_actions = load_frontier_actions(args.frontier_actions)

    env = DummyVecEnv([
        make_env_factory(
            args.env_id,
            args.movement,
            args.skip,
            args.seed,
            rank=0,
            progress_reward_scale=args.progress_reward_scale,
            step_penalty=args.step_penalty,
            stuck_limit=args.stuck_limit,
            stuck_penalty=args.stuck_penalty,
            death_penalty=args.death_penalty,
            flag_reward=args.flag_reward,
            min_jump_hold=args.min_jump_hold,
            max_jump_hold=args.max_jump_hold,
            frontier_actions=frontier_actions,
            milestone_x=tuple(args.milestone_x),
            milestone_bonus=args.milestone_bonus,
            run_bonus=args.run_bonus,
            no_run_penalty=args.no_run_penalty,
        )
    ])
    env = VecFrameStack(env, n_stack=4, channels_order="last")
    model = PPO.load(model_path, env=env, device=args.device)

    rewards: list[float] = []
    x_positions: list[int] = []
    max_x_positions: list[int] = []
    lengths: list[int] = []
    flag_gets: list[bool] = []
    deaths: list[bool] = []
    stuck_timeouts: list[bool] = []
    reward_component_totals: dict[str, float] = {}

    try:
        for episode in range(args.episodes):
            obs = env.reset()
            done = [False]
            total_reward = 0.0
            steps = 0
            last_info = {}
            max_x_pos = 0

            while not done[0]:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, done, infos = env.step(action)
                total_reward += float(reward[0])
                steps += 1
                last_info = infos[0]
                max_x_pos = max(max_x_pos, int(last_info.get("x_pos", 0)))
                for name, value in last_info.get("reward_components", {}).items():
                    reward_component_totals[name] = (
                        reward_component_totals.get(name, 0.0) + float(value)
                    )

            x_pos = int(last_info.get("x_pos", 0))
            rewards.append(total_reward)
            x_positions.append(x_pos)
            max_x_positions.append(max_x_pos)
            lengths.append(steps)
            flag_gets.append(bool(last_info.get("flag_get", False)))
            deaths.append(bool(last_info.get("death_detected", False)))
            stuck_timeouts.append(bool(last_info.get("stuck_timeout", False)))
            print(
                f"episode={episode + 1} reward={total_reward:.2f} "
                f"steps={steps} x_pos={x_pos} max_x_pos={max_x_pos} "
                f"flag_get={flag_gets[-1]} death={deaths[-1]} "
                f"stuck_timeout={stuck_timeouts[-1]}"
            )

        print(
            "summary "
            f"reward_mean={np.mean(rewards):.2f} "
            f"x_pos_mean={np.mean(x_positions):.1f} "
            f"max_x_pos_mean={np.mean(max_x_positions):.1f} "
            f"max_x_pos={np.max(max_x_positions)} "
            f"steps_mean={np.mean(lengths):.1f} "
            f"flag_success_rate={np.mean(flag_gets):.3f} "
            f"death_rate={np.mean(deaths):.3f} "
            f"stuck_timeout_rate={np.mean(stuck_timeouts):.3f}"
        )
        if reward_component_totals:
            component_means = " ".join(
                f"{name}={total / args.episodes:.2f}"
                for name, total in sorted(reward_component_totals.items())
            )
            print(f"reward_components_mean {component_means}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
