from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stuck-limit", type=int, default=120)
    parser.add_argument("--stuck-penalty", type=float, default=25.0)
    parser.add_argument("--death-penalty", type=float, default=100.0)
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
    model_path = resolve_model_path(args.model_path)
    frontier_actions = load_frontier_actions(args.frontier_actions)

    env = DummyVecEnv([
        make_env_factory(
            args.env_id,
            args.movement,
            args.skip,
            args.seed,
            rank=0,
            stuck_limit=args.stuck_limit,
            stuck_penalty=args.stuck_penalty,
            death_penalty=args.death_penalty,
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
    lengths: list[int] = []

    try:
        for episode in range(args.episodes):
            obs = env.reset()
            done = [False]
            total_reward = 0.0
            steps = 0
            last_info = {}

            while not done[0]:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, done, infos = env.step(action)
                total_reward += float(reward[0])
                steps += 1
                last_info = infos[0]

            x_pos = int(last_info.get("x_pos", 0))
            rewards.append(total_reward)
            x_positions.append(x_pos)
            lengths.append(steps)
            print(
                f"episode={episode + 1} reward={total_reward:.2f} "
                f"steps={steps} x_pos={x_pos} flag_get={last_info.get('flag_get', False)}"
            )

        print(
            "summary "
            f"reward_mean={np.mean(rewards):.2f} "
            f"x_pos_mean={np.mean(x_positions):.1f} "
            f"x_pos_max={np.max(x_positions)} "
            f"steps_mean={np.mean(lengths):.1f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
