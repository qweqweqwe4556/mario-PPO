from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from mario_rl.env import make_env_factory
from mario_rl.utils import PROJECT_ROOT, latest_checkpoint, load_frontier_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Super Mario Bros with a trained PPO model.")
    parser.add_argument("--env-id", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--movement", choices=["right", "run-right", "simple", "complex"], default="right")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=3)
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
    parser.add_argument("--delay", type=float, default=0.03, help="Seconds to sleep after each rendered step.")
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
        help="Replay a JSON action prefix and play from the saved NES frontier state.",
    )
    parser.add_argument("--milestone-x", type=int, nargs="*", default=[])
    parser.add_argument("--milestone-bonus", type=float, default=0.0)
    parser.add_argument("--run-bonus", type=float, default=0.0)
    parser.add_argument("--no-run-penalty", type=float, default=0.0)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def resolve_model_path(model_path: Path | None) -> Path:
    if model_path:
        return model_path

    final_model = PROJECT_ROOT / "models" / "final_model.zip"
    if final_model.exists():
        return final_model

    checkpoint = latest_checkpoint(PROJECT_ROOT / "models" / "checkpoints")
    if checkpoint:
        return checkpoint

    raise FileNotFoundError("No trained model found. Run training first.")


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
            render_mode="human",
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
    print(f"Loaded model: {model_path}")

    try:
        for episode in range(args.episodes):
            obs = env.reset()
            done = [False]
            total_reward = 0.0
            while not done[0]:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, done, info = env.step(action)
                total_reward += float(reward[0])
                env.render("human")
                if args.delay > 0:
                    time.sleep(args.delay)

            clean_info = {
                key: value
                for key, value in info[0].items()
                if key != "terminal_observation"
            }
            print(f"Episode {episode + 1}: reward={total_reward:.2f}, info={clean_info}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
