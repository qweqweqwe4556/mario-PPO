from __future__ import annotations

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from mario_rl.env import make_env_factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer a trained right-action PPO model to run-right actions.")
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--source-movement", default="right")
    parser.add_argument("--target-movement", default="run-right")
    parser.add_argument("--source-action-indices", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stuck-limit", type=int, default=180)
    return parser.parse_args()


def make_vec_env(movement: str, seed: int, stuck_limit: int):
    env = DummyVecEnv([
        make_env_factory(
            "SuperMarioBros-1-1-v0",
            movement,
            skip=4,
            seed=seed,
            rank=0,
            stuck_limit=stuck_limit,
        )
    ])
    return VecFrameStack(env, n_stack=4, channels_order="last")


def main() -> None:
    args = parse_args()
    source_env = make_vec_env(args.source_movement, args.seed, args.stuck_limit)
    target_env = make_vec_env(args.target_movement, args.seed, args.stuck_limit)

    try:
        source_model = PPO.load(args.source_model, env=source_env, device=args.device)
        target_model = PPO(
            "CnnPolicy",
            target_env,
            learning_rate=1e-4,
            n_steps=512,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.03,
            verbose=1,
            device=args.device,
        )

        source_state = source_model.policy.state_dict()
        target_state = target_model.policy.state_dict()
        merged_state = {}

        for key, target_value in target_state.items():
            source_value = source_state.get(key)
            if source_value is not None and source_value.shape == target_value.shape:
                merged_state[key] = source_value
            else:
                merged_state[key] = target_value

        action_indices = torch.tensor(args.source_action_indices, dtype=torch.long)
        merged_state["action_net.weight"] = source_state["action_net.weight"].index_select(0, action_indices)
        merged_state["action_net.bias"] = source_state["action_net.bias"].index_select(0, action_indices)

        target_model.policy.load_state_dict(merged_state)
        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        target_model.save(args.output_model)
        print(f"Saved transferred model to {args.output_model}.zip")
    finally:
        source_env.close()
        target_env.close()


if __name__ == "__main__":
    main()
