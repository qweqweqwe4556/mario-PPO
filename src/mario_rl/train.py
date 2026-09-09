from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor

from mario_rl.env import make_env_factory
from mario_rl.utils import (
    PROJECT_ROOT,
    ensure_dir,
    latest_checkpoint,
    load_frontier_actions,
    set_global_seed,
)


class XPosEvalCallback(BaseCallback):
    """Save the model that reaches the farthest average x position."""

    def __init__(
        self,
        eval_env,
        save_path: Path,
        eval_freq: int,
        n_eval_episodes: int,
        deterministic: bool,
        metric: str,
        success_threshold: int = 0,
        label: str = "eval",
    ) -> None:
        super().__init__(verbose=1)
        self.eval_env = eval_env
        self.save_path = save_path
        self.passed_path = save_path.with_name("passed_model")
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.metric = metric
        self.success_threshold = success_threshold
        self.label = label
        self.best_score = float("-inf")

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        x_positions: list[int] = []
        flag_gets = 0
        rewards: list[float] = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = [False]
            total_reward = 0.0
            last_info = {}

            while not done[0]:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, done, infos = self.eval_env.step(action)
                total_reward += float(reward[0])
                last_info = infos[0]

            x_positions.append(int(last_info.get("x_pos", 0)))
            flag_gets += int(bool(last_info.get("flag_get", False)))
            rewards.append(total_reward)

        mean_x_pos = sum(x_positions) / len(x_positions)
        max_x_pos = max(x_positions)
        mean_reward = sum(rewards) / len(rewards)
        successes = (
            sum(x_pos >= self.success_threshold for x_pos in x_positions)
            if self.success_threshold > 0
            else 0
        )
        self.logger.record(f"{self.label}/mean_x_pos", mean_x_pos)
        self.logger.record(f"{self.label}/max_x_pos", max_x_pos)
        self.logger.record(f"{self.label}/flag_gets", flag_gets)
        self.logger.record(f"{self.label}/mean_reward", mean_reward)
        self.logger.record(f"{self.label}/successes", successes)
        self.logger.record(f"{self.label}/success_rate", successes / len(x_positions))
        if self.metric == "success":
            score = successes * 10_000 + mean_x_pos
        else:
            score = max_x_pos if self.metric == "max" else mean_x_pos
        print(
            f"x_pos_eval mode={self.label} mean_x_pos={mean_x_pos:.1f} "
            f"max_x_pos={max_x_pos} "
            f"successes={successes}/{self.n_eval_episodes}@{self.success_threshold} "
            f"best_score={self.best_score:.1f} "
            f"flag_gets={flag_gets}/{self.n_eval_episodes} "
            f"mean_reward={mean_reward:.2f}",
            flush=True,
        )

        if flag_gets > 0:
            self.model.save(self.passed_path)
            print(f"Saved passed model to {self.passed_path}.zip", flush=True)

        if score > self.best_score:
            self.best_score = score
            self.model.save(self.save_path)
            print(f"Saved best x_pos model to {self.save_path}.zip", flush=True)

        return True

    def _on_training_end(self) -> None:
        self.eval_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Super Mario Bros.")
    parser.add_argument("--env-id", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--movement", choices=["right", "run-right", "simple", "complex"], default="right")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--skip", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load-model", type=Path, default=None)
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "runs" / "tensorboard")
    parser.add_argument("--tb-log-name", default="ppo_mario")
    parser.add_argument("--final-model-name", default="final_model")
    parser.add_argument("--checkpoint-every", type=int, default=25_000)
    parser.add_argument("--xpos-eval-every", type=int, default=25_000)
    parser.add_argument("--xpos-eval-episodes", type=int, default=3)
    parser.add_argument("--xpos-eval-stochastic", action="store_true")
    parser.add_argument(
        "--xpos-eval-both",
        action="store_true",
        help="Evaluate and save deterministic and stochastic policies separately.",
    )
    parser.add_argument("--xpos-eval-metric", choices=["mean", "max", "success"], default="mean")
    parser.add_argument("--xpos-success-threshold", type=int, default=0)
    parser.add_argument("--xpos-eval-frontier", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="Stop a PPO update early when the approximate KL exceeds this value.",
    )
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--progress-reward-scale", type=float, default=0.3)
    parser.add_argument("--step-penalty", type=float, default=0.03)
    parser.add_argument("--stuck-limit", type=int, default=120)
    parser.add_argument("--stuck-penalty", type=float, default=25.0)
    parser.add_argument("--death-penalty", type=float, default=100.0)
    parser.add_argument("--min-jump-hold", type=int, default=0)
    parser.add_argument("--max-jump-hold", type=int, default=0)
    parser.add_argument("--frontier-actions", type=Path, default=None)
    parser.add_argument("--frontier-envs", type=int, default=0)
    parser.add_argument("--milestone-x", type=int, nargs="*", default=[])
    parser.add_argument("--milestone-bonus", type=float, default=0.0)
    parser.add_argument("--run-bonus", type=float, default=0.0)
    parser.add_argument("--no-run-penalty", type=float, default=0.0)
    return parser.parse_args()


def build_vec_env(args: argparse.Namespace, frontier_actions: tuple[int, ...]):
    factories = [
        make_env_factory(
            args.env_id,
            args.movement,
            args.skip,
            args.seed,
            rank,
            progress_reward_scale=args.progress_reward_scale,
            step_penalty=args.step_penalty,
            stuck_limit=args.stuck_limit,
            stuck_penalty=args.stuck_penalty,
            death_penalty=args.death_penalty,
            min_jump_hold=args.min_jump_hold,
            max_jump_hold=args.max_jump_hold,
            frontier_actions=frontier_actions if rank < args.frontier_envs else (),
            milestone_x=tuple(args.milestone_x),
            milestone_bonus=args.milestone_bonus,
            run_bonus=args.run_bonus,
            no_run_penalty=args.no_run_penalty,
        )
        for rank in range(args.n_envs)
    ]

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = vec_cls(factories)
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=4, channels_order="last")
    return env


def apply_training_args(model: PPO, args: argparse.Namespace) -> None:
    model.n_steps = args.n_steps
    model.batch_size = args.batch_size
    model.learning_rate = args.learning_rate
    model.lr_schedule = get_schedule_fn(args.learning_rate)
    model.clip_range = get_schedule_fn(args.clip_range)
    model.gamma = args.gamma
    model.gae_lambda = args.gae_lambda
    model.rollout_buffer.gamma = args.gamma
    model.rollout_buffer.gae_lambda = args.gae_lambda
    model.target_kl = args.target_kl
    model.ent_coef = args.ent_coef
    model.vf_coef = args.vf_coef
    model.max_grad_norm = args.max_grad_norm

    if args.reset_optimizer:
        model.policy.optimizer = model.policy.optimizer_class(
            model.policy.parameters(),
            lr=args.learning_rate,
            **model.policy.optimizer_kwargs,
        )
        model._n_updates = 0
    else:
        for param_group in model.policy.optimizer.param_groups:
            param_group["lr"] = args.learning_rate


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    frontier_actions = load_frontier_actions(args.frontier_actions)
    if args.frontier_envs < 0 or args.frontier_envs > args.n_envs:
        raise ValueError("frontier-envs must be between 0 and n-envs")
    if args.frontier_envs > 0 and not frontier_actions:
        raise ValueError("frontier-actions is required when frontier-envs is greater than zero")
    if args.xpos_eval_metric == "success" and args.xpos_success_threshold <= 0:
        raise ValueError("xpos-success-threshold must be positive for success evaluation")

    model_dir = ensure_dir(args.model_dir)
    checkpoint_dir = ensure_dir(model_dir / "checkpoints")
    log_dir = ensure_dir(args.log_dir)

    env = build_vec_env(args, frontier_actions)

    checkpoint = latest_checkpoint(checkpoint_dir) if args.resume else None
    if checkpoint:
        print(f"Resuming from {checkpoint}")
        model = PPO.load(
            checkpoint,
            env=env,
            device=args.device,
            tensorboard_log=str(log_dir),
            custom_objects={
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
            },
        )
        apply_training_args(model, args)
        reset_num_timesteps = False
    elif args.load_model:
        print(f"Loading warm-start model from {args.load_model}")
        model = PPO.load(
            args.load_model,
            env=env,
            device=args.device,
            tensorboard_log=str(log_dir),
            custom_objects={
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
            },
        )
        apply_training_args(model, args)
        reset_num_timesteps = True
    else:
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            target_kl=args.target_kl,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            verbose=1,
            tensorboard_log=str(log_dir),
            device=args.device,
        )
        reset_num_timesteps = True

    callback = CheckpointCallback(
        save_freq=max(args.checkpoint_every // max(args.n_envs, 1), 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_mario",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks = [callback]

    if args.xpos_eval_every > 0:
        eval_modes = (
            [
                ("eval_deterministic", True, "best_deterministic_model"),
                ("eval_stochastic", False, "best_stochastic_model"),
            ]
            if args.xpos_eval_both
            else [
                (
                    "eval_stochastic" if args.xpos_eval_stochastic else "eval_deterministic",
                    not args.xpos_eval_stochastic,
                    "best_xpos_model",
                )
            ]
        )
        eval_specs = [(*mode, ()) for mode in eval_modes]
        if args.xpos_eval_frontier:
            if not frontier_actions:
                raise ValueError("frontier-actions is required for frontier evaluation")
            eval_specs.extend(
                (
                    label.replace("eval_", "eval_frontier_", 1),
                    deterministic,
                    save_name.replace("best_", "best_frontier_", 1),
                    frontier_actions,
                )
                for label, deterministic, save_name in eval_modes
            )

        for label, deterministic, save_name, eval_frontier_actions in eval_specs:
            eval_env = DummyVecEnv([
                make_env_factory(
                    args.env_id,
                    args.movement,
                    args.skip,
                    args.seed + 10_000,
                    rank=0,
                    progress_reward_scale=args.progress_reward_scale,
                    step_penalty=args.step_penalty,
                    stuck_limit=args.stuck_limit,
                    stuck_penalty=args.stuck_penalty,
                    death_penalty=args.death_penalty,
                    min_jump_hold=args.min_jump_hold,
                    max_jump_hold=args.max_jump_hold,
                    frontier_actions=eval_frontier_actions,
                    milestone_x=tuple(args.milestone_x),
                    milestone_bonus=args.milestone_bonus,
                    run_bonus=args.run_bonus,
                    no_run_penalty=args.no_run_penalty,
                )
            ])
            eval_env = VecFrameStack(eval_env, n_stack=4, channels_order="last")
            callbacks.append(
                XPosEvalCallback(
                    eval_env=eval_env,
                    save_path=model_dir / save_name,
                    eval_freq=max(args.xpos_eval_every // max(args.n_envs, 1), 1),
                    n_eval_episodes=args.xpos_eval_episodes,
                    deterministic=deterministic,
                    metric=args.xpos_eval_metric,
                    success_threshold=args.xpos_success_threshold,
                    label=label,
                )
            )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=args.tb_log_name,
            progress_bar=True,
        )
        final_path = model_dir / args.final_model_name
        model.save(final_path)
        print(f"Saved final model to {final_path}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    mp.freeze_support()
    main()
