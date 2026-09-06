from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor

from mario_rl.env import make_env_factory
from mario_rl.utils import PROJECT_ROOT, ensure_dir, latest_checkpoint, set_global_seed


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
    ) -> None:
        super().__init__(verbose=1)
        self.eval_env = eval_env
        self.save_path = save_path
        self.passed_path = save_path.with_name("passed_model")
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.metric = metric
        self.best_x_pos = float("-inf")

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
        self.logger.record("eval/mean_x_pos", mean_x_pos)
        self.logger.record("eval/max_x_pos", max_x_pos)
        self.logger.record("eval/flag_gets", flag_gets)
        self.logger.record("eval/mean_reward", mean_reward)
        score_x_pos = max_x_pos if self.metric == "max" else mean_x_pos
        print(
            f"x_pos_eval mean_x_pos={mean_x_pos:.1f} "
            f"max_x_pos={max_x_pos} "
            f"best_x_pos={self.best_x_pos:.1f} "
            f"flag_gets={flag_gets}/{self.n_eval_episodes} "
            f"mean_reward={mean_reward:.2f}",
            flush=True,
        )

        if flag_gets > 0:
            self.model.save(self.passed_path)
            print(f"Saved passed model to {self.passed_path}.zip", flush=True)

        if score_x_pos > self.best_x_pos:
            self.best_x_pos = score_x_pos
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
    parser.add_argument("--xpos-eval-metric", choices=["mean", "max"], default="mean")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--progress-reward-scale", type=float, default=0.3)
    parser.add_argument("--step-penalty", type=float, default=0.03)
    parser.add_argument("--stuck-limit", type=int, default=120)
    parser.add_argument("--stuck-penalty", type=float, default=25.0)
    parser.add_argument("--run-bonus", type=float, default=0.0)
    parser.add_argument("--no-run-penalty", type=float, default=0.0)
    return parser.parse_args()


def build_vec_env(args: argparse.Namespace):
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
    model.learning_rate = args.learning_rate
    model.lr_schedule = get_schedule_fn(args.learning_rate)
    model.clip_range = get_schedule_fn(args.clip_range)
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

    model_dir = ensure_dir(args.model_dir)
    checkpoint_dir = ensure_dir(model_dir / "checkpoints")
    log_dir = ensure_dir(args.log_dir)

    env = build_vec_env(args)

    checkpoint = latest_checkpoint(checkpoint_dir) if args.resume else None
    if checkpoint:
        print(f"Resuming from {checkpoint}")
        model = PPO.load(
            checkpoint,
            env=env,
            device=args.device,
            tensorboard_log=str(log_dir),
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
                run_bonus=args.run_bonus,
                no_run_penalty=args.no_run_penalty,
            )
        ])
        eval_env = VecFrameStack(eval_env, n_stack=4, channels_order="last")
        callbacks.append(
            XPosEvalCallback(
                eval_env=eval_env,
                save_path=model_dir / "best_xpos_model",
                eval_freq=max(args.xpos_eval_every // max(args.n_envs, 1), 1),
                n_eval_episodes=args.xpos_eval_episodes,
                deterministic=not args.xpos_eval_stochastic,
                metric=args.xpos_eval_metric,
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
