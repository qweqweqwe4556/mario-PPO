from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import gym
import gym_super_mario_bros
import shimmy
import numpy as np
from gym import spaces
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


MOVEMENTS = {
    "right": RIGHT_ONLY,
    "run-right": [["right", "B"], ["right", "A", "B"]],
    "simple": SIMPLE_MOVEMENT,
    "complex": COMPLEX_MOVEMENT,
}


class SkipFrame(gym.Wrapper):
    """Repeat an action for a few frames and return the accumulated reward."""

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self.skip = skip

    def step(self, action: int):
        total_reward = 0.0
        done = False
        info: dict[str, Any] = {}
        obs = None

        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += float(reward)
            if done:
                break

        return obs, total_reward, done, info


class ResizeGrayScaleObservation(gym.ObservationWrapper):
    """Convert RGB frames to 84x84 grayscale observations."""

    def __init__(self, env: gym.Env, shape: tuple[int, int] = (84, 84)) -> None:
        super().__init__(env)
        self.shape = shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shape[0], shape[1], 1),
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self.shape, interpolation=cv2.INTER_AREA)
        return resized[:, :, None].astype(np.uint8)


class MarioRewardWrapper(gym.Wrapper):
    """Reward shaping that favors forward progress and avoids waiting."""

    def __init__(
        self,
        env: gym.Env,
        progress_reward_scale: float = 0.3,
        step_penalty: float = 0.03,
        stuck_limit: int = 120,
        stuck_penalty: float = 25.0,
        action_meanings: list[list[str]] | None = None,
        run_bonus: float = 0.0,
        no_run_penalty: float = 0.0,
    ) -> None:
        super().__init__(env)
        self._last_x_pos = 0
        self._last_life = 2
        self._max_x_pos = 0
        self._steps_without_progress = 0
        self.progress_reward_scale = progress_reward_scale
        self.step_penalty = step_penalty
        self.stuck_limit = stuck_limit
        self.stuck_penalty = stuck_penalty
        self.action_meanings = action_meanings or []
        self.run_bonus = run_bonus
        self.no_run_penalty = no_run_penalty

    def reset(self, **kwargs):
        seed = kwargs.pop("seed", None)
        kwargs.pop("options", None)
        if seed is not None:
            self.env.seed(seed)

        self._last_x_pos = 0
        self._last_life = 2
        self._max_x_pos = 0
        self._steps_without_progress = 0
        return self.env.reset(**kwargs)

    def step(self, action: int):
        obs, reward, done, info = self.env.step(action)

        x_pos = int(info.get("x_pos", self._last_x_pos))
        life = int(info.get("life", self._last_life))
        x_delta = max(0, x_pos - self._last_x_pos)
        max_x_delta = max(0, x_pos - self._max_x_pos)

        if max_x_delta > 0:
            self._max_x_pos = x_pos
            self._steps_without_progress = 0
        else:
            self._steps_without_progress += 1

        shaped_reward = float(reward)
        shaped_reward += self.progress_reward_scale * x_delta
        shaped_reward -= self.step_penalty

        if x_delta == 0:
            shaped_reward -= self.step_penalty

        if 0 <= int(action) < len(self.action_meanings):
            action_buttons = self.action_meanings[int(action)]
            if "B" in action_buttons:
                shaped_reward += self.run_bonus
            else:
                shaped_reward -= self.no_run_penalty

        if life < self._last_life:
            shaped_reward -= 15.0
        if (
            self.stuck_limit > 0
            and self._steps_without_progress >= self.stuck_limit
            and not info.get("flag_get", False)
        ):
            done = True
            shaped_reward -= self.stuck_penalty
            info = dict(info)
            info["stuck_timeout"] = True
        if done and not info.get("flag_get", False):
            shaped_reward -= 10.0
        if info.get("flag_get", False):
            shaped_reward += 200.0

        self._last_x_pos = x_pos
        self._last_life = life

        return obs, shaped_reward, done, info


def make_mario_env(
    env_id: str = "SuperMarioBros-1-1-v0",
    movement: str = "right",
    skip: int = 4,
    seed: int | None = None,
    render_mode: str | None = None,
    progress_reward_scale: float = 0.3,
    step_penalty: float = 0.03,
    stuck_limit: int = 120,
    stuck_penalty: float = 25.0,
    run_bonus: float = 0.0,
    no_run_penalty: float = 0.0,
) -> gym.Env:
    if movement not in MOVEMENTS:
        valid = ", ".join(sorted(MOVEMENTS))
        raise ValueError(f"Unknown movement {movement!r}. Valid values: {valid}")

    env = gym_super_mario_bros.make(env_id)
    action_meanings = MOVEMENTS[movement]
    env = JoypadSpace(env, action_meanings)

    if seed is not None:
        env.seed(seed)

    env = SkipFrame(env, skip=skip)
    env = MarioRewardWrapper(
        env,
        progress_reward_scale=progress_reward_scale,
        step_penalty=step_penalty,
        stuck_limit=stuck_limit,
        stuck_penalty=stuck_penalty,
        action_meanings=action_meanings,
        run_bonus=run_bonus,
        no_run_penalty=no_run_penalty,
    )
    env = ResizeGrayScaleObservation(env)
    return shimmy.GymV21CompatibilityV0(env=env, render_mode=render_mode)


def make_env_factory(
    env_id: str,
    movement: str,
    skip: int,
    seed: int,
    rank: int,
    render_mode: str | None = None,
    progress_reward_scale: float = 0.3,
    step_penalty: float = 0.03,
    stuck_limit: int = 120,
    stuck_penalty: float = 25.0,
    run_bonus: float = 0.0,
    no_run_penalty: float = 0.0,
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        return make_mario_env(
            env_id=env_id,
            movement=movement,
            skip=skip,
            seed=seed + rank,
            render_mode=render_mode,
            progress_reward_scale=progress_reward_scale,
            step_penalty=step_penalty,
            stuck_limit=stuck_limit,
            stuck_penalty=stuck_penalty,
            run_bonus=run_bonus,
            no_run_penalty=no_run_penalty,
        )

    return _init
