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


class JumpHold(gym.Wrapper):
    """Apply minimum and maximum durations to consecutive A-button presses."""

    def __init__(
        self,
        env: gym.Env,
        action_meanings: list[list[str]],
        minimum_steps: int = 0,
        maximum_steps: int = 0,
    ) -> None:
        super().__init__(env)
        self.action_meanings = action_meanings
        self.minimum_steps = max(0, minimum_steps)
        self.maximum_steps = max(0, maximum_steps)
        if self.maximum_steps and self.minimum_steps > self.maximum_steps:
            raise ValueError("minimum jump hold cannot exceed maximum jump hold")
        self._hold_remaining = 0
        self._a_was_down = False
        self._consecutive_a = 0

    def reset(self, **kwargs):
        self._hold_remaining = 0
        self._a_was_down = False
        self._consecutive_a = 0
        return self.env.reset(**kwargs)

    def _with_jump(self, action: int) -> int:
        requested = set(self.action_meanings[action])
        requested.add("A")
        for index, buttons in enumerate(self.action_meanings):
            if set(buttons) == requested:
                return index

        for index, buttons in enumerate(self.action_meanings):
            if "A" in buttons and "right" in buttons:
                return index
        return action

    def _without_jump(self, action: int) -> int:
        requested = set(self.action_meanings[action])
        requested.discard("A")
        for index, buttons in enumerate(self.action_meanings):
            if set(buttons) == requested:
                return index
        return action

    def step(self, action: int):
        action = int(action)
        requested_has_a = "A" in self.action_meanings[action]

        if self._hold_remaining > 0:
            action = self._with_jump(action)
            self._hold_remaining -= 1
        elif requested_has_a and not self._a_was_down:
            self._hold_remaining = max(0, self.minimum_steps - 1)

        applied_has_a = "A" in self.action_meanings[action]
        if (
            applied_has_a
            and self.maximum_steps > 0
            and self._consecutive_a >= self.maximum_steps
        ):
            action = self._without_jump(action)
            applied_has_a = "A" in self.action_meanings[action]
            self._hold_remaining = 0
            self._consecutive_a = 0
        elif applied_has_a:
            self._consecutive_a += 1
        else:
            self._consecutive_a = 0

        self._a_was_down = applied_has_a
        return self.env.step(action)


class FrontierStart(gym.Wrapper):
    """Replay a prefix once, then restore that NES state on future resets."""

    def __init__(self, env: gym.Env, actions: tuple[int, ...]) -> None:
        super().__init__(env)
        self.actions = actions
        self._prepared = False
        self.start_x_pos = 0
        self.start_life = 2

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if self._prepared or not self.actions:
            return obs

        last_info: dict[str, Any] = {}
        for step, action in enumerate(self.actions, start=1):
            obs, _, done, last_info = self.env.step(action)
            if done:
                raise RuntimeError(
                    f"Frontier action prefix terminated at step {step} "
                    f"(x_pos={last_info.get('x_pos', 0)})"
                )

        backup = getattr(self.env.unwrapped, "_backup", None)
        if backup is None:
            raise RuntimeError("The underlying NES environment does not support state backup")
        backup()
        self.start_x_pos = int(last_info.get("x_pos", 0))
        self.start_life = int(last_info.get("life", 2))
        self._prepared = True
        return obs


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
        death_penalty: float = 100.0,
        milestone_x: tuple[int, ...] = (),
        milestone_bonus: float = 0.0,
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
        self.death_penalty = death_penalty
        self.milestone_x = tuple(sorted(set(milestone_x)))
        self.milestone_bonus = milestone_bonus
        self._reached_milestones: set[int] = set()
        self.action_meanings = action_meanings or []
        self.run_bonus = run_bonus
        self.no_run_penalty = no_run_penalty

    def reset(self, **kwargs):
        seed = kwargs.pop("seed", None)
        kwargs.pop("options", None)
        if seed is not None:
            self.env.seed(seed)

        obs = self.env.reset(**kwargs)
        start_x_pos = int(getattr(self.env, "start_x_pos", 0))
        self._last_x_pos = start_x_pos
        self._last_life = int(getattr(self.env, "start_life", 2))
        self._max_x_pos = start_x_pos
        self._steps_without_progress = 0
        self._reached_milestones = {
            threshold for threshold in self.milestone_x if threshold <= start_x_pos
        }
        return obs

    def step(self, action: int):
        obs, reward, done, info = self.env.step(action)

        x_pos = int(info.get("x_pos", self._last_x_pos))
        life = int(info.get("life", self._last_life))
        x_delta = max(0, x_pos - self._last_x_pos)
        max_x_delta = max(0, x_pos - self._max_x_pos)

        previous_max_x = self._max_x_pos
        if max_x_delta > 0:
            self._max_x_pos = x_pos
            self._steps_without_progress = 0
        else:
            self._steps_without_progress += 1

        shaped_reward = float(reward)
        # Reward only genuinely new progress. Revisited positions should not be
        # rewarded again after Mario is pushed backwards.
        shaped_reward += self.progress_reward_scale * max_x_delta
        shaped_reward -= self.step_penalty

        reached_now = [
            threshold
            for threshold in self.milestone_x
            if threshold not in self._reached_milestones
            and previous_max_x < threshold <= x_pos
        ]
        if reached_now:
            self._reached_milestones.update(reached_now)
            shaped_reward += self.milestone_bonus * len(reached_now)
            info = dict(info)
            info["milestones_reached"] = reached_now

        if x_delta == 0:
            shaped_reward -= self.step_penalty

        if 0 <= int(action) < len(self.action_meanings):
            action_buttons = self.action_meanings[int(action)]
            if "B" in action_buttons:
                shaped_reward += self.run_bonus
            else:
                shaped_reward -= self.no_run_penalty

        flag_get = bool(info.get("flag_get", False))
        death_detected = not flag_get and (done or life < self._last_life)
        if death_detected:
            shaped_reward -= self.death_penalty
            info = dict(info)
            info["death_detected"] = True
        if (
            self.stuck_limit > 0
            and self._steps_without_progress >= self.stuck_limit
            and not flag_get
        ):
            done = True
            shaped_reward -= self.stuck_penalty
            info = dict(info)
            info["stuck_timeout"] = True
        if flag_get:
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
    death_penalty: float = 100.0,
    min_jump_hold: int = 0,
    max_jump_hold: int = 0,
    frontier_actions: tuple[int, ...] = (),
    milestone_x: tuple[int, ...] = (),
    milestone_bonus: float = 0.0,
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
    env = JumpHold(
        env,
        action_meanings=action_meanings,
        minimum_steps=min_jump_hold,
        maximum_steps=max_jump_hold,
    )
    if frontier_actions:
        env = FrontierStart(env, actions=frontier_actions)
    env = MarioRewardWrapper(
        env,
        progress_reward_scale=progress_reward_scale,
        step_penalty=step_penalty,
        stuck_limit=stuck_limit,
        stuck_penalty=stuck_penalty,
        death_penalty=death_penalty,
        milestone_x=milestone_x,
        milestone_bonus=milestone_bonus,
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
    death_penalty: float = 100.0,
    min_jump_hold: int = 0,
    max_jump_hold: int = 0,
    frontier_actions: tuple[int, ...] = (),
    milestone_x: tuple[int, ...] = (),
    milestone_bonus: float = 0.0,
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
            death_penalty=death_penalty,
            min_jump_hold=min_jump_hold,
            max_jump_hold=max_jump_hold,
            frontier_actions=frontier_actions,
            milestone_x=milestone_x,
            milestone_bonus=milestone_bonus,
            run_bonus=run_bonus,
            no_run_penalty=no_run_penalty,
        )

    return _init
