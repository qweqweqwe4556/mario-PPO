from __future__ import annotations

import gym
import numpy as np

from mario_rl.env import FrontierStart, JumpHold, MarioRewardWrapper


ACTION_MEANINGS = [[], ["right"], ["right", "A"]]


class RecordingEnv(gym.Env):
    def __init__(self) -> None:
        self.action_space = gym.spaces.Discrete(len(ACTION_MEANINGS))
        self.observation_space = gym.spaces.Box(0, 255, (2, 2, 3), dtype=np.uint8)
        self.actions: list[int] = []
        self.x_pos = 0
        self._reset_x = 0

    def reset(self, **kwargs):
        self.x_pos = self._reset_x
        return np.zeros(self.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        self.actions.append(int(action))
        self.x_pos += 1
        info = {"x_pos": self.x_pos, "life": 2, "flag_get": False}
        return np.zeros(self.observation_space.shape, dtype=np.uint8), 0.0, False, info

    def _backup(self):
        self._reset_x = self.x_pos


def test_jump_hold_enforces_minimum_duration():
    base = RecordingEnv()
    env = JumpHold(base, ACTION_MEANINGS, minimum_steps=3)
    env.reset()

    for action in (2, 1, 1, 1):
        env.step(action)

    assert base.actions == [2, 2, 2, 1]


def test_jump_hold_inserts_release_after_maximum_duration():
    base = RecordingEnv()
    env = JumpHold(base, ACTION_MEANINGS, maximum_steps=3)
    env.reset()

    for _ in range(5):
        env.step(2)

    assert base.actions == [2, 2, 2, 1, 2]


def test_frontier_reset_uses_frontier_as_reward_baseline():
    base = RecordingEnv()
    frontier = FrontierStart(base, actions=(1, 1, 1))
    env = MarioRewardWrapper(
        frontier,
        progress_reward_scale=1.0,
        step_penalty=0.0,
        stuck_limit=0,
        death_penalty=0.0,
    )

    env.reset()
    _, reward, done, info = env.step(1)

    assert frontier.start_x_pos == 3
    assert info["x_pos"] == 4
    assert reward == 1.0
    assert not done

    env.reset()
    assert base.x_pos == 3
