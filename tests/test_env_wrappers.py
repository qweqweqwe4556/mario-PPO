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


class ScriptedRewardEnv(gym.Env):
    def __init__(self, transitions):
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(0, 255, (2, 2, 3), dtype=np.uint8)
        self.transitions = iter(transitions)

    def reset(self, **kwargs):
        return np.zeros(self.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        reward, done, info = next(self.transitions)
        return np.zeros(self.observation_space.shape, dtype=np.uint8), reward, done, info


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


def test_default_reward_does_not_add_a_second_progress_reward():
    base = ScriptedRewardEnv([
        (4.0, False, {"x_pos": 4, "life": 2, "flag_get": False}),
    ])
    env = MarioRewardWrapper(base, step_penalty=0.0, stuck_limit=0)
    env.reset()

    _, reward, _, info = env.step(0)

    assert reward == 4.0
    assert info["reward_components"]["environment"] == 4.0
    assert info["reward_components"]["new_max_progress"] == 0.0


def test_flag_reward_dominates_progress_and_is_reported():
    base = ScriptedRewardEnv([
        (3.0, True, {"x_pos": 3, "life": 2, "flag_get": True}),
    ])
    env = MarioRewardWrapper(
        base,
        step_penalty=0.0,
        stuck_limit=0,
        flag_reward=1000.0,
    )
    env.reset()

    _, reward, done, info = env.step(0)

    assert done
    assert reward == 1003.0
    assert info["reward_components"]["flag_get"] == 1000.0
    assert "death_detected" not in info


def test_milestone_bonus_is_preserved():
    base = ScriptedRewardEnv([
        (5.0, False, {"x_pos": 5, "life": 2, "flag_get": False}),
    ])
    env = MarioRewardWrapper(
        base,
        step_penalty=0.0,
        stuck_limit=0,
        milestone_x=(5,),
        milestone_bonus=250.0,
    )
    env.reset()

    _, reward, _, info = env.step(0)

    assert reward == 255.0
    assert info["reward_components"]["milestone"] == 250.0


def test_terminal_death_does_not_also_trigger_stuck_timeout():
    base = ScriptedRewardEnv([
        (-25.0, True, {"x_pos": 0, "life": 1, "flag_get": False}),
    ])
    env = MarioRewardWrapper(
        base,
        step_penalty=0.0,
        stuck_limit=1,
        stuck_penalty=25.0,
        death_penalty=100.0,
    )
    env.reset()

    _, reward, done, info = env.step(0)

    assert done
    assert reward == -125.0
    assert info["death_detected"]
    assert "stuck_timeout" not in info
    assert info["reward_components"]["stuck_timeout"] == 0.0
