from __future__ import annotations

from mario_rl.env import make_mario_env


def main() -> None:
    env = make_mario_env()
    try:
        obs, _ = env.reset()
        print(f"reset observation shape={obs.shape}, dtype={obs.dtype}")
        total_reward = 0.0
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total_reward += reward
            if terminated or truncated:
                obs, _ = env.reset()
        print(f"smoke test ok, total_reward={total_reward:.2f}, last_info={info}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
