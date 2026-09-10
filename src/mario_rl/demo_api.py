from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mario_rl.env import make_env_factory
from mario_rl.utils import PROJECT_ROOT, load_frontier_actions

RELEASE_PATH = PROJECT_ROOT / "configs" / "release_v0.2.0.json"
MODELS_DIR = PROJECT_ROOT / "models" / "release"
LEVEL_GOAL_X = 3161

app = FastAPI(title="Mario PPO Demo API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_session_lock = asyncio.Lock()
_play_active = False


class EvaluateRequest(BaseModel):
    model_key: str = Field(description="full_level or frontier_x2095")
    episodes: int = Field(default=3, ge=1, le=20)
    device: str = "auto"
    stream_every: int = Field(default=4, ge=1, le=30)


class PlayStartMessage(BaseModel):
    action: str = "start"
    model_key: str = "full_level"
    episodes: int = Field(default=3, ge=1, le=10)
    device: str = "auto"
    delay: float = Field(default=0.02, ge=0.0, le=0.2)
    jpeg_quality: int = Field(default=72, ge=40, le=95)
    scale: float = Field(default=2.0, ge=1.0, le=3.0)


def _load_release() -> dict[str, Any]:
    if not RELEASE_PATH.exists():
        raise HTTPException(status_code=404, detail="release config missing")
    return json.loads(RELEASE_PATH.read_text(encoding="utf-8-sig"))


def _model_catalog() -> dict[str, dict[str, Any]]:
    release = _load_release()
    catalog: dict[str, dict[str, Any]] = {}
    for item in release["models"]:
        asset = item["asset"]
        if asset.startswith("full_level"):
            key = "full_level"
            title = "完整关卡策略"
            blurb = "从 1-1 起点确定性推理，验证到达 x≈2226。"
        else:
            key = "frontier_x2095"
            title = "前沿技能策略"
            blurb = "从 x=2095 状态随机推理，越过 2300 的后半关技能模型。"
        path = MODELS_DIR / asset
        frontier = item.get("frontier_actions")
        catalog[key] = {
            "key": key,
            "title": title,
            "blurb": blurb,
            "asset": asset,
            "path": str(path),
            "exists": path.exists(),
            "deterministic": bool(item.get("deterministic", True)),
            "frontier_actions": (
                str(PROJECT_ROOT / "configs" / frontier) if frontier else None
            ),
            "parameters": item.get("parameters", {}),
            "validation": item.get("validation", {}),
        }
    return catalog


def _resolve_model(model_key: str) -> dict[str, Any]:
    catalog = _model_catalog()
    if model_key not in catalog:
        raise HTTPException(status_code=400, detail=f"Unknown model_key: {model_key}")
    model = catalog[model_key]
    if not model["exists"]:
        raise HTTPException(
            status_code=404,
            detail=f"Model file missing: {model['path']}. Extract models zip to models/release.",
        )
    return model


def _base_env_from_vec(vec_env: Any) -> Any:
    current = vec_env
    if hasattr(current, "venv"):
        current = current.venv
    if hasattr(current, "envs"):
        current = current.envs[0]
    return current


def _grab_rgb(vec_env: Any) -> np.ndarray:
    base = _base_env_from_vec(vec_env)
    screen = getattr(base, "screen", None)
    if screen is None and hasattr(base, "unwrapped"):
        screen = getattr(base.unwrapped, "screen", None)
    if screen is None:
        raise RuntimeError("Cannot read NES screen buffer")
    return np.asarray(screen)


def _encode_jpeg(rgb: np.ndarray, quality: int = 72, scale: float = 2.0) -> bytes:
    frame = rgb
    if scale != 1.0:
        frame = cv2.resize(
            frame,
            (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
            interpolation=cv2.INTER_NEAREST,
        )
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def _jsonable_info(info: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in info.items():
        if key == "terminal_observation":
            continue
        if isinstance(value, np.generic):
            out[key] = value.item()
        elif isinstance(value, np.ndarray):
            continue
        else:
            out[key] = value
    return out


def _progress_payload(
    episode: int,
    steps: int,
    total_reward: float,
    peak_x: int,
    last_info: dict[str, Any],
) -> dict[str, Any]:
    info = _jsonable_info(last_info)
    x_pos = int(info.get("x_pos", 0))
    peak = max(peak_x, x_pos)
    return {
        "episode": episode,
        "steps": steps,
        "reward": round(float(total_reward), 2),
        "x_pos": x_pos,
        "peak_x": peak,
        "progress": min(1.0, peak / LEVEL_GOAL_X),
        "life": info.get("life"),
        "time": info.get("time"),
        "flag_get": bool(info.get("flag_get", False)),
        "stuck_timeout": bool(info.get("stuck_timeout", False)),
        "death_detected": bool(info.get("death_detected", False)),
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "project": "mario-ppo",
        "version": "0.2.0",
        "play_active": _play_active,
    }


@app.get("/api/release")
def release_info() -> dict[str, Any]:
    release = _load_release()
    catalog = _model_catalog()
    return {
        "release": release.get("release"),
        "created_at": release.get("created_at"),
        "environment": release.get("environment"),
        "level_goal_x": LEVEL_GOAL_X,
        "models": list(catalog.values()),
        "notes": [
            "游戏画面已嵌入网页，通过 WebSocket 推送 NES 帧，无需额外弹窗。",
            "单关环境：死亡一次即结束当前 episode，不是攒满 3 命。",
            "发布模型尚未稳定通关（flag_get=False）。",
        ],
        "story": [
            {
                "title": "观测",
                "text": "84×84 灰度 + 4 帧堆叠，动作空间 RIGHT_ONLY，跳帧 4。",
            },
            {
                "title": "奖励",
                "text": "只奖励首次到达的新 x 位置，并惩罚死亡与卡住。",
            },
            {
                "title": "课程",
                "text": "动作前缀恢复到 x=2095，混合完整起点与前沿并行训练。",
            },
            {
                "title": "现状",
                "text": "完整模型到 2226；前沿随机评估可过 2300，最大 2994。",
            },
        ],
    }


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _run_evaluation(req: EvaluateRequest) -> AsyncIterator[str]:
    if _session_lock.locked():
        yield _sse("error", {"message": "已有任务在运行，请稍后再试。"})
        return

    async with _session_lock:
        model_meta = _resolve_model(req.model_key)
        params = model_meta["parameters"]
        frontier_path = model_meta["frontier_actions"]
        frontier_actions = load_frontier_actions(frontier_path)

        yield _sse(
            "status",
            {
                "phase": "loading",
                "message": f"加载模型 {model_meta['asset']} …",
                "model": model_meta,
            },
        )

        loop = asyncio.get_running_loop()

        def setup():
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

            env = DummyVecEnv(
                [
                    make_env_factory(
                        "SuperMarioBros-1-1-v0",
                        "right",
                        4,
                        42,
                        rank=0,
                        stuck_limit=int(params.get("stuck_limit", 120)),
                        stuck_penalty=float(params.get("stuck_penalty", 25)),
                        death_penalty=float(params.get("death_penalty", 100)),
                        max_jump_hold=int(params.get("max_jump_hold", 0)),
                        frontier_actions=frontier_actions,
                        milestone_x=tuple(params.get("milestone_x", [])),
                        milestone_bonus=float(params.get("milestone_bonus", 0)),
                    )
                ]
            )
            env = VecFrameStack(env, n_stack=4, channels_order="last")
            model = PPO.load(model_meta["path"], env=env, device=req.device)
            return env, model

        try:
            env, model = await loop.run_in_executor(None, setup)
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", {"message": f"模型加载失败: {exc}"})
            return

        deterministic = req.model_key != "frontier_x2095"

        yield _sse(
            "status",
            {
                "phase": "running",
                "message": "评估开始",
                "episodes": req.episodes,
                "deterministic": deterministic,
            },
        )

        rewards: list[float] = []
        x_positions: list[int] = []

        try:
            for episode in range(req.episodes):
                obs = await loop.run_in_executor(None, env.reset)
                done = [False]
                total_reward = 0.0
                steps = 0
                last_info: dict[str, Any] = {}
                peak_x = 0

                yield _sse(
                    "episode_start",
                    {"episode": episode + 1, "episodes": req.episodes},
                )

                while not done[0]:

                    def step_once(current_obs=obs):
                        action, _ = model.predict(current_obs, deterministic=deterministic)
                        return env.step(action)

                    obs, reward, done, infos = await loop.run_in_executor(None, step_once)
                    total_reward += float(reward[0])
                    steps += 1
                    last_info = infos[0]
                    x_pos = int(last_info.get("x_pos", 0))
                    peak_x = max(peak_x, x_pos)

                    if steps % req.stream_every == 0 or done[0]:
                        yield _sse(
                            "progress",
                            _progress_payload(
                                episode + 1, steps, total_reward, peak_x, last_info
                            ),
                        )

                x_pos = int(_jsonable_info(last_info).get("x_pos", 0))
                rewards.append(total_reward)
                x_positions.append(x_pos)
                end_payload = _progress_payload(
                    episode + 1, steps, total_reward, peak_x, last_info
                )
                yield _sse(
                    "episode_end",
                    {
                        "episode": end_payload["episode"],
                        "reward": end_payload["reward"],
                        "steps": end_payload["steps"],
                        "x_pos": end_payload["x_pos"],
                        "peak_x": end_payload["peak_x"],
                        "progress": end_payload["progress"],
                        "flag_get": end_payload["flag_get"],
                        "stuck_timeout": end_payload["stuck_timeout"],
                    },
                )

            summary = {
                "episodes": req.episodes,
                "reward_mean": round(sum(rewards) / len(rewards), 2) if rewards else 0,
                "x_pos_mean": round(sum(x_positions) / len(x_positions), 1) if x_positions else 0,
                "x_pos_max": max(x_positions) if x_positions else 0,
                "x_positions": x_positions,
                "rewards": [round(r, 2) for r in rewards],
            }
            yield _sse("summary", summary)
            yield _sse("status", {"phase": "done", "message": "评估完成"})
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", {"message": f"评估中断: {exc}"})
        finally:
            await loop.run_in_executor(None, env.close)


@app.post("/api/evaluate")
async def evaluate(req: EvaluateRequest) -> StreamingResponse:
    return StreamingResponse(
        _run_evaluation(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/play/status")
def play_status() -> dict[str, Any]:
    return {"running": _play_active, "embedded": True}


async def _ws_send_json(websocket: WebSocket, event: str, data: dict[str, Any]) -> None:
    await websocket.send_text(json.dumps({"event": event, "data": data}, ensure_ascii=False))


@app.websocket("/api/play/ws")
async def play_ws(websocket: WebSocket) -> None:
    global _play_active
    await websocket.accept()
    stop_requested = False

    async def listen_for_stop() -> None:
        nonlocal stop_requested
        while not stop_requested:
            try:
                message = await websocket.receive_text()
            except WebSocketDisconnect:
                stop_requested = True
                return
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                continue
            if payload.get("action") == "stop":
                stop_requested = True
                return

    try:
        raw = await websocket.receive_text()
        start = PlayStartMessage.model_validate_json(raw)
        if start.action != "start":
            await _ws_send_json(websocket, "error", {"message": "首条消息必须是 start"})
            await websocket.close()
            return

        if _session_lock.locked():
            await _ws_send_json(websocket, "error", {"message": "已有任务在运行，请稍后再试。"})
            await websocket.close()
            return

        async with _session_lock:
            _play_active = True
            model_meta = _resolve_model(start.model_key)
            params = model_meta["parameters"]
            frontier_actions = load_frontier_actions(model_meta["frontier_actions"])
            deterministic = start.model_key != "frontier_x2095"
            loop = asyncio.get_running_loop()
            stopper = asyncio.create_task(listen_for_stop())

            await _ws_send_json(
                websocket,
                "status",
                {
                    "phase": "loading",
                    "message": f"加载模型 {model_meta['asset']} …",
                    "model": model_meta,
                },
            )

            def setup():
                from stable_baselines3 import PPO
                from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

                env = DummyVecEnv(
                    [
                        make_env_factory(
                            "SuperMarioBros-1-1-v0",
                            "right",
                            4,
                            42,
                            rank=0,
                            stuck_limit=int(params.get("stuck_limit", 120)),
                            stuck_penalty=float(params.get("stuck_penalty", 25)),
                            death_penalty=float(params.get("death_penalty", 100)),
                            max_jump_hold=int(params.get("max_jump_hold", 0)),
                            frontier_actions=frontier_actions,
                            milestone_x=tuple(params.get("milestone_x", [])),
                            milestone_bonus=float(params.get("milestone_bonus", 0)),
                        )
                    ]
                )
                env = VecFrameStack(env, n_stack=4, channels_order="last")
                model = PPO.load(model_meta["path"], env=env, device=start.device)
                return env, model

            try:
                env, model = await loop.run_in_executor(None, setup)
            except Exception as exc:  # noqa: BLE001
                await _ws_send_json(websocket, "error", {"message": f"模型加载失败: {exc}"})
                stopper.cancel()
                _play_active = False
                return

            await _ws_send_json(
                websocket,
                "status",
                {
                    "phase": "running",
                    "message": "页内播放开始",
                    "episodes": start.episodes,
                    "deterministic": deterministic,
                },
            )

            try:
                for episode in range(start.episodes):
                    if stop_requested:
                        break
                    obs = await loop.run_in_executor(None, env.reset)
                    done = [False]
                    total_reward = 0.0
                    steps = 0
                    last_info: dict[str, Any] = {}
                    peak_x = 0

                    await _ws_send_json(
                        websocket,
                        "episode_start",
                        {"episode": episode + 1, "episodes": start.episodes},
                    )

                    # initial frame
                    rgb = await loop.run_in_executor(None, lambda: _grab_rgb(env))
                    jpeg = await loop.run_in_executor(
                        None,
                        lambda: _encode_jpeg(rgb, start.jpeg_quality, start.scale),
                    )
                    await websocket.send_bytes(jpeg)

                    while not done[0] and not stop_requested:
                        t0 = time.perf_counter()

                        def step_once(current_obs=obs):
                            action, _ = model.predict(
                                current_obs, deterministic=deterministic
                            )
                            return env.step(action)

                        obs, reward, done, infos = await loop.run_in_executor(
                            None, step_once
                        )
                        total_reward += float(reward[0])
                        steps += 1
                        last_info = infos[0]
                        x_pos = int(last_info.get("x_pos", 0))
                        peak_x = max(peak_x, x_pos)

                        rgb = await loop.run_in_executor(None, lambda: _grab_rgb(env))
                        jpeg = await loop.run_in_executor(
                            None,
                            lambda: _encode_jpeg(rgb, start.jpeg_quality, start.scale),
                        )
                        await websocket.send_bytes(jpeg)

                        if steps % 2 == 0 or done[0]:
                            payload = _progress_payload(
                                episode + 1, steps, total_reward, peak_x, last_info
                            )
                            peak_x = int(payload["peak_x"])
                            await _ws_send_json(websocket, "progress", payload)

                        elapsed = time.perf_counter() - t0
                        sleep_for = start.delay - elapsed
                        if sleep_for > 0:
                            await asyncio.sleep(sleep_for)

                    end_payload = _progress_payload(
                        episode + 1, steps, total_reward, peak_x, last_info
                    )
                    await _ws_send_json(
                        websocket,
                        "episode_end",
                        {
                            "episode": end_payload["episode"],
                            "reward": end_payload["reward"],
                            "steps": end_payload["steps"],
                            "x_pos": end_payload["x_pos"],
                            "peak_x": end_payload["peak_x"],
                            "progress": end_payload["progress"],
                            "flag_get": end_payload["flag_get"],
                            "stuck_timeout": end_payload["stuck_timeout"],
                        },
                    )

                await _ws_send_json(
                    websocket,
                    "status",
                    {
                        "phase": "done",
                        "message": "播放结束" if not stop_requested else "已停止播放",
                    },
                )
            except WebSocketDisconnect:
                stop_requested = True
            except Exception as exc:  # noqa: BLE001
                await _ws_send_json(websocket, "error", {"message": f"播放中断: {exc}"})
            finally:
                stopper.cancel()
                await loop.run_in_executor(None, env.close)
                _play_active = False
    except WebSocketDisconnect:
        _play_active = False
    except Exception as exc:  # noqa: BLE001
        _play_active = False
        try:
            await _ws_send_json(websocket, "error", {"message": str(exc)})
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    import uvicorn

    uvicorn.run(
        "mario_rl.demo_api:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
    )


if __name__ == "__main__":
    main()
