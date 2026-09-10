import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ControlPanel } from "./components/ControlPanel";
import { GameScreen } from "./components/GameScreen";
import { LevelStage } from "./components/LevelStage";
import { LiveFeed } from "./components/LiveFeed";
import type {
  EpisodeEnd,
  FeedItem,
  ProgressEvent,
  ReleaseInfo,
  Summary,
} from "./types";

function uid() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function playWsUrl() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/api/play/ws`;
}

export default function App() {
  const [release, setRelease] = useState<ReleaseInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState("full_level");
  const [episodes, setEpisodes] = useState(3);
  const [busy, setBusy] = useState(false);
  const [playRunning, setPlayRunning] = useState(false);
  const [frameBlob, setFrameBlob] = useState<Blob | null>(null);
  const [live, setLive] = useState<ProgressEvent | null>(null);
  const [episodeResults, setEpisodeResults] = useState<EpisodeEnd[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [status, setStatus] = useState("选择模型，点击「页内播放」开始演示。");
  const [statusKind, setStatusKind] = useState<"idle" | "live" | "error">("idle");
  const [feed, setFeed] = useState<FeedItem[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const pushFeed = useCallback((text: string) => {
    setFeed((prev) => [{ id: uid(), text }, ...prev].slice(0, 40));
  }, []);

  useEffect(() => {
    fetch("/api/release")
      .then(async (res) => {
        if (!res.ok) throw new Error(`API ${res.status}`);
        return res.json();
      })
      .then((data: ReleaseInfo) => {
        setRelease(data);
        const firstReady = data.models.find((m) => m.exists)?.key ?? data.models[0]?.key;
        if (firstReady) setSelected(firstReady);
      })
      .catch((err: Error) => {
        setError(
          `无法连接演示 API（${err.message}）。请先启动：python -m mario_rl.demo_api`,
        );
      });

    return () => {
      wsRef.current?.close();
    };
  }, []);

  const selectedModel = useMemo(
    () => release?.models.find((m) => m.key === selected) ?? null,
    [release, selected],
  );

  const handleWsPayload = useCallback(
    (event: string, payload: Record<string, unknown>) => {
      if (event === "status") {
        setStatus(String(payload.message ?? ""));
        setStatusKind(payload.phase === "done" ? "idle" : "live");
        pushFeed(String(payload.message ?? event));
        if (payload.phase === "done") setPlayRunning(false);
      } else if (event === "progress") {
        setLive(payload as unknown as ProgressEvent);
      } else if (event === "episode_end") {
        const ep = payload as unknown as EpisodeEnd;
        setEpisodeResults((prev) => [...prev, ep]);
        pushFeed(
          `E${ep.episode} 结束 · x=${ep.x_pos} · reward=${ep.reward}` +
            (ep.flag_get ? " · FLAG" : "") +
            (ep.stuck_timeout ? " · STUCK" : ""),
        );
      } else if (event === "episode_start") {
        pushFeed(`第 ${payload.episode}/${payload.episodes} 局开始`);
      } else if (event === "error") {
        setStatus(String(payload.message ?? "出错"));
        setStatusKind("error");
        setPlayRunning(false);
        pushFeed(`错误 · ${payload.message}`);
      }
    },
    [pushFeed],
  );

  const stopPlay = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "stop" }));
    } else {
      ws?.close();
      setPlayRunning(false);
    }
    pushFeed("请求停止播放");
  }, [pushFeed]);

  const runPlay = useCallback(() => {
    if (busy || playRunning) return;
    wsRef.current?.close();

    setPlayRunning(true);
    setLive(null);
    setEpisodeResults([]);
    setSummary(null);
    setStatusKind("live");
    setStatus("连接页内播放流…");
    pushFeed(`页内播放 · ${selected} · ${episodes} 局`);

    const ws = new WebSocket(playWsUrl());
    ws.binaryType = "blob";
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          action: "start",
          model_key: selected,
          episodes,
          device: "auto",
          delay: 0.02,
          jpeg_quality: 72,
          scale: 2,
        }),
      );
    };

    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        try {
          const msg = JSON.parse(ev.data) as {
            event: string;
            data: Record<string, unknown>;
          };
          handleWsPayload(msg.event, msg.data);
        } catch {
          pushFeed("无法解析服务端消息");
        }
        return;
      }
      setFrameBlob(ev.data as Blob);
    };

    ws.onerror = () => {
      setStatus("WebSocket 连接失败");
      setStatusKind("error");
      setPlayRunning(false);
      pushFeed("错误 · WebSocket 连接失败");
    };

    ws.onclose = () => {
      setPlayRunning(false);
      if (wsRef.current === ws) wsRef.current = null;
    };
  }, [busy, playRunning, selected, episodes, pushFeed, handleWsPayload]);

  const runEvaluate = async () => {
    if (busy || playRunning) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setBusy(true);
    setLive(null);
    setEpisodeResults([]);
    setSummary(null);
    setStatusKind("live");
    setStatus("正在连接评估流…");
    pushFeed(`快速评估 · ${selected} · ${episodes} 局`);

    try {
      const res = await fetch("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_key: selected,
          episodes,
          device: "auto",
          stream_every: 3,
        }),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) {
        throw new Error(`评估请求失败 (${res.status})`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() ?? "";

        for (const chunk of chunks) {
          const lines = chunk.split("\n");
          let event = "message";
          let data = "";
          for (const line of lines) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            if (line.startsWith("data:")) data += line.slice(5).trim();
          }
          if (!data) continue;
          const payload = JSON.parse(data) as Record<string, unknown>;

          if (event === "status") {
            setStatus(String(payload.message ?? ""));
            setStatusKind(payload.phase === "done" ? "idle" : "live");
            pushFeed(String(payload.message ?? event));
          } else if (event === "progress") {
            setLive(payload as unknown as ProgressEvent);
          } else if (event === "episode_end") {
            const ep = payload as unknown as EpisodeEnd;
            setEpisodeResults((prev) => [...prev, ep]);
            pushFeed(`E${ep.episode} 结束 · x=${ep.x_pos} · reward=${ep.reward}`);
          } else if (event === "summary") {
            const s = payload as unknown as Summary;
            setSummary(s);
            setStatus(
              `完成：均值 x=${s.x_pos_mean}，最大 x=${s.x_pos_max}，均值 reward=${s.reward_mean}`,
            );
            setStatusKind("idle");
            pushFeed(
              `汇总 · mean_x=${s.x_pos_mean} · max_x=${s.x_pos_max} · mean_r=${s.reward_mean}`,
            );
          } else if (event === "error") {
            setStatus(String(payload.message ?? "评估出错"));
            setStatusKind("error");
            pushFeed(`错误 · ${payload.message}`);
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setStatus((err as Error).message);
        setStatusKind("error");
        pushFeed(`错误 · ${(err as Error).message}`);
      }
    } finally {
      setBusy(false);
    }
  };

  if (error) {
    return (
      <div className="error-screen">
        <h1 className="brand-mark">
          Mario <span>PPO</span>
        </h1>
        <p>{error}</p>
      </div>
    );
  }

  if (!release) {
    return (
      <div className="loading-screen">
        <h1 className="brand-mark">
          Mario <span>PPO</span>
        </h1>
        <p>正在加载发布信息…</p>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="hero">
        <div>
          <h1 className="brand-mark">
            Mario <span>PPO</span>
          </h1>
          <p className="hero-copy">
            课程演示控制台：游戏画面直接嵌在页面中央，左侧切换模型并播放，右侧输出实时日志与指标。
          </p>
          <div className="hero-meta">
            <span className="chip">
              Release <strong>{release.release}</strong>
            </span>
            <span className="chip">
              Env <strong>{String(release.environment.env_id)}</strong>
            </span>
            <span className="chip">
              Stream <strong>WebSocket</strong>
            </span>
            {selectedModel && (
              <span className="chip">
                Active <strong>{selectedModel.asset}</strong>
              </span>
            )}
          </div>
        </div>
        <div className="hero-side">
          <div className="stat-tile">
            <div className="label">发布验证 · 完整模型</div>
            <div className="value">x = 2226</div>
          </div>
          <div className="stat-tile">
            <div className="label">发布验证 · 前沿模型</div>
            <div className="value">max 2994</div>
          </div>
          <div className="stat-tile">
            <div className="label">现场汇总</div>
            <div className="value">
              {summary ? `max ${summary.x_pos_max}` : live ? `x ${live.peak_x}` : "尚未运行"}
            </div>
          </div>
        </div>
      </header>

      <main className="layout">
        <ControlPanel
          models={release.models}
          selected={selected}
          episodes={episodes}
          busy={busy}
          playRunning={playRunning}
          onSelect={setSelected}
          onEpisodes={setEpisodes}
          onEvaluate={runEvaluate}
          onPlay={runPlay}
          onStopPlay={stopPlay}
        />
        <div className="center-column">
          <GameScreen
            frameBlob={frameBlob}
            playing={playRunning}
            overlay={playRunning && live ? `x=${live.x_pos}` : undefined}
          />
          <LevelStage
            goalX={release.level_goal_x}
            live={live}
            episodes={episodeResults}
            status={status}
            statusKind={statusKind}
          />
        </div>
        <LiveFeed items={feed} />
      </main>

      <section className="story">
        {release.story.map((item) => (
          <article className="story-card" key={item.title}>
            <h3>{item.title}</h3>
            <p>{item.text}</p>
          </article>
        ))}
      </section>

      <section className="panel">
        <h2>演示提示</h2>
        <ul className="notes">
          {release.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}
