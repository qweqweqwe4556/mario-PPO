import { useMemo } from "react";
import type { EpisodeEnd, ProgressEvent } from "../types";

type Props = {
  goalX: number;
  live: ProgressEvent | null;
  episodes: EpisodeEnd[];
  status: string;
  statusKind: "idle" | "live" | "error";
};

export function LevelStage({ goalX, live, episodes, status, statusKind }: Props) {
  const x = live?.peak_x ?? episodes.at(-1)?.peak_x ?? 0;
  const pct = Math.min(100, (x / goalX) * 100);
  const markers = useMemo(
    () => [
      { x: 2095, label: "前沿 2095" },
      { x: 2300, label: "目标 2300" },
      { x: goalX, label: "旗杆" },
    ],
    [goalX],
  );

  return (
    <section className="panel stage">
      <h2>1-1 进度轨道</h2>
      <div className="level-track" aria-label="关卡进度可视化">
        <div className="clouds" />
        <div className="ground-grid" />
        <div className="markers">
          {markers.map((m) => (
            <div
              key={m.x}
              className="marker"
              style={{ left: `${Math.min(96, (m.x / goalX) * 100)}%` }}
            >
              {m.label}
            </div>
          ))}
        </div>
        <div className="flag" />
        <div className="agent" style={{ left: `${pct}%` }} title={`x=${x}`} />
      </div>

      <div className="metrics">
        <div className="metric">
          <div className="k">当前 x_pos</div>
          <div className="v">{live?.x_pos ?? "—"}</div>
        </div>
        <div className="metric">
          <div className="k">本局峰值</div>
          <div className="v">{live?.peak_x ?? "—"}</div>
        </div>
        <div className="metric">
          <div className="k">累计奖励</div>
          <div className="v">{live?.reward?.toFixed?.(1) ?? "—"}</div>
        </div>
        <div className="metric">
          <div className="k">步数 / 时间</div>
          <div className="v">
            {live ? `${live.steps} / ${live.time ?? "—"}` : "—"}
          </div>
        </div>
      </div>

      <div className="episode-bars">
        {episodes.length === 0 ? (
          <div className="status-line">运行评估后，这里会逐局画出到达位置。</div>
        ) : (
          episodes.map((ep) => (
            <div className="bar-row" key={ep.episode}>
              <span>E{ep.episode}</span>
              <div className="bar-track">
                <div
                  className="bar-fill"
                  style={{ width: `${Math.min(100, (ep.peak_x / goalX) * 100)}%` }}
                />
              </div>
              <span>{ep.x_pos}</span>
            </div>
          ))
        )}
      </div>

      <div className={`status-line ${statusKind}`}>{status}</div>
    </section>
  );
}
