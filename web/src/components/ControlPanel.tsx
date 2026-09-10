import type { ModelInfo } from "../types";

type Props = {
  models: ModelInfo[];
  selected: string;
  episodes: number;
  busy: boolean;
  playRunning: boolean;
  onSelect: (key: string) => void;
  onEpisodes: (n: number) => void;
  onEvaluate: () => void;
  onPlay: () => void;
  onStopPlay: () => void;
};

export function ControlPanel({
  models,
  selected,
  episodes,
  busy,
  playRunning,
  onSelect,
  onEpisodes,
  onEvaluate,
  onPlay,
  onStopPlay,
}: Props) {
  return (
    <section className="panel">
      <h2>演示控制</h2>
      {models.map((model) => (
        <button
          key={model.key}
          type="button"
          className={`model-card ${selected === model.key ? "active" : ""}`}
          onClick={() => onSelect(model.key)}
          disabled={busy || playRunning}
        >
          <div className="title">{model.title}</div>
          <p className="blurb">{model.blurb}</p>
          <div className="meta">
            <span className={`tag ${model.exists ? "ok" : "bad"}`}>
              {model.exists ? "模型就绪" : "缺失"}
            </span>
            <span className="tag">
              {model.deterministic ? "deterministic" : "stochastic"}
            </span>
            {model.validation.x_pos_max != null && (
              <span className="tag warn">max {model.validation.x_pos_max}</span>
            )}
          </div>
        </button>
      ))}

      <div className="controls">
        <div className="field">
          <label htmlFor="episodes">局数</label>
          <input
            id="episodes"
            type="number"
            min={1}
            max={10}
            value={episodes}
            disabled={busy || playRunning}
            onChange={(e) => onEpisodes(Number(e.target.value) || 1)}
          />
        </div>
        <div className="actions">
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy || playRunning}
            onClick={onPlay}
          >
            {playRunning ? "页内播放中…" : "页内播放"}
          </button>
          {playRunning ? (
            <button type="button" className="btn btn-ghost" onClick={onStopPlay}>
              停止播放
            </button>
          ) : (
            <button
              type="button"
              className="btn btn-secondary"
              disabled={busy}
              onClick={onEvaluate}
            >
              {busy ? "评估进行中…" : "快速评估（无画面）"}
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
