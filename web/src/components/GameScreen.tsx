import { useEffect, useRef } from "react";

type Props = {
  frameBlob: Blob | null;
  playing: boolean;
  overlay?: string;
};

export function GameScreen({ frameBlob, playing, overlay }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const urlRef = useRef<string | null>(null);

  useEffect(() => {
    if (!frameBlob || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const url = URL.createObjectURL(frameBlob);
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = url;

    const img = new Image();
    img.onload = () => {
      if (canvas.width !== img.width || canvas.height !== img.height) {
        canvas.width = img.width;
        canvas.height = img.height;
      }
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
      if (urlRef.current === url) urlRef.current = null;
    };
    img.src = url;

    return () => {
      // keep last frame; revoke only stale urls in next tick handled above
    };
  }, [frameBlob]);

  useEffect(() => {
    return () => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    };
  }, []);

  return (
    <section className="panel game-panel">
      <div className="game-panel-head">
        <h2>游戏画面</h2>
        <span className={`live-dot ${playing ? "on" : ""}`}>
          {playing ? "LIVE" : "待机"}
        </span>
      </div>
      <div className="game-frame">
        <canvas ref={canvasRef} className="game-canvas" />
        {!frameBlob && (
          <div className="game-placeholder">
            <p>点击「页内播放」后，马里奥画面会直接显示在这里。</p>
          </div>
        )}
        {overlay && <div className="game-overlay">{overlay}</div>}
      </div>
    </section>
  );
}
