import type { FeedItem } from "../types";

type Props = {
  items: FeedItem[];
};

export function LiveFeed({ items }: Props) {
  return (
    <section className="panel">
      <h2>实时日志</h2>
      <div className="feed">
        {items.length === 0 ? (
          <div className="feed-item">等待操作… 选择模型后点击「页内播放」或「快速评估」。</div>
        ) : (
          items.map((item) => (
            <div className="feed-item" key={item.id}>
              <strong>{item.text}</strong>
            </div>
          ))
        )}
      </div>
    </section>
  );
}
