export type ModelInfo = {
  key: string;
  title: string;
  blurb: string;
  asset: string;
  path: string;
  exists: boolean;
  deterministic: boolean;
  frontier_actions: string | null;
  parameters: Record<string, unknown>;
  validation: {
    episodes?: number;
    x_pos_mean?: number;
    x_pos_max?: number;
    flag_gets?: number;
    success_threshold?: number;
    successes?: number;
  };
};

export type ReleaseInfo = {
  release: string;
  created_at: string;
  environment: Record<string, unknown>;
  level_goal_x: number;
  models: ModelInfo[];
  notes: string[];
  story: { title: string; text: string }[];
};

export type ProgressEvent = {
  episode: number;
  steps: number;
  reward: number;
  x_pos: number;
  peak_x: number;
  progress: number;
  life?: number;
  time?: number;
  flag_get?: boolean;
  stuck_timeout?: boolean;
  death_detected?: boolean;
};

export type EpisodeEnd = {
  episode: number;
  reward: number;
  steps: number;
  x_pos: number;
  peak_x: number;
  progress: number;
  flag_get: boolean;
  stuck_timeout: boolean;
};

export type Summary = {
  episodes: number;
  reward_mean: number;
  x_pos_mean: number;
  x_pos_max: number;
  x_positions: number[];
  rewards: number[];
};

export type FeedItem = {
  id: string;
  text: string;
};
