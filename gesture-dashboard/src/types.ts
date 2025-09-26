// types.ts

/** One run file returned by GET /runs */
export interface Run {
  id: string;
  file: string;
  bytes: number;
  started_at: string | null;
}

/** Histogram like { "Open": 123, "Close": 45, "Unknown": 9 } */
export type ClassHistogram = Record<string, number>;

/**
 * Confusion matrix: outer key = ground truth,
 * inner key = predicted label. Keys may be missing,
 * so use Partial for safety.
 */
export type ConfusionMatrix = Record<string, Partial<Record<string, number>>>;

/** Latency summary. Keys may be null when no samples exist. */
export interface Latency {
  mean: number | null;
  p50: number | null;
  p90: number | null;
  p99: number | null;
}

/** Metrics payload from GET /runs/{id}/metrics */
export interface Metrics {
  run_id: string;
  total_frames: number;
  class_histogram: ClassHistogram;
  confusion: ConfusionMatrix;
  latency: Latency;
  labels_seen: string[];
}

/** One raw event line from the JSONL log (shape is flexible) */
export type LogEvent = Record<string, unknown>;

/** Paged events from GET /runs/{id}/events */
export interface PagedEvents {
  run_id: string;
  offset: number;
  limit: number;
  total: number;
  events: LogEvent[];
}

export type Video = {
  id: string;
  file: string;
  bytes: number;
  created_at: string;
  url: string; // /media/recordings/<file>
};
