export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface VideoInfo {
  video_id: string;
  camera_id: string;
  filename: string;
  duration_sec: number;
  fps: number;
  width: number;
  height: number;
  status: "unindexed" | "indexing" | "indexed" | "error";
  track_count: number;
  detection_count: number;
  error?: string | null;
}

export interface IndexJobStatus {
  video_id: string;
  status: "queued" | "running" | "done" | "error";
  progress: number;
  frames_processed: number;
  total_frames: number;
  tracks_found: number;
  detections_stored: number;
  error?: string | null;
}

export interface MatchResult {
  match_id: string;
  video_id: string;
  camera_id: string;
  track_id: number;
  frame_number: number;
  timestamp_sec: number;
  similarity: number;
  bbox: BoundingBox;
  thumbnail_base64: string;
}

export interface SearchResponse {
  query_video_id: string;
  query_bbox: BoundingBox;
  query_thumbnail_base64: string;
  matches: MatchResult[];
  searched_videos: number;
  candidates_scanned: number;
}

export interface FootballPlayerStat {
  track_id: number;
  team: number;
  distance_m: number;
  mean_speed_kmh: number;
  max_speed_kmh: number;
}

export interface FootballFrame {
  frame: number;
  t_sec: number;
  players: {
    id: number; team: number; x: number; y: number; speed_kmh: number;
    bbox_px: [number, number, number, number]; foot_px: [number, number];
  }[];
  ball: { x: number; y: number; px: [number, number] | null; bbox_px: [number, number, number, number] | null } | null;
  owner_team: number | null;
  owner_player_id: number | null;
}

export interface FootballAnalysis {
  video_id: string;
  fps: number;
  stride: number;
  truncated: boolean;
  pitch?: { length: number; width: number };
  frames_sampled: number;
  players: FootballPlayerStat[];
  possession: { shares: Record<string, number>; segments: { team: number; start_frame: number; end_frame: number }[] };
  events: { type: string; track_id: number; start_frame: number; end_frame: number }[];
  frames: FootballFrame[];
}

const BASE = "/api";

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  health(): Promise<{ status: string }> {
    return fetch(`${BASE}/health`).then((r) => asJson(r));
  },

  listVideos(): Promise<VideoInfo[]> {
    return fetch(`${BASE}/videos`).then((r) => asJson(r));
  },

  getVideo(videoId: string): Promise<VideoInfo> {
    return fetch(`${BASE}/videos/${videoId}`).then((r) => asJson(r));
  },

  uploadVideo(cameraId: string, file: File): Promise<VideoInfo> {
    const form = new FormData();
    form.append("camera_id", cameraId);
    form.append("file", file);
    return fetch(`${BASE}/videos`, { method: "POST", body: form }).then((r) => asJson(r));
  },

  startIndexing(videoId: string, sampleEveryNFrames = 5): Promise<IndexJobStatus> {
    return fetch(`${BASE}/videos/${videoId}/index`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_every_n_frames: sampleEveryNFrames }),
    }).then((r) => asJson(r));
  },

  getIndexStatus(videoId: string): Promise<IndexJobStatus> {
    return fetch(`${BASE}/videos/${videoId}/index/status`).then((r) => asJson(r));
  },

  streamUrl(videoId: string): string {
    return `${BASE}/videos/${videoId}/stream`;
  },

  getFootballCalibration(videoId: string): Promise<{ points: number[][] } | null> {
    return fetch(`${BASE}/football/${videoId}/calibration`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  },

  saveFootballCalibration(videoId: string, points: { x: number; y: number }[]): Promise<unknown> {
    return fetch(`${BASE}/football/${videoId}/calibration`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ points }),
    }).then((r) => asJson(r));
  },

  analyzeFootball(videoId: string): Promise<{ frames_available: number }> {
    return fetch(`${BASE}/football/${videoId}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ max_sampled_frames: 600, stride: 5 }),
    }).then((r) => asJson(r));
  },

  getFootballAnalysis(videoId: string): Promise<FootballAnalysis | null> {
    return fetch(`${BASE}/football/${videoId}/analysis`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  },

  search(params: {
    video_id: string;
    frame_number: number;
    bbox: BoundingBox;
    threshold: number;
    exclude_source_video: boolean;
    top_k?: number;
  }): Promise<SearchResponse> {
    return fetch(`${BASE}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ top_k: 25, ...params }),
    }).then((r) => asJson(r));
  },
};
