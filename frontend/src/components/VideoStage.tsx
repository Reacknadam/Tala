import { useEffect, useRef, useState } from "react";
import type { BoundingBox, FootballFrame, VideoInfo } from "../services/api";
import { api } from "../services/api";

const TEAM_FILL: Record<number, string> = { 1: "#d99a3b", 2: "#3fb27f", 0: "#8a919c" };

/** Row-major 3x3 multiply. */
function matMul(a: number[], b: number[]): number[] {
  const o = new Array<number>(9);
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++)
      o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
  return o;
}

/** Apply row-major 3x3 homography to a point; null if behind the plane. */
function xform(m: number[], x: number, y: number): [number, number] | null {
  const w = m[6] * x + m[7] * y + m[8];
  if (Math.abs(w) < 1e-6) return null;
  return [(m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w];
}

/** Pitch-marking segments in meter space (x along length, y across width). */
function pitchFeatures(L: number, W: number): [number, number][][] {
  const feats: [number, number][][] = [
    [[0, 0], [L, 0], [L, W], [0, W], [0, 0]],
    [[L / 2, 0], [L / 2, W]],
  ];
  const circle: [number, number][] = [];
  const r = Math.min(9.15, Math.min(L, W) * 0.18);
  for (let i = 0; i <= 28; i++) {
    const a = (i / 28) * Math.PI * 2;
    circle.push([L / 2 + r * Math.cos(a), W / 2 + r * Math.sin(a)]);
  }
  feats.push(circle);
  if (L >= 80 && W >= 50) {
    feats.push([[0, (W - 40.32) / 2], [16.5, (W - 40.32) / 2], [16.5, (W + 40.32) / 2], [0, (W + 40.32) / 2]]);
    feats.push([[L, (W - 40.32) / 2], [L - 16.5, (W - 40.32) / 2], [L - 16.5, (W + 40.32) / 2], [L, (W + 40.32) / 2]]);
    feats.push([[0, (W - 18.32) / 2], [5.5, (W - 18.32) / 2], [5.5, (W + 18.32) / 2], [0, (W + 18.32) / 2]]);
    feats.push([[L, (W - 18.32) / 2], [L - 5.5, (W - 18.32) / 2], [L - 5.5, (W + 18.32) / 2], [L, (W + 18.32) / 2]]);
  }
  return feats;
}

interface Props {
  video: VideoInfo | null;
  searchMode: boolean;
  onSelect: (bbox: BoundingBox, frameNumber: number) => void;
  selection: BoundingBox | null;
  highlightBox: BoundingBox | null;
  jumpFlash: boolean;
  videoRef: React.RefObject<HTMLVideoElement>;
  calibrationMode?: boolean;
  calibrationPoints?: { x: number; y: number }[];
  onCalibrationPoint?: (x: number, y: number) => void;
  footballFrames?: FootballFrame[] | null;
  footballPitch?: { length: number; width: number; to_px?: number[] } | null;
  detectedPolygon?: number[][] | null;
}

/** Renders the active camera's video with a canvas overlay used for the
 * search-mode click/drag person selection, plus optional highlight boxes
 * (current selection, or the freshly-jumped-to match). All drawing math
 * converts between the video's *displayed* CSS size and its native pixel
 * resolution, because bounding boxes from the backend are always in native
 * video coordinates. */
export function VideoStage({ video, searchMode, onSelect, selection, highlightBox, jumpFlash, videoRef, calibrationMode = false, calibrationPoints = [], onCalibrationPoint, footballFrames = null, footballPitch = null, detectedPolygon = null }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragStart = useRef<{ x: number; y: number } | null>(null);
  const [dragRect, setDragRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null);

  const draw = () => {
    const canvas = canvasRef.current;
    const vid = videoRef.current;
    if (!canvas || !vid || !vid.videoWidth) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / vid.videoWidth;
    const scaleY = canvas.height / vid.videoHeight;

    const drawBox = (box: BoundingBox, color: string, dashed = false) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash(dashed ? [5, 4] : []);
      ctx.strokeRect(box.x1 * scaleX, box.y1 * scaleY, (box.x2 - box.x1) * scaleX, (box.y2 - box.y1) * scaleY);
      ctx.setLineDash([]);
    };

    if (selection) drawBox(selection, "#4b8bf4");
    if (highlightBox) drawBox(highlightBox, "#3fb27f");

    // Region recognized by auto-detect: translucent fill + outline.
    if (detectedPolygon && detectedPolygon.length >= 3) {
      ctx.beginPath();
      detectedPolygon.forEach(([x, y], i) => {
        if (i === 0) ctx.moveTo(x * scaleX, y * scaleY);
        else ctx.lineTo(x * scaleX, y * scaleY);
      });
      ctx.closePath();
      ctx.fillStyle = "rgba(63, 178, 127, 0.15)";
      ctx.fill();
      ctx.strokeStyle = "rgba(63, 178, 127, 0.7)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    if (calibrationPoints.length > 0) {
      if (calibrationPoints.length === 4) {
        ctx.beginPath();
        calibrationPoints.forEach((p, i) => {
          if (i === 0) ctx.moveTo(p.x * scaleX, p.y * scaleY);
          else ctx.lineTo(p.x * scaleX, p.y * scaleY);
        });
        ctx.closePath();
        ctx.fillStyle = "rgba(217, 154, 59, 0.10)";
        ctx.fill();
      }
      ctx.strokeStyle = "#d99a3b";
      ctx.lineWidth = 2;
      ctx.beginPath();
      calibrationPoints.forEach((p, i) => {
        const px = p.x * scaleX;
        const py = p.y * scaleY;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      if (calibrationPoints.length === 4) ctx.closePath();
      ctx.stroke();
      ctx.fillStyle = "#d99a3b";
      ctx.font = "bold 12px sans-serif";
      calibrationPoints.forEach((p, i) => {
        ctx.beginPath();
        ctx.arc(p.x * scaleX, p.y * scaleY, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillText(String(i + 1), p.x * scaleX + 8, p.y * scaleY - 8);
      });
    }

    if (footballFrames && footballFrames.length > 0) {
      const t = vid.currentTime;
      let i = 0;
      while (i < footballFrames.length - 1 && footballFrames[i + 1].t_sec <= t) i++;
      const a = footballFrames[i];
      const b = footballFrames[Math.min(i + 1, footballFrames.length - 1)];
      const span = Math.max(1e-6, b.t_sec - a.t_sec);
      const k = b === a ? 0 : Math.min(1, Math.max(0, (t - a.t_sec) / span));
      const lerp = (u: number, v: number) => u + (v - u) * k;

      // AR pitch markings: re-project pitch-space lines into the current
      // frame via the stored ref->current homography, so lines stay glued
      // to the grass while the broadcast camera pans/zooms.
      const hRef = a.h && b.h ? a.h.map((v, j) => lerp(v, b.h![j])) : a.h;
      if (footballPitch?.to_px && hRef) {
        const m = matMul(hRef, footballPitch.to_px);
        ctx.save();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.75)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 5]);
        for (const feat of pitchFeatures(footballPitch.length, footballPitch.width)) {
          ctx.beginPath();
          let started = false;
          for (const [x, y] of feat) {
            const p = xform(m, x, y);
            if (!p) { started = false; continue; }
            const cx = p[0] * scaleX, cy = p[1] * scaleY;
            if (!started) { ctx.moveTo(cx, cy); started = true; }
            else ctx.lineTo(cx, cy);
          }
          ctx.stroke();
        }
        ctx.restore();
      }

      const byId = new Map(b.players.map((p) => [p.id, p]));
      for (const p of a.players) {
        if (!p.bbox_px || !p.foot_px) continue; // pre-overlay analysis format
        const q = byId.get(p.id);
        const foot: [number, number] = q
          ? [lerp(p.foot_px[0], q.foot_px[0]), lerp(p.foot_px[1], q.foot_px[1])]
          : p.foot_px;
        const bbox = q ? p.bbox_px.map((v, j) => lerp(v, q.bbox_px[j])) : p.bbox_px;
        const speed = q ? lerp(p.speed_kmh, q.speed_kmh) : p.speed_kmh;
        const cx = foot[0] * scaleX;
        const cy = foot[1] * scaleY;
        const rx = Math.max(6, ((bbox[2] - bbox[0]) * scaleX) / 2);
        const color = TEAM_FILL[p.team] ?? TEAM_FILL[0];
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.ellipse(cx, cy, rx, rx * 0.35, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.fillRect(cx - 11, cy + 3, 22, 13);
        ctx.fillStyle = "#0a0b0d";
        ctx.font = "bold 10px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(String(p.id), cx, cy + 13);
        ctx.font = "11px sans-serif";
        ctx.lineWidth = 3;
        ctx.strokeStyle = "rgba(0,0,0,0.7)";
        const label = `${Math.round(speed)} km/h`;
        ctx.strokeText(label, cx, cy + 30);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, cx, cy + 30);
        ctx.textAlign = "left";
        if (a.owner_player_id === p.id) {
          ctx.fillStyle = "#e05a5a";
          ctx.beginPath();
          ctx.moveTo(cx, bbox[1] * scaleY - 4);
          ctx.lineTo(cx - 7, bbox[1] * scaleY - 16);
          ctx.lineTo(cx + 7, bbox[1] * scaleY - 16);
          ctx.closePath();
          ctx.fill();
        }
      }
      const ballPx =
        a.ball?.px && b.ball?.px
          ? [lerp(a.ball.px[0], b.ball.px[0]), lerp(a.ball.px[1], b.ball.px[1])]
          : a.ball?.px;
      if (ballPx) {
        const bx = ballPx[0] * scaleX;
        const by = ballPx[1] * scaleY;
        ctx.fillStyle = "#22c55e";
        ctx.strokeStyle = "#0a0b0d";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(bx, by + 8);
        ctx.lineTo(bx - 7, by - 6);
        ctx.lineTo(bx + 7, by - 6);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
    }

    if (dragRect) {
      ctx.strokeStyle = "#4b8bf4";
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(dragRect.x, dragRect.y, dragRect.w, dragRect.h);
      ctx.fillStyle = "rgba(75, 139, 244, 0.12)";
      ctx.fillRect(dragRect.x, dragRect.y, dragRect.w, dragRect.h);
      ctx.setLineDash([]);
    }
  };

  useEffect(() => {
    draw();
    const onResize = () => draw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selection, highlightBox, dragRect, video?.video_id, calibrationPoints]);

  // Football overlay needs per-frame redraws (video timeupdate fires ~4 Hz).
  useEffect(() => {
    if (!footballFrames || footballFrames.length === 0) return;
    let raf = 0;
    const loop = () => {
      draw();
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [footballFrames, video?.video_id]);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (calibrationMode && video && onCalibrationPoint) {
      const canvas = canvasRef.current!;
      const vid = videoRef.current!;
      const rect = canvas.getBoundingClientRect();
      onCalibrationPoint(
        ((e.clientX - rect.left) / rect.width) * vid.videoWidth,
        ((e.clientY - rect.top) / rect.height) * vid.videoHeight,
      );
      return;
    }
    if (!searchMode || !video) return;
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    dragStart.current = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    videoRef.current?.pause();
    setDragRect({ x: dragStart.current.x, y: dragStart.current.y, w: 0, h: 0 });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!dragStart.current) return;
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const x = Math.min(cx, dragStart.current.x);
    const y = Math.min(cy, dragStart.current.y);
    const w = Math.abs(cx - dragStart.current.x);
    const h = Math.abs(cy - dragStart.current.y);
    setDragRect({ x, y, w, h });
  };

  const handleMouseUp = () => {
    const vid = videoRef.current;
    const canvas = canvasRef.current;
    if (!dragStart.current || !dragRect || !vid || !canvas || !vid.videoWidth) {
      dragStart.current = null;
      return;
    }
    dragStart.current = null;
    if (dragRect.w < 8 || dragRect.h < 8) {
      setDragRect(null);
      return; // ignore accidental clicks / tiny drags
    }
    const scaleX = vid.videoWidth / canvas.getBoundingClientRect().width;
    const scaleY = vid.videoHeight / canvas.getBoundingClientRect().height;
    const bbox: BoundingBox = {
      x1: dragRect.x * scaleX,
      y1: dragRect.y * scaleY,
      x2: (dragRect.x + dragRect.w) * scaleX,
      y2: (dragRect.y + dragRect.h) * scaleY,
    };
    const frameNumber = Math.max(0, Math.round(vid.currentTime * (video?.fps ?? 25)));
    setDragRect(null);
    onSelect(bbox, frameNumber);
  };

  if (!video) {
    return (
      <div className="video-empty">
        No camera selected.
        <br />
        Add a video and select a camera from the left panel.
      </div>
    );
  }

  return (
    <div className="video-stack" ref={containerRef}>
      <video
        ref={videoRef}
        src={api.streamUrl(video.video_id)}
        controls
        onLoadedMetadata={draw}
        onTimeUpdate={draw}
      />
      <canvas
        ref={canvasRef}
        className={searchMode || calibrationMode ? "selecting" : ""}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
      />
      <div className="camera-label">{video.camera_id}</div>
      {jumpFlash && <div className="jump-flash">JUMPED TO MATCH</div>}
    </div>
  );
}
