import { useEffect, useState } from "react";
import { api, type FootballAnalysis, type VideoInfo } from "../services/api";

interface Props {
  video: VideoInfo | null;
  videoRef: React.RefObject<HTMLVideoElement>;
  calibrationPoints: { x: number; y: number }[];
  onCalibrationPointsChange: (points: { x: number; y: number }[]) => void;
  calibrating: boolean;
  onToggleCalibrating: () => void;
  analysis: FootballAnalysis | null;
  onAnalysis: (analysis: FootballAnalysis | null) => void;
  detectedPolygon: number[][] | null;
  onDetectedPolygon: (poly: number[][] | null) => void;
}

const TEAM_FILL: Record<number, string> = { 1: "#d99a3b", 2: "#3fb27f", 0: "#8a919c" };

/** Football mode: pitch calibration (manual clicks or automatic field
 * recognition), 2D tactical view synced to playback, possession shares and
 * per-player distance/speed. Positions are in meters; calibration lives in
 * the *first frame's* pose (the analysis reference). */
export function FootballPanel({ video, videoRef, calibrationPoints, onCalibrationPointsChange, calibrating, onToggleCalibrating, analysis, onAnalysis, detectedPolygon, onDetectedPolygon }: Props) {
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [now, setNow] = useState(0);
  const [pitchLength, setPitchLength] = useState(105);
  const [pitchWidth, setPitchWidth] = useState(68);

  useEffect(() => {
    if (!video) return;
    api.getFootballCalibration(video.video_id).then((cal) => {
      if (cal) {
        onCalibrationPointsChange(cal.points.map(([x, y]) => ({ x, y })));
        if (cal.pitch_length) setPitchLength(cal.pitch_length);
        if (cal.pitch_width) setPitchWidth(cal.pitch_width);
      }
    });
    api.getFootballAnalysis(video.video_id).then(onAnalysis);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [video?.video_id]);

  useEffect(() => {
    const timer = setInterval(() => setNow(videoRef.current?.currentTime ?? 0), 250);
    return () => clearInterval(timer);
  }, [videoRef]);

  if (!video) return <div className="football-panel"><p className="panel-hint">Select a video to analyse it as a match.</p></div>;

  const fullFrame = () => {
    const w = video.width - 16;
    const h = video.height - 16;
    onDetectedPolygon(null);
    onCalibrationPointsChange([{ x: 8, y: h }, { x: 8, y: 8 }, { x: w, y: 8 }, { x: w, y: h }]);
  };

  const detect = async () => {
    setBusy(true);
    setStatus("Recognizing the field on the first frame…");
    try {
      // Calibration lives in the first frame's pose: seek there so the
      // overlayed result is exactly what was detected.
      if (videoRef.current) videoRef.current.currentTime = 0;
      const found = await api.detectPitch(video.video_id, 0);
      if (!found) {
        setStatus("No field recognized — try calibrating manually.");
        return;
      }
      onDetectedPolygon(found.polygon);
      onCalibrationPointsChange(found.points.map(([x, y]) => ({ x, y })));
      setStatus("Field recognized. Adjust corners or dimensions if needed, then save.");
    } finally {
      setBusy(false);
    }
  };

  const save = async () => {
    setBusy(true);
    try {
      await api.saveFootballCalibration(video.video_id, calibrationPoints, pitchLength, pitchWidth);
      setStatus("Calibration saved. You can run the analysis.");
    } catch (err) {
      setStatus(`Calibration failed: ${(err as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const run = async () => {
    setBusy(true);
    setStatus("Analysing sampled frames (tracking, pitch mapping, possession)…");
    try {
      await api.analyzeFootball(video.video_id);
      onAnalysis(await api.getFootballAnalysis(video.video_id));
      setStatus("Analysis complete.");
    } catch (err) {
      setStatus(`Analysis failed: ${(err as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const frame = analysis?.frames.reduce((best, f) => (f.t_sec <= now ? f : best), analysis?.frames[0]);
  const shares = analysis?.possession.shares ?? {};
  const pitchL = analysis?.pitch?.length ?? 105;
  const pitchW = analysis?.pitch?.width ?? 68;
  // Always render the pitch horizontally: transpose coords when the
  // calibrated region is taller than it is long (e.g. a touchline-to-
  // touchline strip is legitimately W > L).
  const rot = pitchW > pitchL;
  const H = Math.max(pitchL, pitchW);
  const V = Math.min(pitchL, pitchW);
  const pt = (x: number, y: number): [number, number] => (rot ? [y, x] : [x, y]);
  const fullSize = H >= 80 && V >= 50;

  return (
    <div className="football-panel">
      <div className="football-controls">
        <strong>Football</strong>
        <button className={calibrating ? "on" : ""} onClick={onToggleCalibrating}>
          {calibrating ? "Clicking pitch corners…" : "Calibrate pitch"}
        </button>
        <button disabled={busy} onClick={detect}>Auto-detect pitch</button>
        <span className="panel-hint">
          {calibrationPoints.length}/4 corners (order: near-left, far-left, far-right, near-right)
        </span>
        <label className="pitch-dims">
          L <input type="number" value={pitchLength} min={10} max={120} onChange={(e) => setPitchLength(+e.target.value || 105)} /> m
        </label>
        <label className="pitch-dims">
          W <input type="number" value={pitchWidth} min={10} max={90} onChange={(e) => setPitchWidth(+e.target.value || 68)} /> m
        </label>
        <button onClick={fullFrame}>Full-frame pitch</button>
        <button onClick={() => { onCalibrationPointsChange([]); onDetectedPolygon(null); }}>Clear</button>
        <button disabled={busy || calibrationPoints.length !== 4} onClick={save}>Save calibration</button>
        <button className="primary" disabled={busy} onClick={run}>{busy ? "Working…" : "Analyse match"}</button>
        {status && <span className="panel-hint">{status}</span>}
      </div>

      {analysis && frame && (
        <div className="football-grid">
          <svg className="pitch" viewBox={`-3 -3 ${H + 6} ${V + 6}`} role="img" aria-label="Tactical pitch view">
            <rect x="0" y="0" width={H} height={V} className="pitch-bg" />
            <line x1={H / 2} y1="0" x2={H / 2} y2={V} className="pitch-line" />
            <circle cx={H / 2} cy={V / 2} r={Math.min(9.15, V * 0.18)} className="pitch-line" fill="none" />
            {fullSize && (
              <>
                <rect x="0" y={(V - 40.32) / 2} width="16.5" height="40.32" className="pitch-line" fill="none" />
                <rect x={H - 16.5} y={(V - 40.32) / 2} width="16.5" height="40.32" className="pitch-line" fill="none" />
                <rect x="0" y={(V - 18.32) / 2} width="5.5" height="18.32" className="pitch-line" fill="none" />
                <rect x={H - 5.5} y={(V - 18.32) / 2} width="5.5" height="18.32" className="pitch-line" fill="none" />
              </>
            )}
            {frame.players.map((p) => {
              const [px, py] = pt(p.x, p.y);
              return (
                <g key={p.id}>
                  <circle cx={px} cy={py} r="1.3" fill={TEAM_FILL[p.team] ?? TEAM_FILL[0]} stroke="#0a0b0d" strokeWidth="0.3" />
                  <text x={px + 1.6} y={py + 0.8} className="pitch-id">{p.id}</text>
                </g>
              );
            })}
            {frame.ball && (() => { const [bx, by] = pt(frame.ball!.x, frame.ball!.y); return <circle cx={bx} cy={by} r="0.9" fill="#fff" stroke="#0a0b0d" strokeWidth="0.3" />; })()}
          </svg>

          <div className="football-stats">
            <div className="possession">
              <div className="poss-row"><span>Team 1</span><div className="bar"><div className="bar-fill t1" style={{ width: `${shares["1"] ?? 0}%` }} /></div><b>{shares["1"] ?? 0}%</b></div>
              <div className="poss-row"><span>Team 2</span><div className="bar"><div className="bar-fill t2" style={{ width: `${shares["2"] ?? 0}%` }} /></div><b>{shares["2"] ?? 0}%</b></div>
              <span className="panel-hint">
                t = {now.toFixed(1)}s · owner: {frame.owner_team ? `Team ${frame.owner_team}` : "—"} ·{" "}
                {analysis.events.length} high-speed runs · {analysis.frames_sampled} frames sampled
                {analysis.truncated ? " (capped)" : ""}
              </span>
            </div>
            <table className="players-table">
              <thead><tr><th>#</th><th>Team</th><th>Dist (m)</th><th>Mean</th><th>Max</th></tr></thead>
              <tbody>
                {[...analysis.players].sort((a, b) => b.distance_m - a.distance_m).slice(0, 12).map((p) => (
                  <tr key={p.track_id}>
                    <td>{p.track_id}</td>
                    <td><span className="team-dot" style={{ background: TEAM_FILL[p.team] ?? TEAM_FILL[0] }} />{p.team || "—"}</td>
                    <td>{p.distance_m.toFixed(0)}</td>
                    <td>{p.mean_speed_kmh.toFixed(1)}</td>
                    <td>{p.max_speed_kmh.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
