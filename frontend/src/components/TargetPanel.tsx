import { useEffect, useRef, useState } from "react";
import type { BoundingBox, VideoInfo } from "../services/api";

interface Props {
  selection: { bbox: BoundingBox; frameNumber: number } | null;
  video: VideoInfo | null;
  videoRef: React.RefObject<HTMLVideoElement>;
  onSearch: (threshold: number, excludeSourceVideo: boolean) => void;
}

export function TargetPanel({ selection, video, videoRef, onSearch }: Props) {
  const [threshold, setThreshold] = useState(0.55);
  const [excludeSourceVideo, setExcludeSourceVideo] = useState(true);
  const [searching, setSearching] = useState(false);
  const [cropUrl, setCropUrl] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Generate crop preview when selection changes
  useEffect(() => {
    if (!selection || !videoRef.current || !video) {
      setCropUrl(null);
      return;
    }

    const vid = videoRef.current;
    const canvas = canvasRef.current;
    if (!canvas || !vid.videoWidth) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size to match the crop
    const cropWidth = selection.bbox.x2 - selection.bbox.x1;
    const cropHeight = selection.bbox.y2 - selection.bbox.y1;
    canvas.width = cropWidth;
    canvas.height = cropHeight;

    // Draw the cropped region
    ctx.drawImage(
      vid,
      selection.bbox.x1,
      selection.bbox.y1,
      cropWidth,
      cropHeight,
      0,
      0,
      cropWidth,
      cropHeight
    );

    // Convert to data URL
    const url = canvas.toDataURL("image/jpeg", 0.85);
    setCropUrl(url);
  }, [selection, video, videoRef]);

  const handleSearch = () => {
    if (!selection) return;
    setSearching(true);
    onSearch(threshold, excludeSourceVideo);
    // Reset searching state after a short delay (search is async but we don't have a callback)
    setTimeout(() => setSearching(false), 500);
  };

  if (!selection) {
    return (
      <div className="panel">
        <div className="panel-header">Target Selection</div>
        <div className="target-empty">
          Enable Search Mode and click/drag on the video to select a person to find across cameras.
        </div>
      </div>
    );
  }

  return (
    <div className="panel">
      <div className="panel-header">Target Selection</div>
      <div className="panel-body">
        <div className="target-content">
          {cropUrl && (
            <div className="target-crop">
              <img src={cropUrl} alt="Selected crop" />
            </div>
          )}
          
          <div className="field-row">
            <div className="field-label">
              <span>Similarity Threshold</span>
              <span>{threshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
            />
          </div>

          <div className="checkbox-row">
            <input
              type="checkbox"
              id="exclude-source"
              checked={excludeSourceVideo}
              onChange={(e) => setExcludeSourceVideo(e.target.checked)}
            />
            <label htmlFor="exclude-source">Exclude source video from results</label>
          </div>

          <button
            className="btn full primary"
            onClick={handleSearch}
            disabled={searching}
          >
            {searching ? "Searching..." : "Find Across Cameras"}
          </button>
        </div>
      </div>
      {/* Hidden canvas for crop generation */}
      <canvas ref={canvasRef} style={{ display: "none" }} />
    </div>
  );
}
