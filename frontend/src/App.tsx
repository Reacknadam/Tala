import { useEffect, useRef, useState } from "react";
import { VideoStage } from "./components/VideoStage";
import { CameraSidebar } from "./components/CameraSidebar";
import { TargetPanel } from "./components/TargetPanel";
import { MatchesPanel } from "./components/MatchesPanel";
import { FootballPanel } from "./components/FootballPanel";
import { api, type BoundingBox, type VideoInfo, type MatchResult, type FootballAnalysis } from "./services/api";

export default function App() {
  const [videos, setVideos] = useState<VideoInfo[]>([]);
  const [activeVideo, setActiveVideo] = useState<VideoInfo | null>(null);
  const [searchMode, setSearchMode] = useState(false);
  const [selection, setSelection] = useState<{ bbox: BoundingBox; frameNumber: number } | null>(null);
  const [matches, setMatches] = useState<MatchResult[]>([]);
  const [highlightBox, setHighlightBox] = useState<BoundingBox | null>(null);
  const [jumpFlash, setJumpFlash] = useState(false);
  const [mode, setMode] = useState<"search" | "football">("search");
  const [calibrationPoints, setCalibrationPoints] = useState<{ x: number; y: number }[]>([]);
  const [calibrating, setCalibrating] = useState(false);
  const [analysis, setAnalysis] = useState<FootballAnalysis | null>(null);
  const [detectedPolygon, setDetectedPolygon] = useState<number[][] | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Load videos on mount
  useEffect(() => {
    api.listVideos().then(setVideos).catch(console.error);
  }, []);

  // Poll videos every 2 seconds to update indexing status
  useEffect(() => {
    const interval = setInterval(() => {
      api.listVideos().then(setVideos).catch(console.error);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleSelectVideo = (video: VideoInfo) => {
    setActiveVideo(video);
    setSelection(null);
    setMatches([]);
    setHighlightBox(null);
    setCalibrationPoints([]);
    setCalibrating(false);
    setAnalysis(null);
    setDetectedPolygon(null);
  };

  const handleSelect = (bbox: BoundingBox, frameNumber: number) => {
    setSelection({ bbox, frameNumber });
  };

  const handleSearch = async (threshold: number, excludeSourceVideo: boolean) => {
    if (!selection || !activeVideo) return;
    
    try {
      const response = await api.search({
        video_id: activeVideo.video_id,
        frame_number: selection.frameNumber,
        bbox: selection.bbox,
        threshold,
        exclude_source_video: excludeSourceVideo,
      });
      setMatches(response.matches);
    } catch (err) {
      console.error("Search failed:", err);
    }
  };

  const handleJumpToMatch = (match: MatchResult) => {
    if (!videoRef.current) return;
    
    // Update active video if different
    if (match.video_id !== activeVideo?.video_id) {
      const video = videos.find(v => v.video_id === match.video_id);
      if (video) {
        setActiveVideo(video);
        // Wait for video to load before seeking
        setTimeout(() => {
          if (videoRef.current) {
            videoRef.current.currentTime = match.timestamp_sec;
            setHighlightBox(match.bbox);
            setJumpFlash(true);
            setTimeout(() => setJumpFlash(false), 1500);
          }
        }, 500);
      }
    } else {
      // Same video, just seek
      videoRef.current.currentTime = match.timestamp_sec;
      setHighlightBox(match.bbox);
      setJumpFlash(true);
      setTimeout(() => setJumpFlash(false), 1500);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="brand">TALA<span className="brand-sub">Video Intelligence</span></div>
        <div className="header-divider" />
        <button
          className={`search-toggle ${mode === "search" ? "on" : ""}`}
          onClick={() => setMode("search")}
        >
          <span className="dot" />
          Search
        </button>
        <button
          className={`search-toggle ${mode === "football" ? "on" : ""}`}
          onClick={() => setMode("football")}
        >
          <span className="dot" />
          Football
        </button>
        {mode === "search" && (
          <button
            className={`search-toggle ${searchMode ? "on" : ""}`}
            onClick={() => setSearchMode(!searchMode)}
          >
            <span className="dot" />
            Search Mode
          </button>
        )}
      </header>

      <div className="main">
        <CameraSidebar
          videos={videos}
          activeVideo={activeVideo}
          onSelectVideo={handleSelectVideo}
          onVideosChange={setVideos}
        />

        <div className="workspace">
          <div className="video-area">
            <VideoStage
              video={activeVideo}
              searchMode={mode === "search" && searchMode}
              onSelect={handleSelect}
              selection={selection?.bbox || null}
              highlightBox={highlightBox}
              jumpFlash={jumpFlash}
              videoRef={videoRef}
              calibrationMode={mode === "football" && calibrating}
              calibrationPoints={calibrationPoints}
              footballFrames={mode === "football" ? analysis?.frames ?? null : null}
              footballPitch={mode === "football" ? analysis?.pitch ?? null : null}
              detectedPolygon={detectedPolygon}
              onCalibrationPoint={(x, y) =>
                setCalibrationPoints((prev) => (prev.length >= 4 ? [{ x, y }] : [...prev, { x, y }]))
              }
            />
          </div>

          {mode === "search" ? (
            <div className="bottom-split">
              <TargetPanel
                selection={selection}
                video={activeVideo}
                videoRef={videoRef}
                onSearch={handleSearch}
              />
              <MatchesPanel
                matches={matches}
                onJumpToMatch={handleJumpToMatch}
              />
            </div>
          ) : (
            <FootballPanel
              video={activeVideo}
              videoRef={videoRef}
              calibrationPoints={calibrationPoints}
              onCalibrationPointsChange={setCalibrationPoints}
              calibrating={calibrating}
              onToggleCalibrating={() => setCalibrating(!calibrating)}
              analysis={analysis}
              onAnalysis={setAnalysis}
              detectedPolygon={detectedPolygon}
              onDetectedPolygon={setDetectedPolygon}
            />
          )}
        </div>
      </div>
    </div>
  );
}
