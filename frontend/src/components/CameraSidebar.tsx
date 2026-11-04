import { useRef, useState } from "react";
import { api, type VideoInfo, type IndexJobStatus } from "../services/api";

interface Props {
  videos: VideoInfo[];
  activeVideo: VideoInfo | null;
  onSelectVideo: (video: VideoInfo) => void;
  onVideosChange: (videos: VideoInfo[]) => void;
}

export function CameraSidebar({ videos, activeVideo, onSelectVideo, onVideosChange }: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [indexingStatus, setIndexingStatus] = useState<Record<string, IndexJobStatus>>({});

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    try {
      const cameraId = file.name.replace(/\.[^/.]+$/, ""); // Use filename without extension as camera_id
      const video = await api.uploadVideo(cameraId, file);
      onVideosChange([...videos, video]);
    } catch (err) {
      console.error("Upload failed:", err);
      alert("Upload failed: " + (err as Error).message);
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleIndex = async (video: VideoInfo) => {
    try {
      const status = await api.startIndexing(video.video_id);
      setIndexingStatus(prev => ({ ...prev, [video.video_id]: status }));
      
      // Poll for status updates
      const pollInterval = setInterval(async () => {
        try {
          const updated = await api.getIndexStatus(video.video_id);
          setIndexingStatus(prev => ({ ...prev, [video.video_id]: updated }));
          
          if (updated.status === "done" || updated.status === "error") {
            clearInterval(pollInterval);
            // Refresh videos list to get updated status
            const updatedVideos = await api.listVideos();
            onVideosChange(updatedVideos);
          }
        } catch (err) {
          console.error("Status check failed:", err);
          clearInterval(pollInterval);
        }
      }, 2000);
    } catch (err) {
      console.error("Indexing failed:", err);
      alert("Indexing failed: " + (err as Error).message);
    }
  };

  const getStatusClass = (video: VideoInfo) => {
    const jobStatus = indexingStatus[video.video_id];
    if (jobStatus?.status === "running" || jobStatus?.status === "queued") return "indexing";
    return video.status;
  };

  const getProgress = (video: VideoInfo) => {
    const jobStatus = indexingStatus[video.video_id];
    if (jobStatus?.status === "running" || jobStatus?.status === "queued") {
      return jobStatus.progress * 100; // Convert 0-1 to 0-100 for CSS
    }
    return video.status === "indexed" ? 100 : 0;
  };

  return (
    <div className="sidebar">
      <div className="camera-list">
        {videos.length === 0 && (
          <div style={{ padding: "16px 12px", color: "var(--text-2)", fontSize: "12px" }}>
            No videos uploaded yet.
          </div>
        )}
        {videos.map((video) => (
          <div
            key={video.video_id}
            className={`camera-item ${activeVideo?.video_id === video.video_id ? "active" : ""}`}
            onClick={() => onSelectVideo(video)}
          >
            <div className="camera-item-top">
              <div className="camera-name">{video.camera_id}</div>
              <span className={`status-dot ${getStatusClass(video)}`} />
            </div>
            <div className="camera-meta">
              {video.filename} · {Math.round(video.duration_sec)}s
            </div>
            {(getStatusClass(video) === "indexing" || video.status === "indexing") && (
              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{ width: `${getProgress(video)}%` }}
                />
              </div>
            )}
            {video.status === "unindexed" && (
              <button
                className="btn small"
                onClick={(e) => {
                  e.stopPropagation();
                  handleIndex(video);
                }}
              >
                Index
              </button>
            )}
            {video.status === "error" && (
              <div style={{ fontSize: "10px", color: "var(--danger)", marginTop: "4px" }}>
                {video.error || "Indexing failed"}
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="sidebar-footer">
        <div className="upload-row">
          <button
            className="btn full primary"
            onClick={handleUploadClick}
            disabled={uploading}
          >
            {uploading ? "Uploading..." : "Upload Video"}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            className="hidden-input"
            onChange={handleFileChange}
          />
        </div>
      </div>
    </div>
  );
}
