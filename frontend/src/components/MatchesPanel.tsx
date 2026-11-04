import { useState } from "react";
import type { MatchResult } from "../services/api";

interface Props {
  matches: MatchResult[];
  onJumpToMatch: (match: MatchResult) => void;
}

export function MatchesPanel({ matches, onJumpToMatch }: Props) {
  const [jumpedMatchId, setJumpedMatchId] = useState<string | null>(null);

  const handleJump = (match: MatchResult) => {
    setJumpedMatchId(match.match_id);
    onJumpToMatch(match);
    setTimeout(() => setJumpedMatchId(null), 2000);
  };

  if (matches.length === 0) {
    return (
      <div className="panel">
        <div className="panel-header">
          <span>Matches</span>
          <span className="badge">0</span>
        </div>
        <div className="matches-empty">
          No matches yet. Select a person in the video and click "Find Across Cameras".
        </div>
      </div>
    );
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <span>Matches</span>
        <span className="badge">{matches.length}</span>
      </div>
      <div className="panel-body">
        {matches.map((match) => (
          <div
            key={match.match_id}
            className={`match-row ${jumpedMatchId === match.match_id ? "jumped" : ""}`}
            onClick={() => handleJump(match)}
          >
            <img
              src={`data:image/jpeg;base64,${match.thumbnail_base64}`}
              alt="Match thumbnail"
              className="match-thumb"
            />
            <div className="match-info">
              <div className="match-camera">{match.camera_id}</div>
              <div className="match-meta">
                {formatTimestamp(match.timestamp_sec)} · Track #{match.track_id}
              </div>
            </div>
            <div className="match-score">
              <div>{(match.similarity * 100).toFixed(0)}%</div>
              <div className="score-bar">
                <div
                  className="score-bar-fill"
                  style={{ width: `${match.similarity * 100}%` }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function formatTimestamp(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}
