"""Unit tests for the TALA Football analytics core (pure functions only).

Run:  python -m unittest discover -s backend/tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.football.analytics import assign_teams, compute_player_stats, possession_timeline
from app.football.pitch import PitchMapping


class PitchTests(unittest.TestCase):
    def setUp(self):
        # Image quad: near-left, far-left, far-right, near-right (100x100 px).
        self.pitch = PitchMapping([(0, 100), (0, 0), (100, 0), (100, 100)], length=105.0, width=68.0)

    def test_center_maps_to_pitch_center(self):
        x, y = self.pitch.to_pitch(50, 50)
        self.assertAlmostEqual(x, 52.5, places=1)
        self.assertAlmostEqual(y, 34.0, places=1)

    def test_corners(self):
        self.assertEqual(tuple(round(v, 1) for v in self.pitch.to_pitch(0, 100)), (0.0, 0.0))
        self.assertEqual(tuple(round(v, 1) for v in self.pitch.to_pitch(100, 0)), (105.0, 68.0))

    def test_outside_returns_none(self):
        self.assertIsNone(self.pitch.to_pitch(500, 500))

    def test_degenerate_quad_rejected(self):
        with self.assertRaises(ValueError):
            PitchMapping([(0, 0), (0, 0), (10, 0), (10, 10)])


class SpeedTests(unittest.TestCase):
    def test_teleport_glitch_rejected(self):
        # One 50 m jump in a single frame is a tracking glitch, not a sprint.
        tracks = {9: [(f, f * 0.4, 0.0) for f in range(10)] + [(10, 50.0, 0.0), (11, 50.4, 0.0)]}
        stats = compute_player_stats(tracks, fps=25.0)[9]
        self.assertLess(stats["distance_m"], 10.0)
        self.assertLess(stats["max_speed_kmh"], 45.0)

    def test_straight_line_speed_and_distance(self):
        # 10 m in 25 frames at 25 fps = 1 s -> 36 km/h, distance 10 m.
        tracks = {7: [(f, f * 0.4, 0.0) for f in range(26)]}
        stats = compute_player_stats(tracks, fps=25.0)[7]
        self.assertAlmostEqual(stats["distance_m"], 10.0, places=1)
        self.assertAlmostEqual(stats["mean_speed_kmh"], 36.0, places=1)

    def test_none_gaps_do_not_fabricate_distance(self):
        tracks = {3: [(0, 0.0, 0.0), (1, None, None), (2, 1.0, 0.0)]}
        stats = compute_player_stats(tracks, fps=10.0)[3]
        self.assertEqual(stats["distance_m"], 0.0)


class TeamTests(unittest.TestCase):
    def test_two_hue_clusters_split(self):
        teams = assign_teams({1: [10, 12, 11], 2: [13, 9, 10], 3: [200, 205, 198], 4: [202, 199, 203]})
        self.assertEqual(teams[1], teams[2])
        self.assertEqual(teams[3], teams[4])
        self.assertNotEqual(teams[1], teams[3])


class PossessionTests(unittest.TestCase):
    def test_shares_and_segments_with_hysteresis(self):
        frames = []
        for f in range(20):
            # Ball glued to player 1 (team 1) for 10 frames, then player 2 (team 2).
            ball = (1.0, 1.0) if f < 10 else (50.0, 1.0)
            frames.append({
                "frame": f, "ball": ball,
                "players": {1: (1.0, 1.0, 1), 2: (50.0, 1.0, 2)},
            })
        result = possession_timeline(frames, fps=10.0, hold_sec=0.4, radius_m=3.0)
        self.assertAlmostEqual(result["shares"][1], 50.0, delta=15.0)
        self.assertAlmostEqual(result["shares"][2], 50.0, delta=15.0)
        self.assertEqual(len(result["segments"]), 2)

    def test_no_ball_no_possession_no_crash(self):
        frames = [{"frame": 0, "ball": None, "players": {1: (0.0, 0.0, 1)}}]
        result = possession_timeline(frames, fps=25.0)
        self.assertEqual(result["segments"], [])
        self.assertEqual(result["shares"], {1: 0.0, 2: 0.0})


if __name__ == "__main__":
    unittest.main()
