"""TALA Football: pitch-mapped player/ball tracking analytics.

Supersedes the public `football_analysis` tutorial project this module was
inspired by. Fixes its core defects by design: calibration is per-video
instead of one hardcoded quad, fps comes from the probed video instead of a
hardcoded 24, possession uses meter distances with hysteresis instead of an
absolute 70px threshold, and team assignment is deterministic.
"""
