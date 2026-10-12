from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from load_signature_analysis.event_detection import (
    EventDetectionConfig,
    detect_on_intervals,
    detect_on_intervals_array,
)


class EventDetectionTests(unittest.TestCase):
    def _write_channel(self, directory: str, samples: list[tuple[float, float]]) -> Path:
        path = Path(directory) / "channel.dat"
        path.write_text(
            "".join(f"{timestamp} {power}\n" for timestamp, power in samples),
            encoding="utf-8",
        )
        return path

    def test_unknown_state_requires_observed_off_before_on_event(self) -> None:
        config = EventDetectionConfig(
            on_threshold_w=100.0,
            off_threshold_w=20.0,
            minimum_on_seconds=12.0,
            merge_gap_seconds=10.0,
            maximum_sample_gap_seconds=30.0,
        )
        samples = [
            (0, 500),
            (6, 0),
            (12, 500),
            (18, 50),
            (24, 500),
            (30, 0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = detect_on_intervals(self._write_channel(directory, samples), config)

        self.assertEqual(len(result.intervals), 1)
        self.assertEqual(result.intervals[0]["start_timestamp"], 12)
        self.assertEqual(result.intervals[0]["onset_uncertainty_seconds"], 6)

    def test_long_gap_resets_state_and_does_not_create_false_onset(self) -> None:
        config = EventDetectionConfig(100.0, 20.0, 12.0, 10.0, 30.0)
        samples = [
            (0, 0),
            (6, 500),
            (12, 500),
            (100, 500),
            (106, 500),
            (112, 0),
            (118, 500),
            (124, 500),
            (130, 500),
            (136, 0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = detect_on_intervals(self._write_channel(directory, samples), config)

        self.assertEqual(result.gap_count, 1)
        self.assertEqual(result.trusted_coverage_segments, ((0, 12), (112, 136)))
        starts = [interval["start_timestamp"] for interval in result.intervals]
        self.assertNotIn(100, starts)
        self.assertIn(118, starts)

    def test_short_off_gap_is_merged_before_duration_filter(self) -> None:
        config = EventDetectionConfig(100.0, 20.0, 20.0, 10.0, 30.0)
        samples = [
            (0, 0),
            (6, 500),
            (12, 500),
            (18, 500),
            (24, 0),
            (30, 500),
            (36, 500),
            (42, 500),
            (48, 0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = detect_on_intervals(self._write_channel(directory, samples), config)

        self.assertEqual(len(result.intervals), 1)
        self.assertEqual(result.intervals[0]["merged_segment_count"], 2)
        self.assertEqual(result.intervals[0]["active_duration_seconds"], 24)

    def test_threshold_boundaries_are_inclusive(self) -> None:
        config = EventDetectionConfig(
            on_threshold_w=100.0,
            off_threshold_w=20.0,
            minimum_on_seconds=0.0,
            merge_gap_seconds=0.0,
            maximum_sample_gap_seconds=30.0,
        )
        samples = [(0, 20), (6, 100), (12, 20)]
        with tempfile.TemporaryDirectory() as directory:
            result = detect_on_intervals(self._write_channel(directory, samples), config)

        self.assertEqual(len(result.intervals), 1)
        self.assertEqual(result.intervals[0]["start_timestamp"], 6)
        self.assertEqual(result.intervals[0]["end_timestamp"], 12)

        with self.assertRaises(ValueError):
            EventDetectionConfig(on_threshold_w=20.0, off_threshold_w=20.0)
        with self.assertRaises(ValueError):
            EventDetectionConfig(on_threshold_w=float("inf"), off_threshold_w=20.0)

    def test_raw_activity_keeps_short_file_boundary_interval(self) -> None:
        config = EventDetectionConfig(100.0, 20.0, 20.0, 10.0, 30.0)
        samples = [(0, 0), (6, 500), (12, 500)]
        with tempfile.TemporaryDirectory() as directory:
            result = detect_on_intervals(self._write_channel(directory, samples), config)

        self.assertEqual(len(result.intervals), 0)
        self.assertEqual(len(result.activity_intervals), 1)
        self.assertTrue(result.activity_intervals[0]["touches_data_boundary"])
        self.assertEqual(result.short_interval_count, 1)

    def test_array_detector_splits_invalid_region_and_requires_new_off(self) -> None:
        config = EventDetectionConfig(100.0, 20.0, 0.0, 10.0, 30.0)
        timestamps = np.arange(0.0, 54.0, 6.0)
        power = np.asarray([0, 500, 500, 0, 500, 0, 500, 500, 0], dtype=float)
        valid = np.ones(timestamps.size, dtype=bool)
        valid[3] = False

        result = detect_on_intervals_array(timestamps, power, valid, config)

        self.assertEqual(result.trusted_coverage_segments, ((0.0, 12.0), (30.0, 48.0)))
        self.assertEqual(
            [interval["start_timestamp"] for interval in result.intervals],
            [6.0, 36.0],
        )
        self.assertEqual(result.intervals[0]["end_reason"], "data_gap")


if __name__ == "__main__":
    unittest.main()
