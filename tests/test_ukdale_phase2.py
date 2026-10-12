from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from load_signature_analysis.event_detection import DetectionResult
from load_signature_analysis.event_features import EventWindowConfig
from load_signature_analysis.ukdale_phase2 import (
    _assign_window_overlap_clusters,
    _extract_relative_mains_windows,
    _load_device_specs,
    _prefilter_events,
)


class UkDalePhase2Tests(unittest.TestCase):
    def test_overlapping_feature_windows_share_a_cluster(self) -> None:
        features = pd.DataFrame(
            [
                {
                    "event_id": "a",
                    "dataset": "uk_dale",
                    "event_timestamp": 100.0,
                    "channel_id": 7,
                },
                {
                    "event_id": "b",
                    "dataset": "uk_dale",
                    "event_timestamp": 200.0,
                    "channel_id": 18,
                },
                {
                    "event_id": "c",
                    "dataset": "uk_dale",
                    "event_timestamp": 500.0,
                    "channel_id": 7,
                },
            ]
        )

        clusters = _assign_window_overlap_clusters(features, EventWindowConfig(), 5)

        self.assertEqual(clusters.iloc[0], clusters.iloc[1])
        self.assertNotEqual(clusters.iloc[1], clusters.iloc[2])

    def test_relative_bins_use_active_power_and_half_open_boundaries(self) -> None:
        event_timestamp = 1_000.0
        config = EventWindowConfig(maximum_raw_gap_seconds=2.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mains.dat"
            rows = []
            for timestamp in range(940, 1_121):
                active = 200.0 if timestamp < event_timestamp else 1_200.0
                if timestamp == 1_120:
                    active = 9_999.0
                rows.append(f"{timestamp} {active} {active + 100.0} 240.0\n")
            path.write_text("".join(rows), encoding="utf-8")

            medians, counts, gaps, audit = _extract_relative_mains_windows(
                path,
                np.asarray([event_timestamp]),
                config,
                chunksize=37,
                invalid_zero_active_when_apparent_above_va=50.0,
            )

        self.assertTrue(np.all(medians[0, : config.pre_bin_count] == 200.0))
        self.assertTrue(np.all(medians[0, config.pre_bin_count :] == 1_200.0))
        self.assertTrue(np.all(counts == 10))
        self.assertEqual(gaps[0], 1.0)
        self.assertEqual(audit["valid_active_record_count"], 181)

    def test_positive_label_gap_rejects_negative_event(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "channel_id": 18,
                    "label": 1,
                    "event_timestamp": 100.0,
                    "end_timestamp": 120.0,
                    "touches_data_boundary": False,
                },
                {
                    "channel_id": 7,
                    "label": 0,
                    "event_timestamp": 300.0,
                    "end_timestamp": 330.0,
                    "touches_data_boundary": False,
                },
                {
                    "channel_id": 7,
                    "label": 0,
                    "event_timestamp": 480.0,
                    "end_timestamp": 510.0,
                    "touches_data_boundary": False,
                },
            ]
        )
        common = {
            "intervals": (),
            "activity_intervals": (),
            "coverage_start_timestamp": 0.0,
            "coverage_end_timestamp": 600.0,
            "valid_record_count": 100,
            "gap_count": 1,
            "duplicate_or_backward_count": 0,
            "short_interval_count": 0,
        }
        detection_results = {
            18: DetectionResult(
                trusted_coverage_segments=((0.0, 200.0), (400.0, 600.0)),
                **common,
            ),
            7: DetectionResult(
                trusted_coverage_segments=((0.0, 600.0),),
                **common,
            ),
        }

        filtered = _prefilter_events(
            events,
            detection_results,
            {18: 1, 7: 0},
            EventWindowConfig(),
            event_merge_tolerance_seconds=10.0,
            maximum_events_per_device=None,
        )

        self.assertEqual(
            filtered.loc[1, "prefilter_rejection_reasons"], "positive_label_gap"
        )
        self.assertTrue(filtered.loc[2, "selected_for_feature_extraction"])

    def test_invalid_zero_active_creates_a_raw_gap(self) -> None:
        event_timestamp = 1_000.0
        config = EventWindowConfig(maximum_raw_gap_seconds=2.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mains.dat"
            rows = []
            for timestamp in range(940, 1_121):
                active = 1_000.0
                apparent = 1_100.0
                if 990 <= timestamp <= 1_010:
                    active = 0.0
                rows.append(f"{timestamp} {active} {apparent} 240.0\n")
            path.write_text("".join(rows), encoding="utf-8")

            _, _, gaps, audit = _extract_relative_mains_windows(
                path,
                np.asarray([event_timestamp]),
                config,
                chunksize=17,
                invalid_zero_active_when_apparent_above_va=50.0,
            )

        self.assertEqual(audit["invalid_zero_active_count"], 21)
        self.assertEqual(gaps[0], 22.0)

    def test_short_positive_activity_still_blocks_negative_window(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "channel_id": 7,
                    "label": 0,
                    "event_timestamp": 300.0,
                    "end_timestamp": 330.0,
                    "touches_data_boundary": False,
                }
            ]
        )
        common = {
            "intervals": (),
            "trusted_coverage_segments": ((0.0, 600.0),),
            "coverage_start_timestamp": 0.0,
            "coverage_end_timestamp": 600.0,
            "valid_record_count": 100,
            "gap_count": 0,
            "duplicate_or_backward_count": 0,
            "short_interval_count": 1,
        }
        detection_results = {
            18: DetectionResult(
                activity_intervals=(
                    {"start_timestamp": 250.0, "end_timestamp": 260.0},
                ),
                **common,
            ),
            7: DetectionResult(activity_intervals=(), **common),
        }

        filtered = _prefilter_events(
            events,
            detection_results,
            {18: 1, 7: 0},
            EventWindowConfig(),
            event_merge_tolerance_seconds=10.0,
            maximum_events_per_device=None,
        )

        self.assertEqual(
            filtered.loc[0, "prefilter_rejection_reasons"],
            "opposite_class_activity_overlap",
        )

    def test_short_negative_activity_also_blocks_positive_window(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "channel_id": 18,
                    "label": 1,
                    "event_timestamp": 300.0,
                    "end_timestamp": 330.0,
                    "touches_data_boundary": False,
                }
            ]
        )
        common = {
            "intervals": (),
            "trusted_coverage_segments": ((0.0, 600.0),),
            "coverage_start_timestamp": 0.0,
            "coverage_end_timestamp": 600.0,
            "valid_record_count": 100,
            "gap_count": 0,
            "duplicate_or_backward_count": 0,
            "short_interval_count": 1,
        }
        detection_results = {
            18: DetectionResult(activity_intervals=(), **common),
            7: DetectionResult(
                activity_intervals=(
                    {
                        "start_timestamp": 250.0,
                        "end_timestamp": 260.0,
                        "onset_uncertainty_seconds": 6.0,
                    },
                ),
                **common,
            ),
        }

        filtered = _prefilter_events(
            events,
            detection_results,
            {18: 1, 7: 0},
            EventWindowConfig(),
            event_merge_tolerance_seconds=10.0,
            maximum_events_per_device=None,
        )

        self.assertEqual(
            filtered.loc[0, "prefilter_rejection_reasons"],
            "opposite_class_activity_overlap",
        )

    def test_inventory_controls_labels_while_config_controls_thresholds(self) -> None:
        inventory = pd.DataFrame(
            [
                {
                    "dataset": "uk_dale",
                    "house_id": 5,
                    "channel_id": 18,
                    "canonical_label": "kettle",
                    "semantic_class": "resistive_heating",
                    "binary_label": 1,
                    "include_in_main_mvp": True,
                    "measurement_unit": "W",
                },
                {
                    "dataset": "uk_dale",
                    "house_id": 5,
                    "channel_id": 7,
                    "canonical_label": "treadmill",
                    "semantic_class": "motor",
                    "binary_label": 0,
                    "include_in_main_mvp": True,
                    "measurement_unit": "W",
                },
            ]
        )
        configured = [
            {
                "channel_id": 18,
                "on_threshold_w": 2_100.0,
                "off_threshold_w": 110.0,
                "label": 0,
            },
            {
                "channel_id": 7,
                "on_threshold_w": 120.0,
                "off_threshold_w": 25.0,
                "label": 1,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.csv"
            inventory.to_csv(path, index=False, encoding="utf-8-sig")
            specs = _load_device_specs(path, 5, configured)

        kettle = next(spec for spec in specs if spec["channel_id"] == 18)
        treadmill = next(spec for spec in specs if spec["channel_id"] == 7)
        self.assertEqual(kettle["label"], 1)
        self.assertEqual(kettle["source_appliance"], "kettle")
        self.assertEqual(kettle["detection"].on_threshold_w, 2_100.0)
        self.assertEqual(treadmill["label"], 0)


if __name__ == "__main__":
    unittest.main()
