"""Contract tests for online_causal_v1_1 grid, labels, atomic outputs and NPZ files."""

from __future__ import annotations

import csv
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from load_signature_analysis.event_detection import DetectionResult
from load_signature_analysis.mvp_preprocessing import HouseKey
from load_signature_analysis.online_features import OnlineFeatureConfig
from load_signature_analysis.online_samples import (
    ChannelSpec,
    GroundTruthChannel,
    _build_core_label_index,
    _candidate_identifier,
    _load_ukdale_house_streaming,
    assess_candidate_horizon,
    assign_candidate_labels,
    assign_overlap_clusters,
    assign_target_active_labels,
    build_fixed_utc_bins_from_records,
    build_online_causal_samples,
    extract_candidate_window,
)


def _detection(
    *,
    starts: tuple[float, ...] = (),
    activities: tuple[tuple[float, float], ...] = (),
    coverage: tuple[tuple[float, float], ...] = ((-1_000.0, 1_000.0),),
) -> DetectionResult:
    """Construct minimal trusted submeter detections for pure label functions."""

    intervals = tuple(
        {
            "start_timestamp": start,
            "end_timestamp": start + 20.0,
            "touches_data_boundary": False,
        }
        for start in starts
    )
    activity_intervals = tuple(
        {
            "start_timestamp": left,
            "end_timestamp": right,
            "touches_data_boundary": False,
        }
        for left, right in activities
    )
    return DetectionResult(
        intervals=intervals,
        activity_intervals=activity_intervals,
        trusted_coverage_segments=coverage,
        coverage_start_timestamp=coverage[0][0] if coverage else None,
        coverage_end_timestamp=coverage[-1][1] if coverage else None,
        valid_record_count=100,
        gap_count=0,
        duplicate_or_backward_count=0,
        short_interval_count=0,
    )


def _channel(
    role: str,
    *,
    channel_id: str = "1",
    starts: tuple[float, ...] = (),
    activities: tuple[tuple[float, float], ...] = (),
    coverage: tuple[tuple[float, float], ...] = ((-1_000.0, 1_000.0),),
) -> GroundTruthChannel:
    """Construct a primary or hard-negative submeter without putting device power in sequence inputs."""

    return GroundTruthChannel(
        dataset="refit",
        house_id="2",
        channel_id=channel_id,
        source_appliance="kettle" if role == "core_positive" else "microwave",
        taxonomy_role=role,
        result=_detection(starts=starts, activities=activities, coverage=coverage),
    )


class FixedGridAndLabelTests(unittest.TestCase):
    """Cover the fixed half-open grid, quality breaks, one-to-one labels and auxiliary activity labels."""

    def test_refit_invalid_rows_and_half_open_bins(self) -> None:
        """Issues or nonpositive aggregate values invalidate a bin; right-boundary samples belong to the next bin."""

        series = build_fixed_utc_bins_from_records(
            [
                (0.0, 100.0, True, ()),
                (9.9, 120.0, True, ()),
                (10.0, 130.0, True, ()),
                (20.0, 0.0, False, ("refit_nonpositive_aggregate",)),
                (30.0, 140.0, False, ("refit_issues",)),
            ]
        )

        self.assertEqual(series.bin_at(0).aggregate_power_w, 110.0)
        self.assertEqual(series.bin_at(10).aggregate_power_w, 130.0)
        self.assertFalse(series.bin_at(20).quality_valid)
        self.assertTrue(np.isnan(series.bin_at(20).aggregate_power_w))
        self.assertIn("refit_issues", series.bin_at(30).quality_flags)

    def test_raw_gap_and_h20_future_causality(self) -> None:
        """Raw gaps over 30 seconds are marked; H20 reads no bins at or after t0+20."""

        records = [
            (float(timestamp), 100.0 if timestamp < 0 else 500.0, True, ())
            for timestamp in range(-80, 151, 10)
        ]
        series = build_fixed_utc_bins_from_records(records)
        baseline = extract_candidate_window(series, 0, 20)
        modified_records = [
            (timestamp, 9_999.0 if timestamp >= 20.0 else power, valid, flags)
            for timestamp, power, valid, flags in records
        ]
        modified = extract_candidate_window(
            build_fixed_utc_bins_from_records(modified_records), 0, 20
        )
        np.testing.assert_equal(baseline.power_w, modified.power_w)
        self.assertEqual(baseline.power_w.size, 8)

        gap_series = build_fixed_utc_bins_from_records(
            [(0.0, 100.0, True, ()), (31.0, 100.0, True, ())]
        )
        self.assertIn("raw_gap_gt_30s", gap_series.bin_at(30).quality_flags)

    def test_nearest_one_to_one_matching_and_background_guard(self) -> None:
        """One actual activation matches at most one candidate; unknown positive coverage cannot become background negative."""

        core = _channel(
            "core_positive",
            starts=(100.0,),
            activities=((100.0, 130.0),),
        )
        rows = [
            {"candidate_id": "a", "candidate_onset_s": 98.0, "candidate_detected_at_s": 118.0},
            {"candidate_id": "b", "candidate_onset_s": 102.0, "candidate_detected_at_s": 122.0},
        ]
        audits, matched, unmatched = assign_candidate_labels(
            rows,
            core_channels=[core],
            hard_channels=[],
            aggregate_max_window_trusted={"a": True, "b": True},
        )
        self.assertEqual(len(matched), 1)
        self.assertEqual(len(unmatched), 0)
        self.assertEqual(audits["a"].matched_event_id is not None, True)
        self.assertIsNone(audits["b"].matched_event_id)

        background_rows = [
            {"candidate_id": "background", "candidate_onset_s": 400.0, "candidate_detected_at_s": 420.0}
        ]
        clean_core = _channel("core_positive", coverage=((0.0, 1_000.0),))
        background, _, _ = assign_candidate_labels(
            background_rows,
            core_channels=[clean_core],
            hard_channels=[],
            aggregate_max_window_trusted={"background": True},
        )
        self.assertEqual(background["background"].label_type, "background_negative")
        gapped_core = _channel("core_positive", coverage=((350.0, 410.0),))
        unknown, _, _ = assign_candidate_labels(
            background_rows,
            core_channels=[gapped_core],
            hard_channels=[],
            aggregate_max_window_trusted={"background": True},
        )
        self.assertEqual(unknown["background"].label_type, "ignore")
        self.assertIn("positive_coverage_gap", unknown["background"].ignore_reasons)

    def test_positive_negative_overlap_and_active_labels(self) -> None:
        """Overlapping primary-positive and hard-negative activity is ignored; auxiliary labels use 1/0/-1."""

        core = _channel(
            "core_positive",
            starts=(100.0,),
            activities=((100.0, 140.0),),
            coverage=((0.0, 500.0),),
        )
        hard = _channel(
            "hard_negative",
            channel_id="2",
            activities=((110.0, 130.0),),
        )
        audits, _, _ = assign_candidate_labels(
            [{"candidate_id": "overlap", "candidate_onset_s": 100.0, "candidate_detected_at_s": 120.0}],
            core_channels=[core],
            hard_channels=[hard],
            aggregate_max_window_trusted={"overlap": True},
        )
        self.assertEqual(audits["overlap"].label_type, "ignore")
        self.assertIn("positive_negative_activity_overlap", audits["overlap"].ignore_reasons)
        labels = assign_target_active_labels(
            [0, 10, 20, 500], bin_seconds=10, core_channels=[core]
        )
        np.testing.assert_array_equal(labels, np.asarray([0, 0, 0, -1], dtype=np.int8))
        active_labels = assign_target_active_labels(
            [100, 110], bin_seconds=10, core_channels=[core]
        )
        np.testing.assert_array_equal(active_labels, np.asarray([1, 1], dtype=np.int8))

    def test_overlap_cluster_uses_dataset_house_joint_key(self) -> None:
        """Different datasets never share an overlap cluster just because their house numbers match."""

        clusters = assign_overlap_clusters(
            [
                {"candidate_id": "u1", "dataset": "uk_dale", "house_id": "2", "candidate_onset_s": 100.0},
                {"candidate_id": "u2", "dataset": "uk_dale", "house_id": "2", "candidate_onset_s": 150.0},
                {"candidate_id": "r1", "dataset": "refit", "house_id": "2", "candidate_onset_s": 100.0},
            ]
        )
        self.assertEqual(clusters["u1"], clusters["u2"])
        self.assertNotEqual(clusters["u1"], clusters["r1"])

    def test_file_boundary_and_long_gap_reject_window(self) -> None:
        """Windows crossing an aggregate-file boundary or a raw gap fail quality checks."""

        records = [
            (float(timestamp), 100.0, True, ())
            for timestamp in range(0, 101, 10)
        ]
        series = build_fixed_utc_bins_from_records(records)
        boundary_window = extract_candidate_window(series, 60, 20)
        accepted, reasons = assess_candidate_horizon(
            boundary_window,
            horizon_seconds=20,
            feature_config=OnlineFeatureConfig(),
        )
        self.assertFalse(accepted)
        self.assertIn("aggregate_file_boundary", reasons)

        gapped = build_fixed_utc_bins_from_records(
            [
                (float(timestamp), 100.0, True, ())
                for timestamp in (-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 50, 60)
            ]
        )
        gap_window = extract_candidate_window(gapped, 40, 20)
        accepted, reasons = assess_candidate_horizon(
            gap_window,
            horizon_seconds=20,
            feature_config=OnlineFeatureConfig(),
        )
        self.assertFalse(accepted)
        self.assertIn("raw_gap_gt_30s", reasons)

    def test_transfer_match_is_recorded_but_not_main_label(self) -> None:
        """Transfer-positive matches retain their audit identity and never become primary training positives."""

        transfer = _channel(
            "extended_clean_positive",
            starts=(100.0,),
            activities=((100.0, 130.0),),
        )
        transfer = GroundTruthChannel(
            dataset=transfer.dataset,
            house_id=transfer.house_id,
            channel_id=transfer.channel_id,
            source_appliance="hair_dryer",
            taxonomy_role="extended_clean_positive",
            result=transfer.result,
        )
        audits, _, _ = assign_candidate_labels(
            [{"candidate_id": "transfer", "candidate_onset_s": 100.0, "candidate_detected_at_s": 120.0}],
            core_channels=[_channel("core_positive", coverage=((0.0, 500.0),))],
            hard_channels=[],
            transfer_channels=[transfer],
            aggregate_max_window_trusted={"transfer": True},
        )
        self.assertIsNone(audits["transfer"].label)
        self.assertEqual(audits["transfer"].label_type, "transfer_positive")
        self.assertEqual(audits["transfer"].label_status, "transfer_only")

    def test_hard_negative_requires_no_core_pollution(self) -> None:
        """Hard-negative matches become label=0 only in reliable windows with no target activity."""

        core = _channel("core_positive", coverage=((0.0, 500.0),))
        hard = _channel("hard_negative", starts=(100.0,), activities=((100.0, 130.0),))
        audits, _, _ = assign_candidate_labels(
            [{"candidate_id": "hard", "candidate_onset_s": 100.0, "candidate_detected_at_s": 120.0}],
            core_channels=[core],
            hard_channels=[hard],
            aggregate_max_window_trusted={"hard": True},
        )
        self.assertEqual(audits["hard"].label, 0)
        self.assertEqual(audits["hard"].label_type, "hard_negative")

    def test_two_core_appliance_types_nearby_are_ambiguous(self) -> None:
        """Kettle and toaster are distinct sources even within core; channel tie-breaking cannot force a label."""

        kettle = _channel("core_positive", channel_id="1", starts=(100.0,), activities=((100.0, 130.0),))
        toaster = GroundTruthChannel(
            dataset="refit",
            house_id="2",
            channel_id="2",
            source_appliance="toaster",
            taxonomy_role="core_positive",
            result=_detection(
                starts=(105.0,),
                activities=((105.0, 135.0),),
                coverage=((-1_000.0, 1_000.0),),
            ),
        )
        audits, _, _ = assign_candidate_labels(
            [{"candidate_id": "two-core", "candidate_onset_s": 100.0, "candidate_detected_at_s": 120.0}],
            core_channels=[kettle, toaster],
            hard_channels=[],
            aggregate_max_window_trusted={"two-core": True},
        )
        self.assertEqual(audits["two-core"].label_type, "ignore")
        self.assertIn("multiple_different_category_onsets", audits["two-core"].ignore_reasons)

    def test_later_core_onset_inside_max_window_invalidates_match(self) -> None:
        """A second unmatched activation within H120 requires ignore even if the first activation matched."""

        core = _channel(
            "core_positive",
            starts=(100.0, 180.0),
            activities=((100.0, 130.0), (180.0, 200.0)),
            coverage=((0.0, 500.0),),
        )
        audits, _, _ = assign_candidate_labels(
            [{"candidate_id": "later", "candidate_onset_s": 100.0, "candidate_detected_at_s": 120.0}],
            core_channels=[core],
            hard_channels=[],
            aggregate_max_window_trusted={"later": True},
        )
        self.assertEqual(audits["later"].label_type, "ignore")
        self.assertIn("target_onset_unmatched_or_nonunique", audits["later"].ignore_reasons)

    def test_unmatched_target_activity_cannot_be_background(self) -> None:
        """Primary-positive activity within the maximum window requires ignore when no unique match exists."""

        core = _channel(
            "core_positive",
            starts=(200.0,),
            activities=((200.0, 215.0),),
            coverage=((0.0, 500.0),),
        )
        audits, _, _ = assign_candidate_labels(
            [{"candidate_id": "ambiguous", "candidate_onset_s": 100.0, "candidate_detected_at_s": 120.0}],
            core_channels=[core],
            hard_channels=[],
            aggregate_max_window_trusted={"ambiguous": True},
        )
        self.assertEqual(audits["ambiguous"].label_type, "ignore")
        self.assertIn("target_onset_unmatched_or_nonunique", audits["ambiguous"].ignore_reasons)

    def test_target_active_unknown_without_full_coverage(self) -> None:
        """An inactive point without complete positive coverage has label -1, not OFF."""

        core = _channel("core_positive", coverage=((0.0, 10.0),))
        labels = assign_target_active_labels([0, 10], bin_seconds=10, core_channels=[core])
        np.testing.assert_array_equal(labels, np.asarray([0, -1], dtype=np.int8))

    def test_precompiled_active_label_index_preserves_interval_semantics(self) -> None:
        """Indexed queries preserve interval semantics, including half-open activity and closed coverage endpoints."""

        core = _channel(
            "core_positive",
            activities=((10.0, 20.0), (40.0, 60.0)),
            coverage=((0.0, 15.0), (30.0, 70.0)),
        )
        bin_left = [0, 10, 20, 30, 40, 50, 60]
        direct = assign_target_active_labels(
            bin_left,
            bin_seconds=10,
            core_channels=[core],
        )
        indexed = assign_target_active_labels(
            bin_left,
            bin_seconds=10,
            core_channels=[core],
            core_label_index=_build_core_label_index([core]),
        )
        np.testing.assert_array_equal(indexed, direct)
        np.testing.assert_array_equal(
            indexed,
            np.asarray([0, 1, -1, 0, 1, 1, 0], dtype=np.int8),
        )

    def test_candidate_identifier_uses_stable_milliseconds(self) -> None:
        """Candidate IDs combine dataset, house and millisecond time without unstable floating-point text."""

        self.assertEqual(
            _candidate_identifier("uk_dale", "5", 1_417_360_000.0),
            "uk_dale_h5_t1417360000000",
        )
        self.assertEqual(
            _candidate_identifier("refit", "2", 1.2344),
            _candidate_identifier("refit", "2", 1.23440000001),
        )

    def test_invalid_onset_bin_is_not_interpolated(self) -> None:
        """Invalid raw measurements at t0 remain NaN; neighboring pre/post values cannot fill them."""

        records = [
            (float(timestamp), 100.0, True, ())
            for timestamp in range(-80, 51, 10)
            if timestamp != 0
        ]
        records.append((0.0, 0.0, False, ("refit_nonpositive_aggregate",)))
        window = extract_candidate_window(build_fixed_utc_bins_from_records(records), 0, 20)
        self.assertFalse(window.quality_valid_mask[6])
        self.assertTrue(np.isnan(window.power_w[6]))

    def test_unmatched_core_onset_is_reported_by_matching_helper(self) -> None:
        """Actual primary-positive activations without an aggregate candidate enter the unmatched list."""

        core = _channel("core_positive", starts=(500.0,), activities=((500.0, 530.0),))
        _, matched, unmatched = assign_candidate_labels(
            [{"candidate_id": "far", "candidate_onset_s": 100.0, "candidate_detected_at_s": 120.0}],
            core_channels=[core],
            hard_channels=[],
            aggregate_max_window_trusted={"far": True},
        )
        self.assertEqual(matched, [])
        self.assertEqual([event.start_timestamp_s for event in unmatched], [500.0])

    def test_no_core_channel_never_yields_background_negative(self) -> None:
        """Without a reliable primary-positive channel, unmatched candidates remain ignore rather than background negatives."""

        audits, _, _ = assign_candidate_labels(
            [{"candidate_id": "unknown", "candidate_onset_s": 100.0, "candidate_detected_at_s": 120.0}],
            core_channels=[],
            hard_channels=[],
            aggregate_max_window_trusted={"unknown": True},
        )
        self.assertEqual(audits["unknown"].label_type, "ignore")
        self.assertIn("no_reliable_core_positive_channel", audits["unknown"].ignore_reasons)

    def test_ukdale_loader_uses_active_w_not_apparent_va(self) -> None:
        """UK-DALE mains uses active W from column two; VA from column three never enters the sequence."""

        with tempfile.TemporaryDirectory() as temporary:
            mains = Path(temporary) / "mains.dat"
            mains.write_text("0 100 1000\n1 100 1000\n10 200 2000\n", encoding="utf-8")
            series, channels = _load_ukdale_house_streaming(
                mains_path=mains,
                specs=[],
                thresholds={},
                bin_seconds=10,
                maximum_raw_gap_seconds=30.0,
                chunksize=10,
                invalid_zero_apparent_threshold_va=50.0,
                maximum_nonmonotonic_records=0,
            )
        self.assertEqual(channels, [])
        self.assertEqual(series.bin_at(0).aggregate_power_w, 100.0)
        self.assertEqual(series.bin_at(10).aggregate_power_w, 200.0)

    def test_h20_post_quality_rule_requires_two_valid_bins(self) -> None:
        """Either invalid H20 post bin fails the rounded-up 80% coverage requirement."""

        records = [
            (float(timestamp), 100.0, True, ())
            for timestamp in range(-80, 71, 10)
            if timestamp != 10
        ]
        records.append((10.0, 0.0, False, ("refit_nonpositive_aggregate",)))
        window = extract_candidate_window(build_fixed_utc_bins_from_records(records), 0, 20)
        accepted, reasons = assess_candidate_horizon(
            window,
            horizon_seconds=20,
            feature_config=OnlineFeatureConfig(),
        )
        self.assertFalse(accepted)
        self.assertIn("insufficient_post_coverage", reasons)


class OnlineOutputTests(unittest.TestCase):
    """Use a tiny REFIT fixture to verify atomic directories, pickle-free NPZ and horizon lengths."""

    def _write_fixture(self, root: Path) -> tuple[Path, Path]:
        """Write a tiny Phase 1 inventory and raw CSV with one core channel."""

        raw = root / "house2.csv"
        rows = ["Unix,Aggregate,Appliance1,Issues"]
        for timestamp in range(0, 271, 10):
            high = 100 <= timestamp < 210
            aggregate = 500.0 if high else 100.0
            appliance = 1_200.0 if high else 0.0
            rows.append(f"{timestamp},{aggregate},{appliance},0")
        raw.write_text("\n".join(rows), encoding="utf-8")
        phase1 = root / "phase1"
        phase1.mkdir()
        (phase1 / "appliance_inventory.csv").write_text(
            "dataset,house_id,channel_id,source_column,source_file,canonical_label,taxonomy_role,taxonomy_confidence,data_use,include_in_main_mvp\n"
            f"refit,2,1,Appliance1,{raw.as_posix()},kettle,core_positive,high,main,true\n",
            encoding="utf-8",
        )
        (phase1 / "aggregate_source_inventory.csv").write_text(
            "dataset,house_id,source_kind,source_file\n"
            f"refit,2,primary_inventory_source,{raw.as_posix()}\n",
            encoding="utf-8",
        )
        taxonomy = root / "taxonomy.yaml"
        taxonomy.write_text("schema_version: test\n", encoding="utf-8")
        config = root / "online.toml"
        config.write_text(
            """
[schema]
schema_version = "online_causal_v1_1"
feature_registry_version = "online_feature_v1_1"
npz_contract_version = "online_sequence_v1_1"

[paths]
appliance_inventory = "phase1/appliance_inventory.csv"
aggregate_source_inventory = "phase1/aggregate_source_inventory.csv"
taxonomy = "taxonomy.yaml"
default_output_dir = "formal"

[streaming]
refit_chunksize = 20
ukdale_chunksize = 20
bin_seconds = 10
maximum_raw_gap_seconds = 30.0
ukdale_invalid_zero_apparent_threshold_va = 50.0

[candidate_detector]
baseline_bin_count = 6
minimum_valid_baseline_bins = 5
positive_step_threshold_w = 300.0
confirmation_bin_count = 2
cooldown_seconds = 30
maximum_raw_gap_seconds = 30.0

[window]
pre_seconds = 60
horizons_seconds = [20, 40, 60, 120]
minimum_pre_coverage = 0.8333333333333334
minimum_post_coverage = 0.8
maximum_consecutive_missing_bins = 1
maximum_raw_gap_seconds = 30.0

[matching]
onset_match_tolerance_seconds = 20.0

[features]
activity_margin_w = 50.0
step_threshold_w = 30.0
relative_epsilon_w = 50.0

[sample_weight]
core_positive = 1.0
hard_negative = 1.0
background_negative = 0.25
refit_missing_issues_matched_positive = 0.5

[label_policy]
allow_refit_missing_issues_matched_positive = true
allow_refit_missing_issues_background_negative = false

[split]
train = ["uk_dale:5", "refit:1", "refit:3", "refit:6", "refit:9", "refit:13"]
validation = ["refit:2", "refit:4"]
test = ["uk_dale:2", "refit:19", "refit:20"]

[thresholds.kettle]
on_threshold_w = 1000.0
off_threshold_w = 100.0
minimum_on_seconds = 20.0
merge_gap_seconds = 15.0
maximum_sample_gap_seconds = 30.0
""".strip(),
            encoding="utf-8",
        )
        return config, raw

    def _add_second_refit_house(self, root: Path) -> None:
        """Add a second house to the minimal Phase 1 fixture to test output ordering across houses."""

        raw = root / "house19.csv"
        rows = ["Unix,Aggregate,Appliance1,Issues"]
        for timestamp in range(0, 271, 10):
            high = 100 <= timestamp < 210
            aggregate = 500.0 if high else 100.0
            appliance = 1_200.0 if high else 0.0
            rows.append(f"{timestamp},{aggregate},{appliance},0")
        raw.write_text("\n".join(rows), encoding="utf-8")
        appliance_inventory = root / "phase1" / "appliance_inventory.csv"
        aggregate_inventory = root / "phase1" / "aggregate_source_inventory.csv"
        with appliance_inventory.open("a", encoding="utf-8") as handle:
            handle.write(
                f"refit,19,1,Appliance1,{raw.as_posix()},kettle,core_positive,high,main,true\n"
            )
        with aggregate_inventory.open("a", encoding="utf-8") as handle:
            handle.write(f"refit,19,primary_inventory_source,{raw.as_posix()}\n")

    def test_atomic_output_and_npz_contract(self) -> None:
        """Commit outputs atomically; all four NPZ files have correct lengths and load without pickle."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = self._write_fixture(root)
            output = root / "online_output"
            result = build_online_causal_samples(
                project_root=root,
                config_path=config,
                output_dir=output,
                houses=["refit:2"],
            )
            self.assertEqual(result["candidate_count"], 1)
            with (output / "aggregate_bins.csv").open(encoding="utf-8", newline="") as handle:
                aggregate_bins = list(csv.DictReader(handle))
            self.assertEqual(len(aggregate_bins), 28)
            self.assertEqual(
                set(aggregate_bins[0]),
                {
                    "dataset",
                    "house_id",
                    "split",
                    "bin_left_s",
                    "bin_right_s",
                    "aggregate_power_w",
                    "raw_valid_sample_count",
                    "quality_valid",
                    "maximum_raw_gap_seconds",
                    "quality_flags",
                },
            )
            for horizon, length in ((20, 8), (40, 10), (60, 12), (120, 18)):
                with np.load(output / f"sequences_h{horizon}.npz", allow_pickle=False) as payload:
                    self.assertEqual(payload["values"].shape[1:], (length, 4))
                    self.assertIn(payload["candidate_ids"].dtype.kind, {"U", "S"})
                    self.assertEqual(payload["relative_bin_indices"].size, length)
                    self.assertIn("target_active_label", payload.files)
                    self.assertNotIn("target_active_labels", payload.files)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["candidate_count"], 1)
            with self.assertRaises(FileExistsError):
                build_online_causal_samples(
                    project_root=root,
                    config_path=config,
                    output_dir=output,
                    houses=["refit:2"],
                )

    def test_formal_mode_rejects_candidate_cap(self) -> None:
        """A candidate cap cannot be used as sampling for a formal dataset."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = self._write_fixture(root)
            with self.assertRaises(ValueError):
                build_online_causal_samples(
                    project_root=root,
                    config_path=config,
                    output_dir=root / "capped",
                    houses=["refit:2"],
                    maximum_candidates_per_house=1,
                    smoke_test=False,
                )

    def test_smoke_cap_is_explicit_in_manifest(self) -> None:
        """A smoke cap is allowed for quick checks, but the manifest must identify its nonformal scope."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = self._write_fixture(root)
            output = root / "smoke_output"
            build_online_causal_samples(
                project_root=root,
                config_path=config,
                output_dir=output,
                houses=["refit:2"],
                smoke_test=True,
                maximum_candidates_per_house=1,
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "smoke")
            self.assertTrue(manifest["candidate_cap_is_nonformal_sampling"])

    def test_manifest_hashes_and_csv_npz_counts_are_consistent(self) -> None:
        """Manifest output hashes refer to the committed directory; CSV, QC and NPZ row counts agree."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = self._write_fixture(root)
            output = root / "checked_output"
            build_online_causal_samples(
                project_root=root,
                config_path=config,
                output_dir=output,
                houses=["refit:2"],
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            candidates = (output / "candidates.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(candidates) - 1, manifest["candidate_count"])
            description = manifest["output_files"]["candidates.csv"]
            self.assertIn("checked_output/candidates.csv", description["path"])
            self.assertEqual(
                description["sha256"],
                hashlib.sha256((output / "candidates.csv").read_bytes()).hexdigest(),
            )
            qc = (output / "candidate_horizon_qc.csv").read_text(encoding="utf-8").splitlines()
            for horizon in (20, 40, 60, 120):
                accepted = sum(
                    1
                    for line in qc[1:]
                    if line.split(",")[4] == str(horizon) and line.split(",")[5] == "1"
                )
                with np.load(output / f"sequences_h{horizon}.npz", allow_pickle=False) as payload:
                    self.assertEqual(payload["candidate_ids"].size, accepted)

    def test_npz_candidate_order_follows_candidates_csv_after_reversed_house_scan(self) -> None:
        """NPZ rows retain canonical candidate order even when house processing order is reversed."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = self._write_fixture(root)
            self._add_second_refit_house(root)
            output = root / "ordered_output"
            # The actual CLI sorts houses; inject reverse scan order to cover disagreement between
            # NPZ append order and the final canonical candidates.csv order.
            with patch(
                "load_signature_analysis.online_samples._parse_house_selection",
                return_value=[HouseKey.parse("refit:19"), HouseKey.parse("refit:2")],
            ):
                build_online_causal_samples(
                    project_root=root,
                    config_path=config,
                    output_dir=output,
                    houses=["refit:19", "refit:2"],
                )
            with (output / "candidates.csv").open(encoding="utf-8", newline="") as handle:
                candidates = list(csv.DictReader(handle))
            with (output / "candidate_horizon_qc.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                qc_rows = list(csv.DictReader(handle))
            for horizon in (20, 40, 60, 120):
                accepted = {
                    row["candidate_id"]
                    for row in qc_rows
                    if int(row["horizon_seconds"]) == horizon and int(row["accepted"]) == 1
                }
                expected = [
                    row["candidate_id"] for row in candidates if row["candidate_id"] in accepted
                ]
                with np.load(output / f"sequences_h{horizon}.npz", allow_pickle=False) as payload:
                    self.assertEqual(payload["candidate_ids"].tolist(), expected)


if __name__ == "__main__":
    unittest.main()
