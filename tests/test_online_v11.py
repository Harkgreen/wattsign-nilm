"""Causality, label and output-contract regression tests for online_causal_v1_1 fixes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from load_signature_analysis.event_detection import DetectionResult
from load_signature_analysis.online_features import OnlineFeatureConfig, extract_online_features
from load_signature_analysis.online_samples import (
    CandidateLabelAudit,
    GroundTruthChannel,
    apply_source_quality_policy,
    assess_candidate_horizon,
    assign_candidate_labels,
    build_fixed_utc_bins_from_records,
    build_online_causal_samples,
    extract_candidate_window,
    load_online_config,
)


def _detection(
    *,
    starts: tuple[float, ...] = (),
    activities: tuple[tuple[float, float], ...] = (),
    coverage: tuple[tuple[float, float], ...] = ((-1_000.0, 1_000.0),),
) -> DetectionResult:
    """Create trusted submeter detections used only by pure label functions."""

    return DetectionResult(
        intervals=tuple(
            {
                "start_timestamp": start,
                "end_timestamp": start + 20.0,
                "touches_data_boundary": False,
            }
            for start in starts
        ),
        activity_intervals=tuple(
            {
                "start_timestamp": left,
                "end_timestamp": right,
                "touches_data_boundary": False,
            }
            for left, right in activities
        ),
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
    channel_id: str,
    appliance: str,
    coverage: tuple[tuple[float, float], ...],
    starts: tuple[float, ...] = (),
    activities: tuple[tuple[float, float], ...] = (),
) -> GroundTruthChannel:
    """Create primary-positive or transfer-positive channels without placing device power in model inputs."""

    return GroundTruthChannel(
        dataset="uk_dale",
        house_id="5",
        channel_id=channel_id,
        source_appliance=appliance,
        taxonomy_role=role,
        result=_detection(starts=starts, activities=activities, coverage=coverage),
    )


def _core_hard_background_audits() -> dict[str, CandidateLabelAudit]:
    """Create one normal core, hard and background label for shared weight and provenance tests."""

    full_coverage = ((-1_000.0, 2_000.0),)
    core = _channel(
        "core_positive",
        channel_id="1",
        appliance="kettle",
        coverage=full_coverage,
        starts=(100.0,),
        activities=((100.0, 140.0),),
    )
    hard = _channel(
        "hard_negative",
        channel_id="2",
        appliance="microwave",
        coverage=full_coverage,
        starts=(400.0,),
        activities=((400.0, 440.0),),
    )
    audits, _, _ = assign_candidate_labels(
        [
            {
                "candidate_id": "core",
                "candidate_onset_s": 100.0,
                "candidate_detected_at_s": 120.0,
            },
            {
                "candidate_id": "hard",
                "candidate_onset_s": 400.0,
                "candidate_detected_at_s": 420.0,
            },
            {
                "candidate_id": "background",
                "candidate_onset_s": 700.0,
                "candidate_detected_at_s": 720.0,
            },
        ],
        core_channels=[core],
        hard_channels=[hard],
    )
    return audits


def _accepted_ids(output: Path, horizon_seconds: int) -> list[str]:
    """Read usable candidate IDs for a horizon in canonical QC order."""

    with (output / "candidate_horizon_qc.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        return [
            row["candidate_id"]
            for row in csv.DictReader(handle)
            if int(row["horizon_seconds"]) == horizon_seconds
            and int(row["accepted"]) == 1
        ]


def _qc_by_horizon(output: Path) -> dict[int, dict[str, str]]:
    """Read the four horizon QC rows of a single-candidate fixture."""

    with (output / "candidate_horizon_qc.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["horizon_seconds"]): row for row in rows}


class OnlineV11Fixture:
    """Create a tiny UK-DALE fixture with independent aggregate and device submeters."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.mains = root / "mains.dat"
        self.submeter = root / "channel_1.dat"
        self.config = root / "online_v11.toml"
        self._write_static_files()
        self.write_mains()
        self.write_config(maximum_consecutive_missing_bins=1)

    def _write_static_files(self) -> None:
        """Write fixed submeters, a Phase 1 inventory and a placeholder taxonomy snapshot."""

        submeter_rows = []
        for timestamp in range(0, 271, 10):
            power_w = 1_200.0 if 100 <= timestamp < 210 else 0.0
            submeter_rows.append(f"{timestamp} {power_w}")
        self.submeter.write_text("\n".join(submeter_rows), encoding="utf-8")

        phase1 = self.root / "phase1"
        phase1.mkdir()
        (phase1 / "appliance_inventory.csv").write_text(
            "dataset,house_id,channel_id,source_column,source_file,canonical_label,"
            "taxonomy_role,taxonomy_confidence,data_use,include_in_main_mvp\n"
            f"uk_dale,5,1,channel_1,{self.submeter.as_posix()},kettle,"
            "core_positive,high,main,true\n",
            encoding="utf-8",
        )
        (phase1 / "aggregate_source_inventory.csv").write_text(
            "dataset,house_id,source_kind,source_file\n"
            f"uk_dale,5,alternate_high_frequency_mains,{self.mains.as_posix()}\n",
            encoding="utf-8",
        )
        (self.root / "taxonomy.yaml").write_text(
            "schema_version: online-causal-v1-1-test\n", encoding="utf-8"
        )

    def write_mains(
        self,
        *,
        invalid_zero_bin_left: int | None = None,
    ) -> None:
        """Write active-W mains, optionally injecting a quality gap after candidate confirmation."""

        rows: list[str] = []
        for timestamp in range(0, 271, 10):
            active_w = 500.0 if 100 <= timestamp < 210 else 100.0
            apparent_va = active_w
            if timestamp == invalid_zero_bin_left:
                active_w = 0.0
                apparent_va = 1_000.0
            rows.append(f"{timestamp} {active_w} {apparent_va}")
        self.mains.write_text("\n".join(rows), encoding="utf-8")

    def write_config(self, *, maximum_consecutive_missing_bins: int) -> None:
        """Write the fixed v1.1 configuration with a test-selectable consecutive-missing limit."""

        self.config.write_text(
            f"""
[schema]
schema_version = "online_causal_v1_1"
feature_registry_version = "online_feature_v1_1"
npz_contract_version = "online_sequence_v1_1"

[paths]
appliance_inventory = "phase1/appliance_inventory.csv"
aggregate_source_inventory = "phase1/aggregate_source_inventory.csv"
taxonomy = "taxonomy.yaml"
default_output_dir = "formal_v11"

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
maximum_consecutive_missing_bins = {maximum_consecutive_missing_bins}
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
train = ["uk_dale:5"]
validation = []
test = []

[thresholds.kettle]
on_threshold_w = 1000.0
off_threshold_w = 100.0
minimum_on_seconds = 20.0
merge_gap_seconds = 15.0
maximum_sample_gap_seconds = 30.0
""".strip(),
            encoding="utf-8",
        )

    def build(self, output_name: str) -> Path:
        """Build a formal output directory that cannot be overwritten."""

        output = self.root / output_name
        build_online_causal_samples(
            project_root=self.root,
            config_path=self.config,
            output_dir=output,
            houses=["uk_dale:5"],
        )
        return output


class OnlineV11LabelAndFeatureTests(unittest.TestCase):
    """Fix transfer-coverage safeguards and sign-change semantics around missing segments."""

    def test_matched_core_starting_before_candidate_is_not_preexisting_load(self) -> None:
        """An earlier activation within tolerance cannot also be labeled as a pre-existing load for the same submeter event."""

        core = _channel(
            "core_positive",
            channel_id="1",
            appliance="electric_heater",
            coverage=((-1_000.0, 2_000.0),),
            starts=(90.0,),
            activities=((90.0, 160.0),),
        )
        audits, _, _ = assign_candidate_labels(
            [
                {
                    "candidate_id": "aggregate-lags-submeter",
                    "candidate_onset_s": 100.0,
                    "candidate_detected_at_s": 120.0,
                }
            ],
            core_channels=[core],
            hard_channels=[],
        )

        audit = audits["aggregate-lags-submeter"]
        self.assertEqual(audit.label, 1)
        self.assertEqual(audit.label_type, "core_positive")
        self.assertEqual(audit.onset_match_error_seconds, 10.0)
        self.assertNotIn("target_already_active_before_candidate", audit.ignore_reasons)

    def test_normal_label_types_have_frozen_confidence_and_weights(self) -> None:
        """Normal-source core/hard/background weights remain 1, 1 and 0.25."""

        audits = _core_hard_background_audits()
        expected = {
            "core": (1, "core_positive", "high", 1.0),
            "hard": (0, "hard_negative", "high", 1.0),
            "background": (
                0,
                "background_negative",
                "weak_unmonitored_background",
                0.25,
            ),
        }
        for candidate_id, contract in expected.items():
            with self.subTest(candidate_id=candidate_id):
                audit = audits[candidate_id]
                self.assertEqual(
                    (
                        audit.label,
                        audit.label_type,
                        audit.label_confidence,
                        audit.sample_weight,
                    ),
                    contract,
                )

    def test_missing_refit_issues_keeps_only_matched_core_at_medium_confidence(self) -> None:
        """Without Issues, retain only matched positives and reduce confidence and weight to medium/0.5."""

        adjusted = apply_source_quality_policy(
            _core_hard_background_audits(),
            source_quality_flags=("refit_issues_flag_unavailable",),
            allow_refit_missing_issues_matched_positive=True,
            allow_refit_missing_issues_background_negative=False,
            missing_issues_positive_sample_weight=0.5,
        )

        core = adjusted["core"]
        self.assertEqual(core.label, 1)
        self.assertEqual(core.label_type, "core_positive")
        self.assertEqual(core.label_status, "matched_source_quality_limited")
        self.assertEqual(core.label_confidence, "medium_missing_refit_issues")
        self.assertEqual(core.sample_weight, 0.5)
        self.assertIn("refit_issues_flag_unavailable", core.source_quality_warnings)

        for candidate_id in ("hard", "background"):
            with self.subTest(candidate_id=candidate_id):
                audit = adjusted[candidate_id]
                self.assertIsNone(audit.label)
                self.assertEqual(audit.label_type, "ignore")
                self.assertEqual(audit.label_status, "ignore")
                self.assertEqual(audit.label_confidence, "none")
                self.assertEqual(audit.sample_weight, 0.0)
                self.assertIn("refit_issues_flag_unavailable", audit.ignore_reasons)
                self.assertIn(
                    "refit_issues_flag_unavailable", audit.source_quality_warnings
                )

    def test_transfer_coverage_gap_blocks_background_negative(self) -> None:
        """Unknown transfer-positive coverage prevents certification as a background negative."""

        core = _channel(
            "core_positive",
            channel_id="1",
            appliance="kettle",
            coverage=((0.0, 500.0),),
        )
        transfer = _channel(
            "extended_clean_positive",
            channel_id="2",
            appliance="hair_dryer",
            coverage=((0.0, 90.0), (230.0, 500.0)),
        )
        audits, _, _ = assign_candidate_labels(
            [
                {
                    "candidate_id": "background-with-transfer-gap",
                    "candidate_onset_s": 100.0,
                    "candidate_detected_at_s": 120.0,
                }
            ],
            core_channels=[core],
            hard_channels=[],
            transfer_channels=[transfer],
            aggregate_max_window_trusted={"background-with-transfer-gap": True},
        )

        audit = audits["background-with-transfer-gap"]
        self.assertEqual(audit.label_type, "ignore")
        self.assertIsNone(audit.label)
        self.assertIn("transfer_positive_coverage_gap", audit.ignore_reasons)

    def test_sign_change_count_does_not_bridge_a_missing_bin(self) -> None:
        """A missing bin breaks the step-sign sequence; signs across a gap do not form a reversal."""

        pre = np.full(6, 100.0)
        post = np.asarray([200.0, np.nan, 100.0, 0.0, 0.0, 0.0])
        post_quality = np.asarray([1, 0, 1, 1, 1, 1], dtype=np.int8)
        row = extract_online_features(
            pre,
            post,
            np.ones(6, dtype=np.int8),
            post_quality,
            horizon_seconds=60,
        )

        self.assertEqual(row["online_temporal__positive_step_count"], 1.0)
        self.assertEqual(row["online_temporal__negative_step_count"], 1.0)
        self.assertEqual(row["online_temporal__sign_change_count"], 0.0)
        self.assertEqual(row["mask__online_temporal__sign_change_count"], 1)


class OnlineV11EndToEndTests(unittest.TestCase):
    """Use the actual builder to check horizon membership, quality configuration and NPZ alignment."""

    def test_config_rejects_noninteger_chunks_horizons_and_old_schema(self) -> None:
        """Validate integer fields by their actual TOML types; do not silently cast bool or float to int."""

        mutations = {
            "refit_chunksize_bool": ("refit_chunksize = 20", "refit_chunksize = true"),
            "refit_chunksize_float": ("refit_chunksize = 20", "refit_chunksize = 20.0"),
            "ukdale_chunksize_bool": ("ukdale_chunksize = 20", "ukdale_chunksize = false"),
            "ukdale_chunksize_float": ("ukdale_chunksize = 20", "ukdale_chunksize = 20.0"),
            "float_horizon": (
                "horizons_seconds = [20, 40, 60, 120]",
                "horizons_seconds = [20.0, 40, 60, 120]",
            ),
            "old_schema": (
                'schema_version = "online_causal_v1_1"',
                'schema_version = "online_causal_v1"',
            ),
        }
        for case_name, (original, replacement) in mutations.items():
            with self.subTest(case_name=case_name), tempfile.TemporaryDirectory() as temporary:
                fixture = OnlineV11Fixture(Path(temporary))
                valid_text = fixture.config.read_text(encoding="utf-8")
                self.assertIn(original, valid_text)
                fixture.config.write_text(
                    valid_text.replace(original, replacement, 1), encoding="utf-8"
                )
                with self.assertRaises(ValueError):
                    load_online_config(fixture.config)

    def test_h20_membership_ignores_later_aggregate_invalid_zero(self) -> None:
        """Changing mains quality after t0+20 cannot change H20 membership."""

        with tempfile.TemporaryDirectory() as temporary:
            fixture = OnlineV11Fixture(Path(temporary))
            clean_output = fixture.build("clean")
            clean_h20 = _accepted_ids(clean_output, 20)

            # With t0=100, bin 120 lies outside half-open H20 but remains inside H60/H120.
            fixture.write_mains(invalid_zero_bin_left=120)
            damaged_output = fixture.build("damaged_after_h20")
            damaged_h20 = _accepted_ids(damaged_output, 20)
            damaged_qc = _qc_by_horizon(damaged_output)

            self.assertEqual(len(clean_h20), 1)
            self.assertEqual(damaged_h20, clean_h20)
            self.assertEqual(int(damaged_qc[20]["accepted"]), 1)
            for horizon in (60, 120):
                self.assertEqual(int(damaged_qc[horizon]["quality_accepted"]), 0)
                self.assertEqual(int(damaged_qc[horizon]["accepted"]), 0)
                self.assertIn(
                    "ukdale_invalid_active_zero",
                    damaged_qc[horizon]["rejection_reasons"].split(";"),
                )

    def test_h20_membership_ignores_long_gap_starting_at_right_boundary(self) -> None:
        """A long gap known only after the right boundary cannot invalidate the final H20 bin retrospectively."""

        complete_records = [
            (
                float(timestamp),
                100.0 if timestamp < 0 else 500.0,
                True,
                (),
            )
            for timestamp in range(-80, 201, 10)
        ]
        delayed_future_records = [
            record
            for record in complete_records
            if record[0] < 20.0 or record[0] >= 100.0
        ]

        complete_window = extract_candidate_window(
            build_fixed_utc_bins_from_records(complete_records), 0, 20
        )
        delayed_window = extract_candidate_window(
            build_fixed_utc_bins_from_records(delayed_future_records), 0, 20
        )
        np.testing.assert_equal(delayed_window.power_w, complete_window.power_w)
        np.testing.assert_array_equal(
            delayed_window.quality_valid_mask,
            complete_window.quality_valid_mask,
        )
        self.assertEqual(delayed_window.quality_flags, complete_window.quality_flags)

        for window in (complete_window, delayed_window):
            accepted, reasons = assess_candidate_horizon(
                window,
                horizon_seconds=20,
                feature_config=OnlineFeatureConfig(),
            )
            self.assertTrue(accepted, reasons)

    def test_h20_membership_does_not_require_a_record_after_right_boundary(self) -> None:
        """Short-window completeness uses only in-boundary data; future records cannot resolve its right file boundary."""

        window_records = [
            (
                float(timestamp),
                100.0 if timestamp < 0 else 500.0,
                True,
                (),
            )
            for timestamp in range(-80, 20, 10)
        ]
        future_extended_records = [
            *window_records,
            (100.0, 500.0, True, ()),
        ]
        without_future = extract_candidate_window(
            build_fixed_utc_bins_from_records(window_records), 0, 20
        )
        with_future = extract_candidate_window(
            build_fixed_utc_bins_from_records(future_extended_records), 0, 20
        )

        self.assertFalse(without_future.touches_file_boundary)
        self.assertFalse(with_future.touches_file_boundary)
        np.testing.assert_equal(without_future.power_w, with_future.power_w)
        np.testing.assert_array_equal(
            without_future.quality_valid_mask,
            with_future.quality_valid_mask,
        )
        for window in (without_future, with_future):
            accepted, reasons = assess_candidate_horizon(
                window,
                horizon_seconds=20,
                feature_config=OnlineFeatureConfig(),
            )
            self.assertTrue(accepted, reasons)

    def test_maximum_consecutive_missing_bins_is_frozen_and_forwardable(self) -> None:
        """The formal configuration fixes this to 1, but underlying QC parameters must still work without hardcoding."""

        records = [
            (float(timestamp), 500.0, True, ())
            for timestamp in range(-100, 101, 10)
            if timestamp != 20
        ]
        window = extract_candidate_window(
            build_fixed_utc_bins_from_records(records), 0, 60
        )
        permissive, permissive_reasons = assess_candidate_horizon(
            window,
            horizon_seconds=60,
            feature_config=OnlineFeatureConfig(),
            maximum_consecutive_missing_bins=1,
        )
        strict, strict_reasons = assess_candidate_horizon(
            window,
            horizon_seconds=60,
            feature_config=OnlineFeatureConfig(),
            maximum_consecutive_missing_bins=0,
        )
        self.assertTrue(permissive, permissive_reasons)
        self.assertFalse(strict)
        self.assertIn("consecutive_missing_bins", strict_reasons)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = OnlineV11Fixture(Path(temporary))
            fixture.write_config(maximum_consecutive_missing_bins=0)
            with self.assertRaisesRegex(
                ValueError, "window.maximum_consecutive_missing_bins"
            ):
                load_online_config(fixture.config)

    def test_npz_has_exact_v11_keys_dtypes_and_row_alignment(self) -> None:
        """Fix ten keys and align QC with sequence_bins for every candidate and time step."""

        with tempfile.TemporaryDirectory() as temporary:
            fixture = OnlineV11Fixture(Path(temporary))
            output = fixture.build("npz_contract")

            with (output / "candidate_horizon_qc.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                qc_rows = list(csv.DictReader(handle))
            with (output / "sequence_bins.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                sequence_rows = list(csv.DictReader(handle))

            expected_keys = {
                "candidate_ids",
                "values",
                "quality_valid_mask",
                "target_active_label",
                "labels",
                "sample_weights",
                "split",
                "sequence_feature_names",
                "relative_bin_indices",
                "horizon_seconds",
            }
            expected_lengths = {20: 8, 40: 10, 60: 12, 120: 18}
            for horizon, sequence_length in expected_lengths.items():
                accepted_rows = [
                    row
                    for row in qc_rows
                    if int(row["horizon_seconds"]) == horizon
                    and int(row["accepted"]) == 1
                ]
                with np.load(
                    output / f"sequences_h{horizon}.npz", allow_pickle=False
                ) as payload:
                    self.assertEqual(set(payload.files), expected_keys)
                    sample_count = len(accepted_rows)
                    self.assertEqual(payload["candidate_ids"].shape, (sample_count,))
                    self.assertEqual(payload["values"].shape, (sample_count, sequence_length, 4))
                    self.assertEqual(
                        payload["quality_valid_mask"].shape,
                        (sample_count, sequence_length),
                    )
                    self.assertEqual(
                        payload["target_active_label"].shape,
                        (sample_count, sequence_length),
                    )
                    self.assertEqual(payload["labels"].shape, (sample_count,))
                    self.assertEqual(payload["sample_weights"].shape, (sample_count,))
                    self.assertEqual(payload["split"].shape, (sample_count,))
                    self.assertEqual(payload["sequence_feature_names"].shape, (4,))
                    self.assertEqual(payload["relative_bin_indices"].shape, (sequence_length,))
                    self.assertEqual(payload["horizon_seconds"].shape, ())

                    self.assertIn(payload["candidate_ids"].dtype.kind, {"U", "S"})
                    self.assertEqual(payload["values"].dtype, np.dtype(np.float64))
                    self.assertEqual(payload["quality_valid_mask"].dtype, np.dtype(np.uint8))
                    self.assertEqual(payload["target_active_label"].dtype, np.dtype(np.int8))
                    self.assertEqual(payload["labels"].dtype, np.dtype(np.int8))
                    self.assertEqual(payload["sample_weights"].dtype, np.dtype(np.float32))
                    self.assertIn(payload["split"].dtype.kind, {"U", "S"})
                    self.assertIn(payload["sequence_feature_names"].dtype.kind, {"U", "S"})
                    self.assertEqual(payload["relative_bin_indices"].dtype, np.dtype(np.int64))
                    self.assertEqual(payload["horizon_seconds"].dtype, np.dtype(np.int64))

                    expected_ids = [row["candidate_id"] for row in accepted_rows]
                    expected_labels = [int(row["label"]) for row in accepted_rows]
                    expected_splits = [row["split"] for row in accepted_rows]
                    self.assertEqual(payload["candidate_ids"].tolist(), expected_ids)
                    self.assertEqual(payload["labels"].tolist(), expected_labels)
                    self.assertEqual(payload["split"].tolist(), expected_splits)
                    self.assertTrue(np.all(np.isfinite(payload["sample_weights"])))
                    self.assertTrue(np.all(payload["sample_weights"] > 0.0))

                    for sample_index, candidate_id in enumerate(expected_ids):
                        expected_relative = payload["relative_bin_indices"].tolist()
                        candidate_bins = {
                            int(row["relative_bin_index"]): row
                            for row in sequence_rows
                            if row["candidate_id"] == candidate_id
                        }
                        expected_power = np.asarray(
                            [
                                float(candidate_bins[index]["aggregate_power_w"])
                                for index in expected_relative
                            ],
                            dtype=np.float64,
                        )
                        expected_quality = np.asarray(
                            [
                                int(candidate_bins[index]["quality_valid_mask"])
                                for index in expected_relative
                            ],
                            dtype=np.uint8,
                        )
                        expected_active = np.asarray(
                            [
                                int(candidate_bins[index]["target_active_label"])
                                for index in expected_relative
                            ],
                            dtype=np.int8,
                        )
                        np.testing.assert_allclose(
                            payload["values"][sample_index, :, 0], expected_power
                        )
                        np.testing.assert_array_equal(
                            payload["quality_valid_mask"][sample_index], expected_quality
                        )
                        np.testing.assert_array_equal(
                            payload["target_active_label"][sample_index], expected_active
                        )

    def test_manifest_versions_match_real_npz_keys_and_empty_weight_shape(self) -> None:
        """The manifest's three versions and key list match actual NPZ; empty arrays retain one-dimensional weights."""

        with tempfile.TemporaryDirectory() as temporary:
            fixture = OnlineV11Fixture(Path(temporary))
            # One invalid-zero bin after H20 lets H60/H120 produce valid empty NPZ files.
            fixture.write_mains(invalid_zero_bin_left=120)
            output = fixture.build("manifest_contract")
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["schema_version"], "online_causal_v1_1")
            self.assertEqual(
                manifest["feature_registry_version"], "online_feature_v1_1"
            )
            self.assertEqual(
                manifest["npz_contract_version"], "online_sequence_v1_1"
            )
            self.assertEqual(
                manifest["config_snapshot"]["schema"],
                {
                    "schema_version": "online_causal_v1_1",
                    "feature_registry_version": "online_feature_v1_1",
                    "npz_contract_version": "online_sequence_v1_1",
                },
            )

            for horizon in (20, 40, 60, 120):
                with np.load(
                    output / f"sequences_h{horizon}.npz", allow_pickle=False
                ) as payload:
                    self.assertEqual(payload.files, manifest["npz_keys"])
            with np.load(output / "sequences_h120.npz", allow_pickle=False) as payload:
                self.assertEqual(payload["candidate_ids"].shape, (0,))
                self.assertEqual(payload["sample_weights"].shape, (0,))
                self.assertEqual(payload["sample_weights"].dtype, np.dtype(np.float32))


if __name__ == "__main__":
    unittest.main()
