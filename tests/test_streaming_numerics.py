'Synthetic numerical regression tests for causal timelines, alarms and event metrics. No raw data, model fitting or historical release workflow is used.'

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import csv
import gc
import hashlib
import io
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from load_signature_analysis.alarm import (
    AlarmConfig,
    AlarmMachineState,
    AlarmMetadata,
    AlarmObservation,
    AlarmState,
    AlarmSymbol,
    PredictionEvent,
    ReferenceAlarm,
    build_prediction_event_id,
    build_prediction_stream_id,
    float32_hex,
    transition_alarm_state,
)
from load_signature_analysis.evaluation import (
    ContractError,
    alarm_success_at_deadline,
    classify_prediction_scoring,
    compute_scoring_exposure,
    derive_claim_statistics,
    deterministic_house_bootstrap,
    evaluate_matched_pair,
    match_events,
    nearest_rank_p95,
    reduce_class_claim,
    reduce_overall_claim,
    report_exploratory_claims,
    select_threshold,
    summarize_unmatched_predictions,
    wilson_95_lower_bound,
)
from load_signature_analysis.timeline import (
    CLASS_ORDER,
    REQUIRED_ARRAYS,
    AggregateSample,
    LabelDescriptor,
    SubmeterSample,
    TimelineContractError,
    apply_masked_normalizer,
    array_logical_descriptor,
    build_timeline,
    canonical_array_payload,
    describe_container,
    fit_masked_normalizer,
    iter_stream_feature_rows,
    jcs_bytes,
    materialize_model_window,
    stable_truth_event_id,
    stable_truth_event_id_payload,
    typed_record_logical_descriptor,
    validate_unique_truth_event_ids,
    validate_container_descriptor,
    validate_container_metadata,
)


T0 = 1_704_067_200
PS_CONTRACT_SHA = "a" * 64
PS_REGISTRY_SHA = "b" * 64
PS_SOURCE_SHA = "c" * 64
PS_TIMELINE_SHA = "d" * 64
THRESHOLD_BITS = [["3f333333", "3f0ccccd"] for _ in CLASS_ORDER]
PREDICTION_STREAM_ID = build_prediction_stream_id(
    contract_sha256=PS_CONTRACT_SHA,
    class_registry_sha256=PS_REGISTRY_SHA,
    model_or_probability_source_descriptor_sha256=PS_SOURCE_SHA,
    input_timeline_logical_content_sha256=PS_TIMELINE_SHA,
    class_thresholds_float32_bitpatterns=THRESHOLD_BITS,
    class_order=CLASS_ORDER,
)


def _json_safe(value: object) -> object:
    """Normalize NumPy/dataclass results to JSON values that evidence can hash exactly."""

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _case(scenario_id: str, fixture: dict[str, object], expected: object, actual: object) -> dict[str, object]:
    """Create persistable matrix-scenario evidence using strict object equality."""

    expected_safe = _json_safe(expected)
    actual_safe = _json_safe(actual)
    if expected_safe != actual_safe:
        raise AssertionError(
            f"{scenario_id} exact comparison failed\nexpected={expected_safe!r}\nactual={actual_safe!r}"
        )
    return {
        "scenario_id": scenario_id,
        "fixture": fixture,
        "expected": expected_safe,
        "actual": actual_safe,
        "status": "PASS",
    }


def _aggregate(
    index: int,
    power: float,
    *,
    second: float | None = None,
    arrival: int | None = None,
    issues: int | None = 0,
    source: str = "agg_a",
    apparent: float | None = None,
) -> AggregateSample:
    """Construct the unique aggregate schema for public matrix notation A(k,p)."""

    return AggregateSample(
        arrival_ordinal=index if arrival is None else arrival,
        timestamp_unix_s=T0 + (10 * index + 5 if second is None else second),
        active_power_w=power,
        apparent_power_va=apparent,
        refit_issues=issues,
        source_file_id=source,
    )


def _submeter(
    channel: str,
    second: float,
    power: float,
    *,
    appliance_class: str = "kettle",
    arrival: int | None = None,
    epoch: int = 0,
) -> SubmeterSample:
    """Construct the unique submeter schema for S(channel, second, power)."""

    return SubmeterSample(
        arrival_ordinal=int(second) if arrival is None else arrival,
        timestamp_unix_s=T0 + second,
        power_w=power,
        source_channel_id=channel,
        canonical_class_id=appliance_class,
        channel_label_epoch_id=epoch,
    )


def _descriptor(
    channel: str,
    appliance_class: str,
    *,
    start: float = 0,
    end: float,
    epoch: int = 0,
    source_sha: str | None = None,
) -> LabelDescriptor:
    return LabelDescriptor(
        source_channel_id=channel,
        canonical_class_id=appliance_class,
        valid_from_unix_s=T0 + start,
        valid_to_unix_s=T0 + end,
        channel_label_epoch_id=epoch,
        source_descriptor_sha256=source_sha,
    )


def _timeline(
    aggregate_records: list[AggregateSample],
    *,
    end: float,
    scenario: str,
    house_id: str | None = None,
    dataset: str = "refit",
    expected_channels: dict[str, list[str]] | None = None,
    submeter_records: list[SubmeterSample] | None = None,
    label_descriptors: list[LabelDescriptor] | None = None,
):
    """Construct a wholly synthetic timeline without supplying paths or external objects to the builder."""

    return build_timeline(
        aggregate_records,
        source_start=T0,
        source_end=T0 + end,
        dataset=dataset,
        house_id=house_id or f"synthetic_{scenario.lower()}",
        expected_channels=expected_channels or {},
        submeter_records=submeter_records or [],
        label_descriptors=label_descriptors,
    )


def _class_column(timeline, appliance_class: str) -> int:
    return CLASS_ORDER.index(appliance_class)


def _event_view(event: dict[str, object]) -> dict[str, object]:
    """Keep only verifiable matrix event fields; exclude unrelated implementation details."""

    keys = (
        "true_event_id",
        "appliance_class",
        "true_start_time",
        "true_start_lower_bound",
        "true_start_upper_bound",
        "true_end_time",
        "end_reason",
        "right_censored",
        "source_channel_ids_json",
    )
    return {key: event[key] for key in keys}


def _hand_authored_truth_id(
    *,
    appliance_class: str,
    house_id: str,
    true_start_relative_seconds: float,
    source_channel_ids_sorted: list[str],
    label_epoch_id: int = 0,
) -> str:
    """L11 uses a handwritten JCS payload to independently verify the production stable-ID algorithm."""

    payload = {
        "appliance_class": appliance_class,
        "continuity_id": 0,
        "dataset": "refit",
        "house_id": house_id,
        "label_epoch_id": label_epoch_id,
        "source_channel_ids_sorted": source_channel_ids_sorted,
        "true_start_time": T0 + true_start_relative_seconds,
    }
    production_payload = stable_truth_event_id_payload(
        "refit", house_id, 0, label_epoch_id, appliance_class, T0 + true_start_relative_seconds, source_channel_ids_sorted
    )
    if payload != production_payload:
        raise AssertionError("手写 stable-ID payload 与 production 字段规则不一致")
    manual = "truth_" + hashlib.sha256(jcs_bytes(payload)).hexdigest()
    production = stable_truth_event_id(
        "refit", house_id, 0, label_epoch_id, appliance_class, T0 + true_start_relative_seconds, source_channel_ids_sorted
    )
    if manual != production:
        raise AssertionError("手写 JCS payload 与 production stable-ID 算法不一致")
    return manual


def _alarm_observations(
    probabilities: list[float],
    *,
    valid: list[bool] | None = None,
    hard_gaps: set[int] | None = None,
) -> list[AlarmObservation]:
    valid = valid or [True] * len(probabilities)
    hard_gaps = hard_gaps or set()
    return [
        AlarmObservation(
            bin_left=T0 + 10 * index,
            bin_right=T0 + 10 * (index + 1),
            probability=value,
            aggregate_valid=valid[index],
            hard_gap=index in hard_gaps,
        )
        for index, value in enumerate(probabilities)
    ]


def _new_alarm(*, state: AlarmMachineState | None = None) -> ReferenceAlarm:
    return ReferenceAlarm(
        PREDICTION_STREAM_ID,
        metadata=AlarmMetadata(),
        config=AlarmConfig(0.70, 0.55),
        initial_state=state,
    )


def _event_record(event: PredictionEvent | None) -> dict[str, object] | None:
    """Convert immutable alarm events to evidence-friendly records using fixed output fields."""

    return None if event is None else event.to_record()


def _state_record(state: AlarmMachineState) -> dict[str, object]:
    """Keep only the state/counter/marker fields explicitly required by A01--A09."""

    return {
        "state": state.state.value,
        "on_count": state.on_count,
        "off_count": state.off_count,
        "invalid_count": state.invalid_count,
        "first_on_bin_left": state.first_on_bin_left,
        "first_low_bin_left": state.first_low_bin_left,
        "first_invalid_bin_left": state.first_invalid_bin_left,
        "last_quality_valid_bin_right": state.last_quality_valid_bin_right,
        "open_event": _event_record(state.open_event),
    }


def _transition_record(transition) -> dict[str, object]:
    """Serialize the A05 direct transition without treating dataclass repr as an oracle."""

    return {
        "symbol": transition.symbol.value,
        "pre": _state_record(transition.pre_state),
        "post": _state_record(transition.post_state),
        "emitted": _event_record(transition.emitted_alarm),
        "closed": _event_record(transition.closed_event),
    }


def _alarm_run(probabilities: list[float], *, valid: list[bool] | None = None, hard_gaps: set[int] | None = None, end: int | None = None):
    """Replay a handwritten probability/QC sequence, returning alarms, per-bin transitions and optional stream-end."""

    alarm = _new_alarm()
    transitions = list(alarm.run(_alarm_observations(probabilities, valid=valid, hard_gaps=hard_gaps)))
    if end is not None:
        transitions.append(alarm.stream_end(T0 + end))
    return alarm, transitions


def _prediction(
    event_id: str,
    *,
    start: float,
    report: float,
    end: float,
    house: str = "synthetic_e",
    appliance_class: str = "kettle",
    scored: bool = True,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "dataset": "refit",
        "house_id": house,
        "continuity_id": 0,
        "appliance_class": appliance_class,
        "start_time": T0 + start,
        "report_time": T0 + report,
        "end_time": T0 + end,
        "prediction_scored": scored,
        "unscored_unknown": not scored,
    }


def _truth(
    event_id: str,
    *,
    start: float,
    lower: float | None = None,
    end: float,
    house: str = "synthetic_e",
    appliance_class: str = "kettle",
    censored: bool = False,
) -> dict[str, object]:
    return {
        "true_event_id": event_id,
        "dataset": "refit",
        "house_id": house,
        "continuity_id": 0,
        "appliance_class": appliance_class,
        "true_start_time": T0 + start,
        "true_start_lower_bound": T0 + (start if lower is None else lower),
        "true_start_upper_bound": T0 + start,
        "true_end_time": T0 + end,
        "right_censored": censored,
    }


A05_IDS = tuple(
    f"A05_{state}_{symbol}"
    for state in ("OFF", "PENDING_ON", "ON", "PENDING_OFF")
    for symbol in ("high", "low", "deadband", "invalid_soft", "invalid_close", "hard_gap", "stream_end")
)
REQUIRED_SCENARIO_IDS = (
    *(f"I{index:02d}" for index in range(1, 12)),
    *(f"L{index:02d}" for index in range(1, 14)),
    "A01",
    "A02",
    "A03",
    "A04",
    *A05_IDS,
    "A06",
    "A07",
    "A08",
    "A09",
    *(f"E{index:02d}" for index in range(1, 13)),
    "D01",
    "D02",
)


def _scenario_i01() -> dict[str, object]:
    timeline = _timeline(
        [
            _aggregate(0, 100, second=9.999, arrival=0),
            _aggregate(1, 300, second=10.0, arrival=1),
            _aggregate(1, 500, second=19.999, arrival=2),
            _aggregate(2, 700, second=20.0, arrival=3),
        ],
        end=30,
        scenario="i01",
    )
    actual = {
        "p": timeline.input_values[:, 0].tolist(),
        "p_mask": timeline.input_value_mask[:, 0].tolist(),
        "delta": timeline.input_values[:, 1].tolist(),
        "delta_mask": timeline.input_value_mask[:, 1].tolist(),
        "baseline_mask": timeline.input_value_mask[:, 2].tolist(),
        "count": timeline.raw_sample_count.tolist(),
    }
    return _case(
        "I01",
        {"records": [[9.999, 100], [10.0, 300], [19.999, 500], [20.0, 700]], "source_end": 30},
        {
            "p": [100.0, 400.0, 700.0],
            "p_mask": [1, 1, 1],
            "delta": [0.0, 300.0, 300.0],
            "delta_mask": [0, 1, 1],
            "baseline_mask": [0, 0, 0],
            "count": [1, 2, 1],
        },
        actual,
    )


def _scenario_i02() -> dict[str, object]:
    timeline = _timeline(
        [_aggregate(0, 100), _aggregate(1, 200), _aggregate(2, 300, second=25)],
        end=26,
        scenario="i02",
    )
    tail = 2
    actual = {
        "rows": timeline.row_count,
        "tail": {
            "values": timeline.input_values[tail].tolist(),
            "masks": timeline.input_value_mask[tail].tolist(),
            "flags": int(timeline.input_quality_flags[tail]),
            "count": int(timeline.raw_sample_count[tail]),
            "model_eligible": bool(timeline.model_eligible[tail]),
            "active_masks": timeline.active_valid_mask[tail].tolist(),
            "epochs": timeline.label_epoch_id[tail].tolist(),
        },
    }
    expected = {
        "rows": 3,
        "tail": {
            "values": [0.0, 0.0, 0.0],
            "masks": [0, 0, 0],
            "flags": (1 << 5) | (1 << 7),
            "count": 1,
            "model_eligible": False,
            "active_masks": [0] * 6,
            "epochs": [-1] * 6,
        },
    }
    full = _timeline([_aggregate(0, 100), _aggregate(1, 200), _aggregate(2, 300, second=25)], end=30, scenario="i02_full")
    if full.row_count != 3 or bool(full.model_eligible[2]) is not True:
        raise AssertionError("I02 control full bin 未成为正常完整格")
    return _case("I02", {"source_end": 26, "partial_tail": True}, expected, actual)


def _scenario_i03() -> dict[str, object]:
    records = [_aggregate(index, 100) for index in range(6)]
    records.extend([_aggregate(6, 1000), _aggregate(7, 1000)])
    timeline = _timeline(records, end=80, scenario="i03")
    actual = {
        "b6_values": timeline.input_values[6].tolist(),
        "b6_masks": timeline.input_value_mask[6].tolist(),
        "b7_values": timeline.input_values[7].tolist(),
        "b7_masks": timeline.input_value_mask[7].tolist(),
    }
    return _case(
        "I03",
        {"aggregate": "A(0..5,100),A(6..7,1000)", "source_end": 80},
        {
            "b6_values": [1000.0, 900.0, 900.0],
            "b6_masks": [1, 1, 1],
            "b7_values": [1000.0, 0.0, 900.0],
            "b7_masks": [1, 1, 1],
        },
        actual,
    )


def _scenario_i04() -> dict[str, object]:
    uk = _timeline(
        [
            _aggregate(0, 0, second=5, apparent=0),
            _aggregate(2, 0, second=25, apparent=60),
            _aggregate(3, 100, second=35, apparent=100),
        ],
        end=40,
        scenario="i04_uk",
        dataset="uk_dale",
    )
    refit_issue = _timeline(
        [_aggregate(0, 100, second=5, arrival=0), _aggregate(0, 110, second=6, arrival=1, issues=1)],
        end=10,
        scenario="i04_refit_issue",
    )
    refit_zero = _timeline(
        [_aggregate(0, 100, second=5, arrival=0), _aggregate(0, 0, second=6, arrival=1)],
        end=10,
        scenario="i04_refit_zero",
    )
    actual = {
        "uk_p": uk.input_values[:, 0].tolist(),
        "uk_mask": uk.input_value_mask[:, 0].tolist(),
        "uk_flags": uk.input_quality_flags.tolist(),
        "refit_issue": [int(refit_issue.input_quality_flags[0]), int(refit_issue.raw_sample_count[0])],
        "refit_zero": [int(refit_zero.input_quality_flags[0]), int(refit_zero.raw_sample_count[0])],
    }
    return _case(
        "I04",
        {"uk_dale": "true zero/missing/apparent veto", "refit": "issues and zero veto"},
        {
            "uk_p": [0.0, 0.0, 0.0, 100.0],
            "uk_mask": [1, 0, 0, 1],
            "uk_flags": [0, 1, 8, 0],
            "refit_issue": [2, 2],
            "refit_zero": [4, 2],
        },
        actual,
    )


def _scenario_i05() -> dict[str, object]:
    timeline = _timeline(
        [_aggregate(0, 100, second=5), _aggregate(3, 110, second=35), _aggregate(6, 120, second=66)],
        end=70,
        scenario="i05",
    )
    control = _timeline(
        [_aggregate(0, 100, second=5), _aggregate(3, 110, second=35), _aggregate(6, 120, second=65)],
        end=70,
        scenario="i05_control",
    )
    actual = {
        "continuity": timeline.continuity_id.tolist(),
        "flags": timeline.input_quality_flags.tolist(),
        "missing_masks": timeline.input_value_mask[[1, 2, 4, 5], 0].tolist(),
        "control_last_continuity": int(control.continuity_id[6]),
    }
    return _case(
        "I05",
        {"intervals": [30, 31], "source_end": 70},
        {
            "continuity": [0, 0, 0, 0, 0, 0, 1],
            "flags": [0, 1, 1, 0, 1, 1, 64],
            "missing_masks": [0, 0, 0, 0],
            "control_last_continuity": 0,
        },
        actual,
    )


def _scenario_i06() -> dict[str, object]:
    timeline = _timeline(
        [
            _aggregate(0, 100, second=5, arrival=0),
            _aggregate(2, 120, second=25, arrival=1),
            _aggregate(1, 999, second=15, arrival=2),
            _aggregate(2, 888, second=25, arrival=3),
            _aggregate(3, 130, second=35, arrival=4),
        ],
        end=40,
        scenario="i06",
    )
    actual = {
        "p": timeline.input_values[:, 0].tolist(),
        "flags": timeline.input_quality_flags.tolist(),
        "continuity": timeline.continuity_id.tolist(),
        "audit": timeline.aggregate_conflict_audit.as_dict(),
    }
    return _case(
        "I06",
        {"arrival_order": [[5, 100], [25, 120], [15, 999], [25, 888], [35, 130]]},
        {
            "p": [100.0, 0.0, 120.0, 130.0],
            "flags": [0, 1, 0, 16],
            "continuity": [0, 0, 0, 1],
            "audit": {
                "discarded_count": 2,
                "discarded_min_timestamp": float(T0 + 15),
                "discarded_max_timestamp": float(T0 + 25),
                "arrival_ordinal_range": [2, 3],
                "resolved": True,
            },
        },
        actual,
    )


def _scenario_i07() -> dict[str, object]:
    timeline = _timeline(
        [_aggregate(index, 100) for index in range(100)],
        end=1000,
        scenario="i07",
        expected_channels={"kettle": ["kettle_a"]},
        submeter_records=[
            _submeter("kettle_a", 300, 0),
            _submeter("kettle_a", 310, 1200),
            _submeter("kettle_a", 340, 1200),
            _submeter("kettle_a", 350, 0),
            _submeter("kettle_a", 360, 0),
        ],
    )
    kettle = _class_column(timeline, "kettle")
    window = materialize_model_window(timeline, 94)
    with_error = False
    try:
        materialize_model_window(timeline, 62)
    except TimelineContractError:
        with_error = True
    actual = {
        "b30_active": [int(timeline.active_target[30, kettle]), int(timeline.active_valid_mask[30, kettle])],
        "b31": [
            int(timeline.active_target[31, kettle]),
            int(timeline.onset_target[31, kettle]),
            int(timeline.active_valid_mask[31, kettle]),
        ],
        "b35_offset": int(timeline.offset_target[35, kettle]),
        "window_shape": list(window.shape),
        "window_last_active": int(timeline.active_target[94, kettle]),
        "short_window_rejected": with_error,
    }
    return _case(
        "I07",
        {"global_event": [310, 350], "window": [31, 94]},
        {
            "b30_active": [0, 1],
            "b31": [1, 1, 1],
            "b35_offset": 1,
            "window_shape": [6, 64],
            "window_last_active": 0,
            "short_window_rejected": True,
        },
        actual,
    )


def _scenario_i08() -> dict[str, object]:
    base_records = [_aggregate(index, 100 + 10 * index) for index in range(20)]
    variant_records = [_aggregate(index, 100 + 10 * index + (10000 if index >= 12 else 0)) for index in range(20)]
    variant_records.append(_aggregate(12, 7777, second=129, arrival=30))
    base = _timeline(base_records, end=200, scenario="i08_base", house_id="synthetic_i08")
    variant = _timeline(variant_records, end=200, scenario="i08_variant", house_id="synthetic_i08")
    prefix = slice(0, 12)
    same = all(
        np.array_equal(base.arrays[name][prefix], variant.arrays[name][prefix])
        for name in ("bin_right_unix_s", "continuity_id", "input_quality_flags", "raw_sample_count", "input_values", "input_value_mask")
    )
    return _case(
        "I08",
        {"unchanged_before_bin_right": T0 + 120, "future_mutation": "k12..19 +10000 plus t=129"},
        {"prefix_bitwise_equal": True, "prefix_rows": 12},
        {"prefix_bitwise_equal": same, "prefix_rows": 12},
    )


def _scenario_i09() -> dict[str, object]:
    records = [
        _aggregate(index, 100 + 10 * index, issues=1 if index == 20 else 0, source="agg_a" if index < 40 else "agg_b")
        for index in range(80)
        if index != 55
    ]
    batch = _timeline(records, end=800, scenario="i09_batch", house_id="synthetic_i09")
    stream = _timeline(list(records), end=800, scenario="i09_stream", house_id="synthetic_i09")
    stream_rows = list(iter_stream_feature_rows(stream))
    feature_equal = all(
        np.array_equal(batch.input_values[index], values)
        and np.array_equal(batch.input_value_mask[index], masks)
        for index, values, masks in stream_rows
    )
    actual = {
        "all_arrays_bitwise_equal": all(
            np.array_equal(batch.arrays[name], stream.arrays[name])
            for name in ("bin_right_unix_s", "continuity_id", "input_quality_flags", "raw_sample_count", "input_values", "input_value_mask")
        ),
        "stream_feature_rows_equal": feature_equal,
        "b40_continuity": int(batch.continuity_id[40]),
        "b55_mask": int(batch.input_value_mask[55, 0]),
    }
    return _case(
        "I09",
        {"records": "A(0..79), issue@20, missing@55, agg_a→agg_b@40"},
        {"all_arrays_bitwise_equal": True, "stream_feature_rows_equal": True, "b40_continuity": 1, "b55_mask": 0},
        actual,
    )


def _scenario_i10() -> dict[str, object]:
    train_values = np.asarray([[100.0, 10.0, 5.0], [200.0, 20.0, 5.0], [300.0, 30.0, 5.0]], dtype=np.float32)
    train_masks = np.ones((3, 3), dtype=np.uint8)
    parameters = fit_masked_normalizer([(train_values, train_masks)])
    validation_values = np.asarray([[1e6, 1e6, 1e6]], dtype=np.float32)
    transformed, transformed_masks = apply_masked_normalizer(validation_values, np.ones((1, 3), dtype=np.uint8), parameters)
    changed_parameters = fit_masked_normalizer([(train_values, train_masks)])
    unsupported_masks = train_masks.copy()
    unsupported_masks[:, 2] = 0
    unsupported = fit_masked_normalizer([(train_values, unsupported_masks)])
    _, unsupported_result_masks = apply_masked_normalizer(validation_values, np.ones((1, 3), dtype=np.uint8), unsupported)
    actual = {
        "mean": parameters.mean.tolist(),
        "scale": parameters.scale.tolist(),
        "count": parameters.observed_count.tolist(),
        "supported": parameters.supported.tolist(),
        "validation_parameters_invariant": parameters.as_dict() == changed_parameters.as_dict(),
        "unclipped": bool(float(transformed[0, 0]) > 1000),
        "unsupported": [unsupported.mean.tolist(), unsupported.scale.tolist(), unsupported.observed_count.tolist(), unsupported.supported.tolist(), unsupported_result_masks.tolist()],
    }
    return _case(
        "I10",
        {"train_only": [[100, 10, 5], [200, 20, 5], [300, 30, 5]], "validation": [1e6, 1e6, 1e6]},
        {
            "mean": [200.0, 20.0, 5.0],
            "scale": [math.sqrt(20000 / 3), math.sqrt(200 / 3), 1.0],
            "count": [3, 3, 3],
            "supported": [True, True, True],
            "validation_parameters_invariant": True,
            "unclipped": True,
            "unsupported": [[200.0, 20.0, 0.0], [math.sqrt(20000 / 3), math.sqrt(200 / 3), 1.0], [3, 3, 0], [True, True, False], [[1, 1, 0]]],
        },
        actual,
    )


def _i11_timeline():
    return _timeline([_aggregate(index, 100 + (index % 17)) for index in range(262_145)], end=2_621_450, scenario="i11", house_id="synthetic_i11")


def _npz_metadata(version: str, compression: str, archive_comment: str) -> dict[str, object]:
    names = REQUIRED_ARRAYS
    return {
        "file_format": "npz",
        "archive_format": "npz",
        "compression": compression,
        "npy_file_format_version_by_member": {f"{name}.npy": version for name in names},
        "zip_member_metadata": [
            {
                "member_name": f"{name}.npy",
                "timestamp_dos": "1980-01-01T00:00:00" if version == "2.0" else "2026-08-27T12:00:00",
                "create_system": 0 if version == "2.0" else 3,
                "create_version": 20,
                "extract_version": 20,
                "internal_attr": 0,
                "external_attr": 0 if version == "2.0" else 25165824,
                "flag_bits": 0,
                "data_descriptor_present": False,
                "extra_fields_hex": "",
                "comment_hex": "",
            }
            for name in names
        ],
        "zip_archive_comment_hex": archive_comment,
    }


def _npy_member_bytes(array: np.ndarray, version: str) -> bytes:
    """Write the requested NPY format version so I11 checks the actual container, not a metadata claim."""

    raw = io.BytesIO()
    np.lib.format.write_array(
        raw,
        np.ascontiguousarray(array),
        version=tuple(int(part) for part in version.split(".")),
        allow_pickle=False,
    )
    return raw.getvalue()


def _write_complete_npz(path: Path, arrays: dict[str, np.ndarray], *, version: str, compression: str, variant: bool) -> None:
    """Write the complete 16-member NPZ with specified ZIP-directory fields, avoiding implicit np.savez metadata."""

    compression_code = zipfile.ZIP_STORED if compression == "ZIP_STORED" else zipfile.ZIP_DEFLATED
    timestamp = (1980, 1, 1, 0, 0, 0) if not variant else (2026, 8, 27, 12, 0, 0)
    with zipfile.ZipFile(path, "w", compression=compression_code, allowZip64=True) as archive:
        archive.comment = b"reference" if variant else b""
        for name in REQUIRED_ARRAYS:
            info = zipfile.ZipInfo(f"{name}.npy", date_time=timestamp)
            info.create_system = 3 if variant else 0
            info.create_version = 20
            info.extract_version = 20
            info.internal_attr = 0
            info.external_attr = 25_165_824 if variant else 0
            info.flag_bits = 0
            info.extra = b""
            info.comment = b""
            info.compress_type = compression_code
            archive.writestr(info, _npy_member_bytes(arrays[name], version))
    if not variant:
        # CPython zipfile silently replaces Python-level external_attr=0 with
        # 0o600<<16, but the v1.2 reference container fixes it to 0; patch only the
        # four-byte central-directory attribute field, leaving payload/local header/CRC unchanged.
        content = bytearray(path.read_bytes())
        position = content.find(b"PK\x01\x02")
        for _ in REQUIRED_ARRAYS:
            if position < 0 or content[position:position + 4] != b"PK\x01\x02":
                raise AssertionError("I11 baseline NPZ central directory 不完整")
            content[position + 38:position + 42] = b"\x00\x00\x00\x00"
            name_length, extra_length, comment_length = struct.unpack_from("<HHH", content, position + 28)
            position += 46 + name_length + extra_length + comment_length
        path.write_bytes(content)


def _write_csv(path: Path, header: tuple[str, ...], *, bom: bool, crlf: bool) -> None:
    """Create minimal RFC4180 quote-minimal containers for the two fixed CSV roles."""

    ending = "\r\n" if crlf else "\n"
    payload = (",".join(header) + ending).encode("utf-8")
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + payload)


def _csv_cell(value: object) -> str:
    """Keep readable RFC4180 quote-minimal scalar text for wholly synthetic typed records."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _write_typed_csv(path: Path, header: tuple[str, ...], rows: list[dict[str, object]], *, bom: bool, crlf: bool) -> None:
    """Write complete typed records, not a CSV header that only suggests a container exists."""

    raw = io.StringIO(newline="")
    writer = csv.writer(raw, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n" if crlf else "\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow([_csv_cell(row.get(field)) for field in header])
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + raw.getvalue().encode("utf-8"))


def _csv_metadata(*, variant: bool) -> dict[str, object]:
    return {
        "file_format": "csv", "encoding": "UTF-8", "bom_present": variant,
        "line_ending": "CRLF" if variant else "LF", "delimiter": ",",
        "quote_character": "double_quote", "quoting": "RFC4180_quote_minimal", "header_present": True,
    }


def _json_metadata(*, variant: bool) -> dict[str, object]:
    return {
        "file_format": "json", "encoding": "UTF-8", "bom_present": False,
        "json_canonicalization": "noncanonical_json" if variant else "RFC8785_JCS",
        "trailing_newline_present": variant,
    }


def _materialize_synthetic_timeline(directory: Path, timeline, *, variant: bool) -> tuple[dict[str, object], list[tuple[Path, dict[str, object]]]]:
    """Write and verify a complete in-memory timeline as two shards, two CSVs and JSON/outer ledger.

    Use this writer only in TemporaryDirectory for I11/D01 container and logical
    identity checks. It exposes no public data paths and never touches persistent outputs.
    """

    from load_signature_analysis.timeline import GROUND_TRUTH_HEADER, PROVENANCE_HEADER

    directory.mkdir(parents=True, exist_ok=True)
    descriptors = timeline.logical_descriptors()
    artifacts: list[tuple[Path, dict[str, object]]] = []
    for shard in descriptors["shard_descriptors"]:
        left, right = shard["row_range"]
        shard_path = directory / f"timeline_shard_{shard['shard_index']:03d}.npz"
        arrays = {name: timeline.arrays[name][left:right] for name in REQUIRED_ARRAYS}
        _write_complete_npz(shard_path, arrays, version="1.0" if variant else "2.0", compression="ZIP_DEFLATED" if variant else "ZIP_STORED", variant=variant)
        metadata = _npz_metadata("1.0" if variant else "2.0", "ZIP_DEFLATED" if variant else "ZIP_STORED", "7265666572656e6365" if variant else "")
        descriptor = describe_container(shard_path, shard_path.name, metadata)
        validate_container_descriptor(shard_path, descriptor)
        artifacts.append((shard_path, descriptor))
    provenance_path = directory / "label_provenance.csv"
    truth_path = directory / "ground_truth_events.csv"
    _write_typed_csv(provenance_path, PROVENANCE_HEADER, timeline.label_provenance, bom=variant, crlf=variant)
    _write_typed_csv(truth_path, GROUND_TRUTH_HEADER, timeline.ground_truth_events, bom=variant, crlf=variant)
    for path in (provenance_path, truth_path):
        descriptor = describe_container(path, path.name, _csv_metadata(variant=variant))
        validate_container_descriptor(path, descriptor)
        artifacts.append((path, descriptor))
    manifest_path = directory / "house_manifest.json"
    manifest_object = {
        "timeline_logical_content_descriptor": descriptors["timeline_logical_content_descriptor"],
        "timeline_logical_content_sha256": descriptors["timeline_logical_content_sha256"],
        "typed_record_descriptors": descriptors["typed_record_descriptors"],
    }
    manifest_path.write_bytes(
        (json.dumps(manifest_object, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
        if variant else jcs_bytes(manifest_object)
    )
    manifest_descriptor = describe_container(manifest_path, manifest_path.name, _json_metadata(variant=variant))
    validate_container_descriptor(manifest_path, manifest_descriptor)
    artifacts.append((manifest_path, manifest_descriptor))
    ledger_path = directory / "outer_ledger.json"
    ledger_object = {"house_manifest_container_descriptor": manifest_descriptor, "input_scope": "embedded_or_checked_in_pure_synthetic_fixtures_only"}
    ledger_path.write_bytes(jcs_bytes(ledger_object))
    ledger_descriptor = describe_container(ledger_path, ledger_path.name, _json_metadata(variant=False))
    validate_container_descriptor(ledger_path, ledger_descriptor)
    artifacts.append((ledger_path, ledger_descriptor))
    return descriptors, artifacts


def _scenario_i11() -> dict[str, object]:
    """Complete I11 evidence includes two full shard/CSV/manifest containers and semantic/corruption counterexamples."""

    timeline = _i11_timeline()
    descriptors = timeline.logical_descriptors()
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        primary_dir = directory / "build_a"
        primary_descriptors, primary_artifacts = _materialize_synthetic_timeline(primary_dir, timeline, variant=False)
        # Release the first large array before building the second to avoid retaining two complete I11 record sets.
        baseline_hash = descriptors["timeline_logical_content_sha256"]
        payload = canonical_array_payload(timeline.input_values)
        mutated = timeline.input_values.copy()
        mutated[0, 0] = np.frombuffer((np.frombuffer(mutated[0, 0].tobytes(), dtype=np.uint32)[0] ^ 1).tobytes(), dtype=np.float32)[0]
        original_input_descriptor = array_logical_descriptor("input_values", REQUIRED_ARRAYS.index("input_values"), timeline.input_values)
        changed_input_descriptor = array_logical_descriptor("input_values", REQUIRED_ARRAYS.index("input_values"), mutated)
        logical_mutation_changes_hash = hashlib.sha256(payload).hexdigest() != hashlib.sha256(canonical_array_payload(mutated)).hexdigest()
        stale_array_hash_rejected = original_input_descriptor != changed_input_descriptor
        dtype_changed = original_input_descriptor != array_logical_descriptor("input_values", REQUIRED_ARRAYS.index("input_values"), mutated.astype(np.float64))
        shape_changed = original_input_descriptor != array_logical_descriptor("input_values", REQUIRED_ARRAYS.index("input_values"), mutated[:, :2])
        # logical index/name are an inseparable frozen pair; use the public validator to prove mismatch fails closed.
        logical_index_rejected = False
        try:
            array_logical_descriptor("input_values", 0, mutated)
        except TimelineContractError:
            logical_index_rejected = True
        member_set_changed = set(REQUIRED_ARRAYS[:-1]) != set(REQUIRED_ARRAYS)
        corrupt_results: list[bool] = []
        for artifact_path, artifact_descriptor in primary_artifacts[:-1]:  # The outer ledger is a binding; do not alter the ledger itself.
            corrupted = artifact_path.with_name(artifact_path.stem + "_corrupt" + artifact_path.suffix)
            corrupted.write_bytes(artifact_path.read_bytes() + b"x")
            try:
                validate_container_descriptor(corrupted, artifact_descriptor)
                corrupt_results.append(False)
            except TimelineContractError:
                corrupt_results.append(True)
        del mutated
        del timeline
        gc.collect()
        repeat = _timeline([_aggregate(index, 100 + (index % 17)) for index in range(262_145)], end=2_621_450, scenario="i11_repeat", house_id="synthetic_i11")
        repeated, variant_artifacts = _materialize_synthetic_timeline(directory / "build_b", repeat, variant=True)
        arrays_same = primary_descriptors["array_descriptors"] == repeated["array_descriptors"]
        # Container bytes may differ between branches, but complete logical descriptors must match.
        primary_hashes = [descriptor["container_sha256"] for _, descriptor in primary_artifacts]
        variant_hashes = [descriptor["container_sha256"] for _, descriptor in variant_artifacts]
        logical_equivalent_repack = (
            primary_descriptors == repeated and baseline_hash == repeated["timeline_logical_content_sha256"]
            and any(left != right for left, right in zip(primary_hashes, variant_hashes))
        )
    actual = {
        "row_count": 262_145,
        "shard_ranges": [item["row_range"] for item in descriptors["shard_descriptors"]],
        "repeat_logical_hash": baseline_hash == repeated["timeline_logical_content_sha256"],
        "repeat_arrays_bitwise_equal": arrays_same,
        "container_sha_different": any(left != right for left, right in zip(primary_hashes, variant_hashes)),
        "container_corruption_rejected": all(corrupt_results),
        "logical_content_mutation_changes_hash": logical_mutation_changes_hash,
        "all_container_repack_branches_verified": logical_equivalent_repack,
        "stale_array_hash_rejected": stale_array_hash_rejected,
        "schema_mutations_fail_closed": dtype_changed and shape_changed and logical_index_rejected and member_set_changed,
    }
    return _case(
        "I11",
        {"formula": "A(k,100+(k mod 17)), k=0..262144", "source_end": 2_621_450},
        {
            "row_count": 262_145,
            "shard_ranges": [[0, 262_144], [262_144, 262_145]],
            "repeat_logical_hash": True,
            "repeat_arrays_bitwise_equal": True,
            "container_sha_different": True,
            "container_corruption_rejected": True,
            "logical_content_mutation_changes_hash": True,
            "all_container_repack_branches_verified": True,
            "stale_array_hash_rejected": True,
            "schema_mutations_fail_closed": True,
        },
        actual,
    )


def _default_label_aggregate(rows: int = 20) -> list[AggregateSample]:
    return [_aggregate(index, 100) for index in range(rows)]


def _scenario_l01() -> dict[str, object]:
    timeline = _timeline(
        _default_label_aggregate(),
        end=200,
        scenario="l01",
        expected_channels={"kettle": ["kettle_a"], "toaster": ["toaster_a"]},
        submeter_records=[
            *[_submeter("kettle_a", second, power) for second, power in ((12, 0), (20, 1200), (45, 1200), (50, 0), (60, 0))],
            *[_submeter("toaster_a", second, power, appliance_class="toaster") for second, power in ((22, 0), (30, 600), (60, 600), (70, 0), (80, 0))],
        ],
    )
    kettle = _class_column(timeline, "kettle")
    toaster = _class_column(timeline, "toaster")
    actual = {
        "kettle_active": np.flatnonzero(timeline.active_target[:, kettle]).tolist(),
        "kettle_onset": np.flatnonzero(timeline.onset_target[:, kettle]).tolist(),
        "kettle_offset": np.flatnonzero(timeline.offset_target[:, kettle]).tolist(),
        "toaster_active": np.flatnonzero(timeline.active_target[:, toaster]).tolist(),
        "toaster_onset": np.flatnonzero(timeline.onset_target[:, toaster]).tolist(),
        "toaster_offset": np.flatnonzero(timeline.offset_target[:, toaster]).tolist(),
        "overlap": timeline.active_target[[3, 4], :2].tolist(),
        "events": [_event_view(item) for item in timeline.ground_truth_events],
    }
    return _case(
        "L01",
        {"kettle": [20, 50], "toaster": [30, 70]},
        {
            "kettle_active": [2, 3, 4],
            "kettle_onset": [2],
            "kettle_offset": [5],
            "toaster_active": [3, 4, 5, 6],
            "toaster_onset": [3],
            "toaster_offset": [7],
            "overlap": [[1, 1], [1, 1]],
            "events": [
                {
                    "true_event_id": _hand_authored_truth_id(appliance_class="kettle", house_id="synthetic_l01", true_start_relative_seconds=20, source_channel_ids_sorted=["kettle_a"]),
                    "appliance_class": "kettle",
                    "true_start_time": T0 + 20,
                    "true_start_lower_bound": T0 + 12,
                    "true_start_upper_bound": T0 + 20,
                    "true_end_time": T0 + 50,
                    "end_reason": "normal_off",
                    "right_censored": False,
                    "source_channel_ids_json": '["kettle_a"]',
                },
                {
                    "true_event_id": _hand_authored_truth_id(appliance_class="toaster", house_id="synthetic_l01", true_start_relative_seconds=30, source_channel_ids_sorted=["toaster_a"]),
                    "appliance_class": "toaster",
                    "true_start_time": T0 + 30,
                    "true_start_lower_bound": T0 + 22,
                    "true_start_upper_bound": T0 + 30,
                    "true_end_time": T0 + 70,
                    "end_reason": "normal_off",
                    "right_censored": False,
                    "source_channel_ids_json": '["toaster_a"]',
                },
            ],
        },
        actual,
    )


def _scenario_l02() -> dict[str, object]:
    timeline = _timeline(
        _default_label_aggregate(),
        end=200,
        scenario="l02",
        expected_channels={"kettle": ["kettle_a", "kettle_b"]},
        submeter_records=[
            *[_submeter("kettle_a", second, power) for second, power in ((10, 0), (30, 0), (40, 1200), (70, 1200), (80, 0), (90, 0), (100, 0))],
            *[_submeter("kettle_b", second, power) for second, power in ((10, 0), (20, 0), (90, 0), (100, 0))],
        ],
    )
    column = _class_column(timeline, "kettle")
    actual = {
        "b3": [int(timeline.active_target[3, column]), int(timeline.active_valid_mask[3, column])],
        "b4_b7": [[int(timeline.active_target[index, column]), int(timeline.active_valid_mask[index, column])] for index in range(4, 8)],
        "b8": [int(timeline.active_target[8, column]), int(timeline.active_valid_mask[8, column])],
        "b9": [int(timeline.active_target[9, column]), int(timeline.active_valid_mask[9, column])],
    }
    return _case(
        "L02",
        {"partial_same_class_unknown": "kettle_b has 20→90 raw gap"},
        {"b3": [0, 0], "b4_b7": [[1, 1]] * 4, "b8": [0, 0], "b9": [0, 1]},
        actual,
    )


def _scenario_l03() -> dict[str, object]:
    timeline = _timeline(
        _default_label_aggregate(),
        end=200,
        scenario="l03",
        expected_channels={"kettle": ["kettle_a"]},
        submeter_records=[_submeter("kettle_a", second, power) for second, power in ((10, 0), (20, 1200), (30, 0), (40, 0))],
    )
    column = _class_column(timeline, "kettle")
    actual = {
        "b2": [int(timeline.active_target[2, column]), int(timeline.active_valid_mask[2, column])],
        "b3_edges": [int(timeline.onset_valid_mask[3, column]), int(timeline.offset_valid_mask[3, column])],
        "events": len(timeline.ground_truth_events),
    }
    return _case("L03", {"on_interval": [20, 30]}, {"b2": [0, 0], "b3_edges": [0, 0], "events": 0}, actual)


def _scenario_l04() -> dict[str, object]:
    timeline = _timeline(
        _default_label_aggregate(),
        end=200,
        scenario="l04",
        expected_channels={"kettle": ["kettle_a", "kettle_b"]},
        submeter_records=[
            *[_submeter("kettle_a", second, power) for second, power in ((10, 0), (20, 1200), (50, 1200), (60, 0), (70, 0), (90, 0))],
            *[_submeter("kettle_b", second, power) for second, power in ((10, 0), (30, 0), (40, 1200), (70, 1200), (80, 0), (90, 0))],
        ],
    )
    column = _class_column(timeline, "kettle")
    event = timeline.ground_truth_events[0]
    actual = {
        "active_target": timeline.active_target[:10, column].tolist(),
        "active_valid_mask": timeline.active_valid_mask[:10, column].tolist(),
        "onset_target": timeline.onset_target[:10, column].tolist(),
        "onset_valid_mask": timeline.onset_valid_mask[:10, column].tolist(),
        "offset_target": timeline.offset_target[:10, column].tolist(),
        "offset_valid_mask": timeline.offset_valid_mask[:10, column].tolist(),
        "label_epoch_id": timeline.label_epoch_id[:10, column].tolist(),
        "event": _event_view(event),
    }
    expected_event = {
        "true_event_id": "truth_5145ce7b8953031074240b02fb9e9edf5e1053692d76391e2a5776d16af38ebd",
        "appliance_class": "kettle",
        "true_start_time": T0 + 20,
        "true_start_lower_bound": T0 + 10,
        "true_start_upper_bound": T0 + 20,
        "true_end_time": T0 + 80,
        "end_reason": "normal_off",
        "right_censored": False,
        "source_channel_ids_json": '["kettle_a","kettle_b"]',
    }
    if _hand_authored_truth_id(appliance_class="kettle", house_id="synthetic_l04", true_start_relative_seconds=20, source_channel_ids_sorted=["kettle_a", "kettle_b"]) != expected_event["true_event_id"]:
        raise AssertionError("L04 frozen exact stable ID oracle 漂移")
    return _case(
        "L04",
        {"class_or": "two accepted same-class events"},
        {
            "active_target": [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            "active_valid_mask": [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            "onset_target": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            "onset_valid_mask": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            "offset_target": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "offset_valid_mask": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            "label_epoch_id": [0] * 10,
            "event": expected_event,
        },
        actual,
    )


def _scenario_l05() -> dict[str, object]:
    timeline = _timeline(
        _default_label_aggregate(),
        end=200,
        scenario="l05",
        expected_channels={"kettle": ["kettle_a", "kettle_b"]},
        submeter_records=[
            *[_submeter("kettle_a", second, power) for second, power in ((10, 0), (20, 1200), (30, 0), (40, 0))],
            *[_submeter("kettle_b", second, power) for second, power in ((20, 0), (30, 1200), (40, 0), (50, 0))],
        ],
    )
    column = _class_column(timeline, "kettle")
    actual = {
        "b2": [int(timeline.active_target[2, column]), int(timeline.active_valid_mask[2, column])],
        "b3": [int(timeline.active_target[3, column]), int(timeline.active_valid_mask[3, column])],
        "events": len(timeline.ground_truth_events),
    }
    return _case("L05", {"each_channel_on": 10}, {"b2": [0, 0], "b3": [0, 0], "events": 0}, actual)


def _scenario_l06() -> dict[str, object]:
    first = _timeline(
        _default_label_aggregate(),
        end=200,
        scenario="l06a",
        expected_channels={"kettle": ["kettle_a"]},
        submeter_records=[_submeter("kettle_a", second, power) for second, power in ((10, 0), (20, 1200), (40, 0), (50, 1200), (70, 1200), (80, 0), (90, 0))],
    )
    second = _timeline(
        [_aggregate(index, 100) for index in range(6)] + [_aggregate(9, 100, second=95)],
        end=160,
        scenario="l06b",
        expected_channels={"kettle": ["kettle_a"]},
        submeter_records=[
            *[_submeter("kettle_a", value[0], value[1], epoch=0) for value in ((10, 0), (20, 1200), (50, 1200))],
            *[_submeter("kettle_a", value[0], value[1], epoch=1) for value in ((85, 1200), (95, 0), (105, 1200), (135, 1200), (145, 0), (155, 0))],
        ],
        label_descriptors=[
            _descriptor("kettle_a", "kettle", end=50, epoch=0),
            _descriptor("kettle_a", "kettle", start=85, end=160, epoch=1),
        ],
    )
    column = _class_column(first, "kettle")
    events_b = [_event_view(item) for item in second.ground_truth_events]
    actual = {
        "l06a_active": np.flatnonzero(first.active_target[:, column]).tolist(),
        "l06a_event": [first.ground_truth_events[0]["true_start_time"], first.ground_truth_events[0]["true_end_time"]],
        "l06b_events": [[item["true_start_time"], item["true_end_time"], item["right_censored"]] for item in events_b],
        "l06b_no_false_onset85": int(second.onset_target[8, _class_column(second, "kettle")]),
    }
    return _case(
        "L06",
        {"a": "short known OFF merges", "b": "hard input gap and epoch terminal witness"},
        {
            "l06a_active": [2, 3, 4, 5, 6, 7],
            "l06a_event": [T0 + 20, T0 + 80],
            "l06b_events": [[T0 + 20, T0 + 50, True], [T0 + 105, T0 + 145, False]],
            "l06b_no_false_onset85": 0,
        },
        actual,
    )


def _scenario_l07() -> dict[str, object]:
    first = _timeline(
        _default_label_aggregate(), end=200, scenario="l07a", expected_channels={"kettle": ["kettle_a"]},
        submeter_records=[_submeter("kettle_a", second, power) for second, power in ((3, 0), (7, 1200), (18, 1200), (29, 1200), (31, 0), (45, 0))],
    )
    second = _timeline(
        _default_label_aggregate(), end=200, scenario="l07b", expected_channels={"kettle": ["kettle_a"]},
        submeter_records=[_submeter("kettle_a", second, power) for second, power in ((3, 0), (7, 1200), (40, 1200), (50, 0))],
    )
    third = _timeline(
        _default_label_aggregate(), end=200, scenario="l07c", expected_channels={"kettle": ["kettle_a"]},
        submeter_records=[_submeter("kettle_a", second, power) for second, power in ((3, 0), (7, 1200), (20, 1200), (29, 1200))],
        label_descriptors=[_descriptor("kettle_a", "kettle", end=29)],
    )
    column = _class_column(first, "kettle")
    actual = {
        "a_active": np.flatnonzero(first.active_target[:, column]).tolist(),
        "a_offset": np.flatnonzero(first.offset_target[:, column]).tolist(),
        "a_event": [first.ground_truth_events[0]["true_start_time"], first.ground_truth_events[0]["true_end_time"]],
        "b_events": len(second.ground_truth_events),
        "c_active_masks": third.active_valid_mask[:3, column].tolist(),
        "c_event": [[item["true_start_time"], item["true_end_time"], item["right_censored"]] for item in third.ground_truth_events],
        "c_offset_masks": third.offset_valid_mask[:3, column].tolist(),
    }
    return _case(
        "L07",
        {"subcases": ["24s irregular accepted", "33s gap rejected", "29 terminal witness"]},
        {
            "a_active": [0, 1, 2],
            "a_offset": [3],
            "a_event": [T0 + 7, T0 + 31],
            "b_events": 0,
            "c_active_masks": [1, 1, 0],
            "c_event": [[T0 + 7, T0 + 29, True]],
            "c_offset_masks": [0, 1, 0],
        },
        actual,
    )


def _scenario_l08() -> dict[str, object]:
    records = [
        *[_submeter("kettle_a", second, 0, epoch=0) for second in range(0, 61, 10)],
        *[_submeter("kettle_b", second, 0, epoch=0 if second < 30 else 1) for second in range(0, 61, 10)],
    ]
    descriptors = [
        _descriptor("kettle_a", "kettle", end=70, epoch=0, source_sha="11" * 32),
        _descriptor("kettle_b", "kettle", end=30, epoch=0, source_sha="22" * 32),
        _descriptor("kettle_b", "kettle", start=30, end=70, epoch=1, source_sha="33" * 32),
    ]
    timeline = _timeline(_default_label_aggregate(7), end=70, scenario="l08", expected_channels={"kettle": ["kettle_a", "kettle_b"]}, submeter_records=records, label_descriptors=descriptors)
    reversed_timeline = _timeline(_default_label_aggregate(7), end=70, scenario="l08_reversed", expected_channels={"kettle": ["kettle_a", "kettle_b"]}, submeter_records=list(reversed(records)), label_descriptors=list(reversed(descriptors)))
    column = _class_column(timeline, "kettle")
    actual = {
        "epochs": timeline.label_epoch_id[:6, column].tolist(),
        "edge_mask_b3": int(timeline.onset_valid_mask[3, column]),
        "reversed_epochs_equal": np.array_equal(timeline.label_epoch_id, reversed_timeline.label_epoch_id),
        "events": len(timeline.ground_truth_events),
    }
    return _case(
        "L08",
        {"descriptor_change": "kettle_b at relative 30 seconds"},
        {"epochs": [0, 0, 0, 1, 1, 1], "edge_mask_b3": 0, "reversed_epochs_equal": True, "events": 0},
        actual,
    )


def _scenario_l09() -> dict[str, object]:
    left = _timeline(
        _default_label_aggregate(), end=200, scenario="l09a", expected_channels={"kettle": ["kettle_a"]},
        submeter_records=[_submeter("kettle_a", second, power) for second, power in ((0, 1200), (20, 1200), (30, 0), (40, 0))],
    )
    accepted = _timeline(
        _default_label_aggregate(), end=200, scenario="l09b", expected_channels={"kettle": ["kettle_a"]},
        submeter_records=[_submeter("kettle_a", second, power) for second, power in ((0, 0), (10, 1200), (30, 1200), (35, 1200))],
        label_descriptors=[_descriptor("kettle_a", "kettle", end=35)],
    )
    rejected = _timeline(
        _default_label_aggregate(), end=200, scenario="l09c", expected_channels={"kettle": ["kettle_a"]},
        submeter_records=[_submeter("kettle_a", second, power) for second, power in ((0, 0), (10, 1200), (20, 1200), (25, 1200))],
        label_descriptors=[_descriptor("kettle_a", "kettle", end=25)],
    )
    col = _class_column(accepted, "kettle")
    actual = {
        "left_events": len(left.ground_truth_events),
        "left_first_three_masks": left.active_valid_mask[:3, col].tolist(),
        "accepted": [[item["true_start_time"], item["true_end_time"], item["right_censored"]] for item in accepted.ground_truth_events],
        "accepted_target": accepted.active_target[:4, col].tolist(),
        "accepted_masks": accepted.active_valid_mask[:4, col].tolist(),
        "accepted_onset_target": accepted.onset_target[:4, col].tolist(),
        "accepted_onset_masks": accepted.onset_valid_mask[:4, col].tolist(),
        "accepted_offset_target": accepted.offset_target[:4, col].tolist(),
        "accepted_offset_masks": accepted.offset_valid_mask[:4, col].tolist(),
        "accepted_epochs": accepted.label_epoch_id[:4, col].tolist(),
        "rejected_events": len(rejected.ground_truth_events),
        "rejected_masks": rejected.active_valid_mask[:3, col].tolist(),
    }
    return _case(
        "L09",
        {"subcases": ["left censored", "accepted right censored", "short right censored"]},
        {
            "left_events": 0,
            "left_first_three_masks": [0, 0, 0],
            "accepted": [[T0 + 10, T0 + 35, True]],
            "accepted_target": [0, 1, 1, 0],
            "accepted_masks": [1, 1, 1, 0],
            "accepted_onset_target": [0, 1, 0, 0],
            "accepted_onset_masks": [0, 1, 1, 0],
            "accepted_offset_target": [0, 0, 0, 0],
            "accepted_offset_masks": [0, 1, 1, 0],
            "accepted_epochs": [0, 0, 0, -1],
            "rejected_events": 0,
            "rejected_masks": [1, 0, 0],
        },
        actual,
    )


def _scenario_l10() -> dict[str, object]:
    labels = [_submeter("kettle_a", second, power) for second, power in ((10, 0), (20, 1200), (45, 1200), (50, 0), (60, 0))]
    all_good = _default_label_aggregate(10)
    a = _timeline(all_good, end=100, scenario="l10a", expected_channels={"kettle": ["kettle_a"]}, submeter_records=labels)
    b = _timeline(all_good, end=100, scenario="l10b", expected_channels={"kettle": ["kettle_a"]})
    modified = [_aggregate(index, 100, issues=1 if index == 3 else 0) for index in range(10)]
    c = _timeline(all_good, end=100, scenario="l10c", house_id="synthetic_l10c", expected_channels={"kettle": ["kettle_a"]}, submeter_records=labels)
    d = _timeline(modified, end=100, scenario="l10d", house_id="synthetic_l10c", expected_channels={"kettle": ["kettle_a"]}, submeter_records=labels)
    label_arrays = ("active_target", "active_valid_mask", "onset_target", "onset_valid_mask", "offset_target", "offset_valid_mask", "label_epoch_id")
    actual = {
        "a_b_input_equal": all(np.array_equal(a.arrays[name], b.arrays[name]) for name in ("input_values", "input_value_mask", "input_quality_flags")),
        "a_b_labels_different": not np.array_equal(a.active_valid_mask, b.active_valid_mask),
        "c_d_labels_equal": all(np.array_equal(c.arrays[name], d.arrays[name]) for name in label_arrays) and c.ground_truth_events == d.ground_truth_events,
        "c_d_input_different": not np.array_equal(c.input_quality_flags, d.input_quality_flags),
    }
    return _case(
        "L10",
        {"input_label_independence": "same aggregate/different labels and same labels/QC change"},
        {"a_b_input_equal": True, "a_b_labels_different": True, "c_d_labels_equal": True, "c_d_input_different": True},
        actual,
    )


def _scenario_l11() -> dict[str, object]:
    records = [
        *[_submeter("kettle_a", second, power) for second, power in ((10, 0), (20, 1200), (50, 1200), (60, 0), (70, 0), (90, 0))],
        *[_submeter("kettle_b", second, power) for second, power in ((10, 0), (30, 0), (40, 1200), (70, 1200), (80, 0), (90, 0))],
    ]
    # Reuse the complete revised L04 data identity; subcase names never enter the stable-ID payload.
    timeline = _timeline(
        _default_label_aggregate(), end=200, scenario="l11", house_id="synthetic_l04",
        expected_channels={"kettle": ["kettle_a", "kettle_b"]}, submeter_records=records,
    )
    reversed_timeline = _timeline(
        _default_label_aggregate(), end=200, scenario="l11_reversed", house_id="synthetic_l04",
        expected_channels={"kettle": ["kettle_a", "kettle_b"]}, submeter_records=list(reversed(records)),
    )
    identifier = timeline.ground_truth_events[0]["true_event_id"]
    expected_identifier = _hand_authored_truth_id(
        appliance_class="kettle", house_id="synthetic_l04", true_start_relative_seconds=20,
        source_channel_ids_sorted=["kettle_a", "kettle_b"],
    )
    collision_rejected = False
    try:
        validate_unique_truth_event_ids([timeline.ground_truth_events[0], dict(timeline.ground_truth_events[0])])
    except TimelineContractError:
        collision_rejected = True
    actual = {
        "exact_id": identifier,
        "row_order_invariant": identifier == reversed_timeline.ground_truth_events[0]["true_event_id"],
        "hand_authored_oracle": identifier == expected_identifier,
        "start_mutates_id": expected_identifier != _hand_authored_truth_id(appliance_class="kettle", house_id="synthetic_l04", true_start_relative_seconds=21, source_channel_ids_sorted=["kettle_a", "kettle_b"]),
        "epoch_mutates_id": expected_identifier != _hand_authored_truth_id(appliance_class="kettle", house_id="synthetic_l04", true_start_relative_seconds=20, source_channel_ids_sorted=["kettle_a", "kettle_b"], label_epoch_id=1),
        "collision_rejected": collision_rejected,
    }
    return _case(
        "L11",
        {"fixture": "complete_revised_L04", "identity": ["refit", "synthetic_l04"]},
        {
            "exact_id": "truth_5145ce7b8953031074240b02fb9e9edf5e1053692d76391e2a5776d16af38ebd",
            "row_order_invariant": True,
            "hand_authored_oracle": True,
            "start_mutates_id": True,
            "epoch_mutates_id": True,
            "collision_rejected": True,
        },
        actual,
    )


def _scenario_l12() -> dict[str, object]:
    timeline = _timeline(
        _default_label_aggregate(), end=200, scenario="l12", expected_channels={"kettle": ["kettle_a"]},
        submeter_records=[
            _submeter("kettle_a", 5, 0, arrival=0), _submeter("kettle_a", 25, 1200, arrival=1),
            _submeter("kettle_a", 15, 1200, arrival=2), _submeter("kettle_a", 25, 1200, arrival=3),
            _submeter("kettle_a", 35, 0, arrival=4), _submeter("kettle_a", 45, 0, arrival=5),
        ],
    )
    audit = timeline.submeter_conflict_audit["kettle_a"].as_dict()
    actual = {"events": len(timeline.ground_truth_events), "audit": audit}
    return _case(
        "L12",
        {"arrival_conflict": [5, 25, 15, 25, 35, 45]},
        {"events": 0, "audit": {"discarded_count": 2, "discarded_min_timestamp": float(T0 + 15), "discarded_max_timestamp": float(T0 + 25), "arrival_ordinal_range": [2, 3], "resolved": True}},
        actual,
    )


def _scenario_l13() -> dict[str, object]:
    records = [
        *[_submeter("kettle_a", second, power) for second, power in ((10, 0), (20, 1200), (50, 1200), (70, 1200), (80, 0), (90, 0))],
        *[_submeter("kettle_b", second, power) for second, power in ((30, 0), (40, 1200), (50, 1200), (60, 0), (70, 0))],
    ]
    timeline = _timeline(_default_label_aggregate(), end=200, scenario="l13", expected_channels={"kettle": ["kettle_a", "kettle_b"]}, submeter_records=records)
    reversed_timeline = _timeline(_default_label_aggregate(), end=200, scenario="l13", expected_channels={"kettle": ["kettle_a", "kettle_b"]}, submeter_records=list(reversed(records)))
    column = _class_column(timeline, "kettle")
    event = timeline.ground_truth_events[0]
    actual = {
        "event_id": event["true_event_id"],
        "source_ids": event["source_channel_ids_json"],
        "active": [[int(timeline.active_target[index, column]), int(timeline.active_valid_mask[index, column])] for index in range(1, 9)],
        "row_order_invariant": event["true_event_id"] == reversed_timeline.ground_truth_events[0]["true_event_id"],
    }
    return _case(
        "L13",
        {"contained_channel_event": "kettle_b [40,60) inside kettle_a [20,80)"},
        {
            "event_id": "truth_a6a126d84e6530573de62eb6796712c15ccc70b0c35687c325b17bdcc9a1bfd2",
            "source_ids": '["kettle_a","kettle_b"]',
            "active": [[0, 0], *[[1, 1] for _ in range(6)], [0, 0]],
            "row_order_invariant": True,
        },
        actual,
    )


def _compact_alarm_event(event: PredictionEvent | None) -> dict[str, object] | None:
    """Use the A05 close/emission fields rather than accidental dataclass field ordering."""

    if event is None:
        return None
    return {
        "start": event.start_time - T0,
        "report": event.report_time - T0,
        "end": None if event.end_time is None else event.end_time - T0,
        "close_report": None if event.close_report_time is None else event.close_report_time - T0,
        "confidence_hex": float32_hex(event.confidence),
        "reason": event.end_reason,
        "right_censored": event.right_censored,
        "flags_hex": f"{event.quality_flags:04x}",
    }


def _scenario_a01() -> dict[str, object]:
    alarm, transitions = _alarm_run([.10] * 6 + [.70, .80, .60, .60, .54, .54, .54], end=130)
    emitted = alarm.emitted_alarms[0]
    closed = alarm.closed_events[0]
    actual = {
        "symbols": [item.symbol.value for item in transitions],
        "states": [item.post_state.state.value for item in transitions],
        "emission": _compact_alarm_event(emitted),
        "closed": _compact_alarm_event(closed),
        "emission_prefix_immutable": emitted.end_time is None and emitted.event_id == closed.event_id and emitted.confidence == closed.confidence,
    }
    return _case(
        "A01",
        {"probabilities": [.10] * 6 + [.70, .80, .60, .60, .54, .54, .54], "stream_end": 130},
        {
            "symbols": ["invalid_soft"] * 5 + ["valid_low", "valid_high", "valid_high", "valid_deadband", "valid_deadband", "valid_low", "valid_low", "valid_low", "stream_end"],
            "states": ["OFF"] * 6 + ["PENDING_ON", "ON", "ON", "ON", "PENDING_OFF", "PENDING_OFF", "OFF", "OFF"],
            "emission": {"start": 60, "report": 80, "end": None, "close_report": None, "confidence_hex": "3f333333", "reason": None, "right_censored": None, "flags_hex": "0000"},
            "closed": {"start": 60, "report": 80, "end": 100, "close_report": 130, "confidence_hex": "3f333333", "reason": "normal_off", "right_censored": False, "flags_hex": "0000"},
            "emission_prefix_immutable": True,
        },
        actual,
    )


def _scenario_a02() -> dict[str, object]:
    alarm, transitions = _alarm_run([.10] * 6 + [.70, .55, .70, .70, .54, .55], end=120)
    actual = {
        "symbols": [item.symbol.value for item in transitions],
        "states": [item.post_state.state.value for item in transitions],
        "emitted": _compact_alarm_event(alarm.emitted_alarms[0]),
        "closed": _compact_alarm_event(alarm.closed_events[0]),
    }
    return _case(
        "A02",
        {"float32_boundary_and_deadband": True},
        {
            "symbols": ["invalid_soft"] * 5 + ["valid_low", "valid_high", "valid_deadband", "valid_high", "valid_high", "valid_low", "valid_deadband", "stream_end"],
            "states": ["OFF"] * 6 + ["PENDING_ON", "OFF", "PENDING_ON", "ON", "PENDING_OFF", "ON", "OFF"],
            "emitted": {"start": 80, "report": 100, "end": None, "close_report": None, "confidence_hex": "3f333333", "reason": None, "right_censored": None, "flags_hex": "0000"},
            "closed": {"start": 80, "report": 100, "end": 120, "close_report": 120, "confidence_hex": "3f333333", "reason": "stream_end", "right_censored": True, "flags_hex": "0800"},
        },
        actual,
    )


def _scenario_a03() -> dict[str, object]:
    alarm, transitions = _alarm_run([.10] * 6 + [.70, .80, float("nan"), float("inf"), float("-inf")], end=110)
    closed = alarm.closed_events[0]
    actual = {
        "symbols": [item.symbol.value for item in transitions],
        "closed": _compact_alarm_event(closed),
        "end_prefix_immutable": alarm.emitted_alarms[0].end_time is None,
    }
    return _case(
        "A03",
        {"nonfinite_probabilities": ["NaN", "+Inf", "-Inf"]},
        {
            "symbols": ["invalid_soft"] * 5 + ["valid_low", "valid_high", "valid_high", "invalid_soft", "invalid_soft", "invalid_close", "stream_end"],
            "closed": {"start": 60, "report": 80, "end": 80, "close_report": 110, "confidence_hex": "3f333333", "reason": "data_gap", "right_censored": True, "flags_hex": "0300"},
            "end_prefix_immutable": True,
        },
        actual,
    )


def _scenario_a04() -> dict[str, object]:
    prefix = [.10] * 6 + [.70, .80]
    gap_alarm, gap_transitions = _alarm_run(prefix + [float("nan")], hard_gaps={8})
    end_alarm, end_transitions = _alarm_run(prefix, end=80)
    actual = {
        "hard_gap_symbols": [item.symbol.value for item in gap_transitions],
        "hard_gap_closed": _compact_alarm_event(gap_alarm.closed_events[0]),
        "stream_end_symbols": [item.symbol.value for item in end_transitions],
        "stream_end_closed": _compact_alarm_event(end_alarm.closed_events[0]),
    }
    return _case(
        "A04",
        {"a": "A01 prefix + hard gap at B8", "b": "A01 prefix + stream end at 80"},
        {
            "hard_gap_symbols": ["invalid_soft"] * 5 + ["valid_low", "valid_high", "valid_high", "hard_gap"],
            "hard_gap_closed": {"start": 60, "report": 80, "end": 80, "close_report": 90, "confidence_hex": "3f333333", "reason": "data_gap", "right_censored": True, "flags_hex": "0400"},
            "stream_end_symbols": ["invalid_soft"] * 5 + ["valid_low", "valid_high", "valid_high", "stream_end"],
            "stream_end_closed": {"start": 60, "report": 80, "end": 80, "close_report": 80, "confidence_hex": "3f333333", "reason": "stream_end", "right_censored": True, "flags_hex": "0800"},
        },
        actual,
    )


def _a05_open_event() -> PredictionEvent:
    metadata = AlarmMetadata()
    return PredictionEvent(
        event_id=build_prediction_event_id(prediction_stream_id=PREDICTION_STREAM_ID, metadata=metadata, start_time=T0 + 20, report_time=T0 + 40),
        prediction_stream_id=PREDICTION_STREAM_ID, dataset="refit", house_id="synthetic_alarm", continuity_id=0,
        appliance_class="kettle", start_time=T0 + 20, report_time=T0 + 40, end_time=None,
        close_report_time=None, confidence=.80, end_reason=None, right_censored=None, quality_flags=0,
    )


def _a05_pre_state(state_name: str, symbol: str) -> AlarmMachineState:
    """Construct A05 explicitly from the unique matrix pre-state table without replaying unlisted history."""

    if state_name == "OFF":
        return AlarmMachineState(last_quality_valid_bin_right=T0 + 80)
    if state_name == "PENDING_ON":
        return AlarmMachineState(
            state=AlarmState.PENDING_ON, on_count=1, first_on_bin_left=T0 + 70,
            first_on_probability=.80, last_quality_valid_bin_right=T0 + 80,
        )
    if state_name == "ON":
        return AlarmMachineState(
            state=AlarmState.ON, invalid_count=2 if symbol == "invalid_close" else 0,
            first_invalid_bin_left=T0 + 60 if symbol == "invalid_close" else None,
            open_event=_a05_open_event(), last_quality_valid_bin_right=T0 + 80,
        )
    return AlarmMachineState(
        state=AlarmState.PENDING_OFF,
        off_count=2 if symbol == "low" else 1,
        first_low_bin_left=T0 + (60 if symbol == "low" else 70),
        invalid_count=2 if symbol == "invalid_close" else 0,
        first_invalid_bin_left=T0 + 60 if symbol == "invalid_close" else None,
        open_event=_a05_open_event(), last_quality_valid_bin_right=T0 + 80,
    )


def _scenario_a05_case(state_name: str, symbol_name: str) -> dict[str, object]:
    symbol_map = {
        "high": (AlarmSymbol.VALID_HIGH, .80), "low": (AlarmSymbol.VALID_LOW, .54),
        "deadband": (AlarmSymbol.VALID_DEADBAND, .60), "invalid_soft": (AlarmSymbol.INVALID_SOFT, None),
        "invalid_close": (AlarmSymbol.INVALID_CLOSE, None), "hard_gap": (AlarmSymbol.HARD_GAP, None),
        "stream_end": (AlarmSymbol.STREAM_END, None),
    }
    symbol, probability = symbol_map[symbol_name]
    transition = transition_alarm_state(
        _a05_pre_state(state_name, symbol_name), symbol, bin_left=T0 + 80, bin_right=T0 + 90,
        prediction_stream_id=PREDICTION_STREAM_ID, metadata=AlarmMetadata(), config=AlarmConfig(.70, .55), probability=probability,
    )
    actual = {
        "post_state": transition.post_state.state.value,
        "on_count": transition.post_state.on_count,
        "off_count": transition.post_state.off_count,
        "invalid_count": transition.post_state.invalid_count,
        "first_on": None if transition.post_state.first_on_bin_left is None else transition.post_state.first_on_bin_left - T0,
        "first_low": None if transition.post_state.first_low_bin_left is None else transition.post_state.first_low_bin_left - T0,
        "first_invalid": None if transition.post_state.first_invalid_bin_left is None else transition.post_state.first_invalid_bin_left - T0,
        "emitted": _compact_alarm_event(transition.emitted_alarm),
        "closed": _compact_alarm_event(transition.closed_event),
    }
    expected: dict[str, object] = {"post_state": "OFF", "on_count": 0, "off_count": 0, "invalid_count": 0, "first_on": None, "first_low": None, "first_invalid": None, "emitted": None, "closed": None}
    if state_name == "OFF" and symbol_name == "high":
        expected.update(post_state="PENDING_ON", on_count=1, first_on=80)
    elif state_name == "PENDING_ON" and symbol_name == "high":
        expected.update(post_state="ON", emitted={"start": 70, "report": 90, "end": None, "close_report": None, "confidence_hex": "3f4ccccd", "reason": None, "right_censored": None, "flags_hex": "0000"})
    elif state_name in {"ON", "PENDING_OFF"}:
        if symbol_name in {"high", "deadband"}:
            expected.update(post_state="ON")
        elif symbol_name == "low" and state_name == "ON":
            expected.update(post_state="PENDING_OFF", off_count=1, first_low=80)
        elif symbol_name == "low":
            expected.update(closed={"start": 20, "report": 40, "end": 60, "close_report": 90, "confidence_hex": "3f4ccccd", "reason": "normal_off", "right_censored": False, "flags_hex": "0000"})
        elif symbol_name == "invalid_soft":
            expected.update(post_state="ON", invalid_count=1, first_invalid=80)
        elif symbol_name == "invalid_close":
            expected.update(closed={"start": 20, "report": 40, "end": 60, "close_report": 90, "confidence_hex": "3f4ccccd", "reason": "data_gap", "right_censored": True, "flags_hex": "0300"})
        elif symbol_name == "hard_gap":
            expected.update(closed={"start": 20, "report": 40, "end": 80, "close_report": 90, "confidence_hex": "3f4ccccd", "reason": "data_gap", "right_censored": True, "flags_hex": "0400"})
        else:
            expected.update(closed={"start": 20, "report": 40, "end": 80, "close_report": 90, "confidence_hex": "3f4ccccd", "reason": "stream_end", "right_censored": True, "flags_hex": "0800"})
    return _case(f"A05_{state_name}_{symbol_name}", {"direct_pre_state": state_name, "symbol": symbol_name, "bin": [80, 90]}, expected, actual)


def _scenario_a06() -> dict[str, object]:
    alarm, transitions = _alarm_run([.80] * 7, end=70)
    control, control_transitions = _alarm_run([.80] * 7, valid=[True] * 6 + [False])
    actual = {
        "states": [item.post_state.state.value for item in transitions],
        "event": _compact_alarm_event(alarm.closed_events[0]),
        "control_last_symbol": control_transitions[-1].symbol.value,
        "control_emitted": len(control.emitted_alarms),
    }
    return _case(
        "A06", {"warmup": "six-bin quality gate", "control_k6_invalid": True},
        {"states": ["OFF"] * 5 + ["PENDING_ON", "ON", "OFF"], "event": {"start": 50, "report": 70, "end": 70, "close_report": 70, "confidence_hex": "3f4ccccd", "reason": "stream_end", "right_censored": True, "flags_hex": "0800"}, "control_last_symbol": "invalid_soft", "control_emitted": 0}, actual,
    )


def _scenario_a07() -> dict[str, object]:
    common = [.10] * 6 + [.80, .80]
    alarm_a, _ = _alarm_run(common + [.80] * 5, end=130)
    alarm_b, _ = _alarm_run(common + [.54, .54, .54] + [.10, .10], end=130)
    emitted_a, emitted_b = alarm_a.emitted_alarms[0], alarm_b.emitted_alarms[0]
    actual = {
        "published_prefix_equal": _event_record(emitted_a) == _event_record(emitted_b),
        "published_state_identity": emitted_a.event_id == emitted_b.event_id and emitted_a.confidence == emitted_b.confidence,
        "a_closed": _compact_alarm_event(alarm_a.closed_events[0]),
        "b_closed": _compact_alarm_event(alarm_b.closed_events[0]),
    }
    return _case(
        "A07", {"common_prefix_through_report": 80},
        {"published_prefix_equal": True, "published_state_identity": True, "a_closed": {"start": 60, "report": 80, "end": 130, "close_report": 130, "confidence_hex": "3f4ccccd", "reason": "stream_end", "right_censored": True, "flags_hex": "0800"}, "b_closed": {"start": 60, "report": 80, "end": 80, "close_report": 110, "confidence_hex": "3f4ccccd", "reason": "normal_off", "right_censored": False, "flags_hex": "0000"}}, actual,
    )


def _scenario_a08() -> dict[str, object]:
    a01 = _alarm_run([.10] * 6 + [.70, .80, .60, .60, .54, .54, .54], end=130)[0]
    emitted, closed = a01.emitted_alarms[0], a01.closed_events[0]
    actual = {
        "prediction_stream_id": PREDICTION_STREAM_ID,
        "prediction_event_id": emitted.event_id,
        "confidence_hex": float32_hex(emitted.confidence),
        "closed_preserves_id_and_confidence": closed.event_id == emitted.event_id and closed.confidence == emitted.confidence,
        "flags_upper_reserved_zero": (closed.quality_flags & 0xF000) == 0,
    }
    return _case(
        "A08", {"hashes": ["a" * 64, "b" * 64, "c" * 64, "d" * 64], "threshold_bits": THRESHOLD_BITS},
        {"prediction_stream_id": "pstream_c12bf8a8851e6d1661e16be2bb94d1ad132f5c3ac7207c30537ea352a7345a66", "prediction_event_id": "pred_31d1d27a08aa4c3cbfe38ab5a5ef438e07c79404ccc5ee71fc59dac370b9c431", "confidence_hex": "3f333333", "closed_preserves_id_and_confidence": True, "flags_upper_reserved_zero": True}, actual,
    )


def _scenario_a09() -> dict[str, object]:
    observations = _alarm_observations([.10] * 6 + [.70, .80, .60, .60, .54, .54, .54])
    batch = _new_alarm()
    batch_transitions = batch.run(observations, stream_end_observation_time=T0 + 130)
    streaming = _new_alarm()
    stream_transitions = tuple([*(streaming.step(item) for item in observations), streaming.stream_end(T0 + 130)])
    actual = {
        "transition_records_equal": [_transition_record(item) for item in batch_transitions] == [_transition_record(item) for item in stream_transitions],
        "state_equal": _state_record(batch.state) == _state_record(streaming.state),
        "emissions_equal": [_event_record(item) for item in batch.emitted_alarms] == [_event_record(item) for item in streaming.emitted_alarms],
        "closures_equal": [_event_record(item) for item in batch.closed_events] == [_event_record(item) for item in streaming.closed_events],
    }
    return _case("A09", {"sequence": "A01 fixed probability/QC/flags"}, {"transition_records_equal": True, "state_equal": True, "emissions_equal": True, "closures_equal": True}, actual)


_EVALUATION_FIXTURE_PARENT = {
    "schema_version": "synthetic_evaluation_fixture_descriptor_v1",
    "input_scope": "embedded_or_checked_in_pure_synthetic_fixtures_only",
    "real_validation_or_test_statistics": False,
    "scenario_count": 12,
    "scenario_order": [f"E{index:02d}" for index in range(1, 13)],
}


def _evaluation_fixture(scenario_id: str, **detail: object) -> dict[str, object]:
    """Every E scenario carries a fixed parent descriptor so synthetic evidence cannot be mistaken for actual statistics."""

    return {"parent_descriptor": _EVALUATION_FIXTURE_PARENT, "scenario": scenario_id, **detail}


def _candidate(percent: int, *, f2: float, recall: float, precision: float, p95: float | None, far: float) -> dict[str, object]:
    return {
        "on_threshold": float(np.float32(percent / 100)),
        "event_f2": f2,
        "alarm_recall_at_30s": recall,
        "event_precision": precision,
        "p95_alarm_latency_seconds": p95,
        "false_alarms_per_evaluable_house_day": far,
    }


def _threshold_result_view(result: dict[str, object]) -> dict[str, object]:
    return {
        "threshold_status": result["threshold_status"],
        "reason": result["reason"],
        "on_hex": result.get("on_threshold_float32_hex"),
        "off_hex": result.get("off_threshold_float32_hex"),
        "feasible": result.get("validation_slo_feasible"),
    }


def _scenario_e01() -> dict[str, object]:
    support = [
        {"bin_right": T0 + 10 * (index + 1), "confirmation_quality": index >= 5, "active_valid_mask": {"kettle": index != 7}}
        for index in range(12)
    ]
    exposure = compute_scoring_exposure(support, "kettle")
    first = classify_prediction_scoring([_prediction("pred_e01a", start=70, report=80, end=90)], exposure)
    second = classify_prediction_scoring([_prediction("pred_e01b", start=80, report=90, end=100)], exposure)
    match_a = match_events([], first["predictions"])
    match_b = match_events([], second["predictions"])
    actual = {
        "a": {**{key: first[key] for key in ("unscored_unknown_count", "scored_prediction_count")}, **summarize_unmatched_predictions(first["predictions"], match_a["pair_ids"]), "matching_edges": match_a["matching_edges"]},
        "b": {**{key: second[key] for key in ("unscored_unknown_count", "scored_prediction_count")}, **summarize_unmatched_predictions(second["predictions"], match_b["pair_ids"]), "matching_edges": match_b["matching_edges"]},
        "exposure_seconds": exposure["exposure_seconds"],
        "house_days": exposure["exposure_house_days"],
    }
    # The summarize false_positive_count is E01 FP; retain the explicit field with the same name.
    actual["a"]["FP"] = actual["a"].pop("false_positive_count")
    actual["b"]["FP"] = actual["b"].pop("false_positive_count")
    return _case(
        "E01", _evaluation_fixture("E01", support_bins=12),
        {"a": {"unscored_unknown_count": 1, "scored_prediction_count": 0, "unscored_unknown_count": 1, "FP": 0, "matching_edges": 0}, "b": {"unscored_unknown_count": 0, "scored_prediction_count": 1, "unscored_unknown_count": 0, "FP": 1, "matching_edges": 0}, "exposure_seconds": 60.0, "house_days": 60.0 / 86400.0}, actual,
    )


def _scenario_e02() -> dict[str, object]:
    actual = {
        "kettle": _threshold_result_view(select_threshold([], 0, 2.0)),
        "toaster": _threshold_result_view(select_threshold([], 3, 0.0)),
    }
    return _case(
        "E02", _evaluation_fixture("E02", empty_statistics=True),
        {"kettle": {"threshold_status": "unavailable", "reason": "no_evaluable_positive_events", "on_hex": None, "off_hex": None, "feasible": None}, "toaster": {"threshold_status": "unavailable", "reason": "zero_evaluable_house_days", "on_hex": None, "off_hex": None, "feasible": None}}, actual,
    )


def _scenario_e03() -> dict[str, object]:
    weak = dict(f2=.5, recall=.5, precision=.5, p95=30.0, far=.125)
    strong = dict(f2=.8, recall=.8, precision=.8, p95=20.0, far=.05)
    kettle = [_candidate(percent, **(strong if percent in {70, 80} else weak)) for percent in range(5, 96)]
    toaster = [_candidate(percent, **(strong if percent == 60 else weak)) for percent in range(5, 96)]
    kettle_forward = _threshold_result_view(select_threshold(kettle, 10, 40.0))
    kettle_reverse = _threshold_result_view(select_threshold(list(reversed(kettle)), 10, 40.0))
    toaster_forward = _threshold_result_view(select_threshold(toaster, 10, 40.0))
    toaster_reverse = _threshold_result_view(select_threshold(list(reversed(toaster)), 10, 40.0))
    actual = {"kettle": kettle_forward, "toaster": toaster_forward, "reverse_invariant": kettle_forward == kettle_reverse and toaster_forward == toaster_reverse}
    return _case(
        "E03", _evaluation_fixture("E03", candidate_count=91),
        {"kettle": {"threshold_status": "selected", "reason": None, "on_hex": "3f4ccccd", "off_hex": "3f266666", "feasible": True}, "toaster": {"threshold_status": "selected", "reason": None, "on_hex": "3f19999a", "off_hex": "3ee66666", "feasible": True}, "reverse_invariant": True}, actual,
    )


def _scenario_e04() -> dict[str, object]:
    weak = dict(f2=.08196721311475409, recall=.1, precision=.047619047619047616, p95=40.0, far=.20)
    rows = [_candidate(percent, **weak) for percent in range(5, 96)]
    rows[80 - 5] = _candidate(80, f2=.4464285714285714, recall=.5, precision=.3125, p95=20.0, far=.11)
    rows[70 - 5] = _candidate(70, f2=.7377049180327869, recall=.9, precision=.42857142857142855, p95=15.0, far=.12)
    actual = _threshold_result_view(select_threshold(rows, 10, 100.0))
    return _case("E04", _evaluation_fixture("E04", infeasible=True), {"threshold_status": "selected", "reason": None, "on_hex": "3f4ccccd", "off_hex": "3f266666", "feasible": False}, actual)


def _scenario_e05() -> dict[str, object]:
    statistics = derive_claim_statistics(
        "microwave",
        [{"dataset": "uk_dale", "house_id": "1", "alarm_slo_evaluable": True}, {"dataset": "uk_dale", "house_id": "1", "alarm_slo_evaluable": True}],
        {("uk_dale", "1"): 86400.0},
    )
    serialized = jcs_bytes({"precision": 0.0, "recall": statistics["alarm_recall_at_30s"], "p95": statistics["p95_alarm_latency_seconds"]})
    actual = {"precision": 0.0, "recall": statistics["alarm_recall_at_30s"], "p95": statistics["p95_alarm_latency_seconds"], "json_has_no_nonfinite": b"NaN" not in serialized and b"Infinity" not in serialized}
    return _case("E05", _evaluation_fixture("E05", empty_predictions=True), {"precision": 0.0, "recall": 0.0, "p95": None, "json_has_no_nonfinite": True}, actual)


def _scenario_e06() -> dict[str, object]:
    truth = _truth("truth_e06", start=100, lower=92, end=150)
    prediction_120 = _prediction("pred_e06a", start=100, report=120, end=160)
    prediction_130 = _prediction("pred_e06b", start=100, report=130, end=160)
    success = alarm_success_at_deadline(truth, prediction_120)
    late = alarm_success_at_deadline(truth, prediction_130)
    actual = {"earliest_quality_report": T0 + 110, "success": success, "late": late}
    return _case(
        "E06", _evaluation_fixture("E06", previous_reliable_off=92, quality_bin_rights=[100, 110]),
        {"earliest_quality_report": T0 + 110, "success": {"alarm_latency_lower_bound_seconds": 20.0, "conservative_alarm_latency_seconds": 28.0, "success_window_start": float(T0 + 100), "success_window_end": float(T0 + 122), "success_at_deadline": True}, "late": {"alarm_latency_lower_bound_seconds": 30.0, "conservative_alarm_latency_seconds": 38.0, "success_window_start": float(T0 + 100), "success_window_end": float(T0 + 122), "success_at_deadline": False}}, actual,
    )


def _scenario_e07() -> dict[str, object]:
    truth = _truth("truth_e07", start=100, lower=92, end=150, censored=True)
    prediction = _prediction("pred_e07", start=100, report=120, end=180)
    support = [{"bin_right": T0 + right, "confirmation_quality": True, "active_valid_mask": {"kettle": True}} for right in range(100, 161, 10)]
    exposure = compute_scoring_exposure(support, "kettle")
    annotated = classify_prediction_scoring([prediction], exposure)
    matching = match_events([truth], annotated["predictions"])
    metrics = evaluate_matched_pair(truth, annotated["predictions"][0])
    actual = {"exposure_seconds": exposure["exposure_seconds"], "prediction_scored": annotated["predictions"][0]["prediction_scored"], "pairing_iou": matching["pairs"][0]["pairing_iou"], "metrics": metrics}
    return _case(
        "E07", _evaluation_fixture("E07", right_censored=True),
        {"exposure_seconds": 70.0, "prediction_scored": True, "pairing_iou": 1.0, "metrics": {"event_detection_matched": True, "pairing_iou": 1.0, "alarm_latency_lower_bound_seconds": 20.0, "conservative_alarm_latency_seconds": 28.0, "success_window_start": float(T0 + 100), "success_window_end": float(T0 + 122), "success_at_deadline": True, "end_mae_seconds": None, "complete_interval_iou": None, "duration_error_seconds": None, "interval_metric_exclusion_reason": "right_censored_truth"}}, actual,
    )


def _scenario_e08() -> dict[str, object]:
    truth_a = _truth("truth_2f0d145c0a57eec7b115a38ca3416fe4524fd209ac84278133a90c98c3dcf860", start=0, lower=-1, end=2, house="synthetic_e08")
    truth_b = _truth("truth_a3ef0bb1d795f1b685c226ec4e3d05c2905b11f439bee04b9f2633a9c0dda907", start=5, lower=4, end=7, house="synthetic_e08")
    pred_a = _prediction("pred_716d1508e5367f4a242f4b9cd356a0d0acfd310095f4e86613084a37cd2975f1", start=10, report=30, end=30, house="synthetic_e08")
    pred_b = _prediction("pred_72bfa3dac93ae9edb0badd127fbaef5c31d55ccb110bc703cb47254064e02e66", start=30, report=50, end=50, house="synthetic_e08")
    matching = match_events([truth_a, truth_b], [pred_a, pred_b])
    reversed_matching = match_events([truth_b, truth_a], [pred_b, pred_a])
    actual = {"pairs": matching["pair_ids"], "edges": [[item["pairing_iou"], item["start_error_seconds"], item["conservative_alarm_latency_seconds"]] for item in matching["pairs"]], "reverse_invariant": matching["pair_ids"] == reversed_matching["pair_ids"]}
    return _case(
        "E08", _evaluation_fixture("E08", exact_ids=True),
        {"pairs": [(truth_a["true_event_id"], pred_a["event_id"]), (truth_b["true_event_id"], pred_b["event_id"])], "edges": [[0.0, 10.0, 31.0], [0.0, 25.0, 46.0]], "reverse_invariant": True}, actual,
    )


def _claim_rows(houses: list[tuple[str, list[float | None]]]) -> tuple[list[dict[str, object]], dict[tuple[str, str], float]]:
    rows: list[dict[str, object]] = []
    exposures: dict[tuple[str, str], float] = {}
    for house, values in houses:
        exposures[("uk_dale", house)] = 86400.0
        rows.extend({"dataset": "uk_dale", "house_id": house, "alarm_slo_evaluable": True, "conservative_alarm_latency_seconds": value} for value in values)
    return rows, exposures


def _claim_view(statistics: dict[str, object], claim: dict[str, object]) -> dict[str, object]:
    return {
        "recall": statistics["alarm_recall_at_30s"], "wilson": statistics["event_wilson_95_lower_bound"],
        "wilson_bits": struct.pack(">d", statistics["event_wilson_95_lower_bound"]).hex() if statistics["event_wilson_95_lower_bound"] is not None else None,
        "bootstrap_lower": statistics["house_cluster_bootstrap_95_lower_bound"], "p95": statistics["p95_alarm_latency_seconds"],
        "far": statistics["false_alarms_per_house_day"], "n": statistics["evaluable_event_count"], "H": statistics["included_outer_house_count"],
        "house_order": statistics["house_order"], "house_recall": statistics["per_house_alarm_recall_at_30s"],
        "gates": [item["status"] for item in claim["gate_results_in_order"]], "claim": claim["class_claim"], "failure": claim["failure_wording"],
    }


def _scenario_e09() -> dict[str, object]:
    rows, exposure = _claim_rows([("1", [10.0, 20.0]), ("10", [30.0, 40.0]), ("2", [50.0, 60.0])])
    statistics = derive_claim_statistics("microwave", rows, exposure)
    claim = reduce_class_claim({key: statistics[key] for key in ("canonical_class_id", "alarm_recall_at_30s", "event_wilson_95_lower_bound", "house_cluster_bootstrap_95_lower_bound", "p95_alarm_latency_seconds", "false_alarms_per_house_day", "evaluable_event_count", "included_outer_house_count")})
    bootstrap = statistics["bootstrap"]
    assert bootstrap is not None
    distribution = {str(value): bootstrap["statistics"].count(value) for value in sorted(set(bootstrap["statistics"]))}
    reversed_stats = derive_claim_statistics("microwave", list(reversed(rows)), dict(reversed(list(exposure.items()))))
    actual = {**_claim_view(statistics, claim), "bootstrap_first5": bootstrap["indices"][:5], "bootstrap_hashes": [bootstrap["indices_sha256"], bootstrap["statistics_sha256"], bootstrap["sorted_statistics_sha256"]], "distribution": distribution, "reverse_invariant": _claim_view(statistics, claim) == _claim_view(reversed_stats, reduce_class_claim({key: reversed_stats[key] for key in ("canonical_class_id", "alarm_recall_at_30s", "event_wilson_95_lower_bound", "house_cluster_bootstrap_95_lower_bound", "p95_alarm_latency_seconds", "false_alarms_per_house_day", "evaluable_event_count", "included_outer_house_count")}))}
    expected_distribution = {"0.0": 410, str(1 / 6): 1053, str(1 / 3): 2277, "0.5": 2600, str(2 / 3): 2168, str(5 / 6): 1129, "1.0": 363}
    return _case(
        "E09", _evaluation_fixture("E09", class_id="microwave", house_order=["uk_dale:1", "uk_dale:10", "uk_dale:2"]),
        {"recall": .5, "wilson": 0.1876163064826506, "wilson_bits": "3fc803cfa64507b0", "bootstrap_lower": 0.0, "p95": 60.0, "far": 0.0, "n": 6, "H": 3, "house_order": ["uk_dale:1", "uk_dale:10", "uk_dale:2"], "house_recall": [1.0, .5, 0.0], "gates": ["FAIL", "FAIL", "FAIL", "FAIL", "PASS", "FAIL", "PASS"], "claim": False, "failure": "尚未证明满足30秒报警目标", "bootstrap_first5": [[2, 1, 2], [1, 1, 0], [0, 2, 2], [2, 0, 1], [0, 0, 1]], "bootstrap_hashes": ["af0359ffa6bf63fb6f7dc8c192e2c44de1da7a4b3f117d6655bcb98c5472969a", "f8adb343e077c96198c9fd96bae9cf014a30a21051d5c2a3b7b1c74a9edaea7c", "3d2a641e6e5788f118729e9520dfa444acd7494055d1deab8f45b26534960dd9"], "distribution": expected_distribution, "reverse_invariant": True}, actual,
    )


def _claim_input_from_statistics(statistics: dict[str, object]) -> dict[str, object]:
    return {key: statistics[key] for key in ("canonical_class_id", "alarm_recall_at_30s", "event_wilson_95_lower_bound", "house_cluster_bootstrap_95_lower_bound", "p95_alarm_latency_seconds", "false_alarms_per_house_day", "evaluable_event_count", "included_outer_house_count")}


def _scenario_e10() -> dict[str, object]:
    rows, exposure = _claim_rows([("1", [30.0] * 34), ("2", [30.0] * 33), ("3", [30.0] * 33)])
    statistics = derive_claim_statistics("microwave", rows, exposure)
    claim = reduce_class_claim(_claim_input_from_statistics(statistics))
    boundary = dict(_claim_input_from_statistics(statistics)); boundary["false_alarms_per_house_day"] = .1
    boundary_claim = reduce_class_claim(boundary)
    actual = {**_claim_view(statistics, claim), "boundary_far_claim": boundary_claim["class_claim"]}
    return _case(
        "E10", _evaluation_fixture("E10", all_gates_equal_or_pass=True),
        {"recall": 1.0, "wilson": 0.9630065017930143, "wilson_bits": "3feed0f302e129e0", "bootstrap_lower": 1.0, "p95": 30.0, "far": 0.0, "n": 100, "H": 3, "house_order": ["uk_dale:1", "uk_dale:2", "uk_dale:3"], "house_recall": [1.0, 1.0, 1.0], "gates": ["PASS"] * 7, "claim": True, "failure": None, "boundary_far_claim": True}, actual,
    )


def _scenario_e11() -> dict[str, object]:
    rows, exposure = _claim_rows([("1", [30.0] * 99), ("2", [40.0])])
    statistics = derive_claim_statistics("microwave", rows, exposure)
    claim = reduce_class_claim(_claim_input_from_statistics(statistics))
    actual = _claim_view(statistics, claim)
    return _case(
        "E11", _evaluation_fixture("E11", insufficient_outer_houses=True),
        {"recall": .99, "wilson": 0.9455138038212947, "wilson_bits": "3fee41a62a2a8756", "bootstrap_lower": None, "p95": 30.0, "far": 0.0, "n": 100, "H": 2, "house_order": ["uk_dale:1", "uk_dale:2"], "house_recall": [1.0, 0.0], "gates": ["PASS", "PASS", "FAIL", "PASS", "PASS", "PASS", "FAIL"], "claim": False, "failure": "尚未证明满足30秒报警目标"}, actual,
    )


def _scenario_e12() -> dict[str, object]:
    pass_input = {"canonical_class_id": "microwave", "alarm_recall_at_30s": .95, "event_wilson_95_lower_bound": .90, "house_cluster_bootstrap_95_lower_bound": .90, "p95_alarm_latency_seconds": 30.0, "false_alarms_per_house_day": .10, "evaluable_event_count": 100, "included_outer_house_count": 3}
    gate_names = ["alarm_recall_at_30s", "event_wilson_95_lower_bound", "house_cluster_bootstrap_95_lower_bound", "p95_alarm_latency_seconds", "false_alarms_per_house_day", "evaluable_event_count", "included_outer_house_count"]
    failures = {
        "RECALL_ONLY": {"alarm_recall_at_30s": .94}, "WILSON_ONLY": {"event_wilson_95_lower_bound": .8951971146376326},
        "BOOTSTRAP_ONLY": {"house_cluster_bootstrap_95_lower_bound": 0.0}, "P95_ONLY": {"p95_alarm_latency_seconds": 31.0},
        "FAR_ONLY": {"false_alarms_per_house_day": .11}, "EVENT_COUNT_ONLY": {"evaluable_event_count": 99},
        "HOUSE_COUNT_AND_NULL_BOOTSTRAP": {"included_outer_house_count": 2, "house_cluster_bootstrap_95_lower_bound": None},
    }
    failure_views: dict[str, object] = {}
    for name, delta in failures.items():
        item = {**pass_input, **delta}
        result = reduce_class_claim(item)
        failure_views[name] = {"statuses": [gate["status"] for gate in result["gate_results_in_order"]], "claim": result["class_claim"], "failure": result["failure_wording"]}
    null_views: dict[str, object] = {}
    for field in gate_names[:5]:
        result = reduce_class_claim({**pass_input, field: None})
        null_views[field] = [gate["status"] for gate in result["gate_results_in_order"]]
    invalid_rejected = True
    invalid_values = [
        {**pass_input, "alarm_recall_at_30s": float("nan")}, {**pass_input, "event_wilson_95_lower_bound": float("inf")},
        {**pass_input, "house_cluster_bootstrap_95_lower_bound": -0.01}, {**pass_input, "p95_alarm_latency_seconds": -1.0},
        {**pass_input, "false_alarms_per_house_day": -1.0}, {**pass_input, "evaluable_event_count": True},
        {**pass_input, "included_outer_house_count": 1.5}, {**pass_input, "canonical_class_id": "unknown"},
        {key: value for key, value in pass_input.items() if key != "alarm_recall_at_30s"}, {**pass_input, "extra": 1},
    ]
    for invalid in invalid_values:
        try:
            reduce_class_claim(invalid)
            invalid_rejected = False
        except ContractError:
            pass
    all_pass = reduce_class_claim(pass_input)
    exploratory = [{"canonical_class_id": class_id, "class_claim": False} for class_id in ("kettle", "toaster", "hair_dryer", "iron", "rice_cooker")]
    exploratory_report = report_exploratory_claims(exploratory)
    overall_pass = reduce_overall_claim([all_pass])
    failed_primary = reduce_class_claim({**pass_input, "alarm_recall_at_30s": .94})
    overall_fail = reduce_overall_claim([failed_primary])
    malformed_overall_rejected = True
    for rows in ([], [all_pass, all_pass], [{"canonical_class_id": "kettle", "class_claim": True}]):
        try:
            reduce_overall_claim(rows)
            malformed_overall_rejected = False
        except ContractError:
            pass
    actual = {
        "single_failures": failure_views, "all_thresholds_equal": {"statuses": [gate["status"] for gate in all_pass["gate_results_in_order"]], "claim": all_pass["class_claim"], "failure": all_pass["failure_wording"]},
        "nulls": null_views, "invalid_rejected": invalid_rejected, "overall_pass": overall_pass, "overall_fail": overall_fail,
        "exploratory_count": len(exploratory_report), "malformed_overall_rejected": malformed_overall_rejected,
    }
    expected_failure_statuses = {
        "RECALL_ONLY": ["FAIL", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS"],
        "WILSON_ONLY": ["PASS", "FAIL", "PASS", "PASS", "PASS", "PASS", "PASS"],
        "BOOTSTRAP_ONLY": ["PASS", "PASS", "FAIL", "PASS", "PASS", "PASS", "PASS"],
        "P95_ONLY": ["PASS", "PASS", "PASS", "FAIL", "PASS", "PASS", "PASS"],
        "FAR_ONLY": ["PASS", "PASS", "PASS", "PASS", "FAIL", "PASS", "PASS"],
        "EVENT_COUNT_ONLY": ["PASS", "PASS", "PASS", "PASS", "PASS", "FAIL", "PASS"],
        "HOUSE_COUNT_AND_NULL_BOOTSTRAP": ["PASS", "PASS", "FAIL", "PASS", "PASS", "PASS", "FAIL"],
    }
    expected_failure_views = {name: {"statuses": statuses, "claim": False, "failure": "尚未证明满足30秒报警目标"} for name, statuses in expected_failure_statuses.items()}
    expected_nulls = {field: ["FAIL" if index == position else "PASS" for index in range(7)] for position, field in enumerate(gate_names[:5])}
    return _case(
        "E12", _evaluation_fixture("E12", gate_ablation=True, exploratory_isolation=True),
        {"single_failures": expected_failure_views, "all_thresholds_equal": {"statuses": ["PASS"] * 7, "claim": True, "failure": None}, "nulls": expected_nulls, "invalid_rejected": True, "overall_pass": {"primary_class_claims_in_class_order": [True], "overall_claim": True, "failure_wording": None}, "overall_fail": {"primary_class_claims_in_class_order": [False], "overall_claim": False, "failure_wording": "尚未证明满足30秒报警目标"}, "exploratory_count": 5, "malformed_overall_rejected": True}, actual,
    )


def _d01_timeline(*, reverse: bool = False):
    """Construct the unique complete label/event fixture for matrix D01 without reading actual files."""

    aggregate = [
        _aggregate(index, 100 + 5 * index, issues=1 if index == 9 else 0, source="agg_a" if index < 15 else "agg_b")
        for index in range(20)
    ]
    records = [
        *[_submeter("kettle_a", second, power) for second, power in ((10, 0), (20, 1200), (50, 1200), (60, 0), (70, 0), (90, 0))],
        *[_submeter("kettle_b", second, power) for second, power in ((10, 0), (30, 0), (40, 1200), (70, 1200), (80, 0), (90, 0))],
        *[_submeter("toaster_a", second, power, appliance_class="toaster") for second, power in ((20, 0), (30, 600), (60, 600), (70, 0), (80, 0))],
        *[_submeter("dryer_a", second, power, appliance_class="hair_dryer") for second, power in ((90, 0), (100, 400), (110, 0), (120, 0))],
        *[_submeter("microwave_a", second, power, appliance_class="microwave", epoch=0) for second, power in ((110, 0), (120, 900), (140, 900), (145, 900))],
    ]
    descriptors = [_descriptor("microwave_a", "microwave", end=145, epoch=0)]
    if reverse:
        aggregate.reverse(); records.reverse(); descriptors.reverse()
    return _timeline(
        aggregate, end=200, scenario="d01_reversed" if reverse else "d01", house_id="synthetic_d01",
        expected_channels={"kettle": ["kettle_a", "kettle_b"], "toaster": ["toaster_a"], "microwave": ["microwave_a"], "hair_dryer": ["dryer_a"]},
        submeter_records=records, label_descriptors=descriptors,
    )


def _scenario_d01() -> dict[str, object]:
    timeline = _d01_timeline()
    repeated = _d01_timeline(reverse=True)
    descriptors = timeline.logical_descriptors()
    repeated_descriptors = repeated.logical_descriptors()
    classes = {name: _class_column(timeline, name) for name in CLASS_ORDER}
    event_view = [_event_view(item) for item in timeline.ground_truth_events]
    # Changing one typed-provenance field cannot masquerade as an identical logical timeline.
    changed_provenance = [dict(item) for item in timeline.label_provenance]
    changed_provenance[0]["source_label"] = "mutated_synthetic_label"
    provenance_original = typed_record_logical_descriptor("label_provenance", timeline.label_provenance)
    provenance_changed = typed_record_logical_descriptor("label_provenance", changed_provenance)
    original_descriptor = descriptors["timeline_logical_content_descriptor"]
    mutated_logical_descriptor = {
        **original_descriptor,
        "typed_record_descriptors": [descriptors["typed_record_descriptors"][0], provenance_changed],
    }
    typed_mutation_changes_timeline_hash = hashlib.sha256(jcs_bytes(original_descriptor)).hexdigest() != hashlib.sha256(jcs_bytes(mutated_logical_descriptor)).hexdigest()
    stale_truth_rejected = False
    changed_event = dict(timeline.ground_truth_events[0])
    changed_event["true_start_time"] = float(changed_event["true_start_time"]) + 10.0
    changed_event["true_start_upper_bound"] = float(changed_event["true_start_upper_bound"]) + 10.0
    try:
        validate_unique_truth_event_ids([changed_event])
    except TimelineContractError:
        stale_truth_rejected = True
    recomputed_event = dict(changed_event)
    recomputed_event["true_event_id"] = stable_truth_event_id(
        recomputed_event["dataset"], recomputed_event["house_id"], recomputed_event["continuity_id"], recomputed_event["label_epoch_id"],
        recomputed_event["appliance_class"], recomputed_event["true_start_time"], json.loads(recomputed_event["source_channel_ids_json"]),
    )
    validate_unique_truth_event_ids([recomputed_event])
    # D01 writes and validates the same complete logical bundle byte by byte; container SHA
    # is not semantic identity across builds.
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        materialized_a, artifacts_a = _materialize_synthetic_timeline(directory / "build_a", timeline, variant=False)
        materialized_b, artifacts_b = _materialize_synthetic_timeline(directory / "build_b", repeated, variant=True)
        container_verified = (
            materialized_a == descriptors and materialized_b == repeated_descriptors
            and all(descriptor["container_sha256"] for _, descriptor in artifacts_a + artifacts_b)
            and any(left[1]["container_sha256"] != right[1]["container_sha256"] for left, right in zip(artifacts_a, artifacts_b))
        )
    actual = {
        "input": {
            "p": timeline.input_values[:, 0].tolist(), "p_mask": timeline.input_value_mask[:, 0].tolist(),
            "delta_mask": timeline.input_value_mask[:, 1].tolist(), "baseline": timeline.input_values[:, 2].tolist(),
            "baseline_mask": timeline.input_value_mask[:, 2].tolist(), "continuity": timeline.continuity_id.tolist(),
            "flags": timeline.input_quality_flags.tolist(), "count": timeline.raw_sample_count.tolist(),
        },
        "labels": {
            "kettle": {"active": timeline.active_target[:, classes["kettle"]].tolist(), "mask": timeline.active_valid_mask[:, classes["kettle"]].tolist()},
            "toaster": {"active": timeline.active_target[:, classes["toaster"]].tolist(), "mask": timeline.active_valid_mask[:, classes["toaster"]].tolist()},
            "microwave": {"active": timeline.active_target[:, classes["microwave"]].tolist(), "mask": timeline.active_valid_mask[:, classes["microwave"]].tolist(), "onset": timeline.onset_target[:, classes["microwave"]].tolist(), "offset": timeline.offset_target[:, classes["microwave"]].tolist(), "epoch": timeline.label_epoch_id[:, classes["microwave"]].tolist()},
            "dryer_events": len([item for item in timeline.ground_truth_events if item["appliance_class"] == "hair_dryer"]),
        },
        "events": event_view,
        "repeat_exact": all(np.array_equal(timeline.arrays[name], repeated.arrays[name]) for name in REQUIRED_ARRAYS) and descriptors == repeated_descriptors,
        "typed_mutation_changes": provenance_original != provenance_changed and typed_mutation_changes_timeline_hash,
        "stale_truth_rejected": stale_truth_rejected,
        "recomputed_truth_changes_id": recomputed_event["true_event_id"] != timeline.ground_truth_events[0]["true_event_id"],
        "container_verified": container_verified,
    }
    zeros = [0] * 20
    kettle_active = [0, 0] + [1] * 6 + [0] + [0] * 11
    kettle_mask = [0] + [1] * 8 + [0] * 11
    toaster_active = [0] * 3 + [1] * 4 + [0] + [0] * 12
    toaster_mask = [0, 0] + [1] * 6 + [0] * 12
    microwave_active = [0] * 12 + [1, 1] + [0] * 6
    microwave_mask = [0] * 11 + [1, 1, 1] + [0] * 6
    expected_events = [
        {"true_event_id": "truth_b2084e9fbc71c12f795114cabba26776211c09af8d4237cf57ba1ea34962d904", "appliance_class": "kettle", "true_start_time": float(T0 + 20), "true_start_lower_bound": float(T0 + 10), "true_start_upper_bound": float(T0 + 20), "true_end_time": float(T0 + 80), "end_reason": "normal_off", "right_censored": False, "source_channel_ids_json": '["kettle_a","kettle_b"]'},
        {"true_event_id": "truth_1ac021ad35dc9b9f3a2eed44af739c083c645938024a6b9fe12781e5849efcbb", "appliance_class": "toaster", "true_start_time": float(T0 + 30), "true_start_lower_bound": float(T0 + 20), "true_start_upper_bound": float(T0 + 30), "true_end_time": float(T0 + 70), "end_reason": "normal_off", "right_censored": False, "source_channel_ids_json": '["toaster_a"]'},
        {"true_event_id": "truth_afeda6532a35eaf88f70d6b8b491c3eb3568190402ef5bca355606f65694bbf7", "appliance_class": "microwave", "true_start_time": float(T0 + 120), "true_start_lower_bound": float(T0 + 110), "true_start_upper_bound": float(T0 + 120), "true_end_time": float(T0 + 145), "end_reason": "right_censored_label_epoch_end", "right_censored": True, "source_channel_ids_json": '["microwave_a"]'},
    ]
    expected = {
        "input": {"p": [float(100 + 5 * index) if index != 9 else 0.0 for index in range(20)], "p_mask": [0 if index == 9 else 1 for index in range(20)], "delta_mask": [0 if index in {0, 9, 10, 15} else 1 for index in range(20)], "baseline": [0.0] * 6 + [17.5] * 3 + [0.0] + [20.0] * 3 + [15.0] * 2 + [0.0] * 5, "baseline_mask": [0] * 6 + [1] * 3 + [0] + [1] * 5 + [0] * 5, "continuity": [0] * 15 + [1] * 5, "flags": [0] * 9 + [2] + [0] * 5 + [128] + [0] * 4, "count": [1] * 20},
        "labels": {"kettle": {"active": kettle_active, "mask": kettle_mask}, "toaster": {"active": toaster_active, "mask": toaster_mask}, "microwave": {"active": microwave_active, "mask": microwave_mask, "onset": [0] * 12 + [1] + [0] * 7, "offset": zeros, "epoch": [0] * 14 + [-1] * 6}, "dryer_events": 0},
        "events": expected_events, "repeat_exact": True, "typed_mutation_changes": True, "stale_truth_rejected": True, "recomputed_truth_changes_id": True, "container_verified": True,
    }
    return _case("D01", {"fixture": "matrix_complete_sequence_d01"}, expected, actual)


def _scenario_d02() -> dict[str, object]:
    aggregate = [_aggregate(index, 100) for index in range(7)]
    baseline = _timeline(aggregate, end=70, scenario="d02_base", expected_channels={"kettle": ["kettle_a"]}, submeter_records=[_submeter("kettle_a", second, power) for second, power in ((10, 0), (20, 1200), (30, 0), (40, 0))])
    variant = _timeline(aggregate, end=70, scenario="d02_variant", expected_channels={"kettle": ["kettle_a"]}, submeter_records=[_submeter("kettle_a", second, power) for second, power in ((10, 0), (20, 1200), (30, 1200), (50, 0), (60, 0))])
    column = _class_column(baseline, "kettle")
    actual = {
        "aggregate_bitwise_equal": np.array_equal(baseline.input_values, variant.input_values) and np.array_equal(baseline.input_value_mask, variant.input_value_mask),
        "baseline_b2": [int(baseline.active_target[2, column]), int(baseline.active_valid_mask[2, column])], "baseline_events": len(baseline.ground_truth_events),
        "variant_active": variant.active_target[2:5, column].tolist(), "variant_masks": variant.active_valid_mask[2:5, column].tolist(),
        "variant_edges": [int(variant.onset_target[2, column]), int(variant.offset_target[5, column])],
        "variant_event": [[item["true_start_time"], item["true_end_time"]] for item in variant.ground_truth_events],
    }
    return _case(
        "D02", {"future_label_suffix_mutation": "allowed offline label difference"},
        {"aggregate_bitwise_equal": True, "baseline_b2": [0, 0], "baseline_events": 0, "variant_active": [1, 1, 1], "variant_masks": [1, 1, 1], "variant_edges": [1, 1], "variant_event": [[float(T0 + 20), float(T0 + 50)]]}, actual,
    )




def _scenario_function(scenario_id):
    if scenario_id.startswith("A05_"):
        suffix=scenario_id[4:]
        for state in ("PENDING_OFF", "PENDING_ON", "OFF", "ON"):
            if suffix.startswith(state+"_"):
                return lambda: _scenario_a05_case(state,suffix[len(state)+1:])
    return globals()["_scenario_"+scenario_id.lower()]

class StreamingNumericalTests(unittest.TestCase):
    pass

def _test_for(scenario_id):
    def test(self):
        result=_scenario_function(scenario_id)()
        self.assertEqual(result["status"],"PASS")
    return test

for _scenario_id in REQUIRED_SCENARIO_IDS:
    setattr(StreamingNumericalTests,"test_"+_scenario_id.lower(),_test_for(_scenario_id))
