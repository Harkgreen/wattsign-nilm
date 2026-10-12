'Event matching, latency and false-alarm evaluation for explicitly supplied records. Numerical definitions are retained from the research reference implementation.'

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import struct
import unicodedata
from typing import Any


class ContractError(ValueError):
    """Raised when the frozen evaluation definition is violated."""


# Constants correspond to the frozen class registry and deployment evaluation criteria.
CLASS_ORDER = ("kettle", "toaster", "microwave", "hair_dryer", "iron", "rice_cooker")
PRIMARY_CLASS_ORDER = ("microwave",)
LOW_RESOURCE_CLASS_ORDER = ("kettle", "toaster", "hair_dryer", "iron", "rice_cooker")
BIN_SECONDS = 10.0
ALARM_DEADLINE_SECONDS = 30.0
FALSE_ALARM_LIMIT = 0.1
WILSON_Z_95 = 1.959963984540054
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_LOWER_RANK = 249
FAILURE_WORDING = "尚未证明满足30秒报警目标"
INT64_MAX = 9_223_372_036_854_775_807


def _nfc_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{field} 必须是非空字符串")
    return unicodedata.normalize("NFC", value)


def _text_key(value: str) -> bytes:
    """Compare NFC-normalized strings by unsigned UTF-8 byte order."""

    return unicodedata.normalize("NFC", value).encode("utf-8")


def _require_class_id(value: Any) -> str:
    class_id = _nfc_text(value, "canonical_class_id")
    if class_id not in CLASS_ORDER:
        raise ContractError("canonical_class_id 不在冻结 K=6 类别轴中")
    return class_id


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{field} 必须是对象")
    return value


def _require_fields(record: Mapping[str, Any], fields: Sequence[str], record_name: str) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise ContractError(f"{record_name} 缺少字段: {', '.join(missing)}")


def _finite_float(value: Any, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{field} 必须是 IEEE-754 binary64 数值")
    converted = float(value)
    if not math.isfinite(converted):
        raise ContractError(f"{field} 不得为 NaN 或 Infinity")
    if nonnegative and converted < 0.0:
        raise ContractError(f"{field} 必须非负")
    return converted


def _nullable_float(value: Any, field: str, *, nonnegative: bool = False) -> float | None:
    if value is None:
        return None
    return _finite_float(value, field, nonnegative=nonnegative)


def _nonnegative_int64(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{field} 必须是非负 signed int64")
    if value < 0 or value > INT64_MAX:
        raise ContractError(f"{field} 超出非负 signed int64 范围")
    return value


def _binary_flag(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    raise ContractError(f"{field} 必须为 0/1 或 bool")


def _float32(value: float) -> float:
    """Round once, and only once, at the specified float32 conversion point."""

    try:
        result = struct.unpack(">f", struct.pack(">f", value))[0]
    except OverflowError as exc:
        raise ContractError("阈值无法表示为 finite float32") from exc
    if not math.isfinite(result):
        raise ContractError("阈值无法表示为 finite float32")
    return result


def _float32_hex(value: float) -> str:
    return struct.pack(">f", _float32(value)).hex()


def _half_open_iou(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    """Compute IoU of two half-open intervals; empty clipped intervals have zero length."""

    if left_end <= left_start or right_end < right_start:
        raise ContractError("事件区间必须是非空半开区间（裁剪前预测可为空）")
    intersection = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    union = (left_end - left_start) + (right_end - right_start) - intersection
    if union <= 0.0:
        raise ContractError("IoU union 不得为零")
    return intersection / union


def _distance_to_closed_interval(value: float, lower: float, upper: float) -> float:
    if lower > upper:
        raise ContractError("真值启动上下界顺序非法")
    if value < lower:
        return lower - value
    if value > upper:
        return value - upper
    return 0.0


def _validate_truth_event(event: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "true_event_id",
        "dataset",
        "house_id",
        "continuity_id",
        "appliance_class",
        "true_start_time",
        "true_start_lower_bound",
        "true_start_upper_bound",
        "true_end_time",
        "right_censored",
    )
    _require_fields(event, required, "truth event")
    start = _finite_float(event["true_start_time"], "true_start_time")
    lower = _finite_float(event["true_start_lower_bound"], "true_start_lower_bound")
    upper = _finite_float(event["true_start_upper_bound"], "true_start_upper_bound")
    end = _finite_float(event["true_end_time"], "true_end_time")
    if not (lower <= upper == start < end):
        raise ContractError("真值必须满足 lower <= upper == true_start < true_end")
    return {
        "true_event_id": _nfc_text(event["true_event_id"], "true_event_id"),
        "dataset": _nfc_text(event["dataset"], "dataset"),
        "house_id": _nfc_text(event["house_id"], "house_id"),
        "continuity_id": _nonnegative_int64(event["continuity_id"], "continuity_id"),
        "appliance_class": _require_class_id(event["appliance_class"]),
        "true_start_time": start,
        "true_start_lower_bound": lower,
        "true_start_upper_bound": upper,
        "true_end_time": end,
        "right_censored": _binary_flag(event["right_censored"], "right_censored"),
        "original": event,
    }


def _validate_prediction_event(event: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "event_id",
        "dataset",
        "house_id",
        "continuity_id",
        "appliance_class",
        "start_time",
        "report_time",
        "end_time",
    )
    _require_fields(event, required, "prediction event")
    start = _finite_float(event["start_time"], "start_time")
    end = _finite_float(event["end_time"], "end_time")
    if end <= start:
        raise ContractError("prediction event 必须满足 start_time < end_time")

    scored = True
    if "prediction_scored" in event:
        scored = _binary_flag(event["prediction_scored"], "prediction_scored")
    if "unscored_unknown" in event:
        unknown = _binary_flag(event["unscored_unknown"], "unscored_unknown")
        if unknown == scored:
            raise ContractError("prediction_scored 与 unscored_unknown 必须互补")
    return {
        "event_id": _nfc_text(event["event_id"], "event_id"),
        "dataset": _nfc_text(event["dataset"], "dataset"),
        "house_id": _nfc_text(event["house_id"], "house_id"),
        "continuity_id": _nonnegative_int64(event["continuity_id"], "continuity_id"),
        "appliance_class": _require_class_id(event["appliance_class"]),
        "start_time": start,
        "report_time": _finite_float(event["report_time"], "report_time"),
        "end_time": end,
        "prediction_scored": scored,
        "original": event,
    }


def _group_key(event: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        event["dataset"],
        event["house_id"],
        event["continuity_id"],
        event["appliance_class"],
    )


def _pairing_iou(truth: Mapping[str, Any], prediction: Mapping[str, Any]) -> float:
    prediction_end = prediction["end_time"]
    if truth["right_censored"]:
        # Clip future prediction tails lacking observable supervision for right-censored truth.
        prediction_end = min(prediction_end, truth["true_end_time"])
        if prediction_end < prediction["start_time"]:
            prediction_end = prediction["start_time"]
    return _half_open_iou(
        truth["true_start_time"],
        truth["true_end_time"],
        prediction["start_time"],
        prediction_end,
    )


@dataclass(frozen=True)
class _PairEdge:
    truth: Mapping[str, Any]
    prediction: Mapping[str, Any]
    pairing_iou: float
    start_error_seconds: float
    conservative_alarm_latency_seconds: float

    @property
    def id_pair(self) -> tuple[str, str]:
        return (self.truth["true_event_id"], self.prediction["event_id"])


def _candidate_edge(truth: Mapping[str, Any], prediction: Mapping[str, Any]) -> _PairEdge | None:
    iou = _pairing_iou(truth, prediction)
    start_error = _distance_to_closed_interval(
        prediction["start_time"],
        truth["true_start_lower_bound"],
        truth["true_start_upper_bound"],
    )
    if iou <= 0.0 and start_error > 30.0:
        return None
    return _PairEdge(
        truth=truth,
        prediction=prediction,
        pairing_iou=iou,
        start_error_seconds=start_error,
        conservative_alarm_latency_seconds=(
            prediction["report_time"] - truth["true_start_lower_bound"]
        ),
    )


def _pair_array_key(edges: Sequence[_PairEdge]) -> tuple[tuple[bytes, bytes], ...]:
    ordered = sorted(
        edges,
        key=lambda edge: (_text_key(edge.truth["true_event_id"]), _text_key(edge.prediction["event_id"])),
    )
    return tuple(
        (_text_key(edge.truth["true_event_id"]), _text_key(edge.prediction["event_id"]))
        for edge in ordered
    )


def _matching_objective(edges: Sequence[_PairEdge]) -> tuple[Any, ...]:
    """Build a Python-comparable minimization key for the five-level objective."""

    # Accumulate left to right in stable-ID order so input row order cannot change the binary64 sum.
    ordered = sorted(
        edges,
        key=lambda edge: (_text_key(edge.truth["true_event_id"]), _text_key(edge.prediction["event_id"])),
    )
    total_iou = 0.0
    total_start_error = 0.0
    total_latency = 0.0
    for edge in ordered:
        total_iou += edge.pairing_iou
        total_start_error += edge.start_error_seconds
        total_latency += edge.conservative_alarm_latency_seconds
    return (
        -len(ordered),
        -total_iou,
        total_start_error,
        total_latency,
        _pair_array_key(ordered),
    )


def _best_group_matching(
    truths: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
) -> list[_PairEdge]:
    """Enumerate all one-to-one matchings of a small synthetic fixture and apply the fixed objective order."""

    ordered_truths = sorted(truths, key=lambda row: _text_key(row["true_event_id"]))
    ordered_predictions = sorted(predictions, key=lambda row: _text_key(row["event_id"]))
    edges_by_truth: list[list[_PairEdge]] = []
    for truth in ordered_truths:
        edges = [
            edge
            for prediction in ordered_predictions
            if (edge := _candidate_edge(truth, prediction)) is not None
        ]
        edges_by_truth.append(edges)

    best: list[_PairEdge] = []
    best_objective: tuple[Any, ...] | None = None

    def visit(index: int, used_predictions: set[str], selected: list[_PairEdge]) -> None:
        nonlocal best, best_objective
        if index == len(ordered_truths):
            objective = _matching_objective(selected)
            if best_objective is None or objective < best_objective:
                best_objective = objective
                best = list(selected)
            return

        # Leaving this truth unmatched is a valid branch, preserving the cardinality objective.
        visit(index + 1, used_predictions, selected)
        for edge in edges_by_truth[index]:
            event_id = edge.prediction["event_id"]
            if event_id in used_predictions:
                continue
            used_predictions.add(event_id)
            selected.append(edge)
            visit(index + 1, used_predictions, selected)
            selected.pop()
            used_predictions.remove(event_id)

    visit(0, set(), [])
    return sorted(
        best,
        key=lambda edge: (_text_key(edge.truth["true_event_id"]), _text_key(edge.prediction["event_id"])),
    )


def match_events(
    truth_events: Sequence[Mapping[str, Any]],
    prediction_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Perform deterministic one-to-one matching of synthetic events.
    
    Unscored predictions (prediction_scored=false or unscored_unknown=true) are reported separately and never enter candidate edges, matching, FP counts or false-alarm rates. pair_ids use stable NFC/UTF-8 ordering of true_event_id,event_id."""

    truths = [_validate_truth_event(_require_mapping(row, "truth event")) for row in truth_events]
    predictions = [
        _validate_prediction_event(_require_mapping(row, "prediction event"))
        for row in prediction_events
    ]
    truth_ids = [row["true_event_id"] for row in truths]
    prediction_ids = [row["event_id"] for row in predictions]
    if len(set(truth_ids)) != len(truth_ids) or len(set(prediction_ids)) != len(prediction_ids):
        raise ContractError("truth_event_id 和 event_id 在一次 matching 内必须各自唯一")

    scored_predictions = [row for row in predictions if row["prediction_scored"]]
    unscored_predictions = [row for row in predictions if not row["prediction_scored"]]
    truths_by_group: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    predictions_by_group: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for truth in truths:
        truths_by_group[_group_key(truth)].append(truth)
    for prediction in scored_predictions:
        predictions_by_group[_group_key(prediction)].append(prediction)

    all_edges: list[_PairEdge] = []
    for group in sorted(set(truths_by_group) | set(predictions_by_group), key=repr):
        all_edges.extend(_best_group_matching(truths_by_group.get(group, ()), predictions_by_group.get(group, ())))
    all_edges.sort(
        key=lambda edge: (_text_key(edge.truth["true_event_id"]), _text_key(edge.prediction["event_id"])),
    )
    matched_truth_ids = {edge.truth["true_event_id"] for edge in all_edges}
    matched_prediction_ids = {edge.prediction["event_id"] for edge in all_edges}
    unmatched_truth_ids = sorted(
        (row["true_event_id"] for row in truths if row["true_event_id"] not in matched_truth_ids),
        key=_text_key,
    )
    unmatched_scored_prediction_ids = sorted(
        (row["event_id"] for row in scored_predictions if row["event_id"] not in matched_prediction_ids),
        key=_text_key,
    )
    unscored_unknown_ids = sorted((row["event_id"] for row in unscored_predictions), key=_text_key)
    return {
        "matching_edges": len(all_edges),
        "pairs": [
            {
                "true_event_id": edge.truth["true_event_id"],
                "event_id": edge.prediction["event_id"],
                "pairing_iou": edge.pairing_iou,
                "start_error_seconds": edge.start_error_seconds,
                "conservative_alarm_latency_seconds": edge.conservative_alarm_latency_seconds,
            }
            for edge in all_edges
        ],
        "pair_ids": [edge.id_pair for edge in all_edges],
        "unmatched_truth_event_ids": unmatched_truth_ids,
        "unmatched_scored_prediction_event_ids": unmatched_scored_prediction_ids,
        "unscored_unknown_prediction_event_ids": unscored_unknown_ids,
    }


def alarm_success_at_deadline(
    truth_event: Mapping[str, Any], prediction_event: Mapping[str, Any], *, deadline_seconds: float = ALARM_DEADLINE_SECONDS
) -> dict[str, Any]:
    """Compute the SLO window and conservative alarm latency for one matched event.
    
    Reporting before the upper bound of true onset cannot count as success. The right boundary is closed, so a conservative latency of exactly 30 seconds can still succeed."""

    truth = _validate_truth_event(_require_mapping(truth_event, "truth event"))
    prediction = _validate_prediction_event(_require_mapping(prediction_event, "prediction event"))
    deadline = _finite_float(deadline_seconds, "deadline_seconds", nonnegative=True)
    if _group_key(truth) != _group_key(prediction):
        raise ContractError("truth 与 prediction 必须属于同一 matching group")
    lower = truth["true_start_lower_bound"]
    upper = truth["true_start_upper_bound"]
    end = truth["true_end_time"]
    report = prediction["report_time"]
    latest = min(lower + deadline, end + BIN_SECONDS)
    return {
        "alarm_latency_lower_bound_seconds": report - upper,
        "conservative_alarm_latency_seconds": report - lower,
        "success_window_start": upper,
        "success_window_end": latest,
        "success_at_deadline": report >= upper and report <= latest,
    }


def evaluate_matched_pair(
    truth_event: Mapping[str, Any], prediction_event: Mapping[str, Any]
) -> dict[str, Any]:
    """Report matched-event metrics, excluding complete-interval metrics for right-censored truth."""

    truth = _validate_truth_event(_require_mapping(truth_event, "truth event"))
    prediction = _validate_prediction_event(_require_mapping(prediction_event, "prediction event"))
    edge = _candidate_edge(truth, prediction)
    if edge is None:
        raise ContractError("这对事件不满足冻结的 matching 条件")
    result = {
        "event_detection_matched": True,
        "pairing_iou": edge.pairing_iou,
        **alarm_success_at_deadline(truth_event, prediction_event),
        "end_mae_seconds": None,
        "complete_interval_iou": None,
        "duration_error_seconds": None,
        "interval_metric_exclusion_reason": None,
    }
    if truth["right_censored"]:
        result["interval_metric_exclusion_reason"] = "right_censored_truth"
        return result
    complete_iou = _half_open_iou(
        truth["true_start_time"],
        truth["true_end_time"],
        prediction["start_time"],
        prediction["end_time"],
    )
    result["end_mae_seconds"] = abs(prediction["end_time"] - truth["true_end_time"])
    result["complete_interval_iou"] = complete_iou
    result["duration_error_seconds"] = abs(
        (prediction["end_time"] - prediction["start_time"])
        - (truth["true_end_time"] - truth["true_start_time"])
    )
    return result


def _mask_for_class(value: Any, appliance_class: str) -> bool:
    if isinstance(value, Mapping):
        if appliance_class not in value:
            raise ContractError("active_valid_mask 未覆盖请求类别")
        return _binary_flag(value[appliance_class], f"active_valid_mask.{appliance_class}")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != len(CLASS_ORDER):
            raise ContractError("active_valid_mask 数组长度必须等于冻结 K=6")
        return _binary_flag(value[CLASS_ORDER.index(appliance_class)], "active_valid_mask")
    return _binary_flag(value, "active_valid_mask")


def compute_scoring_exposure(
    support_bins: Sequence[Mapping[str, Any]], appliance_class: str
) -> dict[str, Any]:
    """Compute per-class scoring exposure from provided support bins.
    
    A bin counts only when confirmation_quality and that class active_valid_mask are both true. This function handles caller-provided synthetic bins and reads no timeline files."""

    class_id = _require_class_id(appliance_class)
    by_right: dict[float, bool] = {}
    for row in support_bins:
        record = _require_mapping(row, "support bin")
        _require_fields(record, ("bin_right", "confirmation_quality", "active_valid_mask"), "support bin")
        right = _finite_float(record["bin_right"], "bin_right")
        if right in by_right:
            raise ContractError("support bin 的 bin_right 必须唯一")
        quality = _binary_flag(record["confirmation_quality"], "confirmation_quality")
        by_right[right] = quality and _mask_for_class(record["active_valid_mask"], class_id)
    included = sorted((right for right, include in by_right.items() if include))
    exposure_seconds = float(len(included)) * BIN_SECONDS
    return {
        "appliance_class": class_id,
        "scoring_exposure_bin_rights": included,
        "exposure_seconds": exposure_seconds,
        "exposure_house_days": exposure_seconds / 86400.0,
    }


def classify_prediction_scoring(
    prediction_events: Sequence[Mapping[str, Any]], scoring_exposure: Mapping[str, Any]
) -> dict[str, Any]:
    """Classify predictions as scored or unscored_unknown at the report-time anchor."""

    exposure = _require_mapping(scoring_exposure, "scoring_exposure")
    _require_fields(exposure, ("scoring_exposure_bin_rights",), "scoring_exposure")
    anchors = {
        _finite_float(value, "scoring_exposure_bin_right")
        for value in exposure["scoring_exposure_bin_rights"]
    }
    annotated: list[dict[str, Any]] = []
    scored = 0
    unknown = 0
    for raw in prediction_events:
        event = _require_mapping(raw, "prediction event")
        if "report_time" not in event:
            raise ContractError("prediction event 缺少 report_time")
        report_time = _finite_float(event["report_time"], "report_time")
        is_scored = report_time in anchors
        copy = dict(event)
        copy["prediction_scored"] = is_scored
        copy["unscored_unknown"] = not is_scored
        annotated.append(copy)
        if is_scored:
            scored += 1
        else:
            unknown += 1
    return {
        "predictions": annotated,
        "scored_prediction_count": scored,
        "unscored_unknown_count": unknown,
    }


def summarize_unmatched_predictions(
    annotated_predictions: Sequence[Mapping[str, Any]], matched_prediction_event_ids: Sequence[str]
) -> dict[str, int]:
    """Count only unmatched scored predictions as FP; unknown predictions never create artificial FP."""

    matched = {_nfc_text(value, "event_id") for value in matched_prediction_event_ids}
    false_positives = 0
    unknown = 0
    scored = 0
    for raw in annotated_predictions:
        event = _require_mapping(raw, "prediction event")
        _require_fields(event, ("event_id", "prediction_scored", "unscored_unknown"), "prediction event")
        event_id = _nfc_text(event["event_id"], "event_id")
        is_scored = _binary_flag(event["prediction_scored"], "prediction_scored")
        is_unknown = _binary_flag(event["unscored_unknown"], "unscored_unknown")
        if is_scored == is_unknown:
            raise ContractError("prediction_scored 与 unscored_unknown 必须互补")
        if is_scored:
            scored += 1
            if event_id not in matched:
                false_positives += 1
        else:
            unknown += 1
    return {
        "scored_prediction_count": scored,
        "unscored_unknown_count": unknown,
        "false_positive_count": false_positives,
    }


_CANDIDATE_FIELDS = (
    "on_threshold",
    "event_f2",
    "alarm_recall_at_30s",
    "event_precision",
    "p95_alarm_latency_seconds",
    "false_alarms_per_evaluable_house_day",
)


def _normalise_threshold_candidate(record: Mapping[str, Any]) -> dict[str, float | None]:
    if set(record) != set(_CANDIDATE_FIELDS):
        raise ContractError("候选统计字段必须精确符合冻结 threshold reducer schema")
    on_input = _finite_float(record["on_threshold"], "on_threshold")
    on = _float32(on_input)
    percent = round(on * 100.0)
    if percent < 5 or percent > 95 or _float32(percent / 100.0) != on:
        raise ContractError("on_threshold 必须是 5..95 的整数百分点并先转为 float32")
    f2 = _finite_float(record["event_f2"], "event_f2", nonnegative=True)
    recall = _finite_float(record["alarm_recall_at_30s"], "alarm_recall_at_30s", nonnegative=True)
    precision = _finite_float(record["event_precision"], "event_precision", nonnegative=True)
    if f2 > 1.0 or recall > 1.0 or precision > 1.0:
        raise ContractError("F2、recall 与 precision 必须在闭区间 [0,1]")
    return {
        "on_threshold": on,
        "event_f2": f2,
        "alarm_recall_at_30s": recall,
        "event_precision": precision,
        "p95_alarm_latency_seconds": _nullable_float(
            record["p95_alarm_latency_seconds"], "p95_alarm_latency_seconds", nonnegative=True
        ),
        "false_alarms_per_evaluable_house_day": _finite_float(
            record["false_alarms_per_evaluable_house_day"],
            "false_alarms_per_evaluable_house_day",
            nonnegative=True,
        ),
    }


def _p95_sort_key(value: float | None) -> float:
    return math.inf if value is None else value


def select_threshold(
    candidate_statistics: Sequence[Mapping[str, Any]],
    evaluable_positive_events: int,
    exposure_house_days: float,
) -> dict[str, Any]:
    """Select on/off thresholds for one class using fixed candidate statistics and ordering.
    
    Candidate statistics must cover the float32 integer percentage points 5 through 95 exactly. Off thresholds follow max(0.01, on-0.15); no other class thresholds are borrowed and validation data is never read."""

    positive_events = _nonnegative_int64(evaluable_positive_events, "evaluable_positive_events")
    days = _finite_float(exposure_house_days, "exposure_house_days", nonnegative=True)
    if positive_events == 0:
        return {
            "threshold_status": "unavailable",
            "reason": "no_evaluable_positive_events",
            "on_threshold": None,
            "off_threshold": None,
        }
    if days == 0.0:
        return {
            "threshold_status": "unavailable",
            "reason": "zero_evaluable_house_days",
            "on_threshold": None,
            "off_threshold": None,
        }
    candidates = [_normalise_threshold_candidate(_require_mapping(row, "candidate statistics")) for row in candidate_statistics]
    expected_grid = {_float32(percent / 100.0) for percent in range(5, 96)}
    supplied_grid = {float(row["on_threshold"]) for row in candidates}
    if len(candidates) != 91 or supplied_grid != expected_grid:
        raise ContractError("候选统计必须且只能覆盖 5..95 的 91 个整数百分点")
    feasible = [
        row
        for row in candidates
        if float(row["false_alarms_per_evaluable_house_day"]) <= FALSE_ALARM_LIMIT
    ]
    if feasible:
        selected = min(
            feasible,
            key=lambda row: (
                -float(row["event_f2"]),
                -float(row["alarm_recall_at_30s"]),
                -float(row["event_precision"]),
                _p95_sort_key(row["p95_alarm_latency_seconds"]),
                -float(row["on_threshold"]),
            ),
        )
        feasible_flag = True
    else:
        selected = min(
            candidates,
            key=lambda row: (
                float(row["false_alarms_per_evaluable_house_day"]),
                -float(row["event_f2"]),
                -float(row["alarm_recall_at_30s"]),
                -float(row["event_precision"]),
                _p95_sort_key(row["p95_alarm_latency_seconds"]),
                -float(row["on_threshold"]),
            ),
        )
        feasible_flag = False
    on = float(selected["on_threshold"])
    on_percent = round(on * 100.0)
    off = _float32(max(1, on_percent - 15) / 100.0)
    return {
        "threshold_status": "selected",
        "reason": None,
        "on_threshold": on,
        "off_threshold": off,
        "validation_slo_feasible": feasible_flag,
        "on_threshold_float32_hex": _float32_hex(on),
        "off_threshold_float32_hex": _float32_hex(off),
    }


def wilson_95_lower_bound(successes: int, trials: int) -> float | None:
    """Compute the lower endpoint of a two-sided 95% Wilson interval with explicit binary64 operations."""

    k = _nonnegative_int64(successes, "successes")
    n = _nonnegative_int64(trials, "trials")
    if k > n:
        raise ContractError("successes 不得大于 trials")
    if n == 0:
        return None
    # Preserve the specified operation order; do not substitute equivalent algebra or fused multiply-add.
    p = float(k) / float(n)
    z = float(WILSON_Z_95)
    z2 = z * z
    first = p + z2 / (2.0 * float(n))
    square_root_term = math.sqrt(
        p * (1.0 - p) / float(n) + z2 / (4.0 * float(n) * float(n))
    )
    numerator = first - z * square_root_term
    denominator = 1.0 + z2 / float(n)
    return numerator / denominator


def nearest_rank_p95(latencies: Sequence[float]) -> float | None:
    """Return nearest-rank P95 using zero-based ceil(.95*n)-1; return null for an empty population."""

    values = [_finite_float(value, "conservative_alarm_latency_seconds", nonnegative=True) for value in latencies]
    if not values:
        return None
    values.sort()
    rank = (95 * len(values) + 99) // 100 - 1
    return values[rank]


def _house_key(dataset: Any, house_id: Any) -> str:
    return f"{_nfc_text(dataset, 'dataset')}:{_nfc_text(house_id, 'house_id')}"


def deterministic_house_bootstrap(
    canonical_class_id: str,
    per_house_recall: Mapping[str, float],
) -> dict[str, Any]:
    """Perform the frozen 10,000-draw deterministic house-cluster bootstrap.
    
    Sample indices derive entirely from the frozen SHA-256 message. Return index/statistic arrays and their fixed-byte-order hashes for exact synthetic-evidence comparison."""

    class_id = _require_class_id(canonical_class_id)
    if not per_house_recall:
        raise ContractError("bootstrap 至少需要一个 included house")
    normalized: dict[str, float] = {}
    for house, recall in per_house_recall.items():
        key = _nfc_text(house, "dataset:house_id")
        if key in normalized:
            raise ContractError("NFC 归一后 house key 重复")
        value = _finite_float(recall, "per_house_recall", nonnegative=True)
        if value > 1.0:
            raise ContractError("per_house_recall 必须在闭区间 [0,1]")
        normalized[key] = value
    house_order = sorted(normalized, key=_text_key)
    house_count = len(house_order)
    if house_count > 256:
        raise ContractError("纯合成 bootstrap fixture 的 H 不得超过 uint8 可表达范围")
    recalls = [normalized[house] for house in house_order]
    indices: list[list[int]] = []
    statistics: list[float] = []
    index_payload = bytearray()
    stat_payload = bytearray()
    for replicate_index in range(BOOTSTRAP_REPLICATES):
        row: list[int] = []
        total = 0.0
        for draw_position in range(house_count):
            # Preserve the published bootstrap byte prefix and sample sequence.
            message = (
                bytes.fromhex("74636e2d67332d686f7573652d626f6f7473747261702d76317c")
                + f"{class_id}|{replicate_index}|{draw_position}".encode("utf-8")
            )
            selected = int.from_bytes(hashlib.sha256(message).digest()[:8], "big") % house_count
            row.append(selected)
            index_payload.append(selected)
            total += recalls[selected]
        statistic = total / float(house_count)
        indices.append(row)
        statistics.append(statistic)
        stat_payload.extend(struct.pack(">d", statistic))
    sorted_statistics = sorted(statistics)
    sorted_payload = b"".join(struct.pack(">d", value) for value in sorted_statistics)
    return {
        "house_order": house_order,
        "indices": indices,
        "statistics": statistics,
        "lower_bound": sorted_statistics[BOOTSTRAP_LOWER_RANK],
        "indices_sha256": hashlib.sha256(index_payload).hexdigest(),
        "statistics_sha256": hashlib.sha256(stat_payload).hexdigest(),
        "sorted_statistics_sha256": hashlib.sha256(sorted_payload).hexdigest(),
    }


def _normalise_exposure_seconds(value: Mapping[Any, Any] | Sequence[Mapping[str, Any]]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    if isinstance(value, Mapping):
        items = value.items()
        for raw_key, raw_seconds in items:
            if isinstance(raw_key, tuple) and len(raw_key) == 2:
                key = _house_key(raw_key[0], raw_key[1])
            else:
                key = _nfc_text(raw_key, "dataset:house_id")
            if key in normalized:
                raise ContractError("NFC 归一后 exposure house key 重复")
            normalized[key] = _finite_float(raw_seconds, "exposure_seconds", nonnegative=True)
        return normalized
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractError("house_exposure_seconds 必须是对象或记录列表")
    for row in value:
        record = _require_mapping(row, "house exposure")
        _require_fields(record, ("dataset", "house_id", "exposure_seconds"), "house exposure")
        key = _house_key(record["dataset"], record["house_id"])
        if key in normalized:
            raise ContractError("house exposure 不得重复")
        normalized[key] = _finite_float(record["exposure_seconds"], "exposure_seconds", nonnegative=True)
    return normalized


def _count_unmatched_scored_predictions(value: int | Sequence[Mapping[str, Any]]) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return _nonnegative_int64(value, "unmatched_scored_predictions")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractError("unmatched_scored_predictions 必须是非负计数或预测记录列表")
    count = 0
    for raw in value:
        record = _require_mapping(raw, "unmatched prediction")
        if "prediction_scored" not in record:
            raise ContractError("unmatched prediction 必须显式提供 prediction_scored")
        if _binary_flag(record["prediction_scored"], "prediction_scored"):
            count += 1
    return count


def derive_claim_statistics(
    canonical_class_id: str,
    truth_match_rows: Sequence[Mapping[str, Any]],
    house_exposure_seconds: Mapping[Any, Any] | Sequence[Mapping[str, Any]],
    unmatched_scored_predictions: int | Sequence[Mapping[str, Any]] = 0,
) -> dict[str, Any]:
    """Recompute deployment-claim statistics from provided synthetic post-matching rows.
    
    Each truth row contains dataset, house_id and alarm_slo_evaluable. Matched truth supplies finite conservative_alarm_latency_seconds; unmatched truth omits it or uses None. Success requires conservative latency <=30. Unmatched truth remains in the recall denominator but not in the P95 population."""

    class_id = _require_class_id(canonical_class_id)
    exposures = _normalise_exposure_seconds(house_exposure_seconds)
    per_house_total: dict[str, int] = defaultdict(int)
    per_house_successes: dict[str, int] = defaultdict(int)
    matched_latencies: list[float] = []
    successes = 0
    evaluable_events = 0
    for raw in truth_match_rows:
        row = _require_mapping(raw, "truth match row")
        _require_fields(row, ("dataset", "house_id", "alarm_slo_evaluable"), "truth match row")
        key = _house_key(row["dataset"], row["house_id"])
        if key not in exposures:
            raise ContractError("每个 truth match row 的 house 必须有预声明 exposure")
        evaluable = _binary_flag(row["alarm_slo_evaluable"], "alarm_slo_evaluable")
        if not evaluable:
            continue
        evaluable_events += 1
        per_house_total[key] += 1
        latency = _nullable_float(
            row.get("conservative_alarm_latency_seconds"),
            "conservative_alarm_latency_seconds",
            nonnegative=True,
        )
        if latency is not None:
            matched_latencies.append(latency)
            if latency <= ALARM_DEADLINE_SECONDS:
                successes += 1
                per_house_successes[key] += 1

    recall = None if evaluable_events == 0 else float(successes) / float(evaluable_events)
    included_houses = sorted(per_house_total, key=_text_key)
    house_recall = {
        house: float(per_house_successes[house]) / float(per_house_total[house])
        for house in included_houses
    }
    bootstrap: dict[str, Any] | None = None
    if len(included_houses) >= 3:
        bootstrap = deterministic_house_bootstrap(class_id, house_recall)
        bootstrap_lower = bootstrap["lower_bound"]
    else:
        # H < 3 cannot produce apparently valid mod-H sampling evidence.
        bootstrap_lower = None
    total_exposure_seconds = 0.0
    for seconds in exposures.values():
        total_exposure_seconds += seconds
    exposure_days = total_exposure_seconds / 86400.0
    false_positive_count = _count_unmatched_scored_predictions(unmatched_scored_predictions)
    false_alarm_rate = None if exposure_days == 0.0 else float(false_positive_count) / exposure_days
    return {
        "canonical_class_id": class_id,
        "alarm_recall_at_30s": recall,
        "event_wilson_95_lower_bound": wilson_95_lower_bound(successes, evaluable_events),
        "house_cluster_bootstrap_95_lower_bound": bootstrap_lower,
        "p95_alarm_latency_seconds": nearest_rank_p95(matched_latencies),
        "false_alarms_per_house_day": false_alarm_rate,
        "evaluable_event_count": evaluable_events,
        "included_outer_house_count": len(included_houses),
        "success_count": successes,
        "false_positive_count": false_positive_count,
        "scoring_exposure_seconds": total_exposure_seconds,
        "scoring_exposure_house_days": exposure_days,
        "house_order": included_houses,
        "per_house_alarm_recall_at_30s": [house_recall[house] for house in included_houses],
        "bootstrap": bootstrap,
    }


_CLAIM_REQUIRED_FIELDS = (
    "canonical_class_id",
    "alarm_recall_at_30s",
    "event_wilson_95_lower_bound",
    "house_cluster_bootstrap_95_lower_bound",
    "p95_alarm_latency_seconds",
    "false_alarms_per_house_day",
    "evaluable_event_count",
    "included_outer_house_count",
)
_CLAIM_FLOAT_FIELDS = (
    "alarm_recall_at_30s",
    "event_wilson_95_lower_bound",
    "house_cluster_bootstrap_95_lower_bound",
    "p95_alarm_latency_seconds",
    "false_alarms_per_house_day",
)
_PROBABILITY_CLAIM_FIELDS = set(_CLAIM_FLOAT_FIELDS[:3])
_CLAIM_GATES = (
    ("alarm_recall_at_30s", ">=", 0.95),
    ("event_wilson_95_lower_bound", ">=", 0.90),
    ("house_cluster_bootstrap_95_lower_bound", ">=", 0.90),
    ("p95_alarm_latency_seconds", "<=", 30.0),
    ("false_alarms_per_house_day", "<=", 0.1),
    ("evaluable_event_count", ">=", 100),
    ("included_outer_house_count", ">=", 3),
)


def _normalise_claim_input(record: Mapping[str, Any]) -> dict[str, Any]:
    if set(record) != set(_CLAIM_REQUIRED_FIELDS):
        missing = sorted(set(_CLAIM_REQUIRED_FIELDS) - set(record))
        extra = sorted(set(record) - set(_CLAIM_REQUIRED_FIELDS))
        details = []
        if missing:
            details.append(f"缺少 {missing}")
        if extra:
            details.append(f"额外 {extra}")
        raise ContractError("claim comparator 输入字段不精确: " + "; ".join(details))
    normalized: dict[str, Any] = {"canonical_class_id": _require_class_id(record["canonical_class_id"])}
    for field in _CLAIM_FLOAT_FIELDS:
        value = _nullable_float(record[field], field, nonnegative=True)
        if field in _PROBABILITY_CLAIM_FIELDS and value is not None and value > 1.0:
            raise ContractError(f"{field} 必须在闭区间 [0,1]")
        normalized[field] = value
    normalized["evaluable_event_count"] = _nonnegative_int64(
        record["evaluable_event_count"], "evaluable_event_count"
    )
    normalized["included_outer_house_count"] = _nonnegative_int64(
        record["included_outer_house_count"], "included_outer_house_count"
    )
    return normalized


def reduce_class_claim(claim_statistics: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate all seven acceptance criteria for one primary class without short-circuiting."""

    row = _normalise_claim_input(_require_mapping(claim_statistics, "claim statistics"))
    gate_results: list[dict[str, Any]] = []
    for gate_name, operator, required_value in _CLAIM_GATES:
        observed = row[gate_name]
        if observed is None:
            status = "FAIL"
        elif operator == ">=":
            status = "PASS" if observed >= required_value else "FAIL"
        else:
            status = "PASS" if observed <= required_value else "FAIL"
        gate_results.append(
            {
                "gate_name": gate_name,
                "observed": observed,
                "required_operator": operator,
                "required_value": required_value,
                "status": status,
            }
        )
    class_claim = all(result["status"] == "PASS" for result in gate_results)
    return {
        "canonical_class_id": row["canonical_class_id"],
        "gate_results_in_order": gate_results,
        "class_claim": class_claim,
        "failure_wording": None if class_claim else FAILURE_WORDING,
    }


def reduce_overall_claim(class_claim_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute overall AND only for complete, unique results in the frozen primary-class order."""

    received: dict[str, bool] = {}
    for raw in class_claim_records:
        row = _require_mapping(raw, "class claim record")
        permitted_keys = {"canonical_class_id", "class_claim"}
        full_reducer_keys = {
            "canonical_class_id",
            "gate_results_in_order",
            "class_claim",
            "failure_wording",
        }
        if set(row) not in (permitted_keys, full_reducer_keys):
            raise ContractError("overall reducer 只接受 class claim record，不能混入 exploratory 或未知字段")
        class_id = _require_class_id(row.get("canonical_class_id"))
        if class_id not in PRIMARY_CLASS_ORDER:
            raise ContractError("low-resource exploratory 类不得进入 primary overall reducer")
        if class_id in received:
            raise ContractError("每个 primary class 必须恰好有一行")
        if not isinstance(row.get("class_claim"), bool):
            raise ContractError("class_claim 必须是 bool")
        received[class_id] = row["class_claim"]
    if tuple(received) != PRIMARY_CLASS_ORDER or len(received) != len(PRIMARY_CLASS_ORDER):
        raise ContractError("overall reducer 必须接收冻结顺序中每个 primary class 的恰好一行")
    primary_claims = [received[class_id] for class_id in PRIMARY_CLASS_ORDER]
    overall = all(primary_claims)
    return {
        "primary_class_claims_in_class_order": primary_claims,
        "overall_claim": overall,
        "failure_wording": None if overall else FAILURE_WORDING,
    }


def report_exploratory_claims(class_claim_records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Keep low-resource results separate so they cannot affect the primary-class overall claim."""

    records: dict[str, Mapping[str, Any]] = {}
    for raw in class_claim_records:
        row = _require_mapping(raw, "exploratory claim record")
        class_id = _require_class_id(row.get("canonical_class_id"))
        if class_id not in LOW_RESOURCE_CLASS_ORDER:
            raise ContractError("exploratory reporter 只接受冻结 low-resource 类")
        if class_id in records:
            raise ContractError("exploratory 类不得重复")
        records[class_id] = row
    return [records[class_id] for class_id in LOW_RESOURCE_CLASS_ORDER if class_id in records]


__all__ = [
    "ALARM_DEADLINE_SECONDS",
    "BIN_SECONDS",
    "BOOTSTRAP_REPLICATES",
    "CLASS_ORDER",
    "ContractError",
    "FAILURE_WORDING",
    "LOW_RESOURCE_CLASS_ORDER",
    "PRIMARY_CLASS_ORDER",
    "alarm_success_at_deadline",
    "classify_prediction_scoring",
    "compute_scoring_exposure",
    "derive_claim_statistics",
    "deterministic_house_bootstrap",
    "evaluate_matched_pair",
    "match_events",
    "nearest_rank_p95",
    "reduce_class_claim",
    "reduce_overall_claim",
    "report_exploratory_claims",
    "select_threshold",
    "summarize_unmatched_predictions",
    "wilson_95_lower_bound",
]
