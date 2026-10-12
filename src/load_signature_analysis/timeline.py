'Causal time grids, quality masks and appliance labels. The six-class, ten-second numerical definition is preserved; this module performs no raw-data discovery or model fitting.'

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
import math
from pathlib import Path
import unicodedata
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np


CLASS_ORDER = (
    "kettle", "toaster", "microwave", "hair_dryer", "iron", "rice_cooker",
)
CLASS_INDEX = {name: index for index, name in enumerate(CLASS_ORDER)}
BIN_SECONDS = 10
CONTEXT_BINS = 64
MAX_RAW_DEPENDENCY_BINS = 70
REQUIRED_ARRAYS = (
    "grid_index", "bin_left_unix_s", "bin_right_unix_s", "continuity_id",
    "input_values", "input_value_mask", "input_quality_flags", "raw_sample_count",
    "maximum_raw_gap_s", "active_target", "active_valid_mask", "onset_target",
    "onset_valid_mask", "offset_target", "offset_valid_mask", "label_epoch_id",
)
GROUND_TRUTH_HEADER = (
    "true_event_id", "dataset", "house_id", "continuity_id", "label_epoch_id",
    "appliance_class", "true_start_time", "previous_reliable_non_on_timestamp",
    "true_start_lower_bound", "true_start_upper_bound", "true_end_time",
    "onset_uncertainty_seconds", "end_reason", "right_censored",
    "input_quality_evaluable", "earliest_quality_eligible_report_time",
    "label_timing_evaluable", "alarm_slo_evaluable", "source_channel_ids_json",
)
GROUND_TRUTH_TYPES = (
    "utf8_string", "utf8_string", "utf8_string", "int32", "int32", "utf8_string",
    "float64_unix_seconds", "nullable_float64_unix_seconds", "float64_unix_seconds",
    "float64_unix_seconds", "float64_unix_seconds", "float64", "utf8_string", "bool",
    "bool", "nullable_float64_unix_seconds", "bool", "bool", "utf8_string",
)
PROVENANCE_HEADER = (
    "dataset", "house_id", "appliance_class", "class_index", "label_epoch_id",
    "start_unix_s", "end_unix_s", "source_channel_id", "source_label",
    "canonical_label", "supervision_role", "trust_tier", "mapping_status",
    "source_descriptor_sha256",
)
PROVENANCE_TYPES = (
    "utf8_string", "utf8_string", "utf8_string", "int32", "int32",
    "float64_unix_seconds", "nullable_float64_unix_seconds", "utf8_string",
    "utf8_string", "utf8_string", "utf8_string", "utf8_string", "utf8_string",
    "sha256_lowercase_hex",
)
THRESHOLDS: dict[str, tuple[float, float, float, float, float]] = {
    # ON, OFF, minimum activity, short OFF merge, maximum native sample gap.
    "kettle": (1000.0, 100.0, 20.0, 15.0, 30.0),
    "toaster": (500.0, 100.0, 20.0, 15.0, 30.0),
    "microwave": (800.0, 100.0, 20.0, 15.0, 30.0),
    "hair_dryer": (300.0, 50.0, 20.0, 15.0, 30.0),
    "iron": (300.0, 50.0, 20.0, 15.0, 30.0),
    "rice_cooker": (20.0, 5.0, 60.0, 15.0, 30.0),
}


class TimelineContractError(ValueError):
    """A synthetic fixture or its logical descriptor violates the frozen timeline definition."""


@dataclass(frozen=True)
class AggregateSample:
    """One raw aggregate P-only record; arrival_ordinal determines quarantine order."""

    arrival_ordinal: int
    timestamp_unix_s: float
    active_power_w: float
    apparent_power_va: float | None = None
    refit_issues: int | None = 0
    source_file_id: str = "agg_a"
    timestamp_parseable: bool = True
    value_parseable: bool = True


@dataclass(frozen=True)
class SubmeterSample:
    """One dedicated submeter record, used only to construct synthetic targets and never as input."""

    arrival_ordinal: int
    timestamp_unix_s: float
    power_w: float
    source_channel_id: str
    canonical_class_id: str
    record_valid: bool = True
    source_file_id: str = "submeter_a"
    channel_label_epoch_id: int = 0
    supervision_role: str = "core_positive"
    trust_tier: str = "high"
    mapping_status: str = "resolved"
    source_descriptor_sha256: str | None = None


@dataclass(frozen=True)
class LabelDescriptor:
    """Valid supervision coverage for a submeter channel, with an exclusive right edge."""

    source_channel_id: str
    canonical_class_id: str
    valid_from_unix_s: float
    valid_to_unix_s: float
    channel_label_epoch_id: int = 0
    supervision_role: str = "core_positive"
    trust_tier: str = "high"
    mapping_status: str = "resolved"
    source_descriptor_sha256: str | None = None
    source_label: str | None = None


@dataclass
class ConflictAudit:
    """Immutable audit summary of arrival-order high-water quarantine."""

    discarded_count: int = 0
    discarded_min_timestamp: float | None = None
    discarded_max_timestamp: float | None = None
    first_arrival_ordinal: int | None = None
    last_arrival_ordinal: int | None = None
    resolved: bool = True

    def discard(self, timestamp: float, ordinal: int) -> None:
        self.discarded_count += 1
        self.discarded_min_timestamp = timestamp if self.discarded_min_timestamp is None else min(self.discarded_min_timestamp, timestamp)
        self.discarded_max_timestamp = timestamp if self.discarded_max_timestamp is None else max(self.discarded_max_timestamp, timestamp)
        self.first_arrival_ordinal = ordinal if self.first_arrival_ordinal is None else min(self.first_arrival_ordinal, ordinal)
        self.last_arrival_ordinal = ordinal if self.last_arrival_ordinal is None else max(self.last_arrival_ordinal, ordinal)

    def as_dict(self) -> dict[str, Any]:
        return {
            "discarded_count": self.discarded_count,
            "discarded_min_timestamp": self.discarded_min_timestamp,
            "discarded_max_timestamp": self.discarded_max_timestamp,
            "arrival_ordinal_range": None if self.first_arrival_ordinal is None else [self.first_arrival_ordinal, self.last_arrival_ordinal],
            "resolved": self.resolved,
        }


@dataclass(frozen=True)
class NormalizerParameters:
    """Three-channel standardizer parameters from inner-training observed values only."""

    mean: np.ndarray
    scale: np.ndarray
    observed_count: np.ndarray
    supported: np.ndarray
    variance_epsilon: float = 1e-12

    @property
    def schema_version(self) -> str:
        return "masked_population_standardizer_parameters_v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "value_channel_order": ["aggregate_active_power_w", "delta_active_power_w", "causal_baseline_delta_w"],
            "variance_epsilon": self.variance_epsilon,
            "ddof": 0,
            "mean": [_json_number(value) for value in self.mean],
            "scale": [_json_number(value) for value in self.scale],
            "observed_count": [int(value) for value in self.observed_count],
            "supported": [bool(value) for value in self.supported],
        }


@dataclass
class Timeline:
    """An in-memory logical representation of a continuous timeline, without public-data I/O."""

    dataset: str
    house_id: str
    source_start_unix_s: float
    source_end_unix_s: float
    arrays: dict[str, np.ndarray]
    ground_truth_events: list[dict[str, Any]] = field(default_factory=list)
    label_provenance: list[dict[str, Any]] = field(default_factory=list)
    aggregate_conflict_audit: ConflictAudit = field(default_factory=ConflictAudit)
    submeter_conflict_audit: dict[str, ConflictAudit] = field(default_factory=dict)
    partial_bin_audit: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        """Support readable, test-friendly access such as ``timeline.input_values``."""
        if name in self.arrays:
            return self.arrays[name]
        raise AttributeError(name)

    @property
    def row_count(self) -> int:
        return int(self.arrays["grid_index"].shape[0])

    @property
    def model_eligible(self) -> np.ndarray:
        """A partial tail can never be a training, inference or alarm endpoint."""
        flags = self.arrays["input_quality_flags"]
        return (flags & np.uint16((1 << 5))) == 0

    def logical_descriptors(self, shard_rows: int = 262_144) -> dict[str, Any]:
        """Return frozen D08 arrays, typed records, shards and the timeline logical descriptor."""
        if not isinstance(shard_rows, int) or shard_rows <= 0:
            raise TimelineContractError("shard_rows 必须为正 int")
        array_descriptors = [array_logical_descriptor(name, index, self.arrays[name]) for index, name in enumerate(REQUIRED_ARRAYS)]
        shards: list[dict[str, Any]] = []
        for shard_index, left in enumerate(range(0, self.row_count, shard_rows)):
            right = min(left + shard_rows, self.row_count)
            part = [array_logical_descriptor(name, index, self.arrays[name][left:right]) for index, name in enumerate(REQUIRED_ARRAYS)]
            descriptor = {
                "schema_version": "timeline_shard_logical_content_v1",
                "shard_index": shard_index,
                "row_range": [left, right],
                "array_descriptors": part,
            }
            shards.append({**descriptor, "shard_logical_content_sha256": _sha256_jcs(descriptor)})
        typed = [
            typed_record_logical_descriptor("ground_truth_events", self.ground_truth_events),
            typed_record_logical_descriptor("label_provenance", self.label_provenance),
        ]
        descriptor = {
            "schema_version": "timeline_logical_content_v1",
            "shard_logical_content_sha256_in_shard_order": [item["shard_logical_content_sha256"] for item in shards],
            "typed_record_descriptors": typed,
        }
        return {
            "array_descriptors": array_descriptors,
            "shard_descriptors": shards,
            "typed_record_descriptors": typed,
            "timeline_logical_content_descriptor": descriptor,
            "timeline_logical_content_sha256": _sha256_jcs(descriptor),
        }

    def model_features(self, end_index: int, normalizer: NormalizerParameters | None = None, *, allow_left_padding: bool = False) -> np.ndarray:
        """Materialize a [C=6,T=64] view; training rejects insufficient real context or windows crossing continuity."""
        return materialize_model_window(self, end_index, normalizer, allow_left_padding=allow_left_padding)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TimelineContractError(message)


def _mapping_value(value: Any, *names: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) and math.isfinite(float(value))


def _as_float(value: Any, label: str) -> float:
    _require(_finite(value), f"{label} 必须是有限数")
    return float(value)


def _as_int(value: Any, label: str) -> int:
    _require(isinstance(value, (int, np.integer)) and not isinstance(value, bool), f"{label} 必须是整数")
    return int(value)


def _nfc(value: str) -> str:
    _require(isinstance(value, str), "字符串字段必须为 str")
    return unicodedata.normalize("NFC", value)


def _json_number(value: Any) -> int | float:
    """Normalize IEEE -0 and serialize integer timestamps without a .0 suffix under RFC8785."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimelineContractError("JCS number 类型非法")
    try:
        numeric = float(value)
    except OverflowError as exc:
        raise TimelineContractError("JCS number 超出 IEEE-754 范围") from exc
    if not math.isfinite(numeric):
        raise TimelineContractError("JCS 禁止 NaN/Inf")
    if numeric == 0:
        return 0
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _jcs_normalize(value: Any) -> Any:
    """Validate I-JSON values for JCS without modifying Unicode content.
    
    NFC belongs to the typed-record definition, not RFC8785. Silent normalization in a general JCS serializer would give different JSON values the same byte representation."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise TimelineContractError("JCS 字符串含未配对 surrogate") from exc
        return value
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return _json_number(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            _require(isinstance(key, str), "JCS object key 必须为 str")
            # Python mappings discard duplicate keys; external JSON parsing checks them with a separate pair hook.
            result[_jcs_normalize(key)] = _jcs_normalize(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_jcs_normalize(item) for item in value]
    raise TimelineContractError(f"不能 JCS 序列化 {type(value).__name__}")


def _jcs_number_text(value: int | float) -> str:
    'ECMAScript/RFC8785 number serialization for an IEEE-754 value.'
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimelineContractError("JCS number 类型非法")
    try:
        binary64 = float(value)
    except OverflowError as exc:
        raise TimelineContractError("JCS number 超出 IEEE-754 范围") from exc
    if isinstance(value, int):
        # JCS consumers use binary64 semantics.  Accept only an integer that survives
        # that conversion, instead of silently hashing a Python-only arbitrary integer.
        _require(math.isfinite(binary64) and int(binary64) == value, "JCS 整数不可由 IEEE-754 精确表示")
    _require(math.isfinite(binary64), "JCS 禁止 NaN/Inf")
    if binary64 == 0.0:
        return "0"

    # ``repr`` is CPython's shortest round-trippable binary64 representation.  RFC8785
    # uses the same shortest-number rule, but has ECMAScript's fixed/scientific boundary
    # and exponent spelling (not Python's ``e-07`` spelling).
    text = repr(binary64).lower()
    if "e" not in text:
        return text[:-2] if text.endswith(".0") else text

    coefficient, exponent_text = text.split("e", 1)
    exponent = int(exponent_text)
    negative = coefficient.startswith("-")
    magnitude = coefficient[1:] if negative else coefficient
    digits = magnitude.replace(".", "")
    decimal_at = (magnitude.find(".") if "." in magnitude else len(magnitude)) + exponent
    sign = "-" if negative else ""
    absolute = abs(binary64)
    if 1e-6 <= absolute < 1e21:
        if decimal_at <= 0:
            return sign + "0." + ("0" * -decimal_at) + digits
        if decimal_at >= len(digits):
            return sign + digits + ("0" * (decimal_at - len(digits)))
        return sign + digits[:decimal_at] + "." + digits[decimal_at:]

    # Retain scientific notation outside the ECMAScript fixed-range, removing only
    # Python's exponent zero-padding.  ``repr`` already supplies the shortest mantissa.
    magnitude = magnitude[:-2] if magnitude.endswith(".0") else magnitude
    exponent_sign = "+" if exponent >= 0 else "-"
    return sign + magnitude + "e" + exponent_sign + str(abs(exponent))


def _jcs_string(value: str) -> str:
    # json.dumps supplies RFC8259's shortest required escapes; _jcs_normalize has
    # already rejected surrogate code points so UTF-8 output is deterministic.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _jcs_encode(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _jcs_string(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _jcs_number_text(value)
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: item[0].encode("utf-16-be"))
        return "{" + ",".join(_jcs_string(key) + ":" + _jcs_encode(item) for key, item in items) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_jcs_encode(item) for item in value) + "]"
    raise TimelineContractError(f"不能 JCS 序列化 {type(value).__name__}")


def jcs_bytes(value: Any) -> bytes:
    """Serialize RFC8785/JCS UTF-8 with no BOM or trailing newline."""
    try:
        return _jcs_encode(_jcs_normalize(value)).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TimelineContractError("JCS 序列化失败") from exc


def _sha256_jcs(value: Any) -> str:
    return hashlib.sha256(jcs_bytes(value)).hexdigest()


def _default_descriptor_sha(channel: str, epoch: int = 0) -> str:
    return hashlib.sha256(f"synthetic-source:{channel}:{epoch}".encode("utf-8")).hexdigest()


def _coerce_aggregate(item: AggregateSample | Mapping[str, Any], fallback_ordinal: int) -> dict[str, Any]:
    return {
        "arrival": _as_int(_mapping_value(item, "arrival_ordinal", "arrival", default=fallback_ordinal), "aggregate arrival_ordinal"),
        "timestamp": _mapping_value(item, "timestamp_unix_s", "timestamp", "time"),
        "power": _mapping_value(item, "active_power_w", "power_w", "power"),
        "apparent": _mapping_value(item, "apparent_power_va", "apparent", default=None),
        "issues": _mapping_value(item, "refit_issues", "issues", default=0),
        "source": _nfc(str(_mapping_value(item, "source_file_id", "source_id", default="agg_a"))),
        "timestamp_parseable": bool(_mapping_value(item, "timestamp_parseable", default=True)),
        "value_parseable": bool(_mapping_value(item, "value_parseable", default=True)),
    }


def _coerce_submeter(item: SubmeterSample | Mapping[str, Any], fallback_ordinal: int) -> dict[str, Any]:
    channel = _mapping_value(item, "source_channel_id", "channel_id", "channel")
    klass = _mapping_value(item, "canonical_class_id", "appliance_class", "class_id")
    _require(isinstance(channel, str) and channel, "分表 source_channel_id 缺失")
    _require(klass in CLASS_INDEX, "分表 canonical_class_id 不在 frozen class axis")
    epoch = _as_int(_mapping_value(item, "channel_label_epoch_id", "epoch_id", default=0), "channel_label_epoch_id")
    return {
        "arrival": _as_int(_mapping_value(item, "arrival_ordinal", "arrival", default=fallback_ordinal), "分表 arrival_ordinal"),
        "timestamp": _mapping_value(item, "timestamp_unix_s", "timestamp", "time"),
        "power": _mapping_value(item, "power_w", "power"),
        "channel": _nfc(channel), "class": _nfc(klass),
        "valid": bool(_mapping_value(item, "record_valid", "valid", default=True)),
        "source": _nfc(str(_mapping_value(item, "source_file_id", "source_id", default="submeter_a"))),
        "epoch": epoch,
        "supervision_role": _nfc(str(_mapping_value(item, "supervision_role", default="core_positive"))),
        "trust_tier": _nfc(str(_mapping_value(item, "trust_tier", default="high"))),
        "mapping_status": _nfc(str(_mapping_value(item, "mapping_status", default="resolved"))),
        "sha": _mapping_value(item, "source_descriptor_sha256", "descriptor_sha", default=None),
    }


def _coerce_descriptor(item: LabelDescriptor | Mapping[str, Any]) -> dict[str, Any]:
    channel = _mapping_value(item, "source_channel_id", "channel_id", "channel")
    klass = _mapping_value(item, "canonical_class_id", "appliance_class", "class_id")
    _require(isinstance(channel, str) and channel and klass in CLASS_INDEX, "label descriptor channel/class 非法")
    start = _as_float(_mapping_value(item, "valid_from_unix_s", "valid_from", "start_unix_s"), "descriptor valid_from")
    end = _as_float(_mapping_value(item, "valid_to_unix_s", "valid_to", "end_unix_s"), "descriptor valid_to")
    _require(start < end, "label descriptor 必须是非空半开区间")
    epoch = _as_int(_mapping_value(item, "channel_label_epoch_id", "epoch_id", default=0), "descriptor epoch")
    sha = _mapping_value(item, "source_descriptor_sha256", "descriptor_sha", default=None)
    if sha is None:
        sha = _default_descriptor_sha(str(channel), epoch)
    _require(isinstance(sha, str) and len(sha) == 64 and all(ch in "0123456789abcdef" for ch in sha), "source_descriptor_sha256 必须为小写 SHA-256")
    return {
        "channel": _nfc(channel), "class": _nfc(klass), "start": start, "end": end,
        "epoch": epoch,
        "role": _nfc(str(_mapping_value(item, "supervision_role", default="core_positive"))),
        "tier": _nfc(str(_mapping_value(item, "trust_tier", default="high"))),
        "mapping": _nfc(str(_mapping_value(item, "mapping_status", default="resolved"))),
        "sha": sha, "source_label": _nfc(str(_mapping_value(item, "source_label", default=channel))),
    }


def _arrival_quarantine(records: Sequence[dict[str, Any]], *, timestamp_key: str = "timestamp") -> tuple[list[dict[str, Any]], ConflictAudit]:
    """Audit strictly in arrival order; the recovery record itself starts new continuity."""
    audit = ConflictAudit()
    accepted: list[dict[str, Any]] = []
    high_water: float | None = None
    quarantined = False
    for record in sorted(records, key=lambda item: item["arrival"]):
        timestamp = record.get(timestamp_key)
        if not _finite(timestamp):
            # Incomparable timestamps cannot be assigned to bins or used as recovery records.
            audit.discarded_count += 1
            audit.resolved = False
            continue
        timestamp = float(timestamp)
        if high_water is None:
            high_water = timestamp
            accepted.append({**record, "timestamp": timestamp, "conflict_recovery": False})
            continue
        if timestamp <= high_water:
            audit.discard(timestamp, record["arrival"])
            quarantined = True
            audit.resolved = False
            continue
        # Update high-water on normal progression and strict recovery; only recovery carries bit4.
        high_water = timestamp
        recovered = quarantined
        if recovered:
            quarantined = False
            audit.resolved = True
        accepted.append({**record, "timestamp": timestamp, "conflict_recovery": recovered})
    return accepted, audit


def _grid_start(source_start: float, source_end: float) -> tuple[float, int, bool]:
    _require(source_start <= source_end, "source_start 不得晚于 source_end")
    _require(source_start % BIN_SECONDS == 0, "source_start 必须位于 UTC 十秒格边界")
    delta = source_end - source_start
    whole = int(math.floor(delta / BIN_SECONDS))
    exact = math.isclose(delta, whole * BIN_SECONDS, rel_tol=0.0, abs_tol=0.0)
    return source_start, whole, not exact


def _aggregate_arrays(
    records: Sequence[dict[str, Any],], *, dataset: str, source_start: float, source_end: float,
) -> tuple[dict[str, np.ndarray], ConflictAudit, dict[str, Any]]:
    """Aggregate P-only bins, quality, continuity and three causal feature channels."""
    accepted, audit = _arrival_quarantine(records)
    _, complete_rows, partial = _grid_start(source_start, source_end)
    rows = complete_rows + int(partial)
    values = np.zeros((rows, 3), dtype=np.float32)
    masks = np.zeros((rows, 3), dtype=np.uint8)
    flags = np.zeros(rows, dtype=np.uint16)
    count = np.zeros(rows, dtype=np.uint16)
    max_gap = np.zeros(rows, dtype=np.float32)
    continuity = np.zeros(rows, dtype=np.int32)
    grouped: list[list[dict[str, Any]]] = [[] for _ in range(rows)]
    usable: list[dict[str, Any]] = []
    for record in accepted:
        timestamp = record["timestamp"]
        if timestamp < source_start or timestamp >= source_end:
            continue
        index = int(math.floor((timestamp - source_start) / BIN_SECONDS))
        if 0 <= index < rows:
            grouped[index].append(record)
            usable.append(record)
    # Derive breaks from time-ordered accepted records; isolated missing bins deliberately preserve continuity.
    sorted_records = sorted(usable, key=lambda item: item["timestamp"])
    break_at: dict[int, int] = {}
    previous: dict[str, Any] | None = None
    current_continuity = 0
    for record in sorted_records:
        index = int(math.floor((record["timestamp"] - source_start) / BIN_SECONDS))
        is_break = False
        bit = 0
        if previous is not None:
            if record["source"] != previous["source"]:
                is_break, bit = True, 7
            elif record["conflict_recovery"]:
                is_break, bit = True, 4
            elif record["timestamp"] - previous["timestamp"] > 30.0:
                is_break, bit = True, 6
        elif record["conflict_recovery"]:
            is_break, bit = True, 4
        if is_break:
            current_continuity += 1
            break_at[index] = break_at.get(index, 0) | (1 << bit)
        record["_raw_gap"] = float(record["timestamp"] - previous["timestamp"]) if previous is not None and record["source"] == previous["source"] else 0.0
        record["_continuity"] = current_continuity
        previous = record
    current = 0
    for index in range(rows):
        if index in break_at:
            current += 1
        continuity[index] = current
    if sorted_records and int(sorted_records[0].get("_continuity", 0)) != 0:
        continuity += int(sorted_records[0]["_continuity"])
    for index, bucket in enumerate(grouped):
        if index == complete_rows and partial:
            # The audited tail count excludes quarantine; its other fields remain unknown.
            count[index] = np.uint16(min(len([r for r in bucket if r["timestamp_parseable"] and _finite(r["timestamp"])]), 65535))
            if len(bucket) > 65535:
                raise TimelineContractError("raw_sample_count 超出 uint16")
            flags[index] = np.uint16((1 << 5) | (1 << 7))
            continue
        if not bucket:
            flags[index] |= np.uint16(1 << 0)
            continue
        parseable = [record for record in bucket if record["timestamp_parseable"] and record["value_parseable"] and _finite(record["timestamp"]) and _finite(record["power"])]
        if len(parseable) > 65535:
            raise TimelineContractError("raw_sample_count 超出 uint16")
        count[index] = np.uint16(len(parseable))
        invalid_bits = 0
        valid_powers: list[float] = []
        for record in parseable:
            power = float(record["power"])
            if dataset == "refit":
                if record["issues"] in (1, True):
                    invalid_bits |= 1 << 1
                if power <= 0.0:
                    invalid_bits |= 1 << 2
            elif dataset == "uk_dale":
                apparent = record["apparent"]
                if power == 0.0 and _finite(apparent) and abs(float(apparent)) > 50.0:
                    invalid_bits |= 1 << 3
            else:
                raise TimelineContractError("dataset 只允许 refit 或 uk_dale 的纯合成 fixture")
            valid_powers.append(power)
        if not parseable:
            invalid_bits |= 1 << 0
        if index in break_at:
            invalid_bits |= break_at[index]
        # Bits4/6/7 describe continuity provenance, not a quality veto on P itself; a vetoed
        # raw record (bits0..3) invalidates the whole bin even if the bin also contains a good record.
        flags[index] |= np.uint16(invalid_bits)
        if (invalid_bits & 0x000F) == 0 and valid_powers:
            median = np.median(np.asarray(valid_powers, dtype=np.float64))
            values[index, 0] = np.float32(median)
            masks[index, 0] = 1
        # Compare only temporally adjacent accepted records from the same source after arrival quarantine; earlier
        # records from another file must not hide a file boundary.
        if bucket:
            max_gap[index] = np.float32(max(float(record.get("_raw_gap", 0.0)) for record in bucket))
    # A terminal partial bin retains continuity but cannot produce a derived value.
    for index in range(1, complete_rows):
        if masks[index, 0] and masks[index - 1, 0] and continuity[index] == continuity[index - 1]:
            values[index, 1] = np.float32(float(values[index, 0]) - float(values[index - 1, 0]))
            masks[index, 1] = 1
        left = max(0, index - 6)
        history = [float(values[prior, 0]) for prior in range(left, index) if masks[prior, 0] and continuity[prior] == continuity[index]]
        # The first baseline still requires all six historical bin positions; only afterward may an isolated
        # invalid bin be skipped (for example D01 B9), satisfying both I03 B6 and D01 B0..B5.
        if masks[index, 0] and index >= 6 and len(history) >= 5:
            baseline = np.median(np.asarray(history, dtype=np.float64))
            values[index, 2] = np.float32(float(values[index, 0]) - float(baseline))
            masks[index, 2] = 1
    arrays = {
        "grid_index": np.arange(rows, dtype=np.int64),
        "bin_left_unix_s": np.asarray([source_start + BIN_SECONDS * index for index in range(rows)], dtype=np.int64),
        "bin_right_unix_s": np.asarray([source_start + BIN_SECONDS * (index + 1) for index in range(rows)], dtype=np.int64),
        "continuity_id": continuity,
        "input_values": values,
        "input_value_mask": masks,
        "input_quality_flags": flags,
        "raw_sample_count": count,
        "maximum_raw_gap_s": max_gap,
    }
    tail = {"present": bool(partial), "source_end_unix_s": source_end, "row_index": complete_rows if partial else None}
    return arrays, audit, tail


def _descriptor_identity(descriptor: Mapping[str, Any]) -> tuple[Any, ...]:
    return (descriptor["channel"], descriptor["sha"], descriptor["epoch"], descriptor["class"], descriptor["role"], descriptor["tier"], descriptor["mapping"])


def _descriptor_at(descriptors: Sequence[dict[str, Any]], timestamp: float) -> dict[str, Any] | None:
    matched = [item for item in descriptors if item["start"] <= timestamp < item["end"]]
    if len(matched) > 1:
        raise TimelineContractError("一个 channel 同时有重叠 label descriptor")
    return matched[0] if matched else None


def _channel_label_processing(
    channel: str, klass: str, raw_records: Sequence[dict[str, Any]], descriptors: Sequence[dict[str, Any]], source_end: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], ConflictAudit]:
    """Apply native hysteresis, then merge, then duration filtering; return grid-query intervals and accepted channel events."""
    accepted, audit = _arrival_quarantine(raw_records)
    accepted = sorted(accepted, key=lambda item: item["timestamp"])
    on_threshold, off_threshold, minimum, merge_gap, max_gap = THRESHOLDS[klass]
    prepared: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    continuity = 0
    previous_state: int | None = None
    for record in accepted:
        timestamp = record["timestamp"]
        if not record["valid"] or not _finite(record["power"]):
            previous_state = None
            previous = None
            continue
        descriptor = _descriptor_at(descriptors, timestamp)
        witness = False
        if descriptor is None and previous is not None:
            candidate = previous["descriptor"]
            if timestamp == candidate["end"] and _descriptor_identity(candidate) == (
                channel, record["sha"] or candidate["sha"], record["epoch"], record["class"], record["supervision_role"], record["trust_tier"], record["mapping_status"]
            ) and record["source"] == previous["source"] and timestamp - previous["timestamp"] <= max_gap:
                witness, descriptor = True, candidate
        if descriptor is None:
            previous_state = None
            previous = None
            continue
        new_continuity = previous is None or record["conflict_recovery"] or record["source"] != previous["source"] or timestamp - previous["timestamp"] > max_gap or _descriptor_identity(descriptor) != _descriptor_identity(previous["descriptor"])
        if new_continuity and previous is not None:
            continuity += 1
            previous_state = None
        if witness:
            prepared.append({**record, "descriptor": descriptor, "continuity": continuity, "witness": True, "state": None})
            previous = prepared[-1]
            continue
        power = float(record["power"])
        if power >= on_threshold:
            state = 1 if previous_state is not None else None
        elif power <= off_threshold:
            state = 0
        else:
            state = previous_state
        prepared.append({**record, "descriptor": descriptor, "continuity": continuity, "witness": False, "state": state, "previous_state": previous_state})
        previous, previous_state = prepared[-1], state
    base: list[dict[str, Any]] = []
    for index, record in enumerate(prepared[:-1]):
        next_record = prepared[index + 1]
        if record["witness"] or record["state"] is None:
            continue
        same = record["continuity"] == next_record["continuity"] and next_record["timestamp"] - record["timestamp"] <= max_gap
        if next_record["witness"]:
            same = same and next_record["timestamp"] == record["descriptor"]["end"] and _descriptor_identity(record["descriptor"]) == _descriptor_identity(next_record["descriptor"])
        else:
            same = same and _descriptor_identity(record["descriptor"]) == _descriptor_identity(next_record["descriptor"])
        if same:
            base.append({"start": record["timestamp"], "end": next_record["timestamp"], "state": record["state"], "record": record, "right_censored": bool(next_record["witness"])})
    # Find an ON candidate first; merge only an explicit short OFF interval bounded by ON at both ends.
    candidates: list[dict[str, Any]] = []
    position = 0
    while position < len(base):
        if base[position]["state"] != 1:
            position += 1
            continue
        start = base[position]["start"]
        end = base[position]["end"]
        members = [position]
        cursor = position
        while cursor + 1 < len(base):
            # Consecutive native ON intervals are already one candidate and need no OFF merge.
            next_interval = base[cursor + 1]
            if next_interval["state"] == 1 and next_interval["start"] == end:
                members.append(cursor + 1)
                end, cursor = next_interval["end"], cursor + 1
                continue
            if cursor + 2 >= len(base):
                break
            off, next_on = base[cursor + 1], base[cursor + 2]
            if off["state"] == 0 and next_on["state"] == 1 and off["start"] == end and off["end"] - off["start"] <= merge_gap and next_on["start"] == off["end"]:
                members.extend([cursor + 1, cursor + 2])
                end, cursor = next_on["end"], cursor + 2
            else:
                break
        candidates.append({"start": start, "end": end, "members": members, "accepted": end - start >= minimum, "right_censored": base[members[-1]]["right_censored"]})
        position = cursor + 1
    accepted_spans: list[dict[str, Any]] = []
    blocked: list[tuple[float, float]] = []
    for candidate in candidates:
        if candidate["accepted"]:
            first = base[candidate["members"][0]]["record"]
            accepted_spans.append({
                "start": candidate["start"], "end": candidate["end"], "right_censored": candidate["right_censored"],
                "censor_kind": "right_censored_label_epoch_end" if candidate["end"] < source_end else "right_censored_input_coverage_end",
                "previous_off": first.get("previous_timestamp_if_off"), "on_record": first,
            })
        else:
            blocked.append((candidate["start"], candidate["end"]))
    # The predecessor is the immediately preceding reliable OFF record in the same epoch and continuity.
    for span in accepted_spans:
        on_record = span["on_record"]
        predecessor = None
        for record in reversed(prepared):
            if record["timestamp"] >= span["start"]:
                continue
            if record["continuity"] == on_record["continuity"] and not record["witness"] and record["state"] == 0 and _descriptor_identity(record["descriptor"]) == _descriptor_identity(on_record["descriptor"]):
                predecessor = record["timestamp"]
                break
        span["previous_off"] = predecessor
    intervals: list[dict[str, Any]] = []
    for unit in base:
        state = "off" if unit["state"] == 0 else "unknown"
        for span in accepted_spans:
            if span["start"] <= unit["start"] and unit["end"] <= span["end"]:
                state = "on"
                break
        if state != "on" and any(left <= unit["start"] and unit["end"] <= right for left, right in blocked):
            state = "unknown"
        intervals.append({"start": unit["start"], "end": unit["end"], "state": state, "descriptor": unit["record"]["descriptor"], "continuity": unit["record"]["continuity"]})
    return intervals, accepted_spans, audit


def _status_at(intervals: Sequence[dict[str, Any]], timestamp: float) -> str:
    for item in intervals:
        if item["start"] <= timestamp < item["end"]:
            return str(item["state"])
    return "unknown"


def _epoch_ids(
    descriptors_by_channel: Mapping[str, Sequence[dict[str, Any]]], channels: Sequence[str], klass: str, decision_times: np.ndarray,
) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
    output = np.full(decision_times.shape[0], -1, dtype=np.int32)
    vectors: list[list[dict[str, Any]]] = []
    previous_key: str | None = None
    next_epoch = -1
    for index, decision in enumerate(decision_times):
        # Near Unix time 1.7e9, subtracting ``1e-9`` vanishes in binary64; use the actual preceding
        # representable float to implement [left,right) semantics at the bin_right decision time.
        point = float(np.nextafter(float(decision), -np.inf))
        vector = []
        for channel in channels:
            descriptor = _descriptor_at(descriptors_by_channel[channel], point)
            if descriptor is None:
                vector = []
                break
            _require(descriptor["class"] == klass, "expected channel 与 descriptor class 冲突")
            vector.append(descriptor)
        if vector:
            vector.sort(key=lambda item: (item["channel"].encode("utf-8"), item["epoch"], item["sha"].encode("utf-8")))
            key = _sha256_jcs([{
                "source_channel_id": item["channel"], "source_descriptor_sha256": item["sha"], "channel_label_epoch_id": item["epoch"],
                "canonical_class_id": item["class"], "supervision_role": item["role"], "trust_tier": item["tier"], "mapping_status": item["mapping"],
            } for item in vector])
            if key != previous_key:
                next_epoch += 1
                previous_key = key
            output[index] = next_epoch
        else:
            previous_key = None
        vectors.append(vector)
    return output, vectors


def _quality_evaluable(arrays: Mapping[str, np.ndarray], event: Mapping[str, Any]) -> tuple[bool, float | None]:
    """Precompute the frozen six-position quality rule without reading model probabilities."""
    p_mask = arrays["input_value_mask"][:, 0]
    continuity = arrays["continuity_id"]
    right = arrays["bin_right_unix_s"]
    start_upper, start_lower, end = float(event["true_start_upper_bound"]), float(event["true_start_lower_bound"]), float(event["true_end_time"])
    ok = np.zeros(len(right), dtype=bool)
    for index in range(5, len(right)):
        range_ = slice(index - 5, index + 1)
        same_cont = bool(np.all(continuity[range_] == continuity[index]))
        ok[index] = same_cont and bool(p_mask[index]) and int(np.sum(p_mask[range_])) >= 5
    latest = min(start_lower + 30.0, end + 10.0)
    for index in range(len(right) - 1):
        if ok[index] and ok[index + 1] and continuity[index] == continuity[index + 1] and right[index] >= start_upper and right[index + 1] <= latest:
            return True, float(right[index + 1])
    return False, None


def stable_truth_event_id_payload(
    dataset: str,
    house_id: str,
    continuity_id: int,
    label_epoch_id: int,
    appliance_class: str,
    true_start_time: float,
    source_channel_ids: Sequence[str],
) -> dict[str, Any]:
    """Construct the unique JCS payload for a frozen stable truth ID.
    
    This API is shared by the production builder, the independent L11 oracle and mutation fixtures. Never copy a payload from an actual event; recompute the ID here after any field change."""
    _require(isinstance(dataset, str) and dataset, "stable truth ID dataset 必须为非空 str")
    _require(isinstance(house_id, str) and house_id, "stable truth ID house_id 必须为非空 str")
    _require(isinstance(appliance_class, str) and appliance_class in CLASS_INDEX, "stable truth ID appliance_class 非法")
    _require(not isinstance(source_channel_ids, (str, bytes)), "stable truth ID source_channel_ids 必须为字符串序列")
    canonical_sources: set[str] = set()
    for source in source_channel_ids:
        _require(isinstance(source, str) and source, "stable truth ID source_channel_id 必须为非空 str")
        canonical_sources.add(_nfc(source))
    _require(canonical_sources, "stable truth ID 至少需要一个 source_channel_id")
    return {
        "dataset": _nfc(dataset),
        "house_id": _nfc(house_id),
        "continuity_id": _as_int(continuity_id, "stable truth ID continuity_id"),
        "label_epoch_id": _as_int(label_epoch_id, "stable truth ID label_epoch_id"),
        "appliance_class": _nfc(appliance_class),
        "true_start_time": _json_number(_as_float(true_start_time, "stable truth ID true_start_time")),
        "source_channel_ids_sorted": sorted(canonical_sources, key=lambda value: value.encode("utf-8")),
    }


def stable_truth_event_id(
    dataset: str,
    house_id: str,
    continuity_id: int,
    label_epoch_id: int,
    appliance_class: str,
    true_start_time: float,
    source_channel_ids: Sequence[str],
) -> str:
    """Compute the reference-defined production stable truth event ID."""
    return "truth_" + _sha256_jcs(
        stable_truth_event_id_payload(
            dataset,
            house_id,
            continuity_id,
            label_epoch_id,
            appliance_class,
            true_start_time,
            source_channel_ids,
        )
    )


def _truth_event_source_channel_ids(event: Mapping[str, Any]) -> list[str]:
    """Read actual event source membership, rejecting non-JCS values or non-string arrays."""
    encoded = event.get("source_channel_ids_json")
    _require(isinstance(encoded, str), "truth event 缺 source_channel_ids_json")
    try:
        parsed = _parse_json_no_duplicate_keys(encoded)
    except TimelineContractError as exc:
        raise TimelineContractError("truth event source_channel_ids_json 非法") from exc
    _require(isinstance(parsed, list) and parsed and all(isinstance(item, str) and item for item in parsed), "truth event source_channel_ids_json 必须为非空字符串数组")
    _require(encoded.encode("utf-8") == jcs_bytes(parsed), "truth event source_channel_ids_json 必须为 JCS")
    canonical = sorted({_nfc(item) for item in parsed}, key=lambda item: item.encode("utf-8"))
    _require(parsed == canonical, "truth event source_channel_ids_json 必须为 NFC、去重且 UTF-8 字节升序")
    return list(parsed)


def validate_unique_truth_event_ids(events: Sequence[Mapping[str, Any]]) -> None:
    """Recompute production IDs and verify uniqueness across truth events, failing closed on collision."""
    seen: dict[str, int] = {}
    for index, event in enumerate(events):
        _require(isinstance(event, Mapping), "truth event 必须为 mapping")
        event_id = event.get("true_event_id")
        _require(isinstance(event_id, str) and event_id.startswith("truth_") and len(event_id) == 70 and all(char in "0123456789abcdef" for char in event_id[6:]), "truth event ID 格式非法")
        expected = stable_truth_event_id(
            event.get("dataset"),
            event.get("house_id"),
            event.get("continuity_id"),
            event.get("label_epoch_id"),
            event.get("appliance_class"),
            event.get("true_start_time"),
            _truth_event_source_channel_ids(event),
        )
        _require(event_id == expected, "truth event ID 与 frozen stable-ID payload 不匹配")
        _require(event_id not in seen, f"stable truth event ID collision: rows {seen.get(event_id)} and {index}")
        seen[event_id] = index


def _class_events(
    timeline_arrays: Mapping[str, np.ndarray], *, dataset: str, house_id: str, source_start: float,
    source_end: float, class_epochs: Mapping[str, np.ndarray], channel_events: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Logically OR accepted per-channel intervals while retaining included channel membership."""
    events: list[dict[str, Any]] = []
    continuity = timeline_arrays["continuity_id"]
    for klass in CLASS_ORDER:
        spans = [dict(item, channel=channel) for channel, items in channel_events.items() for item in items if item["on_record"]["class"] == klass]
        spans.sort(key=lambda item: (item["start"], item["end"], item["channel"].encode("utf-8")))
        while spans:
            current = spans.pop(0)
            members = [current]
            end = current["end"]
            cursor = 0
            while cursor < len(spans):
                item = spans[cursor]
                if item["start"] <= end:
                    end = max(end, item["end"])
                    members.append(item)
                    spans.pop(cursor)
                else:
                    cursor += 1
            start = current["start"]
            source_ids = sorted({item["channel"] for item in members if item["end"] > start and item["start"] < end}, key=lambda value: value.encode("utf-8"))
            # Find the aggregate continuity and epoch of the ten-second bin containing the event start.
            index = int(math.floor((start - source_start) / BIN_SECONDS))
            if not (0 <= index < len(continuity)):
                continue
            cont = int(continuity[index])
            epoch = int(class_epochs[klass][index])
            if epoch < 0:
                continue
            end_members = [item for item in members if item["end"] == end]
            censored = any(item["right_censored"] for item in end_members)
            end_reason = "normal_off"
            if censored:
                end_reason = "right_censored_label_epoch_end" if any(item["censor_kind"] == "right_censored_label_epoch_end" for item in end_members) else "right_censored_input_coverage_end"
            starters = [item for item in members if item["start"] == start]
            reference = min(starters, key=lambda item: item["channel"].encode("utf-8"))
            previous = reference.get("previous_off")
            uncertainty = None if previous is None else start - float(previous)
            event = {
                "dataset": dataset, "house_id": house_id, "continuity_id": cont, "label_epoch_id": epoch,
                "appliance_class": klass, "true_start_time": _json_number(start),
                "previous_reliable_non_on_timestamp": None if previous is None else _json_number(previous),
                "true_start_lower_bound": None if previous is None else _json_number(previous),
                "true_start_upper_bound": _json_number(start), "true_end_time": _json_number(end),
                "onset_uncertainty_seconds": None if uncertainty is None else _json_number(uncertainty),
                "end_reason": end_reason, "right_censored": bool(censored),
                "source_channel_ids_json": jcs_bytes(source_ids).decode("utf-8"),
            }
            if previous is None:
                # Left censoring must not create an accepted event; this guard makes future fixtures fail closed.
                raise TimelineContractError("accepted truth event 缺少可靠 OFF predecessor")
            quality, earliest = _quality_evaluable(timeline_arrays, event)
            timing = 0.0 < float(uncertainty) <= 10.0
            event["input_quality_evaluable"] = quality
            event["earliest_quality_eligible_report_time"] = None if earliest is None else _json_number(earliest)
            event["label_timing_evaluable"] = timing
            event["alarm_slo_evaluable"] = quality and timing
            event["true_event_id"] = stable_truth_event_id(dataset, house_id, cont, epoch, klass, start, source_ids)
            events.append(event)
    events.sort(key=lambda item: (item["dataset"], item["house_id"], item["continuity_id"], CLASS_INDEX[item["appliance_class"]], float(item["true_start_time"]), item["true_event_id"]))
    validate_unique_truth_event_ids(events)
    return events


def build_timeline(
    aggregate_records: Iterable[AggregateSample | Mapping[str, Any]], *, source_end_unix_s: float | None = None,
    source_end: float | None = None, source_start_unix_s: float | None = None, source_start: float | None = None,
    dataset: str = "refit", house_id: str = "synthetic", expected_channels: Mapping[str, Sequence[str]] | None = None,
    submeter_records: Iterable[SubmeterSample | Mapping[str, Any]] = (), label_descriptors: Iterable[LabelDescriptor | Mapping[str, Any]] | Mapping[str, Any] | None = None,
) -> Timeline:
    """Build a purely synthetic reference timeline.
    
    ``expected_channels`` maps ``{canonical_class_id: [source_channel_id,...]}``; unlisted classes remain unknown. ``label_descriptors`` accepts a flat sequence or ``{channel: descriptor/list}``. All inputs are in-memory fixtures; the runner must establish their synthetic provenance."""
    end = source_end_unix_s if source_end_unix_s is not None else source_end
    _require(end is not None, "source_end 必须显式提供")
    end = _as_float(end, "source_end")
    raw_aggregate = [_coerce_aggregate(item, index) for index, item in enumerate(aggregate_records)]
    if source_start_unix_s is None:
        source_start_unix_s = source_start
    if source_start_unix_s is None:
        finite_timestamps = [float(item["timestamp"]) for item in raw_aggregate if _finite(item["timestamp"])]
        _require(finite_timestamps, "无 aggregate record 时必须显式 source_start")
        source_start_unix_s = math.floor(min(finite_timestamps) / BIN_SECONDS) * BIN_SECONDS
    start = _as_float(source_start_unix_s, "source_start")
    _require(dataset in ("refit", "uk_dale"), "dataset 只允许 refit/uk_dale synthetic fixture")
    arrays, aggregate_audit, partial_audit = _aggregate_arrays(raw_aggregate, dataset=dataset, source_start=start, source_end=end)
    rows = arrays["grid_index"].shape[0]
    target = np.zeros((rows, len(CLASS_ORDER)), dtype=np.uint8)
    target_mask = np.zeros_like(target)
    onset = np.zeros_like(target)
    onset_mask = np.zeros_like(target)
    offset = np.zeros_like(target)
    offset_mask = np.zeros_like(target)
    epochs = np.full((rows, len(CLASS_ORDER)), -1, dtype=np.int32)
    expected = {name: list(items) for name, items in (expected_channels or {}).items()}
    for klass, channels in expected.items():
        _require(klass in CLASS_INDEX and len(set(channels)) == len(channels), "expected_channels 必须使用唯一 frozen class/channel")
    raw_sub = [_coerce_submeter(item, index) for index, item in enumerate(submeter_records)]
    descriptor_items: list[dict[str, Any]] = []
    if label_descriptors:
        iterable: Iterable[Any]
        if isinstance(label_descriptors, Mapping):
            flattened: list[Any] = []
            for channel, item in label_descriptors.items():
                if isinstance(item, Mapping) and "source_channel_id" not in item:
                    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                        flattened.extend([{**entry, "source_channel_id": channel} if isinstance(entry, Mapping) else entry for entry in item])
                    else:
                        flattened.append({**item, "source_channel_id": channel})
                elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                    flattened.extend(item)
                else:
                    flattened.append(item)
            iterable = flattened
        else:
            iterable = label_descriptors
        descriptor_items = [_coerce_descriptor(item) for item in iterable]
    # Expected channels without an explicit descriptor use the common [source_start, source_end) default.
    known_channels = {item["channel"] for item in descriptor_items}
    for klass, channels in expected.items():
        for channel in channels:
            if channel not in known_channels:
                descriptor_items.append({"channel": channel, "class": klass, "start": start, "end": end, "epoch": 0, "role": "core_positive", "tier": "high", "mapping": "resolved", "sha": _default_descriptor_sha(channel), "source_label": channel})
    descriptors_by_channel: dict[str, list[dict[str, Any]]] = {}
    for descriptor in descriptor_items:
        descriptors_by_channel.setdefault(descriptor["channel"], []).append(descriptor)
    for channel, items in descriptors_by_channel.items():
        items.sort(key=lambda item: (item["start"], item["end"], item["epoch"]))
        for left, right in zip(items, items[1:]):
            _require(left["end"] <= right["start"], f"channel {channel} descriptor overlap")
    sub_by_channel: dict[str, list[dict[str, Any]]] = {}
    for record in raw_sub:
        sub_by_channel.setdefault(record["channel"], []).append(record)
        if record["channel"] not in descriptors_by_channel:
            # Records from undeclared expected channels must not silently become supervision.
            raise TimelineContractError("submeter record 未在 expected/descriptors 中声明")
    intervals_by_channel: dict[str, list[dict[str, Any]]] = {}
    accepted_events_by_channel: dict[str, list[dict[str, Any]]] = {}
    sub_audits: dict[str, ConflictAudit] = {}
    for klass, channels in expected.items():
        for channel in channels:
            _require(channel in descriptors_by_channel, "expected channel 缺少 descriptor")
            interval, events, audit = _channel_label_processing(channel, klass, sub_by_channel.get(channel, []), descriptors_by_channel[channel], end)
            intervals_by_channel[channel] = interval
            accepted_events_by_channel[channel] = events
            sub_audits[channel] = audit
    # Decision time is the right edge; read labels strictly at right-epsilon.
    decisions = arrays["bin_right_unix_s"].astype(np.float64)
    class_epochs: dict[str, np.ndarray] = {}
    provenance: list[dict[str, Any]] = []
    for klass in CLASS_ORDER:
        class_index = CLASS_INDEX[klass]
        channels = expected.get(klass, [])
        if not channels:
            continue
        class_epoch, vectors = _epoch_ids(descriptors_by_channel, channels, klass, decisions)
        class_epochs[klass] = class_epoch
        epochs[:, class_index] = class_epoch
        for index, decision in enumerate(decisions.astype(np.float64)):
            point = float(np.nextafter(float(decision), -np.inf))
            vector = vectors[index]
            if not vector:
                continue
            statuses = [_status_at(intervals_by_channel[channel], float(point)) for channel in channels]
            if any(status == "on" for status in statuses):
                target[index, class_index] = 1
                target_mask[index, class_index] = 1
            elif all(status == "off" for status in statuses):
                target_mask[index, class_index] = 1
            # partial terminal row cannot acquire labels even if descriptor happens to cover it.
            if arrays["input_quality_flags"][index] & np.uint16(1 << 5):
                target[index, class_index] = 0
                target_mask[index, class_index] = 0
                epochs[index, class_index] = -1
        # Provenance has one row per class epoch/descriptor; channel epochs are recorded with their class epoch.
        seen_provenance: set[tuple[Any, ...]] = set()
        for index, vector in enumerate(vectors):
            for descriptor in vector:
                key = (int(class_epoch[index]), descriptor["channel"], descriptor["sha"])
                if key in seen_provenance:
                    continue
                seen_provenance.add(key)
                provenance.append({
                    "dataset": dataset, "house_id": house_id, "appliance_class": klass, "class_index": class_index,
                    "label_epoch_id": int(class_epoch[index]), "start_unix_s": _json_number(descriptor["start"]), "end_unix_s": _json_number(descriptor["end"]),
                    "source_channel_id": descriptor["channel"], "source_label": descriptor["source_label"], "canonical_label": klass,
                    "supervision_role": descriptor["role"], "trust_tier": descriptor["tier"], "mapping_status": descriptor["mapping"], "source_descriptor_sha256": descriptor["sha"],
                })
        for index in range(1, rows):
            if target_mask[index, class_index] and target_mask[index - 1, class_index] and arrays["continuity_id"][index] == arrays["continuity_id"][index - 1] and epochs[index, class_index] == epochs[index - 1, class_index] and epochs[index, class_index] >= 0:
                onset_mask[index, class_index] = 1
                offset_mask[index, class_index] = 1
                onset[index, class_index] = np.uint8(target[index, class_index] == 1 and target[index - 1, class_index] == 0)
                offset[index, class_index] = np.uint8(target[index, class_index] == 0 and target[index - 1, class_index] == 1)
    arrays.update({
        "active_target": target, "active_valid_mask": target_mask, "onset_target": onset, "onset_valid_mask": onset_mask,
        "offset_target": offset, "offset_valid_mask": offset_mask, "label_epoch_id": epochs,
    })
    for klass in CLASS_ORDER:
        class_epochs.setdefault(klass, np.full(rows, -1, dtype=np.int32))
    events = _class_events(arrays, dataset=dataset, house_id=house_id, source_start=start, source_end=end, class_epochs=class_epochs, channel_events=accepted_events_by_channel)
    provenance.sort(key=lambda item: tuple(_sort_value(item[name]) for name in PROVENANCE_HEADER))
    return Timeline(dataset=dataset, house_id=house_id, source_start_unix_s=start, source_end_unix_s=end, arrays=arrays,
                    ground_truth_events=events, label_provenance=provenance, aggregate_conflict_audit=aggregate_audit,
                    submeter_conflict_audit=sub_audits, partial_bin_audit=partial_audit)


def fit_masked_normalizer(timelines_or_values: Iterable[Timeline | tuple[np.ndarray, np.ndarray]], *, variance_epsilon: float = 1e-12) -> NormalizerParameters:
    """Fit three-channel parameters with ddof=0 only on observed cells from supplied training timelines."""
    _require(_finite(variance_epsilon) and float(variance_epsilon) >= 0.0, "variance_epsilon 非法")
    observations: list[list[float]] = [[], [], []]
    for item in timelines_or_values:
        if isinstance(item, Timeline):
            values, masks = item.input_values, item.input_value_mask
        else:
            values, masks = item
        values = np.asarray(values)
        masks = np.asarray(masks)
        _require(values.shape == masks.shape and values.ndim == 2 and values.shape[1] == 3, "normalizer 输入必须为 [N,3] values/masks")
        for channel in range(3):
            observations[channel].extend(float(value) for value in values[:, channel][masks[:, channel].astype(bool)])
    mean = np.zeros(3, dtype=np.float64)
    scale = np.ones(3, dtype=np.float64)
    count = np.zeros(3, dtype=np.int64)
    supported = np.zeros(3, dtype=bool)
    for channel, observed in enumerate(observations):
        count[channel] = len(observed)
        if observed:
            values = np.asarray(observed, dtype=np.float64)
            mean[channel] = np.mean(values, dtype=np.float64)
            variance = np.mean((values - mean[channel]) ** 2, dtype=np.float64)
            scale[channel] = 1.0 if variance <= variance_epsilon else math.sqrt(float(variance))
            supported[channel] = True
    return NormalizerParameters(mean=mean, scale=scale, observed_count=count, supported=supported, variance_epsilon=float(variance_epsilon))


def apply_masked_normalizer(values: np.ndarray, masks: np.ndarray, parameters: NormalizerParameters) -> tuple[np.ndarray, np.ndarray]:
    """Compute in float64 and convert once to float32; invalid or unsupported entries always use canonical +0 and mask=0."""
    raw_values = np.asarray(values)
    raw_masks = np.asarray(masks)
    _require(raw_values.shape == raw_masks.shape and raw_values.ndim == 2 and raw_values.shape[1] == 3, "normalizer transform 输入必须为 [N,3]")
    output_values = np.zeros(raw_values.shape, dtype=np.float32)
    output_masks = np.asarray(raw_masks, dtype=np.uint8).copy()
    for channel in range(3):
        if not bool(parameters.supported[channel]):
            output_masks[:, channel] = 0
            continue
        observed = output_masks[:, channel].astype(bool)
        output_values[observed, channel] = np.asarray((raw_values[observed, channel].astype(np.float64) - parameters.mean[channel]) / parameters.scale[channel], dtype=np.float32)
    output_values[output_values == 0] = np.float32(0.0)
    output_values[output_masks == 0] = np.float32(0.0)
    return output_values, output_masks


def materialize_model_window(timeline: Timeline, end_index: int, normalizer: NormalizerParameters | None = None, *, allow_left_padding: bool = False) -> np.ndarray:
    """Return float32 ``[6,64]`` in frozen channel order; only the last bin is the v1 readout."""
    _require(isinstance(end_index, int) and 0 <= end_index < timeline.row_count, "end_index 越界")
    _require(bool(timeline.model_eligible[end_index]), "partial tail 不能作为 model endpoint")
    left = end_index - CONTEXT_BINS + 1
    values = timeline.input_values
    masks = timeline.input_value_mask
    if left < 0 and not allow_left_padding:
        raise TimelineContractError("训练窗口必须有 64 个真实格")
    real_left = max(left, 0)
    if real_left <= end_index:
        cont = timeline.continuity_id[real_left:end_index + 1]
        if not np.all(cont == cont[-1]):
            raise TimelineContractError("window 不得跨 continuity break")
        if not np.all(timeline.model_eligible[real_left:end_index + 1]):
            raise TimelineContractError("window 不得含 terminal partial row")
    scaled_values, scaled_masks = (apply_masked_normalizer(values, masks, normalizer) if normalizer is not None else (values.astype(np.float32, copy=True), masks.astype(np.uint8, copy=True)))
    output_values = np.zeros((CONTEXT_BINS, 3), dtype=np.float32)
    output_masks = np.zeros((CONTEXT_BINS, 3), dtype=np.float32)
    destination_left = max(0, -left)
    output_values[destination_left:] = scaled_values[real_left:end_index + 1]
    output_masks[destination_left:] = scaled_masks[real_left:end_index + 1]
    return np.concatenate([output_values.T, output_masks.T], axis=0).astype(np.float32, copy=False)


def iter_stream_feature_rows(timeline: Timeline) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    """Export the same builder float32 features per closed bin for bit-level I09 comparisons with batch output."""
    for index in range(timeline.row_count):
        if timeline.model_eligible[index]:
            yield index, timeline.input_values[index].copy(), timeline.input_value_mask[index].copy()


def canonical_array_payload(array: np.ndarray) -> bytes:
    """D08 payload: little-endian, C-contiguous and with every floating -0 normalized to +0."""
    value = np.asarray(array)
    _require(value.dtype.kind in "iufb", "logical array dtype 不在 frozen numeric 闭集")
    dtype = value.dtype.newbyteorder("<")
    result = np.ascontiguousarray(value.astype(dtype, copy=False))
    if result.dtype.kind == "f":
        result = result.copy()
        result[result == 0] = 0
    return result.tobytes(order="C")


def array_logical_descriptor(array_name: str, logical_index: int, array: np.ndarray) -> dict[str, Any]:
    _require(array_name in REQUIRED_ARRAYS and logical_index == REQUIRED_ARRAYS.index(array_name), "array name/logical_index 不匹配 frozen 顺序")
    value = np.asarray(array)
    return {"array_name": array_name, "logical_index": logical_index, "dtype": str(value.dtype), "shape": [int(item) for item in value.shape], "canonical_payload_sha256": hashlib.sha256(canonical_array_payload(value)).hexdigest()}


def _sort_value(value: Any) -> tuple[int, Any]:
    if value is None:
        return (0, "")
    if isinstance(value, str):
        return (1, value.encode("utf-8"))
    return (1, value)


def _typed_record_value(value: Any, declared_type: str) -> Any:
    """Apply typed-record field types and NFC rules before general JCS serialization."""
    nullable = declared_type.startswith("nullable_")
    if value is None:
        _require(nullable, f"{declared_type} 字段不得为 null")
        return None
    actual_type = declared_type.removeprefix("nullable_")
    if actual_type == "utf8_string":
        _require(isinstance(value, str), "typed utf8_string 字段必须为 str")
        return _nfc(value)
    if actual_type == "sha256_lowercase_hex":
        _require(isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value), "typed SHA-256 字段非法")
        return value
    if actual_type == "bool":
        _require(isinstance(value, bool), "typed bool 字段必须为 bool")
        return value
    if actual_type == "int32":
        number = _as_int(value, "typed int32")
        _require(-(2 ** 31) <= number < 2 ** 31, "typed int32 越界")
        return number
    if actual_type in {"float64", "float64_unix_seconds"}:
        return _json_number(_as_float(value, f"typed {actual_type}"))
    raise TimelineContractError(f"未知 typed record 字段类型 {declared_type}")


def typed_record_logical_descriptor(role: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Hash logical CSV records using the frozen declared column types and order."""
    if role == "ground_truth_events":
        header, types, version = GROUND_TRUTH_HEADER, GROUND_TRUTH_TYPES, "ground_truth_events_v1"
        ordered = sorted(records, key=lambda item: (item["dataset"], item["house_id"], int(item["continuity_id"]), CLASS_INDEX[item["appliance_class"]], float(item["true_start_time"]), item["true_event_id"]))
    elif role == "label_provenance":
        header, types, version = PROVENANCE_HEADER, PROVENANCE_TYPES, "label_provenance_v1"
        ordered = sorted(records, key=lambda item: tuple(_sort_value(item.get(name)) for name in header))
    else:
        raise TimelineContractError("typed record artifact role 非法")
    rows = [
        [_typed_record_value(item.get(column), declared_type) for column, declared_type in zip(header, types)]
        for item in ordered
    ]
    return {"artifact_role": role, "schema_version": version, "column_order": list(header), "column_types": list(types), "row_count": len(rows), "logical_records_sha256": _sha256_jcs(rows)}


_ZIP_MEMBER_METADATA_FIELDS = {
    "member_name", "timestamp_dos", "create_system", "create_version", "extract_version",
    "internal_attr", "external_attr", "flag_bits", "data_descriptor_present",
    "extra_fields_hex", "comment_hex",
}
_UTF8_BOM = b"\xef\xbb\xbf"


def _require_lowercase_hex(value: Any, label: str) -> None:
    _require(isinstance(value, str) and len(value) % 2 == 0 and all(char in "0123456789abcdef" for char in value), f"{label} 必须为小写偶数位 hex")


def _require_nonnegative_metadata_int(value: Any, label: str) -> None:
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"{label} 必须为非负 int")


def validate_container_metadata(metadata: Mapping[str, Any]) -> None:
    """Validate D08 format_metadata union fields, types and allowed values without reading files."""
    _require(isinstance(metadata, Mapping) and isinstance(metadata.get("file_format"), str), "container metadata 缺 file_format")
    kind = metadata["file_format"]
    expected = {
        "npz": {"file_format", "archive_format", "compression", "npy_file_format_version_by_member", "zip_member_metadata", "zip_archive_comment_hex"},
        "csv": {"file_format", "encoding", "bom_present", "line_ending", "delimiter", "quote_character", "quoting", "header_present"},
        "json": {"file_format", "encoding", "bom_present", "json_canonicalization", "trailing_newline_present"},
    }
    _require(kind in expected and set(metadata) == expected[kind], "container format_metadata 非法闭集")
    if kind == "npz":
        _require(metadata["archive_format"] == "npz" and metadata["compression"] in {"ZIP_STORED", "ZIP_DEFLATED"}, "npz metadata 非法")
        members = metadata["npy_file_format_version_by_member"]
        expected_members = {f"{name}.npy" for name in REQUIRED_ARRAYS}
        _require(isinstance(members, Mapping) and set(members) == expected_members and all(isinstance(version, str) and version in {"1.0", "2.0"} for version in members.values()), "npy member metadata 非法")
        zip_members = metadata["zip_member_metadata"]
        _require(isinstance(zip_members, list) and len(zip_members) == len(REQUIRED_ARRAYS), "zip member metadata 数量非法")
        for name, item in zip(REQUIRED_ARRAYS, zip_members):
            _require(isinstance(item, Mapping) and set(item) == _ZIP_MEMBER_METADATA_FIELDS and item["member_name"] == f"{name}.npy", "zip member metadata 顺序/字段非法")
            _require(isinstance(item["timestamp_dos"], str), "zip timestamp_dos 必须为 str")
            for field_name in ("create_system", "create_version", "extract_version", "internal_attr", "external_attr", "flag_bits"):
                _require_nonnegative_metadata_int(item[field_name], f"zip {field_name}")
            _require(isinstance(item["data_descriptor_present"], bool), "zip data_descriptor_present 必须为 bool")
            _require_lowercase_hex(item["extra_fields_hex"], "zip extra_fields_hex")
            _require_lowercase_hex(item["comment_hex"], "zip comment_hex")
        _require_lowercase_hex(metadata["zip_archive_comment_hex"], "zip_archive_comment_hex")
    elif kind == "csv":
        _require(metadata["encoding"] == "UTF-8" and metadata["line_ending"] in {"LF", "CRLF"} and metadata["delimiter"] == "," and metadata["quote_character"] == "double_quote" and metadata["quoting"] == "RFC4180_quote_minimal" and isinstance(metadata["bom_present"], bool) and isinstance(metadata["header_present"], bool), "csv metadata 非法")
    else:
        _require(metadata["encoding"] == "UTF-8" and metadata["json_canonicalization"] in {"RFC8785_JCS", "noncanonical_json"} and isinstance(metadata["bom_present"], bool) and isinstance(metadata["trailing_newline_present"], bool), "json metadata 非法")


def _zip_timestamp_dos(info: zipfile.ZipInfo) -> str:
    year, month, day, hour, minute, second = info.date_time
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"


def _actual_zip_member_metadata(info: zipfile.ZipInfo) -> dict[str, Any]:
    return {
        "member_name": info.filename,
        "timestamp_dos": _zip_timestamp_dos(info),
        "create_system": info.create_system,
        "create_version": info.create_version,
        "extract_version": info.extract_version,
        "internal_attr": info.internal_attr,
        "external_attr": info.external_attr,
        "flag_bits": info.flag_bits,
        "data_descriptor_present": bool(info.flag_bits & 0x08),
        "extra_fields_hex": info.extra.hex(),
        "comment_hex": getattr(info, "comment", b"").hex(),
    }


def _inspect_npz_bytes(content: bytes, metadata: Mapping[str, Any]) -> None:
    """Bind each declaration to the ZIP directory and every actual NPY member."""
    expected_names = [f"{name}.npy" for name in REQUIRED_ARRAYS]
    compression = zipfile.ZIP_STORED if metadata["compression"] == "ZIP_STORED" else zipfile.ZIP_DEFLATED
    try:
        with zipfile.ZipFile(io.BytesIO(content), "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            _require(names == expected_names and len(set(names)) == len(names), "NPZ 实际成员集合或顺序与 frozen required_arrays 不符")
            _require(archive.comment.hex() == metadata["zip_archive_comment_hex"], "NPZ zip_archive_comment_hex 与实际字节不符")
            for name, info, declared_member in zip(expected_names, infos, metadata["zip_member_metadata"]):
                _require(info.compress_type == compression, f"NPZ {name} compression 与 metadata 不符")
                _require(not (info.flag_bits & 0x01), f"NPZ {name} 不得为加密 member")
                _require(_actual_zip_member_metadata(info) == dict(declared_member), f"NPZ {name} zip member metadata 与实际字节不符")
                payload = archive.read(info)
                _require(payload.startswith(b"\x93NUMPY") and len(payload) >= 8, f"NPZ {name} 不是有效 NPY member")
                version = f"{payload[6]}.{payload[7]}"
                _require(version == metadata["npy_file_format_version_by_member"][name], f"NPZ {name} NPY 版本与 metadata 不符")
                array = np.load(io.BytesIO(payload), allow_pickle=False)
                _require(isinstance(array, np.ndarray) and array.dtype.kind in "iufb", f"NPZ {name} NPY payload dtype 非法")
    except TimelineContractError:
        raise
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        raise TimelineContractError("NPZ container 无法按 frozen contract 检查") from exc


def _csv_record_end(text: str, position: int, line_ending: str) -> int | None:
    if position >= len(text):
        return None
    if line_ending == "CRLF" and text.startswith("\r\n", position):
        return position + 2
    if line_ending == "LF" and text[position] == "\n":
        return position + 1
    if text[position] in "\r\n":
        raise TimelineContractError("CSV 实际 record line ending 与 metadata 不符")
    return None


def _parse_rfc4180_quote_minimal(text: str, line_ending: str) -> list[list[str]]:
    """Strictly parse RFC4180 CSV and require every quote pair to be necessary under a minimal writer."""
    if not text:
        return []
    rows: list[list[str]] = []
    position = 0
    length = len(text)
    while position < length:
        row: list[str] = []
        while True:
            if position == length:
                # A terminal comma denotes one final empty field.
                row.append("")
                rows.append(row)
                return rows
            quoted = text[position] == '"'
            if quoted:
                position += 1
                field_chars: list[str] = []
                closed = False
                while position < length:
                    character = text[position]
                    if character == '"':
                        if position + 1 < length and text[position + 1] == '"':
                            field_chars.append('"')
                            position += 2
                            continue
                        position += 1
                        closed = True
                        break
                    field_chars.append(character)
                    position += 1
                _require(closed, "CSV quoted field 未闭合")
                field = "".join(field_chars)
                _require(any(character in ',"\r\n' for character in field), "CSV quote 不符合 RFC4180_quote_minimal")
            else:
                start = position
                while position < length and text[position] not in ",\r\n":
                    _require(text[position] != '"', "CSV 未引用 field 含 quote")
                    position += 1
                field = text[start:position]
            row.append(field)
            if position == length:
                rows.append(row)
                return rows
            if text[position] == ",":
                position += 1
                continue
            next_position = _csv_record_end(text, position, line_ending)
            _require(next_position is not None, "CSV field 后必须是 delimiter、record ending 或 EOF")
            rows.append(row)
            position = next_position
            break
    return rows


def _validate_csv_header_claim(rows: Sequence[Sequence[str]], header_present: bool, project_relative_path: str | None) -> None:
    """Check actual headers for the two frozen CSV roles; generic CSV must have at least one record."""
    known_headers = {
        "ground_truth_events.csv": GROUND_TRUTH_HEADER,
        "label_provenance.csv": PROVENANCE_HEADER,
    }
    filename = Path(project_relative_path).name if project_relative_path else ""
    expected_header = known_headers.get(filename)
    if expected_header is not None:
        actual_header_present = bool(rows) and tuple(rows[0]) == expected_header
        _require(actual_header_present == header_present, "CSV header_present 与冻结 artifact 实际首行不符")
    elif header_present:
        _require(bool(rows), "CSV 声称 header_present=true 但没有首 record")


def _inspect_csv_bytes(content: bytes, metadata: Mapping[str, Any], project_relative_path: str | None) -> None:
    bom_present = content.startswith(_UTF8_BOM)
    _require(bom_present == metadata["bom_present"], "CSV BOM bit 与实际字节不符")
    payload = content[len(_UTF8_BOM):] if bom_present else content
    _require(not payload.startswith(_UTF8_BOM), "CSV 不得包含重复 UTF-8 BOM")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise TimelineContractError("CSV 不是真实 UTF-8") from exc
    rows = _parse_rfc4180_quote_minimal(text, metadata["line_ending"])
    _validate_csv_header_claim(rows, metadata["header_present"], project_relative_path)


def _parse_json_no_duplicate_keys(text: str) -> Any:
    def reject_constant(value: str) -> Any:
        raise TimelineContractError(f"JSON 禁止非有限常量 {value}")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, "JSON object 不得有 duplicate key")
            result[key] = value
        return result

    try:
        parsed = json.loads(text, object_pairs_hook=object_pairs, parse_constant=reject_constant)
    except TimelineContractError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TimelineContractError("JSON 不是有效 RFC8259 text") from exc
    # Reject escaped unpaired surrogates and other values JCS/I-JSON cannot represent.
    _jcs_normalize(parsed)
    return parsed


def _inspect_json_bytes(content: bytes, metadata: Mapping[str, Any]) -> None:
    bom_present = content.startswith(_UTF8_BOM)
    _require(bom_present == metadata["bom_present"], "JSON BOM bit 与实际字节不符")
    payload = content[len(_UTF8_BOM):] if bom_present else content
    _require(not payload.startswith(_UTF8_BOM), "JSON 不得包含重复 UTF-8 BOM")
    _require(payload.endswith(b"\n") == metadata["trailing_newline_present"], "JSON trailing_newline bit 与实际字节不符")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise TimelineContractError("JSON 不是真实 UTF-8") from exc
    parsed = _parse_json_no_duplicate_keys(text)
    canonical = jcs_bytes(parsed)
    is_jcs = payload == canonical
    if metadata["json_canonicalization"] == "RFC8785_JCS":
        _require(is_jcs, "JSON 声称 RFC8785_JCS 但实际字节不是 JCS")
    else:
        _require(not is_jcs, "JSON 声称 noncanonical_json 但实际字节已经是 JCS")


def inspect_container_metadata(
    path: str | Path,
    format_metadata: Mapping[str, Any],
    *,
    project_relative_path: str | None = None,
) -> None:
    """Check that declared metadata matches actual NPZ/CSV/JSON container bytes item by item."""
    file_path = Path(path)
    _require(file_path.is_file(), "container 文件不存在")
    validate_container_metadata(format_metadata)
    content = file_path.read_bytes()
    kind = format_metadata["file_format"]
    if kind == "npz":
        _inspect_npz_bytes(content, format_metadata)
    elif kind == "csv":
        _inspect_csv_bytes(content, format_metadata, project_relative_path)
    else:
        _inspect_json_bytes(content, format_metadata)


def describe_container(path: str | Path, project_relative_path: str, format_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Describe a written container, rejecting metadata inconsistent with its actual bytes."""
    _require(isinstance(project_relative_path, str) and project_relative_path, "project_relative_path 必须为非空 str")
    file_path = Path(path)
    inspect_container_metadata(file_path, format_metadata, project_relative_path=project_relative_path)
    content = file_path.read_bytes()
    return {"project_relative_path": project_relative_path.replace("\\", "/"), "size_bytes": len(content), "container_sha256": hashlib.sha256(content).hexdigest(), "format_metadata": dict(format_metadata)}


def validate_container_descriptor(path: str | Path, descriptor: Mapping[str, Any]) -> None:
    """Detect false metadata, corruption or replacement; container identity is excluded from logical identity."""
    required = {"project_relative_path", "size_bytes", "container_sha256", "format_metadata"}
    _require(isinstance(descriptor, Mapping) and set(descriptor) == required, "container descriptor 字段必须精确闭集")
    _require(isinstance(descriptor["project_relative_path"], str) and descriptor["project_relative_path"], "container descriptor project_relative_path 非法")
    _require(isinstance(descriptor["size_bytes"], int) and not isinstance(descriptor["size_bytes"], bool) and descriptor["size_bytes"] >= 0, "container descriptor size_bytes 非法")
    _require(isinstance(descriptor["container_sha256"], str) and len(descriptor["container_sha256"]) == 64 and all(char in "0123456789abcdef" for char in descriptor["container_sha256"]), "container descriptor SHA-256 非法")
    actual = describe_container(path, descriptor["project_relative_path"], descriptor["format_metadata"])
    _require(actual["size_bytes"] == descriptor["size_bytes"] and actual["container_sha256"] == descriptor["container_sha256"], "container_sha256 或 size_bytes 不匹配")


__all__ = [
    "AggregateSample", "SubmeterSample", "LabelDescriptor", "Timeline", "TimelineContractError",
    "NormalizerParameters", "build_timeline", "fit_masked_normalizer", "apply_masked_normalizer",
    "materialize_model_window", "iter_stream_feature_rows", "array_logical_descriptor",
    "canonical_array_payload", "typed_record_logical_descriptor", "jcs_bytes",
    "stable_truth_event_id_payload", "stable_truth_event_id", "validate_unique_truth_event_ids",
    "inspect_container_metadata", "describe_container", "validate_container_descriptor",
    "validate_container_metadata", "CLASS_ORDER", "REQUIRED_ARRAYS",
]
