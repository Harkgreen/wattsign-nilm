"""Deterministic streaming alarm state machine with quality-aware hysteresis.

Transitions consume supplied probabilities and continuity/quality indicators.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import hashlib
import json
import math
import struct
from typing import Any, Iterable, Mapping, Sequence


BIN_SECONDS = 10
ON_CONFIRMATION_BINS = 2
OFF_CONFIRMATION_BINS = 3
WARMUP_BINS = 6
WARMUP_MINIMUM_VALID_BINS = 5

# reference definition frozen class axis.  This module deliberately owns the literal rather
# than importing a timeline/config reader: the reference alarm must stay fully
# synthetic and must not acquire any I/O or artifact dependency.
FROZEN_CLASS_ORDER = (
    "kettle",
    "toaster",
    "microwave",
    "hair_dryer",
    "iron",
    "rice_cooker",
)
FROZEN_CLASS_COUNT = len(FROZEN_CLASS_ORDER)

QUALITY_FLAG_NONFINITE_PROBABILITY = 1 << 8
QUALITY_FLAG_INVALID_CLOSE = 1 << 9
QUALITY_FLAG_HARD_GAP_CLOSE = 1 << 10
QUALITY_FLAG_STREAM_END_CLOSE = 1 << 11
QUALITY_FLAG_RESERVED_MASK = 0xF000


class AlarmContractError(ValueError):
    """Raised when input violates the frozen reference alarm definition.
    
    Fail closed: callers must not silently process incomplete input, mixed continuity segments or ambiguous identities as a normal probability stream."""


class AlarmState(StrEnum):
    """The four frozen alarm states."""

    OFF = "OFF"
    PENDING_ON = "PENDING_ON"
    ON = "ON"
    PENDING_OFF = "PENDING_OFF"


class AlarmSymbol(StrEnum):
    """The seven input symbols in the frozen transition table."""

    VALID_HIGH = "valid_high"
    VALID_LOW = "valid_low"
    VALID_DEADBAND = "valid_deadband"
    INVALID_SOFT = "invalid_soft"
    INVALID_CLOSE = "invalid_close"
    HARD_GAP = "hard_gap"
    STREAM_END = "stream_end"


def _integer(value: int, *, label: str) -> int:
    """Accept integer timestamps and continuity IDs; reject booleans and implicit truncation."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise AlarmContractError(f"{label} 必须是整数")
    return value


def _uint8(value: int, *, label: str) -> int:
    value = _integer(value, label=label)
    if not 0 <= value <= 0xFF:
        raise AlarmContractError(f"{label} 必须在 0..255")
    return value


def _uint16(value: int, *, label: str) -> int:
    value = _integer(value, label=label)
    if not 0 <= value <= 0xFFFF:
        raise AlarmContractError(f"{label} 必须在 0..65535")
    if value & QUALITY_FLAG_RESERVED_MASK:
        raise AlarmContractError(f"{label} 的保留 bits 12..15 必须为零")
    return value


def float32(value: float | int) -> float:
    """Convert explicitly once to IEEE-754 binary32.
    
    Compare all probabilities and thresholds after this conversion so Python binary64 text parsing cannot change the frozen >= or < boundary semantics."""

    if isinstance(value, (bool, str, bytes)):
        raise AlarmContractError("float32 输入必须是数值而不是 bool/文本")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise AlarmContractError("float32 输入无法转换为数值") from exc
    try:
        return struct.unpack(">f", struct.pack(">f", numeric))[0]
    except OverflowError:
        return math.copysign(math.inf, numeric)


def float32_hex(value: float | int) -> str:
    """Return float32 IEEE-754 bits as eight lowercase big-endian hexadecimal digits."""

    return struct.pack(">f", float32(value)).hex()


def _finite_float32(value: float | int, *, label: str) -> float:
    result = float32(value)
    if not math.isfinite(result):
        raise AlarmContractError(f"{label} 必须是有限 float32")
    return result


def _sha256_lower_hex(value: str, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AlarmContractError(f"{label} 必须是 64 位小写 SHA-256")
    if value != value.lower() or any(character not in "0123456789abcdef" for character in value):
        raise AlarmContractError(f"{label} 必须是 64 位小写 SHA-256")
    return value


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Produce RFC 8785 JCS bytes for ID payloads containing integers, strings and arrays.
    
    These payloads have no floating-point JSON numbers; float32 values first become the prescribed hexadecimal strings. Sorted keys, compact separators and UTF-8 without a BOM or newline match the required JCS subset."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AlarmContractError("ID payload 不能被 RFC8785 JCS 序列化") from exc
    return text.encode("utf-8")


def _validate_frozen_class_order(class_order: Sequence[str] | None) -> tuple[str, ...]:
    'Resolve only the reference definition class axis; callers cannot select another one.\n\n    The class order is intentionally absent from the prediction-stream payload:\n    it is a contract-level constant.  Accepting an explicit value is retained\n    only as a fail-closed assertion for callers that carry the axis alongside\n    their threshold array.\n    '

    if class_order is None:
        return FROZEN_CLASS_ORDER
    if isinstance(class_order, (str, bytes)):
        raise AlarmContractError("class_order 必须严格等于固定六类 class_order")
    try:
        resolved = tuple(class_order)
    except TypeError as exc:
        raise AlarmContractError("class_order 必须严格等于固定六类 class_order") from exc
    if resolved != FROZEN_CLASS_ORDER:
        raise AlarmContractError("class_order 必须严格等于固定六类 class_order")
    return resolved


def _validate_threshold_bitpatterns(
    bitpatterns: Sequence[Sequence[str]], *, class_order: Sequence[str] | None
) -> list[list[str]]:
    """Validate threshold bit patterns in the frozen prediction-stream payload order."""

    if isinstance(bitpatterns, (str, bytes)) or not bitpatterns:
        raise AlarmContractError("class_thresholds_float32_bitpatterns 不能为空二维数组")
    if class_order is not None and len(bitpatterns) != len(class_order):
        raise AlarmContractError("阈值数组长度必须等于冻结 class_order 长度")
    result: list[list[str]] = []
    for index, pair in enumerate(bitpatterns):
        if isinstance(pair, (str, bytes)) or len(pair) != 2:
            raise AlarmContractError(f"第 {index} 个阈值必须是 [on_hex, off_hex]")
        on_hex, off_hex = pair
        for label, item in (("on", on_hex), ("off", off_hex)):
            if (
                not isinstance(item, str)
                or len(item) != 8
                or item != item.lower()
                or any(character not in "0123456789abcdef" for character in item)
            ):
                raise AlarmContractError(f"第 {index} 个 {label} 阈值不是八位小写 float32 hex")
        on_value = struct.unpack(">f", bytes.fromhex(on_hex))[0]
        off_value = struct.unpack(">f", bytes.fromhex(off_hex))[0]
        if not math.isfinite(on_value) or not math.isfinite(off_value) or not off_value < on_value:
            raise AlarmContractError(f"第 {index} 个阈值必须是有限且 off < on 的 float32")
        result.append([on_hex, off_hex])
    return result


def threshold_bitpatterns(class_thresholds: Sequence[Sequence[float | int]]) -> list[list[str]]:
    """Convert [on, off] thresholds in the frozen class order to ID bit patterns."""

    if isinstance(class_thresholds, (str, bytes)) or not class_thresholds:
        raise AlarmContractError("class_thresholds 不能为空二维数组")
    pairs: list[list[str]] = []
    for index, pair in enumerate(class_thresholds):
        if isinstance(pair, (str, bytes)) or len(pair) != 2:
            raise AlarmContractError(f"第 {index} 个阈值必须是 [on, off]")
        on_value = _finite_float32(pair[0], label=f"第 {index} 个 on threshold")
        off_value = _finite_float32(pair[1], label=f"第 {index} 个 off threshold")
        if not 0.0 <= off_value < on_value <= 1.0:
            raise AlarmContractError(f"第 {index} 个阈值必须满足 0 <= off < on <= 1")
        pairs.append([float32_hex(on_value), float32_hex(off_value)])
    return pairs


def build_prediction_stream_id(
    *,
    contract_sha256: str,
    class_registry_sha256: str,
    model_or_probability_source_descriptor_sha256: str,
    input_timeline_logical_content_sha256: str,
    class_thresholds_float32_bitpatterns: Sequence[Sequence[str]] | None = None,
    class_thresholds: Sequence[Sequence[float | int]] | None = None,
    class_order: Sequence[str] | None = None,
) -> str:
    """Create an immutable prediction-stream ID.
    
    The explicit input_timeline_logical_content_sha256 argument prevents confusing a container hash with the logical timeline descriptor. Extract timeline_logical_content_sha256 from a complete descriptor before calling."""

    if (class_thresholds_float32_bitpatterns is None) == (class_thresholds is None):
        raise AlarmContractError("必须且只能提供阈值 bitpatterns 或 float32 阈值")
    frozen_class_order = _validate_frozen_class_order(class_order)
    if class_thresholds is not None:
        patterns = threshold_bitpatterns(class_thresholds)
        if len(patterns) != FROZEN_CLASS_COUNT:
            raise AlarmContractError("阈值数组必须严格为冻结 class_order 的 K=6")
    else:
        patterns = _validate_threshold_bitpatterns(
            class_thresholds_float32_bitpatterns, class_order=frozen_class_order
        )
    payload = {
        "class_registry_sha256": _sha256_lower_hex(class_registry_sha256, label="class_registry_sha256"),
        "class_thresholds_float32_bitpatterns": patterns,
        "contract_sha256": _sha256_lower_hex(contract_sha256, label="contract_sha256"),
        "input_timeline_descriptor_sha256": _sha256_lower_hex(
            input_timeline_logical_content_sha256,
            label="input_timeline_logical_content_sha256",
        ),
        "model_or_probability_source_descriptor_sha256": _sha256_lower_hex(
            model_or_probability_source_descriptor_sha256,
            label="model_or_probability_source_descriptor_sha256",
        ),
    }
    return "pstream_" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class AlarmConfig:
    """Fixed float32 thresholds injected into the synthetic reference alarm."""

    on_threshold: float = 0.70
    off_threshold: float = 0.55

    def __post_init__(self) -> None:
        on_value = _finite_float32(self.on_threshold, label="on_threshold")
        off_value = _finite_float32(self.off_threshold, label="off_threshold")
        if not 0.0 <= off_value < on_value <= 1.0:
            raise AlarmContractError("阈值必须满足 0 <= off < on <= 1")
        object.__setattr__(self, "on_threshold", on_value)
        object.__setattr__(self, "off_threshold", off_value)


@dataclass(frozen=True, slots=True)
class AlarmMetadata:
    """Stable group identity written to each prediction event."""

    dataset: str = "refit"
    house_id: str = "synthetic_alarm"
    continuity_id: int = 0
    appliance_class: str = "kettle"

    def __post_init__(self) -> None:
        for label, value in (
            ("dataset", self.dataset),
            ("house_id", self.house_id),
            ("appliance_class", self.appliance_class),
        ):
            if not isinstance(value, str) or not value:
                raise AlarmContractError(f"{label} 必须是非空字符串")
        if self.appliance_class not in FROZEN_CLASS_ORDER:
            raise AlarmContractError("appliance_class 必须是固定六类 class_order 中的类别")
        _integer(self.continuity_id, label="continuity_id")


@dataclass(frozen=True, slots=True)
class PredictionEvent:
    """An emitted alarm snapshot or its later closure record.
    
    Closing fields such as end_time are None at ON emission. Closing creates a new immutable record with the same event ID; it never mutates the previously emitted_alarm object."""

    event_id: str
    prediction_stream_id: str
    dataset: str
    house_id: str
    continuity_id: int
    appliance_class: str
    start_time: int
    report_time: int
    end_time: int | None
    close_report_time: int | None
    confidence: float
    end_reason: str | None
    right_censored: bool | None
    quality_flags: int

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id.startswith("pred_"):
            raise AlarmContractError("event_id 必须以 pred_ 开头")
        _sha256_lower_hex(self.event_id[len("pred_") :], label="event_id")
        _prediction_stream_id(self.prediction_stream_id)
        AlarmMetadata(
            dataset=self.dataset,
            house_id=self.house_id,
            continuity_id=self.continuity_id,
            appliance_class=self.appliance_class,
        )
        start_time = _integer(self.start_time, label="start_time")
        report_time = _integer(self.report_time, label="report_time")
        if report_time < start_time:
            raise AlarmContractError("report_time 不得早于 start_time")
        if self.end_time is None:
            if self.close_report_time is not None or self.end_reason is not None or self.right_censored is not None:
                raise AlarmContractError("未关闭 event 不能带关闭字段")
        else:
            end_time = _integer(self.end_time, label="end_time")
            close_report_time = _integer(self.close_report_time, label="close_report_time")
            if end_time < start_time or close_report_time < end_time:
                raise AlarmContractError("关闭时间戳顺序不合法")
            if self.end_reason not in {"normal_off", "data_gap", "stream_end"}:
                raise AlarmContractError("end_reason 不在冻结枚举中")
            if not isinstance(self.right_censored, bool):
                raise AlarmContractError("已关闭 event 必须带 bool right_censored")
            if self.right_censored != (self.end_reason in {"data_gap", "stream_end"}):
                raise AlarmContractError("right_censored 与 end_reason 不一致")
        confidence = _finite_float32(self.confidence, label="confidence")
        object.__setattr__(self, "confidence", confidence)
        _uint16(self.quality_flags, label="quality_flags")

    def to_record(self) -> dict[str, Any]:
        """Return a plain record in the frozen event-output field order."""

        return {
            "event_id": self.event_id,
            "prediction_stream_id": self.prediction_stream_id,
            "dataset": self.dataset,
            "house_id": self.house_id,
            "continuity_id": self.continuity_id,
            "appliance_class": self.appliance_class,
            "start_time": self.start_time,
            "report_time": self.report_time,
            "end_time": self.end_time,
            "close_report_time": self.close_report_time,
            "confidence": self.confidence,
            "end_reason": self.end_reason,
            "right_censored": self.right_censored,
            "quality_flags": self.quality_flags,
        }


def _prediction_stream_id(value: str) -> str:
    if not isinstance(value, str) or not value.startswith("pstream_"):
        raise AlarmContractError("prediction_stream_id 必须以 pstream_ 开头")
    _sha256_lower_hex(value[len("pstream_") :], label="prediction_stream_id")
    return value


def build_prediction_event_id(
    *,
    prediction_stream_id: str,
    metadata: AlarmMetadata,
    start_time: int,
    report_time: int,
) -> str:
    """Create a prediction event ID from the six frozen stable fields."""

    payload = {
        "appliance_class": metadata.appliance_class,
        "continuity_id": _integer(metadata.continuity_id, label="continuity_id"),
        "dataset": metadata.dataset,
        "house_id": metadata.house_id,
        "prediction_stream_id": _prediction_stream_id(prediction_stream_id),
        "report_time": _integer(report_time, label="report_time"),
        "start_time": _integer(start_time, label="start_time"),
    }
    return "pred_" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class AlarmObservation:
    """Synthetic active-probability and QC input for one complete 10-second bin."""

    bin_left: int
    bin_right: int
    probability: float
    aggregate_valid: bool
    continuity_id: int = 0
    input_quality_flags: int = 0
    hard_gap: bool = False

    def __post_init__(self) -> None:
        left = _integer(self.bin_left, label="bin_left")
        right = _integer(self.bin_right, label="bin_right")
        if right - left != BIN_SECONDS:
            raise AlarmContractError("reference alarm 只接受完整 10 秒时间格")
        if not isinstance(self.aggregate_valid, bool):
            raise AlarmContractError("aggregate_valid 必须是 bool")
        if not isinstance(self.hard_gap, bool):
            raise AlarmContractError("hard_gap 必须是 bool")
        _integer(self.continuity_id, label="continuity_id")
        _uint8(self.input_quality_flags, label="input_quality_flags")
        object.__setattr__(self, "probability", float32(self.probability))


@dataclass(frozen=True, slots=True)
class AlarmMachineState:
    """Serializable reference-alarm state supporting direct injection.
    
    Synthetic state-table fixtures use this type for the four explicit pre-states. Normal streaming callers need not edit internal state."""

    state: AlarmState = AlarmState.OFF
    on_count: int = 0
    off_count: int = 0
    invalid_count: int = 0
    first_on_bin_left: int | None = None
    first_on_probability: float | None = None
    first_low_bin_left: int | None = None
    first_invalid_bin_left: int | None = None
    pending_quality_flags: int = 0
    open_event: PredictionEvent | None = None
    last_quality_valid_bin_right: int | None = None

    def __post_init__(self) -> None:
        try:
            state = AlarmState(self.state)
        except ValueError as exc:
            raise AlarmContractError("未知 reference-alarm 状态") from exc
        object.__setattr__(self, "state", state)
        for label, value in (
            ("on_count", self.on_count),
            ("off_count", self.off_count),
            ("invalid_count", self.invalid_count),
        ):
            value = _integer(value, label=label)
            if value < 0:
                raise AlarmContractError(f"{label} 不能为负")
        for label, value in (
            ("first_on_bin_left", self.first_on_bin_left),
            ("first_low_bin_left", self.first_low_bin_left),
            ("first_invalid_bin_left", self.first_invalid_bin_left),
            ("last_quality_valid_bin_right", self.last_quality_valid_bin_right),
        ):
            if value is not None:
                _integer(value, label=label)
        _uint8(self.pending_quality_flags, label="pending_quality_flags")
        if self.first_on_probability is not None:
            object.__setattr__(
                self,
                "first_on_probability",
                _finite_float32(self.first_on_probability, label="first_on_probability"),
            )
        if state is AlarmState.PENDING_ON:
            if self.open_event is not None or self.on_count < 1 or self.first_on_bin_left is None:
                raise AlarmContractError("PENDING_ON 必须带 first_on marker 且不带 open event")
            if self.first_on_probability is None:
                raise AlarmContractError("PENDING_ON 必须带 first_on_probability")
        elif state in {AlarmState.ON, AlarmState.PENDING_OFF}:
            if self.open_event is None:
                raise AlarmContractError("ON/PENDING_OFF 必须带 open event")
        elif self.open_event is not None:
            raise AlarmContractError("OFF 不能带 open event")
        if state is AlarmState.PENDING_OFF and (self.off_count < 1 or self.first_low_bin_left is None):
            raise AlarmContractError("PENDING_OFF 必须带 first_low marker")


@dataclass(frozen=True, slots=True)
class AlarmTransition:
    """Complete comparable evidence from one transition."""

    pre_state: AlarmMachineState
    post_state: AlarmMachineState
    symbol: AlarmSymbol
    emitted_alarm: PredictionEvent | None = None
    closed_event: PredictionEvent | None = None


def _state_with(
    state: AlarmMachineState,
    *,
    next_state: AlarmState,
    on_count: int = 0,
    off_count: int = 0,
    invalid_count: int = 0,
    first_on_bin_left: int | None = None,
    first_on_probability: float | None = None,
    first_low_bin_left: int | None = None,
    first_invalid_bin_left: int | None = None,
    pending_quality_flags: int = 0,
    open_event: PredictionEvent | None = None,
    last_quality_valid_bin_right: int | None = None,
) -> AlarmMachineState:
    """Construct state centrally to avoid stale transition markers."""

    return AlarmMachineState(
        state=next_state,
        on_count=on_count,
        off_count=off_count,
        invalid_count=invalid_count,
        first_on_bin_left=first_on_bin_left,
        first_on_probability=first_on_probability,
        first_low_bin_left=first_low_bin_left,
        first_invalid_bin_left=first_invalid_bin_left,
        pending_quality_flags=pending_quality_flags,
        open_event=open_event,
        last_quality_valid_bin_right=(
            state.last_quality_valid_bin_right
            if last_quality_valid_bin_right is None
            else last_quality_valid_bin_right
        ),
    )


def _event_with_flags(event: PredictionEvent, *, input_quality_flags: int, extra_flags: int = 0) -> PredictionEvent:
    flags = _uint16(event.quality_flags | input_quality_flags | extra_flags, label="quality_flags")
    return replace(event, quality_flags=flags)


def _new_event(
    *,
    prediction_stream_id: str,
    metadata: AlarmMetadata,
    start_time: int,
    report_time: int,
    confidence: float,
    quality_flags: int,
) -> PredictionEvent:
    return PredictionEvent(
        event_id=build_prediction_event_id(
            prediction_stream_id=prediction_stream_id,
            metadata=metadata,
            start_time=start_time,
            report_time=report_time,
        ),
        prediction_stream_id=prediction_stream_id,
        dataset=metadata.dataset,
        house_id=metadata.house_id,
        continuity_id=metadata.continuity_id,
        appliance_class=metadata.appliance_class,
        start_time=start_time,
        report_time=report_time,
        end_time=None,
        close_report_time=None,
        confidence=confidence,
        end_reason=None,
        right_censored=None,
        quality_flags=quality_flags,
    )


def _close_event(
    state: AlarmMachineState,
    *,
    end_time: int,
    close_report_time: int,
    end_reason: str,
    input_quality_flags: int,
    close_flag: int = 0,
) -> PredictionEvent:
    if state.open_event is None:
        raise AlarmContractError("关闭 transition 缺少 open event")
    event = _event_with_flags(
        state.open_event,
        input_quality_flags=input_quality_flags,
        extra_flags=close_flag,
    )
    return replace(
        event,
        end_time=_integer(end_time, label="end_time"),
        close_report_time=_integer(close_report_time, label="close_report_time"),
        end_reason=end_reason,
        right_censored=end_reason in {"data_gap", "stream_end"},
    )


def _valid_symbol(symbol: AlarmSymbol) -> bool:
    return symbol in {
        AlarmSymbol.VALID_HIGH,
        AlarmSymbol.VALID_LOW,
        AlarmSymbol.VALID_DEADBAND,
    }


def transition_alarm_state(
    state: AlarmMachineState,
    symbol: AlarmSymbol | str,
    *,
    bin_left: int,
    bin_right: int,
    prediction_stream_id: str,
    metadata: AlarmMetadata,
    config: AlarmConfig = AlarmConfig(),
    probability: float | int | None = None,
    input_quality_flags: int = 0,
) -> AlarmTransition:
    """Perform one frozen state transition without IO.
    
    Direct state injection supplies one of the seven defined symbols and does not inspect QC history. Normal streaming callers should use ReferenceAlarm.step to resolve symbols using the six-bin quality rule."""

    if not isinstance(state, AlarmMachineState):
        raise AlarmContractError("state 必须是 AlarmMachineState")
    try:
        resolved_symbol = AlarmSymbol(symbol)
    except ValueError as exc:
        raise AlarmContractError("未知 alarm transition symbol") from exc
    left = _integer(bin_left, label="bin_left")
    right = _integer(bin_right, label="bin_right")
    if right < left:
        raise AlarmContractError("bin_right 不得早于 bin_left")
    flags = _uint8(input_quality_flags, label="input_quality_flags")
    _prediction_stream_id(prediction_stream_id)
    if not isinstance(metadata, AlarmMetadata):
        raise AlarmContractError("metadata 必须是 AlarmMetadata")
    if not isinstance(config, AlarmConfig):
        raise AlarmContractError("config 必须是 AlarmConfig")
    probability32 = None if probability is None else float32(probability)
    if resolved_symbol is AlarmSymbol.VALID_HIGH:
        if probability32 is None or not math.isfinite(probability32) or probability32 < config.on_threshold:
            raise AlarmContractError("valid_high 必须带 >= on_threshold 的有限 float32 概率")
    if resolved_symbol is AlarmSymbol.VALID_LOW:
        if probability32 is not None and (not math.isfinite(probability32) or not probability32 < config.off_threshold):
            raise AlarmContractError("valid_low 概率必须是 < off_threshold 的有限 float32")
    if resolved_symbol is AlarmSymbol.VALID_DEADBAND:
        if probability32 is not None and (
            not math.isfinite(probability32)
            or probability32 < config.off_threshold
            or probability32 >= config.on_threshold
        ):
            raise AlarmContractError("valid_deadband 概率必须位于冻结 deadband")

    # Set bit8 only for non-finite probability. A direct invalid fixture omits probability;
    # its matrix-defined input is NaN, so it also sets bit8.
    nonfinite_probability = probability32 is None or not math.isfinite(probability32)
    invalid_extra = QUALITY_FLAG_NONFINITE_PROBABILITY if nonfinite_probability else 0
    valid_last = right if _valid_symbol(resolved_symbol) else state.last_quality_valid_bin_right

    def off_state() -> AlarmMachineState:
        return _state_with(state, next_state=AlarmState.OFF, last_quality_valid_bin_right=valid_last)

    def on_state(event: PredictionEvent, *, invalid_count: int = 0, first_invalid: int | None = None) -> AlarmMachineState:
        return _state_with(
            state,
            next_state=AlarmState.ON,
            invalid_count=invalid_count,
            first_invalid_bin_left=first_invalid,
            open_event=event,
            last_quality_valid_bin_right=valid_last,
        )

    def close(
        *, end_time: int, reason: str, close_flag: int = 0
    ) -> AlarmTransition:
        closed = _close_event(
            state,
            end_time=end_time,
            close_report_time=right,
            end_reason=reason,
            input_quality_flags=flags,
            close_flag=close_flag,
        )
        return AlarmTransition(
            pre_state=state,
            post_state=off_state(),
            symbol=resolved_symbol,
            closed_event=closed,
        )

    if state.state is AlarmState.OFF:
        if resolved_symbol is AlarmSymbol.VALID_HIGH:
            post = _state_with(
                state,
                next_state=AlarmState.PENDING_ON,
                on_count=1,
                first_on_bin_left=left,
                first_on_probability=probability32,
                pending_quality_flags=flags,
                last_quality_valid_bin_right=valid_last,
            )
        else:
            post = off_state()
        return AlarmTransition(state, post, resolved_symbol)

    if state.state is AlarmState.PENDING_ON:
        if resolved_symbol is AlarmSymbol.VALID_HIGH:
            if state.on_count + 1 != ON_CONFIRMATION_BINS:
                raise AlarmContractError("冻结 PENDING_ON 只能以第二格确认 ON")
            assert state.first_on_bin_left is not None
            assert state.first_on_probability is not None
            confidence = float32(min(state.first_on_probability, probability32))
            event = _new_event(
                prediction_stream_id=prediction_stream_id,
                metadata=metadata,
                start_time=state.first_on_bin_left,
                report_time=right,
                confidence=confidence,
                quality_flags=_uint16(state.pending_quality_flags | flags, label="quality_flags"),
            )
            post = on_state(event)
            return AlarmTransition(state, post, resolved_symbol, emitted_alarm=event)
        return AlarmTransition(state, off_state(), resolved_symbol)

    # ON and PENDING_OFF both have a dataclass-validated open event.
    assert state.open_event is not None
    if resolved_symbol is AlarmSymbol.HARD_GAP:
        return close(end_time=left, reason="data_gap", close_flag=QUALITY_FLAG_HARD_GAP_CLOSE)
    if resolved_symbol is AlarmSymbol.STREAM_END:
        if state.last_quality_valid_bin_right is None:
            raise AlarmContractError("stream_end 前缺少 last_quality_valid_bin_right")
        return close(
            end_time=state.last_quality_valid_bin_right,
            reason="stream_end",
            close_flag=QUALITY_FLAG_STREAM_END_CLOSE,
        )
    if resolved_symbol is AlarmSymbol.INVALID_CLOSE:
        if state.invalid_count + 1 != OFF_CONFIRMATION_BINS or state.first_invalid_bin_left is None:
            raise AlarmContractError("invalid_close 必须恰为第三个连续 invalid")
        return close(
            end_time=state.first_invalid_bin_left,
            reason="data_gap",
            close_flag=invalid_extra | QUALITY_FLAG_INVALID_CLOSE,
        )
    if resolved_symbol is AlarmSymbol.INVALID_SOFT:
        if state.invalid_count + 1 >= OFF_CONFIRMATION_BINS:
            raise AlarmContractError("第三个连续 invalid 必须解析为 invalid_close")
        event = _event_with_flags(
            state.open_event,
            input_quality_flags=flags,
            extra_flags=invalid_extra,
        )
        post = on_state(
            event,
            invalid_count=state.invalid_count + 1,
            first_invalid=state.first_invalid_bin_left if state.invalid_count else left,
        )
        return AlarmTransition(state, post, resolved_symbol)

    if state.state is AlarmState.ON:
        if resolved_symbol is AlarmSymbol.VALID_LOW:
            event = _event_with_flags(state.open_event, input_quality_flags=flags)
            post = _state_with(
                state,
                next_state=AlarmState.PENDING_OFF,
                off_count=1,
                first_low_bin_left=left,
                open_event=event,
                last_quality_valid_bin_right=valid_last,
            )
        else:  # valid_high and valid_deadband cancel opposing pending/invalid runs.
            event = _event_with_flags(state.open_event, input_quality_flags=flags)
            post = on_state(event)
        return AlarmTransition(state, post, resolved_symbol)

    # Three consecutive valid-low bins close PENDING_OFF; high/deadband restores ON.
    if resolved_symbol is AlarmSymbol.VALID_LOW:
        next_off_count = state.off_count + 1
        if next_off_count == OFF_CONFIRMATION_BINS:
            assert state.first_low_bin_left is not None
            with_current_flags = replace(
                state,
                open_event=_event_with_flags(state.open_event, input_quality_flags=flags),
            )
            closed = _close_event(
                with_current_flags,
                end_time=state.first_low_bin_left,
                close_report_time=right,
                end_reason="normal_off",
                input_quality_flags=0,
            )
            return AlarmTransition(state, off_state(), resolved_symbol, closed_event=closed)
        if next_off_count > OFF_CONFIRMATION_BINS:
            raise AlarmContractError("PENDING_OFF 的 off_count 超过冻结上限")
        event = _event_with_flags(state.open_event, input_quality_flags=flags)
        post = _state_with(
            state,
            next_state=AlarmState.PENDING_OFF,
            off_count=next_off_count,
            first_low_bin_left=state.first_low_bin_left,
            open_event=event,
            last_quality_valid_bin_right=valid_last,
        )
        return AlarmTransition(state, post, resolved_symbol)

    event = _event_with_flags(state.open_event, input_quality_flags=flags)
    return AlarmTransition(state, on_state(event), resolved_symbol)


def _confirmation_quality(history: Sequence[AlarmObservation], observation: AlarmObservation) -> bool:
    """Compute confirmation quality from the current bin and previous five bins."""

    if len(history) != WARMUP_BINS - 1:
        return False
    positions = [*history, observation]
    continuity_id = observation.continuity_id
    if any(item.continuity_id != continuity_id for item in positions):
        return False
    if any(left.bin_right != right.bin_left for left, right in zip(positions, positions[1:])):
        return False
    if not observation.aggregate_valid:
        return False
    return sum(int(item.aggregate_valid) for item in positions) >= WARMUP_MINIMUM_VALID_BINS


def classify_observation(
    *,
    state: AlarmMachineState,
    observation: AlarmObservation,
    confirmation_quality: bool,
    config: AlarmConfig = AlarmConfig(),
) -> AlarmSymbol:
    """Resolve one observation into one of seven transition symbols using frozen priority."""

    if observation.hard_gap:
        return AlarmSymbol.HARD_GAP
    probability = observation.probability
    if not confirmation_quality or not math.isfinite(probability):
        if state.state in {AlarmState.ON, AlarmState.PENDING_OFF} and state.invalid_count + 1 >= OFF_CONFIRMATION_BINS:
            return AlarmSymbol.INVALID_CLOSE
        return AlarmSymbol.INVALID_SOFT
    if probability >= config.on_threshold:
        return AlarmSymbol.VALID_HIGH
    if probability < config.off_threshold:
        return AlarmSymbol.VALID_LOW
    return AlarmSymbol.VALID_DEADBAND


class ReferenceAlarm:
    """Independent read-only synthetic active-probability reference alarm.
    
    Instances hold no file paths, model objects or threshold-search interface. step is the streaming entry; run calls the same step sequentially for strict batch/stream equivalence checks."""

    def __init__(
        self,
        prediction_stream_id: str,
        *,
        metadata: AlarmMetadata = AlarmMetadata(),
        config: AlarmConfig = AlarmConfig(),
        initial_state: AlarmMachineState | None = None,
    ) -> None:
        self._prediction_stream_id = _prediction_stream_id(prediction_stream_id)
        if not isinstance(metadata, AlarmMetadata):
            raise AlarmContractError("metadata 必须是 AlarmMetadata")
        if not isinstance(config, AlarmConfig):
            raise AlarmContractError("config 必须是 AlarmConfig")
        self._metadata = metadata
        self._config = config
        self._state = initial_state if initial_state is not None else AlarmMachineState()
        if not isinstance(self._state, AlarmMachineState):
            raise AlarmContractError("initial_state 必须是 AlarmMachineState")
        if self._state.open_event is not None:
            if self._state.open_event.prediction_stream_id != self._prediction_stream_id:
                raise AlarmContractError("initial_state open event 的 prediction_stream_id 不匹配")
            open_event = self._state.open_event
            if (
                open_event.dataset != self._metadata.dataset
                or open_event.house_id != self._metadata.house_id
                or open_event.continuity_id != self._metadata.continuity_id
                or open_event.appliance_class != self._metadata.appliance_class
            ):
                raise AlarmContractError("initial_state open event 的稳定身份不匹配")
        self._history: list[AlarmObservation] = []
        self._last_observation: AlarmObservation | None = None
        self._ended = False
        self._emitted_alarms: list[PredictionEvent] = []
        self._closed_events: list[PredictionEvent] = []
        self._event_ids: set[str] = (
            {self._state.open_event.event_id} if self._state.open_event is not None else set()
        )

    @property
    def state(self) -> AlarmMachineState:
        """Return the current immutable state snapshot."""

        return self._state

    @property
    def emitted_alarms(self) -> tuple[PredictionEvent, ...]:
        """Return the emitted alarm prefix at report_time, without rewriting earlier records."""

        return tuple(self._emitted_alarms)

    @property
    def closed_events(self) -> tuple[PredictionEvent, ...]:
        """Return closure records without replacing earlier emitted_alarms snapshots."""

        return tuple(self._closed_events)

    def _validate_order(self, observation: AlarmObservation) -> None:
        previous = self._last_observation
        if previous is None:
            return
        if observation.bin_left < previous.bin_right:
            raise AlarmContractError("observation 时间不能回退或重叠")
        if observation.continuity_id == previous.continuity_id:
            if observation.bin_left != previous.bin_right:
                raise AlarmContractError("同一 continuity 的 10 秒 observation 不得缺格")
        elif not observation.hard_gap and self._state.state in {AlarmState.ON, AlarmState.PENDING_OFF}:
            raise AlarmContractError("活动 event 跨 continuity 必须由 hard_gap fail-closed 关闭")

    def _append_quality_history(self, observation: AlarmObservation) -> None:
        if observation.hard_gap:
            # Any new continuity after a hard gap must satisfy the six-bin warm-up again.
            self._history.clear()
            return
        if self._last_observation is not None and observation.continuity_id != self._last_observation.continuity_id:
            self._history.clear()
        self._history.append(observation)
        del self._history[: max(0, len(self._history) - (WARMUP_BINS - 1))]

    def _record(self, transition: AlarmTransition) -> AlarmTransition:
        if transition.emitted_alarm is not None:
            event_id = transition.emitted_alarm.event_id
            if event_id in self._event_ids:
                raise AlarmContractError("prediction event ID collision，必须 fail closed")
            self._event_ids.add(event_id)
            self._emitted_alarms.append(transition.emitted_alarm)
        if transition.closed_event is not None:
            if transition.closed_event.event_id not in self._event_ids:
                raise AlarmContractError("关闭记录引用了未发射的 prediction event")
            self._closed_events.append(transition.closed_event)
        self._state = transition.post_state
        return transition

    def step(self, observation: AlarmObservation) -> AlarmTransition:
        """Process one complete 10-second bin and return transition evidence."""

        if self._ended:
            raise AlarmContractError("stream_end 后不得再输入 observation")
        if not isinstance(observation, AlarmObservation):
            raise AlarmContractError("step 只接受 AlarmObservation")
        if observation.continuity_id != self._metadata.continuity_id:
            raise AlarmContractError("observation continuity_id 必须与本 reference stream 身份一致")
        self._validate_order(observation)
        quality = _confirmation_quality(self._history[-(WARMUP_BINS - 1) :], observation)
        symbol = classify_observation(
            state=self._state,
            observation=observation,
            confirmation_quality=quality,
            config=self._config,
        )
        transition = transition_alarm_state(
            self._state,
            symbol,
            bin_left=observation.bin_left,
            bin_right=observation.bin_right,
            prediction_stream_id=self._prediction_stream_id,
            metadata=self._metadata,
            config=self._config,
            probability=observation.probability,
            input_quality_flags=observation.input_quality_flags,
        )
        self._record(transition)
        self._append_quality_history(observation)
        self._last_observation = observation
        return transition

    def transition_symbol(
        self,
        symbol: AlarmSymbol | str,
        *,
        bin_left: int,
        bin_right: int,
        probability: float | int | None = None,
        input_quality_flags: int = 0,
    ) -> AlarmTransition:
        """Apply a direct symbol transition for a single-bin state-table fixture."""

        if self._ended:
            raise AlarmContractError("stream_end 后不得再执行 transition")
        transition = transition_alarm_state(
            self._state,
            symbol,
            bin_left=bin_left,
            bin_right=bin_right,
            prediction_stream_id=self._prediction_stream_id,
            metadata=self._metadata,
            config=self._config,
            probability=probability,
            input_quality_flags=input_quality_flags,
        )
        self._record(transition)
        return transition

    def stream_end(self, observation_time: int) -> AlarmTransition:
        """End the stream at its actual source-end observation time; never invent a normal offset."""

        if self._ended:
            raise AlarmContractError("stream_end 只能调用一次")
        observation_time = _integer(observation_time, label="stream_end observation_time")
        if self._last_observation is not None and observation_time < self._last_observation.bin_right:
            raise AlarmContractError("stream_end observation_time 不得早于最后 observation 右边界")
        left = (
            self._state.last_quality_valid_bin_right
            if self._state.last_quality_valid_bin_right is not None
            else observation_time
        )
        transition = transition_alarm_state(
            self._state,
            AlarmSymbol.STREAM_END,
            bin_left=left,
            bin_right=observation_time,
            prediction_stream_id=self._prediction_stream_id,
            metadata=self._metadata,
            config=self._config,
        )
        self._record(transition)
        self._ended = True
        return transition

    def run(
        self,
        observations: Iterable[AlarmObservation],
        *,
        stream_end_observation_time: int | None = None,
    ) -> tuple[AlarmTransition, ...]:
        """Replay a fixed synthetic sequence through the same step path as streaming."""

        transitions = [self.step(observation) for observation in observations]
        if stream_end_observation_time is not None:
            transitions.append(self.stream_end(stream_end_observation_time))
        return tuple(transitions)


__all__ = [
    "AlarmConfig",
    "AlarmContractError",
    "AlarmMachineState",
    "AlarmMetadata",
    "AlarmObservation",
    "AlarmState",
    "AlarmSymbol",
    "AlarmTransition",
    "BIN_SECONDS",
    "FROZEN_CLASS_COUNT",
    "FROZEN_CLASS_ORDER",
    "OFF_CONFIRMATION_BINS",
    "ON_CONFIRMATION_BINS",
    "PredictionEvent",
    "ReferenceAlarm",
    "WARMUP_BINS",
    "WARMUP_MINIMUM_VALID_BINS",
    "build_prediction_event_id",
    "build_prediction_stream_id",
    "classify_observation",
    "float32",
    "float32_hex",
    "threshold_bitpatterns",
    "transition_alarm_state",
]
