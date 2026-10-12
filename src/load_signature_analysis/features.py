"""Physically interpretable features for synchronized voltage/current waveforms."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


HARMONICS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

FULL_FEATURES = (
    "mains_frequency_hz",
    "voltage_rms",
    "current_rms",
    "active_power_w",
    "apparent_power_va",
    "power_factor_abs",
    "phase_difference_deg",
    "voltage_current_correlation",
    "current_crest_factor",
    "current_form_factor",
    "current_skewness",
    "current_kurtosis",
    "current_thd",
    "vi_loop_area_normalized",
    "linear_residual_ratio",
    *tuple(f"current_h{order}_ratio" for order in HARMONICS),
)

# Removes absolute power/current magnitude so that the result measures waveform
# shape rather than simply memorising the typical wattage of each appliance.
SHAPE_FEATURES = tuple(
    name
    for name in FULL_FEATURES
    if name not in {"voltage_rms", "current_rms", "active_power_w", "apparent_power_va"}
)


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal))))


def _safe_ratio(numerator: float, denominator: float, default: float = np.nan) -> float:
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return default
    return float(numerator / denominator)


def estimate_mains_frequency(
    voltage: np.ndarray,
    sample_rate_hz: float,
    lower_hz: float = 45.0,
    upper_hz: float = 65.0,
) -> float:
    """Estimate the grid fundamental from the strongest voltage FFT bin."""

    centered = np.asarray(voltage, dtype=np.float64) - float(np.mean(voltage))
    if centered.size < 8:
        return np.nan
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(centered.size)))
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate_hz)
    mask = (frequencies >= lower_hz) & (frequencies <= upper_hz)
    if not np.any(mask):
        return np.nan
    candidate_indices = np.flatnonzero(mask)
    peak_index = int(candidate_indices[np.argmax(spectrum[mask])])
    return float(frequencies[peak_index])


def _complex_amplitude(signal: np.ndarray, sample_rate_hz: float, frequency_hz: float) -> complex:
    times = np.arange(signal.size, dtype=np.float64) / sample_rate_hz
    kernel = np.exp(-2j * np.pi * frequency_hz * times)
    return 2.0 * np.dot(signal, kernel) / signal.size


def _standardized_moment(signal: np.ndarray, order: int) -> float:
    centered = signal - float(np.mean(signal))
    std = float(np.std(centered))
    if std < 1e-12:
        return np.nan
    return float(np.mean(np.power(centered / std, order)))


def _normalized_vi_area(voltage: np.ndarray, current: np.ndarray, samples_per_cycle: int) -> float:
    if samples_per_cycle < 4 or voltage.size < samples_per_cycle:
        return np.nan
    v_cycle = voltage[-samples_per_cycle:].astype(np.float64, copy=True)
    i_cycle = current[-samples_per_cycle:].astype(np.float64, copy=True)
    v_scale = float(np.max(np.abs(v_cycle)))
    i_scale = float(np.max(np.abs(i_cycle)))
    if v_scale < 1e-12 or i_scale < 1e-12:
        return np.nan
    x = v_cycle / v_scale
    y = i_cycle / i_scale
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def extract_electrical_features(
    current: np.ndarray,
    voltage: np.ndarray,
    sample_rate_hz: float,
    nominal_mains_hz: float = 60.0,
    steady_cycles: int = 30,
) -> dict[str, float]:
    """Extract power, harmonic, phase and V-I geometry features.

    The final ``steady_cycles`` are used because PLAID records an off-to-on
    transition followed by steady operation.  No appliance metadata enters the
    calculation, which keeps the feature cache suitable for grouped evaluation.
    """

    current = np.asarray(current, dtype=np.float64).reshape(-1)
    voltage = np.asarray(voltage, dtype=np.float64).reshape(-1)
    usable = min(current.size, voltage.size)
    if usable < 1_000:
        raise ValueError(f"waveform is too short: {usable} samples")
    current = current[-usable:]
    voltage = voltage[-usable:]
    finite = np.isfinite(current) & np.isfinite(voltage)
    current = current[finite]
    voltage = voltage[finite]
    if current.size < 1_000:
        raise ValueError("waveform has too few finite samples")

    frequency_hz = estimate_mains_frequency(voltage, sample_rate_hz)
    if not np.isfinite(frequency_hz):
        frequency_hz = nominal_mains_hz
    samples_per_cycle = max(4, int(round(sample_rate_hz / frequency_hz)))
    steady_count = min(current.size, steady_cycles * samples_per_cycle)
    current = current[-steady_count:] - float(np.mean(current[-steady_count:]))
    voltage = voltage[-steady_count:] - float(np.mean(voltage[-steady_count:]))

    # Orient current so that imported files with an inverted CT do not create a
    # negative active-power feature for an otherwise identical appliance.
    if float(np.mean(current * voltage)) < 0:
        current = -current

    voltage_rms = _rms(voltage)
    current_rms = _rms(current)
    active_power = float(np.mean(voltage * current))
    apparent_power = voltage_rms * current_rms
    power_factor = abs(_safe_ratio(active_power, apparent_power))

    voltage_fundamental = _complex_amplitude(voltage, sample_rate_hz, frequency_hz)
    current_fundamental = _complex_amplitude(current, sample_rate_hz, frequency_hz)
    phase_difference = np.angle(current_fundamental / voltage_fundamental, deg=True)
    phase_difference = float((phase_difference + 180.0) % 360.0 - 180.0)

    current_h1 = abs(current_fundamental)
    harmonic_amplitudes: dict[int, float] = {}
    harmonic_ratios: dict[int, float] = {}
    for order in HARMONICS:
        amplitude = abs(_complex_amplitude(current, sample_rate_hz, order * frequency_hz))
        harmonic_amplitudes[order] = float(amplitude)
        harmonic_ratios[order] = _safe_ratio(amplitude, current_h1)
    thd = _safe_ratio(
        float(np.sqrt(sum(value * value for value in harmonic_amplitudes.values()))),
        current_h1,
    )

    current_mean_abs = float(np.mean(np.abs(current)))
    correlation = float(np.corrcoef(voltage, current)[0, 1])
    linear_scale = _safe_ratio(float(np.dot(current, voltage)), float(np.dot(voltage, voltage)))
    if np.isfinite(linear_scale) and current_rms > 1e-12:
        residual_ratio = _rms(current - linear_scale * voltage) / current_rms
    else:
        residual_ratio = np.nan

    features: dict[str, float] = {
        "mains_frequency_hz": float(frequency_hz),
        "voltage_rms": voltage_rms,
        "current_rms": current_rms,
        "active_power_w": active_power,
        "apparent_power_va": apparent_power,
        "power_factor_abs": power_factor,
        "phase_difference_deg": phase_difference,
        "voltage_current_correlation": correlation,
        "current_crest_factor": _safe_ratio(float(np.max(np.abs(current))), current_rms),
        "current_form_factor": _safe_ratio(current_rms, current_mean_abs),
        "current_skewness": _standardized_moment(current, 3),
        "current_kurtosis": _standardized_moment(current, 4),
        "current_thd": thd,
        "vi_loop_area_normalized": _normalized_vi_area(voltage, current, samples_per_cycle),
        "linear_residual_ratio": float(residual_ratio),
    }
    features.update({f"current_h{order}_ratio": harmonic_ratios[order] for order in HARMONICS})
    return features


def feature_vector(features: Mapping[str, float], names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([features[name] for name in names], dtype=np.float64)

