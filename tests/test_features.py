from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from load_signature_analysis.features import extract_electrical_features  # noqa: E402


class ElectricalFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_rate = 12_800.0
        self.frequency = 50.0
        self.time = np.arange(int(self.sample_rate)) / self.sample_rate
        self.voltage = 325.0 * np.sin(2 * np.pi * self.frequency * self.time)

    def test_in_phase_resistive_signal(self) -> None:
        current = 5.0 * np.sin(2 * np.pi * self.frequency * self.time)
        features = extract_electrical_features(current, self.voltage, self.sample_rate, 50.0)
        self.assertGreater(features["power_factor_abs"], 0.99)
        self.assertLess(abs(features["phase_difference_deg"]), 1.0)
        self.assertLess(features["vi_loop_area_normalized"], 0.01)
        self.assertLess(features["linear_residual_ratio"], 0.01)

    def test_lagging_signal_has_loop_area(self) -> None:
        current = 5.0 * np.sin(2 * np.pi * self.frequency * self.time - np.pi / 4)
        features = extract_electrical_features(current, self.voltage, self.sample_rate, 50.0)
        self.assertAlmostEqual(features["power_factor_abs"], np.sqrt(0.5), places=2)
        self.assertLess(features["phase_difference_deg"], -40.0)
        self.assertGreater(features["vi_loop_area_normalized"], 1.0)

    def test_third_harmonic_ratio(self) -> None:
        fundamental = 5.0 * np.sin(2 * np.pi * self.frequency * self.time)
        third = 1.0 * np.sin(2 * np.pi * 3 * self.frequency * self.time)
        features = extract_electrical_features(fundamental + third, self.voltage, self.sample_rate, 50.0)
        self.assertAlmostEqual(features["current_h3_ratio"], 0.2, places=2)
        self.assertGreater(features["current_thd"], 0.19)


if __name__ == "__main__":
    unittest.main()

