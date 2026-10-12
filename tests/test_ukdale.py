from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from load_signature_analysis.ukdale import DeviceSpec, extract_channel_intervals  # noqa: E402


class UkDaleIntervalTests(unittest.TestCase):
    def test_short_off_gap_is_merged_into_session(self) -> None:
        spec = DeviceSpec("test_heater", ("test_heater",), 100.0, 6.0, 20.0)
        samples = [
            (0, 0),
            (6, 500),
            (12, 500),
            (18, 0),
            (24, 500),
            (30, 500),
            (36, 0),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "channel.dat"
            path.write_text("".join(f"{timestamp} {power}\n" for timestamp, power in samples), encoding="utf-8")
            intervals = extract_channel_intervals(path, spec)
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0]["start_timestamp"], 6)
        self.assertEqual(intervals[0]["end_timestamp"], 36)
        self.assertEqual(intervals[0]["max_active_power_w"], 500)


if __name__ == "__main__":
    unittest.main()

