from pathlib import Path
import tempfile
import unittest

from load_signature_analysis.inventory import parse_ukdale_meter_models


class MeasurementInventoryTests(unittest.TestCase):
    def test_ukdale_meter_model_parser(self) -> None:
        content = """appliances: []
elec_meters:
  1:
    data_location: house_1/channel_1.dat
    device_model: EcoManagerWholeHouseTx
  2:
    data_location: house_1/channel_2.dat
    device_model: CurrentCostTx
building_type: test
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "building1.yaml"
            path.write_text(content, encoding="utf-8")
            result = parse_ukdale_meter_models(path)
        self.assertEqual(result, {1: "EcoManagerWholeHouseTx", 2: "CurrentCostTx"})


if __name__ == "__main__":
    unittest.main()
