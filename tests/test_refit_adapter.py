from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from load_signature_analysis.data_adapters import RefitCsvAdapter


class RefitCsvAdapterTests(unittest.TestCase):
    """Fix the REFIT adapter quality contract using a tiny synthetic wide table."""

    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.project_root = Path(self._temporary_directory.name)
        raw_dir = self.project_root / "raw"
        raw_dir.mkdir()

        # House 1 emulates original REFIT files without an Issues column.
        pd.DataFrame(
            {
                "Unix": [1_000.0, 1_007.0, 1_014.0],
                "Aggregate": [500.0, 0.0, 1_200.0],
                "Appliance1": [0.0, 0.0, 1_100.0],
            }
        ).to_csv(raw_dir / "House_1.csv", index=False)

        # House 2 emulates CLEAN files; Issues=1 and Aggregate<=0 each create missing observations.
        pd.DataFrame(
            {
                "Unix": [2_000.0, 2_007.0, 2_014.0, 2_021.0],
                "Aggregate": [500.0, 1_500.0, 0.0, 800.0],
                "Appliance1": [0.0, 1_200.0, 0.0, 700.0],
                "Appliance2": [50.0, 1_000.0, 0.0, 100.0],
                "Issues": [0.0, 1.0, 0.0, 0.0],
            }
        ).to_csv(raw_dir / "House_2.csv", index=False)

        appliance_rows = [
            self._appliance_row(1, 1, "Appliance1", "kettle", "core_positive", 1),
            self._appliance_row(2, 1, "Appliance1", "kettle", "core_positive", 1),
            self._appliance_row(2, 2, "Appliance2", "microwave", "hard_negative", 0),
            self._appliance_row(
                2, 3, "Appliance3", "iron", "extended_positive", 1
            ),
            self._appliance_row(
                2,
                4,
                "Appliance4",
                "toaster",
                "core_positive",
                1,
                include_in_main_mvp=False,
            ),
        ]
        pd.DataFrame(appliance_rows).to_csv(
            self.project_root / "appliance_inventory.csv", index=False
        )
        pd.DataFrame(
            [
                self._aggregate_row(1, "raw/House_1.csv", eligible=True),
                self._aggregate_row(2, "raw/House_2.csv", eligible=True),
                self._aggregate_row(3, "raw/missing.csv", eligible=False),
            ]
        ).to_csv(self.project_root / "aggregate_inventory.csv", index=False)

        # A small chunksize also verifies that cross-chunk assembly preserves mask semantics.
        self.adapter = RefitCsvAdapter(
            self.project_root,
            Path("appliance_inventory.csv"),
            Path("aggregate_inventory.csv"),
            chunksize=2,
        )

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    @staticmethod
    def _appliance_row(
        house_id: int,
        channel_id: int,
        source_column: str,
        canonical_label: str,
        taxonomy_role: str,
        binary_label: int,
        *,
        include_in_main_mvp: bool = True,
    ) -> dict[str, object]:
        return {
            "dataset": "refit",
            "house_id": house_id,
            "channel_id": channel_id,
            "source_column": source_column,
            "canonical_label": canonical_label,
            "semantic_class": "synthetic_test",
            "binary_label": binary_label,
            "taxonomy_confidence": "high",
            "measurement_unit": "W",
            "include_in_main_mvp": include_in_main_mvp,
            "taxonomy_role": taxonomy_role,
        }

    @staticmethod
    def _aggregate_row(
        house_id: int, source_file: str, *, eligible: bool
    ) -> dict[str, object]:
        return {
            "dataset": "refit",
            "house_id": house_id,
            "source_kind": "primary_inventory_source",
            "source_file": source_file,
            "active_power_physically_compatible": True,
            "house_policy_primary_eligible": eligible,
        }

    def test_inventory_filters_house_and_taxonomy_policy(self) -> None:
        self.assertEqual(self.adapter.list_houses(), [1, 2])

        default_channels = self.adapter.appliance_channels(2)
        self.assertEqual(
            [(channel.channel_id, channel.taxonomy_role) for channel in default_channels],
            [(1, "core_positive"), (2, "hard_negative")],
        )
        extended = self.adapter.appliance_channels(2, ["extended_positive"])
        self.assertEqual([channel.channel_id for channel in extended], [3])

    def test_issues_and_nonpositive_aggregate_rows_are_invalid(self) -> None:
        channels = self.adapter.appliance_channels(2)
        house = self.adapter.load_house(2, channels)

        np.testing.assert_array_equal(
            house.aggregate_valid_mask, [True, False, False, True]
        )
        np.testing.assert_array_equal(
            house.appliance_valid_mask, [True, False, False, True]
        )
        self.assertTrue(house.quality_field_available)
        self.assertEqual(house.audit["issues_record_count"], 1)
        self.assertEqual(house.audit["aggregate_nonpositive_count"], 1)
        self.assertEqual(house.audit["aggregate_valid_record_count"], 2)
        np.testing.assert_allclose(
            house.appliance_power_w[1], [0.0, 1_200.0, 0.0, 700.0]
        )

    def test_house_without_issues_field_still_uses_aggregate_validity(self) -> None:
        channels = self.adapter.appliance_channels(1)
        house = self.adapter.load_house(1, channels)

        self.assertFalse(house.quality_field_available)
        np.testing.assert_array_equal(
            house.aggregate_valid_mask, [True, False, True]
        )
        np.testing.assert_array_equal(
            house.appliance_valid_mask, [True, False, True]
        )
        self.assertEqual(house.audit["issues_record_count"], 0)
        self.assertEqual(house.audit["aggregate_nonpositive_count"], 1)


if __name__ == "__main__":
    unittest.main()
