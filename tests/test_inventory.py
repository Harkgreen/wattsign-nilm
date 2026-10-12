from pathlib import Path
import tempfile
import unittest

from load_signature_analysis.inventory import (
    classify_appliance,
    parse_refit_appliance_labels,
    scan_refit_house,
    scan_ukdale_house,
)


class InventoryTests(unittest.TestCase):
    def test_refit_readme_parser_handles_wrapped_and_missing_comma_rows(self) -> None:
        content = """House 5
0.Aggregate, 1.Fridge-Freezer, 2.Tumble Dryer 3.Washing Machine, 4.Dishwasher,
5.Computer Site, 6.Television Site, 7.Combination Microwave, 8.Kettle, 9.Toaster
\t!NOTES
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "README.md"
            path.write_text(content, encoding="utf-8")
            labels = parse_refit_appliance_labels(path)
        self.assertEqual(labels[5][0], "Aggregate")
        self.assertEqual(labels[5][2], "Tumble Dryer")
        self.assertEqual(labels[5][3], "Washing Machine")
        self.assertEqual(labels[5][9], "Toaster")

    def test_dedicated_kettle_is_core_positive(self) -> None:
        result = classify_appliance("uk_dale", 2, 8, "kettle")
        self.assertEqual(result["taxonomy_role"], "core_positive")
        self.assertTrue(result["include_in_main_mvp"])

    def test_medium_confidence_main_channel_is_not_in_primary_mvp(self) -> None:
        result = classify_appliance("refit", 1, 7, "Computer Site")
        self.assertEqual(result["data_use"], "main")
        self.assertEqual(result["taxonomy_confidence"], "medium")
        self.assertFalse(result["include_in_main_mvp"])

    def test_ukdale_kettle_radio_is_ignored(self) -> None:
        result = classify_appliance("uk_dale", 4, 3, "kettle_radio")
        self.assertEqual(result["taxonomy_role"], "ignore")

    def test_ukdale_house1_kettle_meter_is_weak_only(self) -> None:
        result = classify_appliance("uk_dale", 1, 10, "kettle")
        self.assertEqual(result["taxonomy_role"], "ignore")
        self.assertEqual(result["data_use"], "weak_label_only")
        self.assertEqual(result["weak_label_candidate"], "kettle")

    def test_refit_mixed_hair_dryer_plug_is_ignored(self) -> None:
        result = classify_appliance("refit", 17, 9, "Plug Site (Bedroom)")
        self.assertEqual(result["taxonomy_role"], "ignore")
        self.assertEqual(result["anomaly_code"], "mixed_plug_site")
        self.assertEqual(result["data_use"], "weak_label_only")
        self.assertEqual(result["weak_label_candidate"], "hair_dryer")

    def test_refit_house17_kettle_misc_channel_is_ignored(self) -> None:
        result = classify_appliance("refit", 17, 8, "Kettle")
        self.assertEqual(result["taxonomy_role"], "ignore")
        self.assertEqual(result["anomaly_code"], "mixed_target_and_unknown")

    def test_pv_house_semantics_are_separate_from_data_use(self) -> None:
        result = classify_appliance("refit", 11, 7, "Kettle")
        self.assertEqual(result["semantic_class"], "core_positive")
        self.assertEqual(result["data_use"], "robustness_only")
        self.assertFalse(result["include_in_main_mvp"])

    def test_uncertain_refit_heater_is_ignored(self) -> None:
        result = classify_appliance("refit", 16, 3, "Electric Heater (1)?")
        self.assertEqual(result["taxonomy_role"], "ignore")

    def test_refit_metadata_conflict_is_ignored(self) -> None:
        result = classify_appliance("refit", 15, 8, "Kettle")
        self.assertEqual(result["taxonomy_role"], "ignore")
        self.assertEqual(result["anomaly_code"], "metadata_conflict")

    def test_house_anomaly_maps_are_dataset_specific(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            refit_path = root / "House_1.csv"
            refit_path.write_text(
                "Unix,Aggregate,Appliance1,Appliance2,Appliance3,Appliance4,"
                "Appliance5,Appliance6,Appliance7,Appliance8,Appliance9\n"
                "100,100,0,0,0,0,0,0,0,0,0\n"
                "107,110,0,0,0,0,0,0,0,0,0\n",
                encoding="utf-8",
            )
            refit_house, _ = scan_refit_house(
                refit_path,
                {0: "Aggregate", **{index: "Not Used" for index in range(1, 10)}},
                chunksize=1,
            )

            uk_house_path = root / "house_1"
            uk_house_path.mkdir()
            (uk_house_path / "labels.dat").write_text("1 aggregate\n", encoding="utf-8")
            (uk_house_path / "channel_1.dat").write_text(
                "100 100\n107 110\n", encoding="utf-8"
            )
            uk_house, _ = scan_ukdale_house(uk_house_path, chunksize=1)

        self.assertEqual(refit_house["house_anomaly_code"], "clean_copy_truncated")
        self.assertEqual(uk_house["house_anomaly_code"], "submeters_switched_off")


if __name__ == "__main__":
    unittest.main()
