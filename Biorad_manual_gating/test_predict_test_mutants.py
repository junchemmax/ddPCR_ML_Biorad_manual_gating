import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mutant_well_analysis import features_from_gate_data
from predict_test_mutants import (
    compare_plates, count_summary, manual_quadrants, plate_identity, top_candidates,
)
from train_and_predict_gate_model import assign_quadrants


class TestMutantInference(unittest.TestCase):
    def test_plate_identity_does_not_merge_exports(self) -> None:
        self.assertEqual(plate_identity(Path("experiment_123_A01_Amplitude.csv")), ("experiment_123", "A01"))
        with self.assertRaises(ValueError):
            plate_identity(Path("experiment_Amplitude.csv"))

    def test_manual_labels_exclude_unknown_and_preserve_quadrants(self) -> None:
        frame = pd.DataFrame({
            "Ch1Amplitude": [20, 20, 0, 0, 10, 10],
            "Ch2Amplitude": [20, 0, 0, 20, 10, 10],
            "mut": ["1", "1", "0", "0", "u", ""],
            "wt": ["1", "0", "0", "1", "0", "1"],
            "Unnamed: 4": [np.nan] * 6,
        })
        quadrants, excluded = manual_quadrants(frame)
        self.assertEqual(quadrants.tolist(), [1, 2, 3, 4])
        self.assertEqual(excluded, 2)
        self.assertEqual(count_summary(quadrants), {"quadrant": "Q1|Q2|Q3|Q4", "count": "1|1|1|1"})
        predicted = assign_quadrants(frame[["Ch1Amplitude", "Ch2Amplitude"]].to_numpy(), 10, 10)
        self.assertEqual(predicted.tolist(), [1, 2, 3, 4, 3, 3])

    def test_unlabeled_inference_needs_no_mutant_annotations(self) -> None:
        data = pd.DataFrame([
            {"dataset": "a", "parent_dataset": "plate_a", "well": "A01",
             "quadrant": "Q1|Q2|Q3|Q4", "count": "1|1|100|100"},
            {"dataset": "b", "parent_dataset": "plate_b", "well": "A01",
             "quadrant": "Q1|Q2|Q3|Q4", "count": "10|10|100|100"},
        ], index=[4, 7])
        features = features_from_gate_data(data)
        self.assertEqual(features["dataset"].tolist(), ["a", "b"])
        self.assertEqual(features["n_called"].tolist(), [202, 220])
        np.testing.assert_allclose(features["plate_mean_rate"], [2 / 202, 20 / 220])
        self.assertNotIn("is_mutant", features)

    def test_top_candidates_exclude_flags_and_preserve_ties(self) -> None:
        data = pd.DataFrame({
            "well": ["A01", "A02", "A03", "A04"],
            "candidate_rank": [1, 1, 3, 4], "poisson_red_flag": [False, False, True, False],
        })
        self.assertEqual(top_candidates(data)["well"].tolist(), ["A01", "A02"])

    def test_comparison_does_not_choose_arbitrary_tie(self) -> None:
        data = pd.DataFrame({
            "project": ["MIP_1"] * 2, "parent_dataset": ["plate"] * 2,
            "well": ["A01", "A02"], "candidate_rank": [1, 1], "poisson_red_flag": [False, False],
        })
        comparison = compare_plates(data, data.copy())
        self.assertEqual(comparison.iloc[0]["trial1_top_wells"], "A01|A02")
        self.assertTrue(comparison.iloc[0]["top_candidate_agreement"])


if __name__ == "__main__":
    unittest.main()
