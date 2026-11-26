"""Algorithm contracts for scaled BindingDB C1f."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.bindingdb_c1f import assign_remaining_targets, protein_features


class BindingDBC1fTests(unittest.TestCase):
    def test_protein_features_are_deterministic_and_normalized(self):
        sequences = ["ACDEFGHIK", "ACDEFGHIK", "YYYYYYYYY"]
        first = protein_features(sequences)
        second = protein_features(sequences)
        np.testing.assert_allclose(first.toarray(), second.toarray())
        self.assertAlmostEqual(float(first[0].multiply(first[1]).sum()), 1.0)
        self.assertLess(float(first[0].multiply(first[2]).sum()), 1.0)

    def test_remaining_target_assignment_populates_every_target(self):
        frame = pd.DataFrame({
            "target_id": [f"t{i}" for i in range(9)],
            "sequence": ["ACDEFGHIK" + chr(65 + i) * 3 for i in range(9)],
            "degree": np.arange(1, 10),
        })
        features = protein_features(frame.sequence.tolist())
        anchors = np.array([0, 1, 2])
        labels, confidence = assign_remaining_targets(
            frame, features, anchors, np.array(["train", "val", "test"]),
            k=3, balance_weight=2.0, seed=42,
        )
        self.assertEqual(len(labels), len(frame))
        self.assertFalse(pd.isna(labels).any())
        self.assertTrue(np.isfinite(confidence).all())


if __name__ == "__main__":
    unittest.main()
