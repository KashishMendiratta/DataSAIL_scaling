"""Contracts for the bounded BindingDB predictive baseline."""
from __future__ import annotations

import unittest

import numpy as np

from experiments.analysis.bindingdb_predictive_baseline import FeatureEncoder, metrics


class BindingDBPredictiveBaselineTests(unittest.TestCase):
    def test_interaction_features_include_all_rows(self):
        rows = [("o1", "CCO", "ACDEFGHIK", "Ki", 7.0),
                ("o2", "CCN", "YYYYYYYYY", "IC50", 6.0)]
        matrix, labels = FeatureEncoder().transform(rows)
        self.assertEqual(matrix.shape[0], 2)
        self.assertEqual(len(labels), 2)
        self.assertGreater(matrix.nnz, 0)

    def test_perfect_predictions_have_zero_error(self):
        values = np.asarray([5.0, 6.0, 7.0])
        result = metrics(values, values)
        self.assertEqual(result["mae"], 0.0)
        self.assertAlmostEqual(result["r2"], 1.0)


if __name__ == "__main__":
    unittest.main()
