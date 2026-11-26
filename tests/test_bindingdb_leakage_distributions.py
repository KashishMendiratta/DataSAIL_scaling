"""Contracts for BindingDB leakage-distribution analysis."""
from __future__ import annotations

import unittest

import numpy as np

from experiments.analysis.bindingdb_leakage_distributions import (
    deterministic_sample, summarize,
)


class BindingDBLeakageDistributionTests(unittest.TestCase):
    def test_sampling_is_reproducible_and_capped(self):
        rows = [(str(index), index) for index in range(100)]
        self.assertEqual(deterministic_sample(rows, 10, 42), deterministic_sample(rows, 10, 42))
        self.assertEqual(len(deterministic_sample(rows, 10, 42)), 10)

    def test_summary_reports_threshold_fractions(self):
        result = summarize(np.asarray([0.4, 0.7, 0.8, 1.0]))
        self.assertEqual(result["max"], 1.0)
        self.assertEqual(result["fraction_ge_0.7"], 0.75)
        self.assertEqual(result["fraction_at_1"], 0.25)


if __name__ == "__main__":
    unittest.main()
