"""Focused contracts for scaled BindingDB C2."""
from __future__ import annotations

import unittest

import numpy as np

from src.bindingdb_c2 import fingerprint_similarity, molecular_fingerprints


class BindingDBC2Tests(unittest.TestCase):
    def test_fingerprint_similarity_is_symmetric(self):
        fingerprints = molecular_fingerprints(["CCO", "CCN", "c1ccccc1"])
        matrix = fingerprint_similarity(fingerprints, np.array([0, 1, 2]))
        np.testing.assert_allclose(matrix, matrix.T)
        np.testing.assert_allclose(np.diag(matrix), 1.0)


if __name__ == "__main__":
    unittest.main()
