"""Contracts for final BindingDB C1e validation."""
from __future__ import annotations

import sqlite3
import unittest

import numpy as np

from src.bindingdb_c1e_validation import (
    UnionFind, certify_ligand_maximum, summarize,
)


class BindingDBC1eValidationTests(unittest.TestCase):
    def test_summary_reports_distribution(self):
        result = summarize(np.array([1, 2, 3, 10]))
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["max"], 10)
        self.assertEqual(result["median"], 2.5)

    def test_union_find_merges_bipartite_component(self):
        groups = UnionFind(5)
        groups.union(0, 3)
        groups.union(1, 3)
        groups.union(2, 4)
        self.assertEqual(groups.find(0), groups.find(1))
        self.assertNotEqual(groups.find(0), groups.find(2))

    def test_cross_split_identical_fingerprint_certifies_global_maximum(self):
        with sqlite3.connect(":memory:") as connection:
            connection.execute(
                "CREATE TABLE chemical_assignments (chemical_id TEXT, canonical_smiles TEXT, split TEXT)"
            )
            connection.executemany(
                "INSERT INTO chemical_assignments VALUES (?, ?, ?)",
                [
                    ("left", "CC[C@H](O)C", "train"),
                    ("right", "CC[C@@H](O)C", "test"),
                ],
            )
            result = certify_ligand_maximum(connection)
        self.assertEqual(result["maximum_cross_split_ligand_similarity"], 1.0)
        self.assertEqual(result["certificate"], "identical_fingerprint_cross_split")


if __name__ == "__main__":
    unittest.main()
