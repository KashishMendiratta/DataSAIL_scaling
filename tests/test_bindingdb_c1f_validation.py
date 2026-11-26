"""Focused contracts for the BindingDB C1f validator."""
from __future__ import annotations

import sqlite3
import unittest

from src.bindingdb_c1f_validation import exact_protein_similarity


class BindingDBC1fValidationTests(unittest.TestCase):
    def test_exact_cross_split_protein_maximum(self):
        with sqlite3.connect(":memory:") as connection:
            connection.execute("ATTACH DATABASE ':memory:' AS prepared")
            connection.execute("CREATE TABLE prepared.targets (target_id TEXT, sequence TEXT)")
            connection.execute("CREATE TABLE assignments (target_id TEXT, split TEXT)")
            rows = [("a", "ACDEFGHIK", "train"), ("b", "ACDEFGHIK", "val"),
                    ("c", "YYYYYYYYY", "test")]
            connection.executemany("INSERT INTO prepared.targets VALUES (?, ?)",
                                   [(target, sequence) for target, sequence, _ in rows])
            connection.executemany("INSERT INTO assignments VALUES (?, ?)",
                                   [(target, split) for target, _, split in rows])
            result = exact_protein_similarity(connection, chunk_size=1)
        self.assertAlmostEqual(result["maximum_cross_split_protein_similarity"], 1.0)
        self.assertEqual(result["calculation"], "exact exhaustive cross-split target pairs")


if __name__ == "__main__":
    unittest.main()
