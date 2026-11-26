"""Focused contracts for C2 validation helpers."""
from __future__ import annotations

import sqlite3
import unittest

from src.bindingdb_c2_validation import regime


class BindingDBC2ValidationTests(unittest.TestCase):
    def test_regime_reports_both_endpoint_types_cold(self):
        with sqlite3.connect(":memory:") as connection:
            connection.execute("ATTACH DATABASE ':memory:' AS prepared")
            connection.executescript("""
                CREATE TABLE chemical_assignments (chemical_id TEXT, split TEXT);
                CREATE TABLE target_assignments (target_id TEXT, split TEXT);
                CREATE TABLE ligand_assignments (ligand_id TEXT, chemical_id TEXT, split TEXT);
                CREATE TABLE interaction_assignments (observation_id TEXT, split TEXT);
                CREATE TABLE prepared.interactions (observation_id TEXT, ligand_id TEXT, target_id TEXT);
            """)
            connection.executemany("INSERT INTO chemical_assignments VALUES (?,?)",
                                   [("c1", "train"), ("c2", "test")])
            connection.executemany("INSERT INTO target_assignments VALUES (?,?)",
                                   [("t1", "train"), ("t2", "test")])
            connection.executemany("INSERT INTO ligand_assignments VALUES (?,?,?)",
                                   [("l1", "c1", "train"), ("l2", "c2", "test")])
            connection.execute("INSERT INTO prepared.interactions VALUES ('o1','l1','t2')")
            connection.execute("INSERT INTO interaction_assignments VALUES ('o1',NULL)")
            result = regime(connection)
        self.assertEqual(result["unseen_test_ligand_rate"], 1.0)
        self.assertEqual(result["unseen_test_target_rate"], 1.0)
        self.assertEqual(result["dropped_endpoint_split_pairs"][0]["count"], 1)


if __name__ == "__main__":
    unittest.main()
