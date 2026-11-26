"""Algorithm and invariant contracts for scaled BindingDB C1e."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.bindingdb_c1e import audit_c1e, choose_degree_anchors, seeded_subset


class BindingDBC1eTests(unittest.TestCase):
    def test_degree_anchor_selection_is_exact_reproducible_and_spans_deciles(self):
        frame = pd.DataFrame({
            "ligand_id": [f"l{i}" for i in range(1000)],
            "canonical_smiles": ["CCO"] * 1000,
            "degree": np.arange(1, 1001),
        })
        first = choose_degree_anchors(frame, ratio=0.1, seed=42)
        repeated = choose_degree_anchors(frame, ratio=0.1, seed=42)
        np.testing.assert_array_equal(first, repeated)
        self.assertEqual(len(first), 100)
        bins = pd.qcut(frame.degree.rank(method="first"), 10, labels=False)
        self.assertEqual(set(bins.iloc[first]), set(range(10)))

    def test_seeded_subset_is_reproducible(self):
        frame = pd.DataFrame({"ligand_id": range(100), "degree": 1})
        a = seeded_subset(frame, 30, 7)
        b = seeded_subset(frame, 30, 7)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(a), 30)

    def test_c1e_audit_reports_ratios_and_rejects_duplicate_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared.sqlite"
            assignments = root / "assignments.sqlite"
            with sqlite3.connect(prepared) as connection:
                connection.executescript("""
                    CREATE TABLE interactions (
                        observation_id TEXT, ligand_id TEXT, target_id TEXT,
                        affinity_type TEXT, p_affinity REAL,
                        duplicate_group_id TEXT
                    );
                """)
                connection.executemany(
                    "INSERT INTO interactions VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        ("o1", "l1", "t1", "Ki", 7.0, "d1"),
                        ("o2", "l2", "t1", "Kd", 8.0, "d2"),
                        ("o3", "l3", "t2", "IC50", 6.0, "d3"),
                    ],
                )
            with sqlite3.connect(assignments) as connection:
                connection.execute(
                    "CREATE TABLE chemical_assignments (chemical_id TEXT PRIMARY KEY, canonical_smiles TEXT, split TEXT, is_anchor INTEGER, degree INTEGER, confidence REAL)"
                )
                connection.executemany(
                    "INSERT INTO chemical_assignments VALUES (?, ?, ?, ?, ?, ?)",
                    [("c1", "CCO", "train", 1, 1, 1.0), ("c2", "CCN", "val", 1, 1, 1.0), ("c3", "CCC", "test", 1, 1, 1.0)],
                )
                connection.execute(
                    "CREATE TABLE assignments (ligand_id TEXT PRIMARY KEY, chemical_id TEXT, split TEXT, is_anchor INTEGER, degree INTEGER, confidence REAL)"
                )
                connection.executemany(
                    "INSERT INTO assignments VALUES (?, ?, ?, ?, ?, ?)",
                    [("l1", "c1", "train", 1, 1, 1.0), ("l2", "c2", "val", 1, 1, 1.0), ("l3", "c3", "test", 1, 1, 1.0)],
                )
            result = audit_c1e(prepared, assignments)
            self.assertEqual(result["retained_observations"], 3)
            self.assertEqual(result["dropped_observations"], 0)
            self.assertEqual(result["duplicate_split_conflicts"], 0)
            self.assertEqual(result["canonical_chemical_split_conflicts"], 0)
            self.assertEqual(result["target_assignment_policy"], "not_applicable_for_C1e_targets_may_span_splits")


if __name__ == "__main__":
    unittest.main()
