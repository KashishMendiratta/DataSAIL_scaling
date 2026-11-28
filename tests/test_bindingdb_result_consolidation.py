"""Regression tests for accepted BindingDB result consolidation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.analysis.consolidate_bindingdb_results import split_rows


class BindingDBResultConsolidationTests(unittest.TestCase):
    def write_bundle(self, root: Path, regime: str, *, c2: bool) -> None:
        directory = root / "splits" / regime.lower()
        directory.mkdir(parents=True)
        audit = {
            "retained_observations": 80 if c2 else 100,
            "dropped_cross_split_observations" if c2 else "dropped_observations": 20 if c2 else 0,
        }
        manifest = {
            "n_canonical_chemicals": 10, "n_ligand_ids": 11, "n_targets": 3,
            "runtime_seconds": {"total": 1.5}, "peak_memory_mb": 12,
            "audit": audit,
        }
        accounting = {
            "retained_observations": 80 if c2 else 100,
            "dropped_cross_split_observations": 20 if c2 else 0,
        }
        if c2:
            accounting.update({
                "retained_counts": {"train": 56, "val": 16, "test": 8},
                "retained_ratios": {"train": .7, "val": .2, "test": .1},
                "requested_ratios": {"train": .7, "val": .2, "test": .1},
            })
            distributions = {"retained_affinity": [
                {"split": "train", "affinity_type": "Ki", "count": 56}
            ]}
        else:
            accounting.update({
                "actual_interaction_counts": {"train": 70, "val": 20, "test": 10},
                "actual_interaction_ratios": {"train": .7, "val": .2, "test": .1},
                "requested_interaction_ratios": {"train": .7, "val": .2, "test": .1},
            })
            distributions = {"affinity": [
                {"split": "train", "affinity_type": "Ki", "count": 70}
            ]}
        validation = {
            "validation_status": "passed", "checks": {
                "accounting": accounting, "distributions": distributions,
            },
        }
        (directory / "run_manifest.json").write_text(json.dumps(manifest))
        (directory / "validation_report.json").write_text(json.dumps(validation))

    def test_all_regime_accounting_schemas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root, "C1e", c2=False)
            self.write_bundle(root, "C1f", c2=False)
            self.write_bundle(root, "C2", c2=True)
            rows, affinity = split_rows(root)
        by_regime = {row["regime"]: row for row in rows}
        self.assertEqual(by_regime["C1e"]["dropped_observations"], 0)
        self.assertEqual(by_regime["C2"]["dropped_observations"], 20)
        self.assertAlmostEqual(by_regime["C2"]["retention_fraction"], .8)
        self.assertEqual(len(affinity), 3)


if __name__ == "__main__":
    unittest.main()
