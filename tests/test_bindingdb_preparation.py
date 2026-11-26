"""Contract tests for the streaming BindingDB ingestion stage."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.dataset_download.prepare_bindingdb import (
    matching_columns,
    parse_affinity,
    prepare_bindingdb,
)


COLUMNS = [
    "BindingDB Reactant_set_id",
    "Ligand SMILES",
    "Ligand InChI Key",
    "BindingDB MonomerID",
    "Target Name",
    "Ki (nM)",
    "Kd (nM)",
    "IC50 (nM)",
    "EC50 (nM)",
    "BindingDB Target Chain Sequence",
    "UniProt (SwissProt) Primary ID of Target Chain",
]


def write_fixture(path: Path) -> None:
    rows = [
        ["1", "CCO", "KEY1", "10", "Target A", "10", "20", "", "", "ACDE", "P00001"],
        ["2", "OCC", "KEY1", "10", "Target A", "", "", "<20", "", "ACDE", "P00001"],
        ["3", "not-smiles", "KEY2", "11", "Target A", "", "", "", "30", "ACDE", "P00001"],
        ["4", "CCN", "KEY3", "12", "Target B", "", "", "", "", "FGHI", "P00002"],
        ["5", "CCC", "KEY4", "13", "Target C", "50", "", "", "", "", "P00003"],
    ]
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, sep="\t", index=False)


class BindingDBPreparationTests(unittest.TestCase):
    def test_parse_affinity_preserves_censoring_relation(self):
        self.assertEqual(parse_affinity("< 12.5"), ("<", 12.5))
        self.assertEqual(parse_affinity("100"), ("=", 100.0))
        self.assertIsNone(parse_affinity("n/a"))
        self.assertIsNone(parse_affinity("0"))

    def test_numbered_release_chain_columns_are_discovered(self):
        columns = [
            "BindingDB Target Chain Sequence 1",
            "BindingDB Target Chain Sequence 2",
            "Unrelated",
        ]
        self.assertEqual(
            matching_columns(columns, "BindingDB Target Chain Sequence"), columns[:2]
        )

    def test_preparation_accounts_for_every_source_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source = tmp_path / "bindingdb.tsv"
            output = tmp_path / "prepared"
            write_fixture(source)

            manifest = prepare_bindingdb(source, output_dir=output, chunk_size=2)

            self.assertEqual(manifest["source_measurements"], 5)
            self.assertEqual(manifest["retained_source_measurements"], 2)
            self.assertEqual(manifest["excluded_source_measurements"], 3)
            self.assertEqual(manifest["affinity_observations"], 3)
            self.assertEqual(
                manifest["affinity_observations_by_type"],
                {"Ki": 1, "Kd": 1, "IC50": 1, "EC50": 0},
            )
            self.assertTrue(manifest["source_accounting_valid"])
            self.assertEqual(manifest["unique_ligands"], 1)
            self.assertEqual(manifest["unique_targets"], 1)
            self.assertGreater(manifest["peak_memory_mb"], 0)
            self.assertGreaterEqual(manifest["runtime_seconds"]["total"], 0)
            interactions = pd.read_csv(output / "interactions.tsv.gz", sep="\t")
            measurements = pd.read_csv(output / "measurements.tsv.gz", sep="\t")
            exclusions = pd.read_csv(output / "exclusions.tsv.gz", sep="\t")
            self.assertEqual(set(interactions["affinity_relation"]), {"=", "<"})
            self.assertEqual(set(interactions["affinity_type"]), {"Ki", "Kd", "IC50"})
            self.assertEqual(measurements["observation_count"].sum(), 3)
            self.assertEqual(set(exclusions["reason"]), {
                "invalid_ligand_smiles",
                "no_valid_affinity",
                "missing_target_sequence",
            })
            self.assertEqual(len(measurements), 5)
            self.assertEqual((measurements["status"] == "retained").sum(), 2)
            self.assertEqual((measurements["status"] == "excluded").sum(), 3)
            self.assertEqual(
                json.loads((output / "preparation_manifest.json").read_text()), manifest
            )

    def test_preparation_refuses_to_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source = tmp_path / "bindingdb.tsv"
            output = tmp_path / "prepared"
            write_fixture(source)
            prepare_bindingdb(source, output_dir=output)
            with self.assertRaises(FileExistsError):
                prepare_bindingdb(source, output_dir=output)

    def test_preparation_fails_on_missing_required_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source = tmp_path / "bad.tsv"
            pd.DataFrame({"Ligand SMILES": ["CCO"]}).to_csv(source, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "Unexpected BindingDB schema"):
                prepare_bindingdb(source, output_dir=tmp_path / "prepared")


if __name__ == "__main__":
    unittest.main()
