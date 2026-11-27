"""Contracts for the sequence-aware BindingDB dual encoder."""
from __future__ import annotations

import unittest

import torch

from experiments.analysis.bindingdb_sequence_dual_encoder import (
    MAX_PROTEIN_LENGTH, SequenceDualEncoder, SequenceEncoder,
)


class BindingDBSequenceDualEncoderTests(unittest.TestCase):
    def test_sequence_encoder_and_model_preserve_rows(self):
        rows = [("o1", "CCO", "ACDEFGHIK", "Ki", 7.0),
                ("o2", "CCN", "Y" * (MAX_PROTEIN_LENGTH + 10), "IC50", 6.0)]
        encoder = SequenceEncoder()
        ligand, protein, affinity, labels = encoder.transform(rows)
        output = SequenceDualEncoder()(
            torch.from_numpy(ligand), torch.from_numpy(protein), torch.from_numpy(affinity),
        )
        self.assertEqual(tuple(output.shape), (2,))
        self.assertEqual(len(labels), 2)
        self.assertEqual(encoder.diagnostics()["truncated_protein_rows"], 1)


if __name__ == "__main__":
    unittest.main()
