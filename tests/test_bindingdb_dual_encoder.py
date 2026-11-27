"""Contracts for the nonlinear BindingDB dual encoder."""
from __future__ import annotations

import unittest

import torch

from experiments.analysis.bindingdb_dual_encoder import DenseEncoder, DualEncoder


class BindingDBDualEncoderTests(unittest.TestCase):
    def test_encoder_and_model_preserve_batch_rows(self):
        rows = [("o1", "CCO", "ACDEFGHIK", "Ki", 7.0),
                ("o2", "CCN", "YYYYYYYYY", "IC50", 6.0)]
        ligand, protein, affinity, labels = DenseEncoder().transform(rows)
        model = DualEncoder()
        output = model(torch.from_numpy(ligand), torch.from_numpy(protein),
                       torch.from_numpy(affinity))
        self.assertEqual(tuple(output.shape), (2,))
        self.assertEqual(len(labels), 2)


if __name__ == "__main__":
    unittest.main()
