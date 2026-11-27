"""Contracts for cached-ESM BindingDB encoding."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.analysis.bindingdb_esm_dual_encoder import CachedESMEncoder, ESMDualEncoder


class BindingDBESMDualEncoderTests(unittest.TestCase):
    def test_cached_encoder_and_model_preserve_rows(self):
        import hashlib
        sequence = "ACDEFGHIK"
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.npz"
            np.savez(cache, sequence_sha256=np.asarray([hashlib.sha256(sequence.encode()).hexdigest()]),
                     embedding=np.ones((1, 320), dtype=np.float32))
            encoded = CachedESMEncoder(cache).transform([
                ("o1", "CCO", sequence, "Ki", 7.0),
            ])
        output = ESMDualEncoder()(torch.from_numpy(encoded[0]), torch.from_numpy(encoded[1]),
                                  torch.from_numpy(encoded[2]))
        self.assertEqual(tuple(output.shape), (1,))


if __name__ == "__main__":
    unittest.main()
