"""Contracts for consuming DataSAIL's in-memory assignment result."""
import unittest

from src.datasail_adapter import assignments_from_result


class DataSAILAdapterTests(unittest.TestCase):
    def test_extracts_requested_technique_and_run(self):
        result = (
            {"C1e": [{"l1": "train"}, {"l1": "test"}]},
            {},
            {"C1e": [{("l1", "t1"): "train"}]},
        )
        assignments = assignments_from_result(result, "C1e", run=0)
        self.assertEqual(assignments.e_splits, {"l1": "train"})
        self.assertEqual(assignments.f_splits, {})
        self.assertEqual(assignments.interaction_splits, {("l1", "t1"): "train"})

    def test_rejects_missing_c1e_assignments(self):
        with self.assertRaisesRegex(ValueError, "no e-assignments"):
            assignments_from_result(({}, {}, {}), "C1e")

    def test_rejects_non_package_return_shape(self):
        with self.assertRaisesRegex(ValueError, "must return"):
            assignments_from_result({"C1e": []}, "C1e")

    def test_extracts_c1f_target_assignments(self):
        result = (None, {"C1f": [{"t1": "test"}]}, None)
        assignments = assignments_from_result(result, "C1f")
        self.assertEqual(assignments.f_splits, {"t1": "test"})


if __name__ == "__main__":
    unittest.main()
