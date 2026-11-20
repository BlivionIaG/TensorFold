"""The GLP parity comparator rejects mismatched values, modes and layer identities."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import glp_parity as parity

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


class Compare(unittest.TestCase):
    """The comparison the script runs on the two readers' views of one file."""

    def setUp(self):
        if not HAVE_NUMPY:
            self.skipTest("numpy is required")
        self.meta = {
            "glp.mode": "project",
            "glp.spec_version": 3,
            "glp.hook_point": "hc_fold_pre_engram",
            "glp.alpha_default": "1.0",
            "glp.layer_ids_zero_based": "7",
        }
        self.tensors = {"direction.7": numpy.zeros(8, dtype=numpy.float32)}
        self.tensors["direction.7"][3] = 1.0
        self.ours = {
            "mode": "project",
            "spec_version": 3,
            "hook_point": "hc_fold_pre_engram",
            "alpha": 1.0,
            "method": "",
            "contrast": "",
            "base_model": "",
            "base_revision": "",
            "sha256": "0" * 64,
            "layers": [7],
        }
        self.blob = self.tensors["direction.7"].astype("<f4").tobytes()

    def test_the_same_file_is_the_same_view(self):
        problems: list[str] = []
        parity.compare(self.meta, self.tensors, self.ours, self.blob, problems)
        self.assertEqual([], problems)

    def test_a_different_layer_is_reported(self):
        problems: list[str] = []
        ours = dict(self.ours, layers=[8])
        parity.compare(self.meta, self.tensors, ours, self.blob, problems)
        self.assertEqual(1, len(problems))
        self.assertIn("layer ids differ", problems[0])

    def test_a_different_number_is_reported(self):
        problems: list[str] = []
        blob = bytearray(self.blob)
        blob[0] = 0x3F  # a different first element
        parity.compare(self.meta, self.tensors, self.ours, bytes(blob), problems)
        self.assertEqual(1, len(problems))
        self.assertIn("direction.7 differs", problems[0])

    def test_a_layer_list_that_does_not_match_the_tensors_is_reported(self):
        problems: list[str] = []
        meta = dict(self.meta, **{"glp.layer_ids_zero_based": "9"})
        parity.compare(meta, self.tensors, self.ours, self.blob, problems)
        self.assertEqual(1, len(problems))
        self.assertIn("names layers", problems[0])

    def test_a_different_mode_is_reported(self):
        problems: list[str] = []
        parity.compare(self.meta, self.tensors, dict(self.ours, mode="add"), self.blob, problems)
        self.assertEqual(1, len(problems))
        self.assertIn("glp.mode", problems[0])

    def test_a_different_spec_version_is_reported(self):
        problems: list[str] = []
        parity.compare(self.meta, self.tensors, dict(self.ours, spec_version=2), self.blob, problems)
        self.assertEqual(1, len(problems))
        self.assertIn("glp.spec_version", problems[0])

    def test_a_project_direction_that_is_not_unit_by_the_reference_is_reported(self):
        problems: list[str] = []
        tensors = {"direction.7": self.tensors["direction.7"] * 2.0}
        blob = tensors["direction.7"].astype("<f4").tobytes()
        parity.compare(self.meta, tensors, self.ours, blob, problems)
        self.assertEqual(1, len(problems))
        self.assertIn("squared norm", problems[0])

    def test_an_add_direction_of_any_length_is_the_same_view(self):
        problems: list[str] = []
        tensors = {"direction.7": self.tensors["direction.7"] * 2.0}
        blob = tensors["direction.7"].astype("<f4").tobytes()
        parity.compare(self.meta | {"glp.mode": "add"}, tensors, dict(self.ours, mode="add"), blob, problems)
        self.assertEqual([], problems)

    def test_the_specs_base_fields_are_the_base_the_two_readers_compare(self):
        meta = dict(self.meta, **{
            "general.base_model.count": 1,
            "general.base_model.0.repo_url": "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017",
            "general.base_model.0.version": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        })
        ours = dict(self.ours, base_model="deepseek-ai/DeepSeek-V4.1-Flash-1017",
                    base_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
        problems: list[str] = []
        parity.compare(meta, self.tensors, ours, self.blob, problems)
        self.assertEqual([], problems)

    def test_a_different_commit_pin_is_reported(self):
        meta = dict(self.meta, **{
            "general.base_model.0.repo_url": "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017",
            "general.base_model.0.version": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        })
        ours = dict(self.ours, base_model="deepseek-ai/DeepSeek-V4.1-Flash-1017", base_revision="")
        problems: list[str] = []
        parity.compare(meta, self.tensors, ours, self.blob, problems)
        self.assertEqual(1, len(problems))
        self.assertIn("commit pin", problems[0])

    def test_organization_and_name_name_the_repo_when_there_is_no_url(self):
        meta = dict(self.meta, **{
            "general.base_model.0.organization": "deepseek-ai",
            "general.base_model.0.name": "DeepSeek-V4.1-Flash-1017",
        })
        self.assertEqual(("deepseek-ai/DeepSeek-V4.1-Flash-1017", ""), parity.base_of(meta))
        problems: list[str] = []
        parity.compare(meta, self.tensors, self.ours, self.blob, problems)
        self.assertEqual(1, len(problems))
        self.assertIn("base repo", problems[0])

    def test_a_producers_own_glp_base_keys_name_no_base(self):
        meta = dict(self.meta, **{"glp.base_model": "someone/Other", "glp.base_revision": "0000000"})
        self.assertEqual(("", ""), parity.base_of(meta))


if __name__ == "__main__":
    unittest.main()
