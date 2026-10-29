"""Reject incomplete serving evidence and incomplete or inconsistent layer traces."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("measure", Path(__file__).resolve().parents[1] / "exl3_offload_measure.py")
measure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(measure)


def report():
    row = dict(complete=True, tokens=16, token_sha256="a" * 64, output_sha256="b" * 64)
    rep = dict(failed=0, unmeasured=0, errors=[], evidence=[row])
    return dict(tokens=16, cells=[dict(prompt="code", temperature=0, streams=1, failed=0, unmeasured=0, timing_valid=True, reps=[rep])])


def log():
    return "\n".join("EXL3-CACHE v=1 rank=0 world=1 phase=decode cache=expert-layer kind=decoder layer=0 lines=16 device_bytes=100 line_bytes=4 timing_scope=last_device_invocation last_device_service_ns=10 last_device_copy_ns=8 last_device_exposed_copy_wait_ns=2" for _ in range(5))


class MeasureTests(unittest.TestCase):
    def test_full_hashes_and_raw_event_samples(self):
        result = measure.measure(log(), report(), report(), 0, expected_layers=1)
        self.assertEqual(result[0]["service_ns"], [10] * 5)

    def test_refuse_bad_reference_and_incomplete_replies(self):
        for field, value in (("token_sha256", "a" * 12), ("output_sha256", "c" * 64), ("complete", False), ("tokens", 15)):
            candidate = copy.deepcopy(report())
            candidate["cells"][0]["reps"][0]["evidence"][0][field] = value
            with self.assertRaises(ValueError):
                measure.measure(log(), candidate, report(), 0, expected_layers=1)

    def test_refuse_partial_layer_coverage_geometry_changes_and_runtime_errors(self):
        with self.assertRaises(ValueError):
            measure.measure(log(), report(), report(), 0)
        for bad in (log() + "\nCUDA_ERROR_INVALID_VALUE", log().replace("lines=16", "lines=17", 1), log().splitlines()[0]):
            with self.assertRaises(ValueError):
                measure.measure(bad, report(), report(), 0, expected_layers=1)


if __name__ == "__main__":
    unittest.main()
