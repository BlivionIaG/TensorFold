"""Cost allocation must use measured costs and conserve actual device bytes."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("cost", Path(__file__).resolve().parents[1] / "exl3_offload_cost.py")
cost = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cost)


def fixture():
    identity = {key: "a" * 64 for key in ("build_sha256", "checkpoint_sha256", "workload_sha256", "settings_sha256")}
    layers = []
    for index, times in enumerate(((100, 90, 80), (500, 200, 100))):
        points = [dict(identity=identity.copy(), outputs_equal=True, lines=lines, variable_device_bytes=lines * 256 + 128, service_ns=[ns] * 5) for lines, ns in zip((10, 11, 12), times)]
        layers.append(dict(layer=index, points=points))
    return dict(schema=1, identity=identity, rank=0, world=2, uniform_lines=11, layers=layers)


class CostTests(unittest.TestCase):
    def test_measured_miss_service_moves_line_to_expensive_layer(self):
        result = cost.propose(fixture())
        self.assertEqual(result["lines"], (10, 12))
        self.assertEqual(result["candidate_surrogate_ns"], 200)
        self.assertEqual(result["variable_device_bytes"], result["baseline_variable_device_bytes"])
        self.assertEqual(result["environment"], {"TF_FLASHNEXT_EXL3_LINE_DELTAS_R0": "0:-1,1:+1"})

    def test_full_variable_bytes_override_apparent_latency_win(self):
        profile = fixture()
        profile["layers"][1]["points"][2]["variable_device_bytes"] += 4
        self.assertEqual(cost.propose(profile)["lines"], (11, 11))

    def test_refuse_mixed_identity_unverified_and_insufficient_samples(self):
        for field, value in (("identity", {}), ("outputs_equal", False), ("service_ns", [1])):
            profile = fixture()
            profile["layers"][0]["points"][0][field] = value
            with self.assertRaises(ValueError):
                cost.propose(profile)

    def test_refuse_missing_baseline_duplicate_and_unordered_layers(self):
        profiles = [fixture() for _ in range(3)]
        profiles[0]["layers"][0]["points"].pop(1)
        profiles[1]["layers"][0]["points"].append(copy.deepcopy(profiles[1]["layers"][0]["points"][0]))
        profiles[2]["layers"].reverse()
        for profile in profiles:
            with self.assertRaises(ValueError):
                cost.propose(profile)


if __name__ == "__main__":
    unittest.main()
