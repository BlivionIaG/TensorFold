"""The production reduction tree must not be replaced by an FP64 sum followed by one FP32 cast."""
import unittest
try:
    import numpy as np
except ImportError:
    np = None
if np is not None:
    from text_rounding_reference import norm_rows, sum32


@unittest.skipIf(np is None, 'NumPy is required for the normalization rounding replay')
class TextRoundingTests(unittest.TestCase):
    def test_positive_lane_sums_round_at_each_butterfly(self):
        values = np.ones(32, dtype=np.float32)
        values[0] = 2 ** 24
        self.assertEqual(sum32(values), 2 ** 24 + 30)
        self.assertEqual(np.float32(values.astype(np.float64).sum()), 2 ** 24 + 32)
        self.assertEqual(values[0], 2 ** 24)

    def test_hidden_and_head_norms_keep_rows_independent(self):
        for width in (128, 2048):
            values = np.ones((2, width), dtype=np.float32)
            values[1] *= 2
            result = norm_rows(values, np.ones(width), 1e-5)
            np.testing.assert_array_equal(result, np.ones_like(values))
            np.testing.assert_array_equal(values[1], np.full(width, 2))

    def test_bad_geometry_or_non_bf16_inputs_are_refused(self):
        for value, gain in ((np.ones((1, 64)), np.ones(64)),
                            (np.full((1, 128), 1 + 2 ** -30), np.ones(128)),
                            (np.ones((1, 128)), np.ones(127))):
            with self.assertRaises(ValueError):
                norm_rows(value, gain, 1e-5)


if __name__ == '__main__':
    unittest.main()
