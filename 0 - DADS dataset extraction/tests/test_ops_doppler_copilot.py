import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from augment_v5.modifiers.ops_doppler import doppler_segmented
except Exception:
    doppler_segmented = None


class TestDopplerSegmentedCopilot(unittest.TestCase):
    def setUp(self):
        self.sr = 22050
        t = np.linspace(0, 1.0, self.sr, endpoint=False)
        self.sig = (np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def test_v_zero_identity(self):
        if doppler_segmented is None:
            self.skipTest("doppler_segmented not available (scipy missing)")
        out, meta = doppler_segmented(self.sig.copy(), {"v_mps": 0.0, "direction": 1, "segments": 16, "max_ratio": 1.02, "sr": self.sr})
        np.testing.assert_allclose(out, self.sig, atol=1e-4)

    def test_directionality_and_determinism(self):
        if doppler_segmented is None:
            self.skipTest("doppler_segmented not available (scipy missing)")
        out1, m1 = doppler_segmented(self.sig.copy(), {"v_mps": 10.0, "direction": 1, "segments": 16, "max_ratio": 1.02, "sr": self.sr})
        out2, m2 = doppler_segmented(self.sig.copy(), {"v_mps": 10.0, "direction": -1, "segments": 16, "max_ratio": 1.02, "sr": self.sr})
        self.assertFalse(np.array_equal(out1, out2))
        self.assertFalse(np.array_equal(out1, self.sig))
        self.assertIn("doppler_v_mps", m1)


if __name__ == '__main__':
    unittest.main()
