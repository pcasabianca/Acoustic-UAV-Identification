import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from augment_v5.modifiers.ops_distance import distance_atten, trajectory_atten


class TestTrajectoryAttenCopilot(unittest.TestCase):
    def setUp(self):
        self.sr = 22050
        t = np.linspace(0, 1.0, self.sr, endpoint=False)
        self.sig = (np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def test_start_eq_end_matches_distance_atten(self):
        out_const, m_const = distance_atten(self.sig.copy(), {"distance_m": 10.0, "ref_m": 1.0})
        out_traj, m_traj = trajectory_atten(self.sig.copy(), {"distance_start_m": 10.0, "distance_end_m": 10.0, "ref_m": 1.0, "curve": "linear"})
        np.testing.assert_allclose(out_const, out_traj, atol=1e-5)
        self.assertIn("distance_start_m", m_traj)


if __name__ == '__main__':
    unittest.main()
