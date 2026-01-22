import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from augment_v5.modifiers.chain import apply_chain


class TestOrchestratorsCopilot(unittest.TestCase):
    def setUp(self):
        self.sr = 22050
        t = np.linspace(0, 1.0, self.sr, endpoint=False)
        self.sig = (np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def test_env_orch_runs(self):
        chain = [{"op": "env_orch", "params": {"distance_mode": "static", "distance_m": 100.0, "sr": self.sr, "doppler": {"enabled": False}}}]
        out, meta = apply_chain(self.sig.copy(), chain, {"seed": 0})
        self.assertEqual(len(out), len(self.sig))

    def test_hardware_orch_runs(self):
        chain = [{"op": "hardware_orch", "params": {"noise_rms_dbfs": -60, "adc_bits": 12, "hp_hz": 100, "sr": self.sr}}]
        out, meta = apply_chain(self.sig.copy(), chain, {"seed": 0})
        self.assertEqual(len(out), len(self.sig))

    def test_drone_orch_runs(self):
        chain = [{"op": "drone_orch", "params": {"flutter_db": 1.0, "flutter_hz": 0.7, "sr": self.sr}}]
        out, meta = apply_chain(self.sig.copy(), chain, {"seed": 0})
        self.assertEqual(len(out), len(self.sig))


if __name__ == '__main__':
    unittest.main()
