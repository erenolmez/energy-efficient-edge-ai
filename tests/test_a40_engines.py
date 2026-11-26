import unittest

from hpc.a40_engines import (accuracy_checks, aggregate, canonical_bus, canonical_uuid, match_device,
                             window_metrics, ROLES)


class A40EngineHelperTests(unittest.TestCase):
    def test_uuid_match_ignores_prefix_and_case(self):
        devices = [(0, "GPU-AAAA-1111", "00000000:3B:00.0"), (1, b"GPU-bbbb-2222", "00000000:86:00.0")]
        self.assertEqual(match_device(devices, torch_uuid="bbbb-2222"), (1, "uuid"))

    def test_pci_fallback_and_no_silent_index_zero(self):
        devices = [(0, "GPU-a", "00000000:3B:00.0"), (1, "GPU-b", "00000000:86:00.0")]
        self.assertEqual(match_device(devices, torch_uuid="zzz", torch_bus="0000:86:00.0"), (1, "pci_bus_id"))
        with self.assertRaises(RuntimeError):
            match_device(devices, torch_uuid="zzz", torch_bus=None)
        self.assertEqual(canonical_bus("0000:3B:00.0"), canonical_bus("00000000:3b:00.0"))
        self.assertEqual(canonical_uuid("MIG-ABC"), "abc")

    def test_window_metrics_energy_per_image_and_idle(self):
        m = window_metrics([0.01] * 100, energy_mj=2000.0, images=800, idle_w=50.0)
        self.assertAlmostEqual(m["latency_ms"], 10.0)
        self.assertAlmostEqual(m["ms_per_image"], 1.25)
        self.assertAlmostEqual(m["energy_mj_per_image"], 2.5)
        self.assertAlmostEqual(m["avg_power_w"], 2.0)
        self.assertAlmostEqual(m["energy_above_idle_mj_per_image"], (2000 - 50000) / 800)
        self.assertIsNone(window_metrics([0.01], energy_mj=0, images=8, idle_w=None)["energy_mj_per_image"])

    def test_aggregate_takes_median_and_batch_energy(self):
        rows = [dict(variant="trt_fp16", model="m", level="engine", batch=8, latency_ms=v, latency_p90_ms=v,
                     ms_per_image=v / 8, energy_mj_per_image=e, avg_power_w=100.0)
                for v, e in ((1.0, 2.0), (3.0, 4.0), (2.0, 3.0))]
        (agg,) = aggregate(rows)
        self.assertEqual(agg["latency_ms"], 2.0)
        self.assertEqual(agg["latency_ms_range"], [1.0, 3.0])
        self.assertEqual(agg["energy_mj"], 24.0)
        self.assertIsNone(agg["energy_above_idle_mj_per_image"])

    def test_accuracy_checks_flag_engine_and_resize_drift(self):
        good = {r: (80.0, 100.0) for r in ROLES}
        acc = {"torch_fp32_cpu": {"m": good},
               "trt_fp16_cpu": {"m": {r: (79.9, 99.5) for r in ROLES}},
               "trt_fp16_gpu": {"m": {r: (78.0, 95.0) for r in ROLES}}}
        checks = accuracy_checks(acc, ["m"])
        self.assertTrue(all(c["engine_ok"] for c in checks))
        self.assertFalse(any(c["resize_ok"] for c in checks))


if __name__ == "__main__":
    unittest.main()
