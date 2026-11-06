import unittest
from pc_tests.model_pipeline.analysis.audit_telemetry_boundaries_v2 import reconcile


def row(count=2,power=0):
    return dict(index=0,timed_telemetry_samples=count,software_powercap_samples=power,software_thermal_samples=0,
        start_reading=dict(utc='2026-11-04T00:00:00+00:00',clock_reasons=0),
        end_reading=dict(utc='2026-11-04T00:00:01+00:00',clock_reasons=0))


def sample(second,reason=0):
    return dict(utc=f'2026-11-04T00:00:{second:06.3f}+00:00',clock_reasons=reason)


class BoundaryTests(unittest.TestCase):
    def test_extra_edge_power_retained(self):
        r=reconcile(row(),[sample(.2),sample(.8),sample(1.004,4)])
        self.assertEqual(r['recorded_powercap_samples'],0)
        self.assertEqual(r['conservative_powercap_samples'],1)

    def test_edge_included_in_snapshot(self):
        r=reconcile(row(3,1),[sample(.2),sample(.8),sample(1.001,4)])
        self.assertEqual(r['phase_labelled_samples'],3)

    def test_missing_sample_rejected(self):
        with self.assertRaises(AssertionError): reconcile(row(3),[sample(.2),sample(.8)])

    def test_unexplained_interior_extra_rejected(self):
        with self.assertRaises(AssertionError): reconcile(row(),[sample(.2),sample(.5),sample(.8)])

    def test_far_spill_rejected(self):
        with self.assertRaises(AssertionError): reconcile(row(),[sample(.2),sample(.8),sample(1.1)])

    def test_edge_thermal_not_discarded(self):
        with self.assertRaises(AssertionError): reconcile(row(),[sample(.2),sample(.8),sample(1.004,32)])


if __name__=='__main__': unittest.main()
