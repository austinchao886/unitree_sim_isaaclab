import math
import unittest
from action_provider.kinematic_window import KinematicWindow


class WindowTests(unittest.TestCase):
    def build(self):
        w=KinematicWindow('session')
        for i in range(201):w.add(i*.005,[.2,0.,0.],.78,.03,.3)
        return w
    def test_time_weighted_constant_velocity(self):
        s=self.build().snapshot()
        self.assertTrue(s['ready']);self.assertEqual(s['sample_count'],201)
        self.assertAlmostEqual(s['mean_planar_speed_m_s'],.2)
        self.assertAlmostEqual(s['mean_planar_velocity_norm_m_s'],.2)
    def test_peak_between_status_updates_is_preserved(self):
        w=KinematicWindow('s')
        for i in range(201):w.add(i*.005,[.6 if i==100 else .2,0.,0.],.78,.03,.3)
        self.assertEqual(w.snapshot()['peak_planar_speed_m_s'],.6)
        for i in range(201,302):w.add(i*.005,[.2,0.,0.],.78,.03,.3)
        self.assertEqual(w.snapshot()['peak_planar_speed_m_s'],.2)
    def test_missing_gap_and_bad_data_are_not_ready(self):
        w=self.build();w.add(1.1,[.2,0.,0.],.78,.03,.3)
        self.assertFalse(w.snapshot()['ready'])
        w.add(1.2,[math.nan,0.,0.],.78,.03,.3)
        self.assertFalse(w.snapshot()['ready']);self.assertEqual(w.snapshot()['sample_count'],0)
    def test_duplicate_clock_resets_evidence(self):
        w=self.build();w.add(1.,[.2,0.,0.],.78,.03,.3)
        self.assertFalse(w.snapshot()['ready'])
        self.assertEqual(w.snapshot()['error'],'nonmonotonic clock')
    def test_signed_velocity_cancellation_does_not_hide_speed(self):
        w=KinematicWindow('s')
        for i in range(201):w.add(i*.005,[.2 if i%2 else -.2,0.,0.],.78,.03,.3)
        s=w.snapshot();self.assertTrue(s['ready'])
        self.assertAlmostEqual(s['mean_planar_velocity_norm_m_s'],0.)
        self.assertAlmostEqual(s['mean_planar_speed_m_s'],.2)


if __name__=='__main__':unittest.main()
