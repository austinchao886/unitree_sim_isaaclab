import unittest
from action_provider.command_timing import CommandTiming


class CommandTimingTests(unittest.TestCase):
    def test_hidden_gap_persists_after_recent_fast_samples(self):
        timing=CommandTiming()
        timing.accept(1.);timing.accept(1.6)
        timing.observe_age(.6)
        for i in range(1,5001):
            timing.accept(1.6+i*.02);timing.observe_age(.01)
        snapshot=timing.snapshot()
        self.assertEqual(snapshot["accepted_count"],5002)
        self.assertEqual(snapshot["interval_counts_over"],[1,1,1,0])
        self.assertEqual(snapshot["age_observation_counts_over"],[1,1,1,0])
        self.assertAlmostEqual(snapshot["max_accepted_interval_s"],.6)
    def test_first_command_has_no_prestart_interval(self):
        timing=CommandTiming();timing.accept(1000.)
        self.assertEqual(timing.snapshot()["interval_counts_over"],[0]*4)
    def test_reject_invalid_and_snapshot_is_copy(self):
        timing=CommandTiming();timing.accept(1.)
        for value in (1.,.5,float("nan"),float("inf")):
            with self.assertRaises(ValueError):timing.accept(value)
        for value in (-1.,float("nan")):
            with self.assertRaises(ValueError):timing.observe_age(value)
        snapshot=timing.snapshot();snapshot["interval_counts_over"][0]=99
        self.assertEqual(timing.snapshot()["interval_counts_over"][0],0)


if __name__=="__main__":unittest.main()
