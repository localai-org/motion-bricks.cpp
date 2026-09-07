"""Offline temporal-boundary regressions; no GPU or weights."""
import unittest
import numpy as np
from analyze_sonic import active_physics
from check_sonic_boundaries import match_command


class TimingTests(unittest.TestCase):
    def test_release_does_not_include_last_suspended_sample(self):
        data={'time':np.array([1.,1.005]),'wall_time':np.array([25.,25.005])}
        play={'sim_time':1.,'wall_time':25.001}
        np.testing.assert_array_equal(active_physics(data,play),[False,True])

    def test_duplicate_targets_do_not_claim_latency(self):
        rows=[{'index':0,'command_time':1.},{'index':1,'command_time':2.}]
        self.assertIsNone(match_command(rows,1.004))

    def test_future_command_rejected(self):
        with self.assertRaisesRegex(AssertionError,'before production'):
            match_command([{'command_time':2.}],1.)

    def test_stale_command_rejected(self):
        with self.assertRaisesRegex(AssertionError,'stale'):
            match_command([{'command_time':1.}],1.1)

    def test_unique_command_matched(self):
        row={'index':1,'command_time':1.}
        self.assertIs(match_command([row],1.004),row)


if __name__=='__main__': unittest.main()
