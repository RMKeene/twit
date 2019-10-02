
from twit import *
import numpy as np
import numpy.testing as npt
import unittest
import numpy.testing as nt
from typing import List, Tuple

import twitc

class TestTwitc(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_AA_interp(self):
        self.assertAlmostEqual(twitc.twit_interp(0, 10, 0.0, 5.0, 0), 0.0)
        self.assertAlmostEqual(twitc.twit_interp(0, 10, 0.0, 5.0, 11), 5.5)
        self.assertAlmostEqual(twitc.twit_interp(10, 0, 0.0, 5.0, 1), 4.5)

        self.assertAlmostEqual(twitc.twit_interp(10, 0, 5.0, 0.0, 10), 5.0)
        self.assertAlmostEqual(twitc.twit_interp(10, 0, 5.0, 0.0, 1), 0.5)
        self.assertAlmostEqual(twitc.twit_interp(0, 0, 1.0, 5.0, 11), 1.0)

        pass

    def test_AAM_interp(self):
        self.assertFalse(twitc.outside_range(0, 10, 5))
        self.assertFalse(twitc.outside_range(0, 10, 10))
        self.assertFalse(twitc.outside_range(0, 10, 0))
        self.assertFalse(twitc.outside_range(10, 0, 5))
        self.assertFalse(twitc.outside_range(10, 0, 10))
        self.assertFalse(twitc.outside_range(10, 0, 0))
        self.assertTrue(twitc.outside_range(0, 10, -5))
        self.assertTrue(twitc.outside_range(0, 10, 11))
        self.assertTrue(twitc.outside_range(0, 10, -1))
        self.assertTrue(twitc.outside_range(10, 0, 500))
        self.assertFalse(twitc.outside_range(10, 0, 3))
        self.assertFalse(twitc.outside_range(10, 0, 2))

        pass


    def test_AB_range_series(self):
        t1 = ((3, 4), (0.6666666666666666, 0.3333333333333333))
        t2 = ((4, 5, 6), (0.25, 0.5, 0.25))
        t3 = ((6, 7, 8), (0.25, 0.5, 0.25))
        t4 = ((8, 9), (0.3333333333333333, 0.6666666666666666))
        a : tuple = twitc.find_range_series_multipliers(1, 4, 3, 9, 1)
        #print(a)
        nt.assert_almost_equal(a, t1)
        a : tuple = twitc.find_range_series_multipliers(1, 4, 3, 9, 2)
        #print(a)
        nt.assert_almost_equal(a, t2)
        a : tuple = twitc.find_range_series_multipliers(1, 4, 3, 9, 3)
        nt.assert_almost_equal(a, t3)
        a : tuple = twitc.find_range_series_multipliers(1, 4, 3, 9, 4)
        nt.assert_almost_equal(a, t4)

        t1 = ((4, 5, 6), (0.25, 0.5, 0.25))
        t2 = ((6, 7, 8), (0.25, 0.5, 0.25))
        t3 = ((8, 9), (0.3333333333333333, 0.6666666666666666))
        t4 = None
        a : tuple = twitc.find_range_series_multipliers(0, 3, 3, 9, 1)
        nt.assert_almost_equal(a, t1)
        a : tuple = twitc.find_range_series_multipliers(0, 3, 3, 9, 2)
        nt.assert_almost_equal(a, t2)
        a : tuple = twitc.find_range_series_multipliers(0, 3, 3, 9, 3)
        nt.assert_almost_equal(a, t3)
        a : tuple = twitc.find_range_series_multipliers(3, 0, 3, 9, 1)
        nt.assert_almost_equal(a, t1)
        a : tuple = twitc.find_range_series_multipliers(0, 3, 9, 3, 2)
        nt.assert_almost_equal(a, t2)
        a : tuple = twitc.find_range_series_multipliers(0, 3, 3, 9, 3)
        nt.assert_almost_equal(a, t3)


        # One to Many
        a : tuple = twitc.find_range_series_multipliers(1, 1, 3, 9, 1)
        nt.assert_almost_equal(a, ((3, 4, 5, 6, 7, 8, 9), (0.14285714285714288, 
                                   0.14285714285714288, 0.14285714285714288, 0.14285714285714288, 0.14285714285714288, 
                                   0.14285714285714288, 0.14285714285714288)))

        # Test out of range narrow_idx.
        with self.assertRaises(Exception):
            a : tuple = twitc.find_range_series_multipliers(0, 3, 3, 9, 4)
        with self.assertRaises(Exception):
            a : tuple = twitc.find_range_series_multipliers(None, None, 3, 9, 2)
        with self.assertRaises(Exception):
            a : tuple = twitc.find_range_series_multipliers(0, 3, None, None, 2)
        with self.assertRaises(Exception):
            a : tuple = twitc.find_range_series_multipliers(0, 3, 3, 9, -1)
        with self.assertRaises(Exception):
            a : tuple = twitc.find_range_series_multipliers(-1, 3, 3, 9, 1)
        with self.assertRaises(Exception):
            a : tuple = twitc.find_range_series_multipliers(0, 3, 3, -9, 1)

        pass


    def test_EA_compute_twit_single_dimension(self):
        # 1 to 1
        t = twitc.compute_twit_single_dimension(1, 1, 5, 5, 10.0, 10.0)
        nt.assert_almost_equal(t, ([1], [5], [10.]))
        # 1 to 4
        t = twitc.compute_twit_single_dimension(1, 1, 2, 5, 5.0, 10.0)
        nt.assert_almost_equal(t, ([1,1,1,1], [2,3,4,5], [5., 6.666666666666667, 8.333333333333334, 10.0]))
        # 4 to 1
        t = twitc.compute_twit_single_dimension(2, 5, 1, 1, 5.0, 10.0)
        nt.assert_almost_equal(t, ([2,3,4,5], [1,1,1,1], [1.25, 1.66666667, 2.08333333, 2.5]))
        # 1 to 4 reverse W
        t = twitc.compute_twit_single_dimension(1, 1, 2, 5, 10.0, 5.0)
        nt.assert_almost_equal(t, ([1,1,1,1], [2,3,4,5], [10.0, 8.333333333333334, 6.666666666666667, 5.]))
        # 4 to 1 rev W
        t = twitc.compute_twit_single_dimension(2, 5, 1, 1, 10.0, 5.0)
        nt.assert_almost_equal(t, ([2,3,4,5], [1,1,1,1], [2.5, 2.08333333, 1.66666667, 1.25]))

        # 3 to 5
        t = twitc.compute_twit_single_dimension(1, 3, 0, 4, 5.0, 10.0)
        nt.assert_almost_equal(t, ([1, 1, 2, 2, 2, 3, 3], [0, 1, 1, 2, 3, 3, 4], [5., 3.571428571428571, 2.678571428571429, 7.5, 3.7500000000000004, 5.0, 10.0]))
        # 5 to 3
        t = twitc.compute_twit_single_dimension(1, 4, 0, 2, 5.0, 10.0)
        nt.assert_almost_equal(t, ([1, 2, 2, 3, 3, 4], [0, 0, 1, 1, 2, 2], [3.75, 1.66666667, 3.33333333, 4.16666667, 2.08333333, 7.5]))
        # 5 to 2
        t = twitc.compute_twit_single_dimension(0, 4, 0, 1, 1.0, 1.0)
        nt.assert_almost_equal(t, ([0, 1, 2, 3, 1, 2, 3, 4], [0, 0, 0, 0, 1, 1, 1, 1], [0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.4]))
        pass

    def test_FA_compute_twit_multi_dimension(self):
        t = twitc.compute_twit_multi_dimension(2, np.array([0, 3, 0, 4, 0, 4, 0, 1], dtype=np.int64), np.array([0.0, 1.0, 1.0, 1.0], dtype=np.double))
        self.assertEqual(2, t[0])
        self.assertEqual(8, t[1][0])
        self.assertEqual(8, t[2][0])
        nt.assert_array_almost_equal(t[1][1], [0, 0, 1, 1, 2, 2, 3, 3])
        nt.assert_array_almost_equal(t[1][2], [0, 1, 1, 2, 2, 3, 3, 4])
        nt.assert_array_almost_equal(t[1][3], [0.0, 0.0625, 0.1875, 0.25, 0.25, 0.5625, 0.1875, 1.0])
        nt.assert_array_almost_equal(t[2][1], [0, 1, 2, 3, 1, 2, 3, 4])
        nt.assert_array_almost_equal(t[2][2], [0, 0, 0, 0, 1, 1, 1, 1])
        nt.assert_array_almost_equal(t[2][3], [0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.4])
        pass

    def test_FA_multi_axis_interpolation(self):
        pass


if __name__ == '__main__':

    unittest.main(verbosity=2)
