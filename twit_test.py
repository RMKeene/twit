"""
Test for Tensor Weighted Interpolation Transfers (twit module) twit.py
"""
from twit import *
import numpy as np
import numpy.testing as npt
import unittest
import numpy.testing as nt
from typing import List, Tuple

class TestTwit(unittest.TestCase):
    def setUp(self):
        pass


    def test_AA_interp(self):
        self.assertAlmostEqual(twit_interp((0, 10), (0.0, 5.0), 0), 0.0)
        self.assertAlmostEqual(twit_interp((0, 10), (0.0, 5.0), 11), 5.5)
        self.assertAlmostEqual(twit_interp((10, 0), (0.0, 5.0), 1), 4.5)

        self.assertAlmostEqual(twit_interp((10, 0), (5.0, 0.0), 10), 5.0)
        self.assertAlmostEqual(twit_interp((10, 0), (5.0, 0.0), 1), 0.5)
        self.assertAlmostEqual(twit_interp((0, 0), (1.0, 5.0), 11), 1.0)
        pass


    def test_AAA_match_shape_lengths(self):
        self.assertAlmostEqual(match_shape_lengths((4,2), (3,5)), ((4,2), (3,5)))
        self.assertAlmostEqual(match_shape_lengths((7, 4, 2), (3, 5)), ((7, 4, 2), (1, 3, 5)))
        self.assertAlmostEqual(match_shape_lengths((4, 2), (9, 3, 5)), ((1, 4, 2), (9, 3, 5)))
        self.assertAlmostEqual(match_shape_lengths((2,), (9, 3, 5)), ((1, 1, 2), (9, 3, 5)))
        pass


    def test_AAB_gen_twit_param_from_shapes(self):
        p = gen_twit_param_from_shapes((3,4), (3,4))
        npt.assert_array_almost_equal(p, (((0, 2), (0, 2), (1.0, 1.0)), ((0, 3), (0, 3), (1.0, 1.0))))
        p = gen_twit_param(np.zeros((3,4)), np.zeros((3,4)))
        npt.assert_array_almost_equal(p, (((0, 2), (0, 2), (1.0, 1.0)), ((0, 3), (0, 3), (1.0, 1.0))))
        
        p = gen_twit_param_from_shapes((3,4), (7,3,4))
        npt.assert_array_almost_equal(p, (((0, 0), (0, 6), (1.0, 1.0)),((0, 2), (0, 2), (1.0, 1.0)), ((0, 3), (0, 3), (1.0, 1.0))))
        pass


    def test_BB_parse_prims(self):
        a = split_strip(",,", sep=',')
        self.assertEqual(a, [])

        a = split_strip("1, 2 ,3 , 4,5 ")
        # Note: sep=' ' by default so yes, this is a correct assert.
        self.assertEqual(a, ['1,', '2', ',3', ',', '4,5'])
        a = split_strip("1, 2 ,3 , 4,5 ", sep=',')
        self.assertEqual(a, ['1', '2', '3', '4', '5'])
        a = split_strip(None)
        self.assertEqual(a, [])
        a = split_strip("    ", sep=',')
        self.assertEqual(a, [])
        a = split_strip("  ,  ", sep=',')
        self.assertEqual(a, [])
        
        a = tryParseFloat()
        self.assertEqual(a, (0.0, False))
        a = tryParseFloat(None, 3.14159)
        self.assertEqual(a, (3.14159, False))
        a = tryParseFloat("", 3.14159)
        self.assertEqual(a, (3.14159, False))
        a = tryParseFloat("w3", 3.14159)
        self.assertEqual(a, (3.14159, False))
        a = tryParseFloat("3w", 3.14159)
        self.assertEqual(a, (3.14159, False))
        a = tryParseFloat("12.3", 3.14159)
        self.assertEqual(a, (12.3, True))
        a = tryParseFloat("-7.2", 3.14159)
        self.assertEqual(a, (-7.2, True))
        a = tryParseFloat("-", 3.14159)
        self.assertEqual(a, (3.14159, False))
        a = tryParseFloat("-.3", 3.14159)
        self.assertEqual(a, (-0.3, True))

        a = tryParseInt()
        self.assertEqual(a, (0, False))
        a = tryParseInt(None, 3)
        self.assertEqual(a, (3, False))
        a = tryParseInt("", 3)
        self.assertEqual(a, (3, False))
        a = tryParseInt("w4", 3)
        self.assertEqual(a, (3, False))
        a = tryParseInt("4w", 3)
        self.assertEqual(a, (3, False))
        a = tryParseInt("12", 3)
        self.assertEqual(a, (12, True))
        a = tryParseInt("-7", 3)
        self.assertEqual(a, (-7, True))
        a = tryParseInt("-", 3)
        self.assertEqual(a, (3, False))
        a = tryParseInt("-3", 3)
        self.assertEqual(a, (-3, True))

    def test_BC_string_parsing(self):
        a = twit_string_to_ranges_internal(0, 10, 0, 3, 0.1, 0.8, "{5,17} <2,3> [-0.1, 0.9]")
        assert(a == (None, 'Invalid source range'))
        a = twit_string_to_ranges_internal(0, 10, 0, 3, 0.1, 0.8, "{5,2} <2,3> [-0.1, 0.9]")
        nt.assert_almost_equal(a, ((5, 2), (2, 3), (-0.1, 0.9)))
        a = twit_string_to_ranges_internal(0, 10, 0, 3, 0.1, 0.8, "{5,2} <2,-3> [-0.1, 0.9]")
        assert(a == (None, 'Invalid destination range'))
        a = twit_string_to_ranges_internal(0, 10, 0, 3, 0.1, 0.8, None)
        nt.assert_almost_equal(a, ((0, 10), (0, 3), (0.1, 0.8)))
        a = twit_string_to_ranges_internal(0, 10, 0, 3, 0.1, 0.8, "")
        nt.assert_almost_equal(a, ((0, 10), (0, 3), (0.1, 0.8)))
        a = twit_string_to_ranges_internal(0, 10, 0, 3, 0.1, 0.8, "{5,2}")
        nt.assert_almost_equal(a, ((5, 2), (0, 3), (0.1, 0.8)))
        a = twit_string_to_ranges_internal(0, 10, 0, 3, 0.1, 0.8, "<2,3>")
        nt.assert_almost_equal(a, ((0, 10), (2, 3), (0.1, 0.8)))
        a = twit_string_to_ranges_internal(0, 10, 0, 3, 0.1, 0.8, "[-0.1, 0.9]")
        nt.assert_almost_equal(a, ((0, 10), (0, 3), (-0.1, 0.9)))
    
    
    def test_BD_tensor_string_parsing_defaults(self):
        b = np.zeros((12,10,4)) 
        c = np.zeros((100,200))
        a = twit_str_to_ranges(b, 1, c, 1, "{5,2} <2,-3> [-0.1, 0.9]")
        assert(a == (None, 'Invalid destination range'))
        a = twit_str_to_ranges(b, 1, c, 1, "{5,2} <2,3> [-0.1, 0.9]")
        nt.assert_almost_equal(a, ((5, 2), (2, 3), (-0.1, 0.9)))


    def test_CA_find_range_series_multipliers(self):
        t1 = ((3, 0.6666666666666666), (4, 0.3333333333333333))
        t2 = ((4, 0.25), (5, 0.5), (6, 0.25))
        t3 = ((6, 0.25), (7, 0.5), (8, 0.25))
        t4 = ((8, 0.3333333333333333), (9, 0.6666666666666666))
        a : tuple = find_range_series_multipliers((1, 4), (3,9), 1)
        nt.assert_almost_equal(a, t1)
        a : tuple = find_range_series_multipliers((1, 4), (3,9), 2)
        nt.assert_almost_equal(a, t2)
        a : tuple = find_range_series_multipliers((1, 4), (3,9), 3)
        nt.assert_almost_equal(a, t3)
        a : tuple = find_range_series_multipliers((1, 4), (3,9), 4)
        nt.assert_almost_equal(a, t4)

        t1 = ((4, 0.25), (5, 0.5), (6, 0.25))
        t2 = ((6, 0.25), (7, 0.5), (8, 0.25))
        t3 = ((8, 0.3333333333333333), (9, 0.6666666666666666))
        t4 = None
        a : tuple = find_range_series_multipliers((0, 3), (3,9), 1)
        nt.assert_almost_equal(a, t1)
        a : tuple = find_range_series_multipliers((0, 3), (3,9), 2)
        nt.assert_almost_equal(a, t2)
        a : tuple = find_range_series_multipliers((0, 3), (3,9), 3)
        nt.assert_almost_equal(a, t3)
        a : tuple = find_range_series_multipliers((3, 0), (3,9), 1)
        nt.assert_almost_equal(a, t1)
        a : tuple = find_range_series_multipliers((0, 3), (9,3), 2)
        nt.assert_almost_equal(a, t2)
        a : tuple = find_range_series_multipliers((0, 3), (3,9), 3)
        nt.assert_almost_equal(a, t3)


        # One to Many
        a : tuple = find_range_series_multipliers((1, 1), (3,9), 1)
        nt.assert_almost_equal(a, ((3, 0.14285714285714288), 
                                   (4, 0.14285714285714288), 
                                   (5, 0.14285714285714288), 
                                   (6, 0.14285714285714288), 
                                   (7, 0.14285714285714288), 
                                   (8, 0.14285714285714288), 
                                   (9, 0.14285714285714288)))

        # Test out of range narrow_idx.
        with self.assertRaises(Exception):
            a : tuple = find_range_series_multipliers((0, 3), (3,9), 4)
        with self.assertRaises(Exception):
            a : tuple = find_range_series_multipliers(None, (3,9), 2)
        with self.assertRaises(Exception):
            a : tuple = find_range_series_multipliers((0, 3), None, 2)
        with self.assertRaises(Exception):
            a : tuple = find_range_series_multipliers((0,), (3,9), 2)
        with self.assertRaises(Exception):
            a : tuple = find_range_series_multipliers((0,3), (3,), 2)
        with self.assertRaises(Exception):
            a : tuple = find_range_series_multipliers((), (3,9), 2)
        with self.assertRaises(Exception):
            a : tuple = find_range_series_multipliers((0,3), (), 2)
        with self.assertRaises(Exception):
            a : tuple = find_range_series_multipliers((0,3), (3, 9), -1)
        with self.assertRaises(Exception):
            a : tuple = find_range_series_multipliers((-1,3), (3, 9), 1)
        with self.assertRaises(Exception):
            a : tuple = find_range_series_multipliers((0,3), (3, -9), 1)

        pass


    def series_exec(self, title: str, creation_func, args, expected: Tuple, skip_asserts: bool=False):
        if skip_asserts:
            print()
            print(title)
            print("SKIP ASSERTS - NO TEST DONE")
        N = 0
        a = creation_func(*args)
        for r in a:
            if skip_asserts:
                print(r)
            else:
                self.assertAlmostEqual(expected[N], r)
            N += 1
        if not skip_asserts:
            nt.assert_equal(N, len(expected))

        pass


    def test_EA_single_axis_discrete_interpolation(self):
        self.series_exec("One to One", twit_single_axis_discrete,
                                  (((1.0, 5.0, 10.0),(1.0, 5.0, 10.0)),),
                                  ((1, 5, 10.0),))
        self.series_exec("One to 4", twit_single_axis_discrete,
                                  (((1.0, 2.0, 5.0),(1.0, 5.0, 10.0)),),
                                  ((1, 2, 5.0),
                                    (1, 3, 6.666666666666667),
                                    (1, 4, 8.333333333333334),
                                    (1, 5, 10.0)))
        self.series_exec("4 to One", twit_single_axis_discrete,
                                  (((2, 1, 5.0),(5.0, 1.0, 10.0)),),
                                  ((2, 1, 1.25),
                                   (3, 1, 1.6666666666666667),
                                   (4, 1, 2.0833333333333335),
                                   (5, 1, 2.5)))
        self.series_exec("One to 4, rev W", twit_single_axis_discrete,
                                  (((1.0, 2.0, 10.0),(1.0, 5.0, 5.0)),),
                                  ((1, 2, 10.0),
                                    (1, 3, 8.333333333333334),
                                    (1, 4, 6.666666666666666),
                                    (1, 5, 5.0)))
        self.series_exec("4 to One, rev W", twit_single_axis_discrete,
                                  (((2.0, 1.0, 10.0),(5.0, 1.0, 5.0)),),
                                  ((2, 1, 2.5),
                                    (3, 1, 2.0833333333333335),
                                    (4, 1, 1.6666666666666665),
                                    (5, 1, 1.25)))

        self.series_exec("Three to Five", twit_single_axis_discrete,
                                  (((1.0, 0.0, 5.0),(3.0, 4.0, 10.0)),),
                                  ((1, 0, 5.0),
                                    (1, 1, 3.571428571428571),
                                    (2, 1, 2.678571428571429),
                                    (2, 2, 7.5),
                                    (2, 3, 3.7500000000000004),
                                    (3, 3, 5.0),
                                    (3, 4, 10.0)))

        self.series_exec("Five to Three", twit_single_axis_discrete,
                                  (((1.0, 0.0, 5.0),(4.0, 2.0, 10.0)),),
                                  ((1, 0, 3.75),
                                    (2, 0, 1.666666666666667),
                                    (2, 1, 3.3333333333333335),
                                    (3, 1, 4.166666666666667),
                                    (3, 2, 2.0833333333333335),
                                    (4, 2, 7.499999999999999)))
        pass


    def test_FA_multi_axis_interpolation(self):
        self.series_exec("Multi Interp Discrete - Fan In", twit,
                         ((((0,3),(0,4),(0.1,0.3)), ((1,3),(4,4), (1.0, 1.0))),), 
                         (((0, 0, 0.10000000000000002), (1, 4, 0.3333333333333333)),
                            ((0, 0, 0.10000000000000002), (2, 4, 0.3333333333333333)),
                            ((0, 0, 0.10000000000000002), (3, 4, 0.3333333333333333)),
                            ((0, 1, 0.0375), (1, 4, 0.3333333333333333)),
                            ((0, 1, 0.0375), (2, 4, 0.3333333333333333)),
                            ((0, 1, 0.0375), (3, 4, 0.3333333333333333)),
                            ((1, 1, 0.11249999999999999), (1, 4, 0.3333333333333333)),
                            ((1, 1, 0.11249999999999999), (2, 4, 0.3333333333333333)),
                            ((1, 1, 0.11249999999999999), (3, 4, 0.3333333333333333)),
                            ((1, 2, 0.08888888888888886), (1, 4, 0.3333333333333333)),
                            ((1, 2, 0.08888888888888886), (2, 4, 0.3333333333333333)),
                            ((1, 2, 0.08888888888888886), (3, 4, 0.3333333333333333)),
                            ((2, 2, 0.11111111111111115), (1, 4, 0.3333333333333333)),
                            ((2, 2, 0.11111111111111115), (2, 4, 0.3333333333333333)),
                            ((2, 2, 0.11111111111111115), (3, 4, 0.3333333333333333)),
                            ((2, 3, 0.17307692307692307), (1, 4, 0.3333333333333333)),
                            ((2, 3, 0.17307692307692307), (2, 4, 0.3333333333333333)),
                            ((2, 3, 0.17307692307692307), (3, 4, 0.3333333333333333)),
                            ((3, 3, 0.07692307692307691), (1, 4, 0.3333333333333333)),
                            ((3, 3, 0.07692307692307691), (2, 4, 0.3333333333333333)),
                            ((3, 3, 0.07692307692307691), (3, 4, 0.3333333333333333)),
                            ((3, 4, 0.3), (1, 4, 0.3333333333333333)),
                            ((3, 4, 0.3), (2, 4, 0.3333333333333333)),
                            ((3, 4, 0.3), (3, 4, 0.3333333333333333))))
        self.series_exec("Multi Interp Discrete - Fan out", twit,
                         ((((0,4),(0,3),(1.0,1.0)), ((1,3),(0,3), (1.0, 1.0))),), 
                         (((0, 0, 0.8), (1, 0, 1.0)),
                            ((0, 0, 0.8), (1, 1, 0.3333333333333333)),
                            ((0, 0, 0.8), (2, 1, 0.6666666666666666)),
                            ((0, 0, 0.8), (2, 2, 0.6)),
                            ((0, 0, 0.8), (3, 2, 0.4)),
                            ((0, 0, 0.8), (3, 3, 1.0)),
                            ((1, 0, 0.2), (1, 0, 1.0)),
                            ((1, 0, 0.2), (1, 1, 0.3333333333333333)),
                            ((1, 0, 0.2), (2, 1, 0.6666666666666666)),
                            ((1, 0, 0.2), (2, 2, 0.6)),
                            ((1, 0, 0.2), (3, 2, 0.4)),
                            ((1, 0, 0.2), (3, 3, 1.0)),
                            ((1, 1, 0.6000000000000001), (1, 0, 1.0)),
                            ((1, 1, 0.6000000000000001), (1, 1, 0.3333333333333333)),
                            ((1, 1, 0.6000000000000001), (2, 1, 0.6666666666666666)),
                            ((1, 1, 0.6000000000000001), (2, 2, 0.6)),
                            ((1, 1, 0.6000000000000001), (3, 2, 0.4)),
                            ((1, 1, 0.6000000000000001), (3, 3, 1.0)),
                            ((2, 1, 0.39999999999999997), (1, 0, 1.0)),
                            ((2, 1, 0.39999999999999997), (1, 1, 0.3333333333333333)),
                            ((2, 1, 0.39999999999999997), (2, 1, 0.6666666666666666)),
                            ((2, 1, 0.39999999999999997), (2, 2, 0.6)),
                            ((2, 1, 0.39999999999999997), (3, 2, 0.4)),
                            ((2, 1, 0.39999999999999997), (3, 3, 1.0)),
                            ((2, 2, 0.4), (1, 0, 1.0)),
                            ((2, 2, 0.4), (1, 1, 0.3333333333333333)),
                            ((2, 2, 0.4), (2, 1, 0.6666666666666666)),
                            ((2, 2, 0.4), (2, 2, 0.6)),
                            ((2, 2, 0.4), (3, 2, 0.4)),
                            ((2, 2, 0.4), (3, 3, 1.0)),
                            ((3, 2, 0.5999999999999999), (1, 0, 1.0)),
                            ((3, 2, 0.5999999999999999), (1, 1, 0.3333333333333333)),
                            ((3, 2, 0.5999999999999999), (2, 1, 0.6666666666666666)),
                            ((3, 2, 0.5999999999999999), (2, 2, 0.6)),
                            ((3, 2, 0.5999999999999999), (3, 2, 0.4)),
                            ((3, 2, 0.5999999999999999), (3, 3, 1.0)),
                            ((3, 3, 0.2), (1, 0, 1.0)),
                            ((3, 3, 0.2), (1, 1, 0.3333333333333333)),
                            ((3, 3, 0.2), (2, 1, 0.6666666666666666)),
                            ((3, 3, 0.2), (2, 2, 0.6)),
                            ((3, 3, 0.2), (3, 2, 0.4)),
                            ((3, 3, 0.2), (3, 3, 1.0)),
                            ((4, 3, 0.8), (1, 0, 1.0)),
                            ((4, 3, 0.8), (1, 1, 0.3333333333333333)),
                            ((4, 3, 0.8), (2, 1, 0.6666666666666666)),
                            ((4, 3, 0.8), (2, 2, 0.6)),
                            ((4, 3, 0.8), (3, 2, 0.4)),
                            ((4, 3, 0.8), (3, 3, 1.0))))
        pass
    
    
    def test_GA_tensor_shape_length_match(self):
        t3, t4 = match_tensor_shape_lengths(np.zeros((5, 10)), np.ones((5, 10)))
        self.assertEqual(t3.shape, (5, 10))
        self.assertEqual(t4.shape, (5, 10))
        t3, t4 = match_tensor_shape_lengths(np.zeros((5, 10)), np.ones((6, 5, 10)))
        self.assertEqual(t3.shape, (1, 5, 10))
        self.assertEqual(t4.shape, (6, 5, 10))
        t3, t4 = match_tensor_shape_lengths(np.zeros((2, 5, 10)), np.ones((6, 5, 10)))
        self.assertEqual(t3.shape, (2, 5, 10))
        self.assertEqual(t4.shape, (6, 5, 10))
        t3, t4 = match_tensor_shape_lengths(np.zeros((2, 5, 10)), np.ones((5, 10)))
        self.assertEqual(t3.shape, (2, 5, 10))
        self.assertEqual(t4.shape, (1, 5, 10))
        t3, t4 = match_tensor_shape_lengths(np.zeros((2, 5, 10)), np.ones((10)))
        self.assertEqual(t3.shape, (2, 5, 10))
        self.assertEqual(t4.shape, (1, 1, 10))
        pass


    def test_HA_apply_twit(self):
        print()
        t1 = np.ones((5))
        t2 = np.ones((4))
        t = twit(([(0,4), (0,3), (1.0, 1.0)],))
        apply_twit(t1, t2, twt=t, preclear=True)
        a = [1., 1., 1., 1.]
        npt.assert_array_almost_equal(t2, a)

        t1 = np.ones((5))
        t2 = np.ones((4))
        t = twit(([(0,4), (0,3), (0.0, 1.0)],))
        apply_twit(t1, t2, twt=t, preclear=True)
        a = [0.05, 0.35, 0.65, 0.95]
        npt.assert_array_almost_equal(t2, a)

        t1 = np.ones((5))
        t1[1] = 0.5
        t2 = np.ones((4))
        t = twit(([(0,4), (0,3), (0.0, 1.0)],))
        apply_twit(t1, t2, twt=t, preclear=True)
        a = [0.025, 0.275, 0.65, 0.95]
        npt.assert_array_almost_equal(t2, a)

        t1 = np.ones((5, 3))
        t1[1, 1] = 0.5
        t2 = np.ones((4, 5))
        t = twit(([(0, 4), (0, 3), (1.0, 1.0)], [(0, 2), (0, 4), (1.0, 1.0)]))
        apply_twit(t1, t2, twt=t, preclear=True)
        a = [[1. , 0.95714286, 0.9 , 0.95714286, 1.],
             [1. , 0.87142857, 0.7 , 0.87142857, 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.]]
        npt.assert_array_almost_equal(t2, a)

        t1 = np.ones((5, 3))
        t1[1, 1] = 0.5
        t2 = np.ones((4, 5))
        t = twit(([(0, 4), (0, 3), (0.0, 1.0)], [(0, 2), (0, 4), (1.0, 1.0)]))
        apply_twit(t1, t2, twt=t, preclear=True)
        a = [[0.05, 0.03928571, 0.025, 0.03928571, 0.05],
             [0.35, 0.31785714, 0.275, 0.31785714, 0.35],
             [0.65, 0.65, 0.65, 0.65, 0.65],
             [0.95, 0.95, 0.95, 0.95, 0.95]]
        npt.assert_array_almost_equal(t2, a)

        t1 = np.ones((5, 4, 3))
        t1[3, 2, 1] = 0.5
        t2 = np.ones((3, 5, 1))
        t = twit(([(1,3), (0,2), (0.9, 0.5)], [(2, 3), (0,1), (1.0, 1.0)], [(0,2), (0,0), (1.0, 1.0)]))
        apply_twit(t1, t2, twt=t, preclear=True)

        a = [[[0.9], [0.9], [1.], [1.], [1.]], 
             [[0.7], [0.7], [1.], [1.], [1.]],
             [[0.41666667], [0.5], [1.], [1.], [1.]]]
        npt.assert_array_almost_equal(t2, a)

        t1 = np.ones((5, 4, 3))
        t1[1, 1, 1] = 0.5
        t2 = np.ones((3, 5, 1))
        # Reverse outermost axis of source
        t = twit(([(3,1), (0,2), (0.9, 0.5)], [(2, 3), (0,1), (1.0, 1.0)], [(0,2), (0,0), (1.0, 1.0)]))
        apply_twit(t1, t2, twt=t, preclear=True)
        #print(t2)
        a = [[[0.9], [0.9], [1.], [1.], [1.]],
             [[0.7], [0.7], [1.], [1.], [1.]],
             [[0.5], [0.5], [1.], [1.], [1.]]]
        npt.assert_array_almost_equal(t2, a)

        t2 = np.ones((3, 5, 1))
        c = make_twit_cache(t)
        apply_twit(t1, t2, cache=c, preclear=True)
        npt.assert_array_almost_equal(t2, a)
        apply_twit(t1, t2, cache=c, preclear=True)
        npt.assert_array_almost_equal(t2, a)
        # You can't have both cache and twt given.
        with self.assertRaises(AttributeError):
            apply_twit(t1, t2, cache=c, twt=t, preclear=True)

        # And now the big overall test.
        t1 = np.ones((5, 4, 3))
        t1[3, 2, 1] = 0.5
        t2 = np.ones((3, 5, 1))
        tensor_transfer(t1, t2, preclear=True)
        a = [[[1.], [1.], [1.], [1.], [1.]],
            [[1.], [1.], [0.97685185], [0.97115385], [1.]],
             [[1.], [1.], [0.9691358], [0.96153846], [1.]]]
        npt.assert_array_almost_equal(t2, a)
        t2 = np.ones((3, 5, 1))
        tensor_transfer(t1, t2, preclear=False)
        a = [[[2.], [2.], [2.], [2.], [2.]],
            [[2.], [2.], [1.97685185], [1.97115385], [2.]],
            [[2.], [2.], [1.9691358], [1.96153846], [2.]]]
        npt.assert_array_almost_equal(t2, a)
        t2 = np.ones((3, 5, 1))
        tensor_transfer(t1, t2, preclear=True, weight_axis=1, weight_range=(0.0, 1.0))
        a = [[[0.], [0.25], [0.5], [0.75], [1.]],
                  [[0.], [0.25], [0.48842593], [0.72836538], [1.]],
                  [[0.], [0.25], [0.4845679],  [0.72115385], [1.]]]
        npt.assert_array_almost_equal(t2, a)
        
        t1 = np.ones((5, 4))
        t1[3, 2] = 0.5
        t2 = np.ones((3, 5, 2))
        tensor_transfer(t1, t2, preclear = True, weight_axis = 1, weight_range = (0.0, 1.0))
        a = [[[0., 0.], [0.25, 0.25], [0.5, 0.5], [0.6875, 0.625], [1., 1.]],
             [[0., 0.], [0.25, 0.25], [0.5, 0.5], [0.6875, 0.625], [1., 1.]],
             [[0., 0.], [0.25, 0.25], [0.5, 0.5], [0.6875, 0.625], [1., 1.]]]
        npt.assert_array_almost_equal(t2, a)

        with self.assertRaises(AttributeError): 
            t1 = np.ones((5, 4))
            t1[3, 2] = 0.5
            # Zero length shape here raises assert.
            t2 = np.zeros(())
            tensor_transfer(t1, t2, preclear=True)
            # Lol, simply the average value of t1.  t1 is 19 ones, and a single
            # 0.5
            # so 19.5/20 is 0.975
            a = [0.975]
            npt.assert_array_almost_equal(t2, a)
        pass

 
if __name__ == '__main__':

    unittest.main(verbosity=2)

