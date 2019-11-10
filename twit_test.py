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


	def test_AAA_match_shape_lengths(self):
		self.assertAlmostEqual(match_shape_lengths((4,2), (3,5)), ((4,2), (3,5)))
		self.assertAlmostEqual(match_shape_lengths((7, 4, 2), (3, 5)), ((7, 4, 2), (1, 3, 5)))
		self.assertAlmostEqual(match_shape_lengths((4, 2), (9, 3, 5)), ((1, 4, 2), (9, 3, 5)))
		self.assertAlmostEqual(match_shape_lengths((2,), (9, 3, 5)), ((1, 1, 2), (9, 3, 5)))
		pass


	def test_AAB_gen_twit_param_from_shapes(self):
		p = gen_twit_param_from_shapes((3,4), (3,4))
		self.assertEqual(p[0], 2)
		npt.assert_array_almost_equal(p[1], [0, 2, 0, 2, 0, 3, 0, 3])
		npt.assert_array_almost_equal(p[2], [1., 1., 1., 1.])
		p = gen_twit_param(np.zeros((3,4)), np.zeros((3,4)))
		self.assertEqual(p[0], 2)
		npt.assert_array_almost_equal(p[1], [0, 2, 0, 2, 0, 3, 0, 3])
		npt.assert_array_almost_equal(p[2], [1., 1., 1., 1.])
		
		p = gen_twit_param_from_shapes((3,4), (7,3,4))
		self.assertEqual(p[0], 3)
		npt.assert_array_almost_equal(p[1], [0, 0, 0, 6, 0, 2, 0, 2, 0, 3, 0, 3])
		npt.assert_array_almost_equal(p[2], [1., 1., 1., 1., 1., 1.])
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


	def test_AA_interp(self):
		self.assertAlmostEqual(twit_interp(0, 10, 0.0, 5.0, 0), 0.0)
		self.assertAlmostEqual(twit_interp(0, 10, 0.0, 5.0, 11), 5.5)
		self.assertAlmostEqual(twit_interp(10, 0, 0.0, 5.0, 1), 4.5)

		self.assertAlmostEqual(twit_interp(10, 0, 5.0, 0.0, 10), 5.0)
		self.assertAlmostEqual(twit_interp(10, 0, 5.0, 0.0, 1), 0.5)
		self.assertAlmostEqual(twit_interp(0, 0, 1.0, 5.0, 11), 1.0)

		pass

	def test_AAM_interp(self):
		self.assertFalse(outside_range(0, 10, 5))
		self.assertFalse(outside_range(0, 10, 10))
		self.assertFalse(outside_range(0, 10, 0))
		self.assertFalse(outside_range(10, 0, 5))
		self.assertFalse(outside_range(10, 0, 10))
		self.assertFalse(outside_range(10, 0, 0))
		self.assertTrue(outside_range(0, 10, -5))
		self.assertTrue(outside_range(0, 10, 11))
		self.assertTrue(outside_range(0, 10, -1))
		self.assertTrue(outside_range(10, 0, 500))
		self.assertFalse(outside_range(10, 0, 3))
		self.assertFalse(outside_range(10, 0, 2))

		pass


	def test_AB_range_series(self):
		t1 = ((3, 4), (0.6666666666666666, 0.3333333333333333))
		t2 = ((4, 5, 6), (0.25, 0.5, 0.25))
		t3 = ((6, 7, 8), (0.25, 0.5, 0.25))
		t4 = ((8, 9), (0.3333333333333333, 0.6666666666666666))
		a : range_series = find_range_series_multipliers(1, 4, 3, 9, 1)
		#print(a)
		nt.assert_almost_equal(a.idxs, [3, 4])
		nt.assert_almost_equal(a.values, [0.6666666666666666, 0.3333333333333333])

		a : range_series = find_range_series_multipliers(1, 4, 3, 9, 2)
		nt.assert_almost_equal(a.idxs, [4, 5, 6])
		nt.assert_almost_equal(a.values, [0.25, 0.5, 0.25])

		a : range_series = find_range_series_multipliers(1, 4, 3, 9, 3)
		nt.assert_almost_equal(a.idxs, [6, 7, 8])
		nt.assert_almost_equal(a.values, [0.25, 0.5, 0.25])

		a : range_series = find_range_series_multipliers(1, 4, 3, 9, 4)
		nt.assert_almost_equal(a.idxs, [8, 9])
		nt.assert_almost_equal(a.values, [0.3333333333333333, 0.6666666666666666])

		t1 = ((4, 5, 6), (0.25, 0.5, 0.25))
		t2 = ((6, 7, 8), (0.25, 0.5, 0.25))
		t3 = ((8, 9), (0.3333333333333333, 0.6666666666666666))
		t4 = None
		a : range_series = find_range_series_multipliers(0, 3, 3, 9, 1)
		nt.assert_almost_equal(a.idxs, t1[0])
		nt.assert_almost_equal(a.values, t1[1])
		a : range_series = find_range_series_multipliers(0, 3, 3, 9, 2)
		nt.assert_almost_equal(a.idxs, t2[0])
		nt.assert_almost_equal(a.values, t2[1])
		a : range_series = find_range_series_multipliers(0, 3, 3, 9, 3)
		nt.assert_almost_equal(a.idxs, t3[0])
		nt.assert_almost_equal(a.values, t3[1])
		a : range_series = find_range_series_multipliers(3, 0, 3, 9, 1)
		nt.assert_almost_equal(a.idxs, t1[0])
		nt.assert_almost_equal(a.values, t1[1])
		a : range_series = find_range_series_multipliers(0, 3, 9, 3, 2)
		nt.assert_almost_equal(a.idxs, t2[0])
		nt.assert_almost_equal(a.values, t2[1])
		a : range_series = find_range_series_multipliers(0, 3, 3, 9, 3)
		nt.assert_almost_equal(a.idxs, t3[0])
		nt.assert_almost_equal(a.values, t3[1])


		# One to Many
		a : range_series = find_range_series_multipliers(1, 1, 3, 9, 1)
		nt.assert_almost_equal(a.idxs, (3, 4, 5, 6, 7, 8, 9))
		nt.assert_almost_equal(a.values, (0.14285714285714288, 
								   0.14285714285714288, 0.14285714285714288, 0.14285714285714288, 0.14285714285714288, 
								   0.14285714285714288, 0.14285714285714288))

		# Test out of range narrow_idx.
		with self.assertRaises(Exception):
			a : range_series = find_range_series_multipliers(0, 3, 3, 9, 4)
		with self.assertRaises(Exception):
			a : range_series = find_range_series_multipliers(None, None, 3, 9, 2)
		with self.assertRaises(Exception):
			a : range_series = find_range_series_multipliers(0, 3, None, None, 2)
		with self.assertRaises(Exception):
			a : range_series = find_range_series_multipliers(0, 3, 3, 9, -1)
		with self.assertRaises(Exception):
			a : range_series = find_range_series_multipliers(-1, 3, 3, 9, 1)
		with self.assertRaises(Exception):
			a : range_series = find_range_series_multipliers(0, 3, 3, -9, 1)

		pass


	def test_EA_compute_twit_single_dimension(self):
		# 1 to 1
		t: twit_single_axis = compute_twit_single_dimension(1, 1, 5, 5, 10.0, 10.0)
		nt.assert_almost_equal(t.length, 1)
		nt.assert_almost_equal(t.srcidxs, ([1]))
		nt.assert_almost_equal(t.dstidxs, ([5]))
		nt.assert_almost_equal(t.weights, ([10.]))
		# 1 to 4
		t = compute_twit_single_dimension(1, 1, 2, 5, 5.0, 10.0)
		nt.assert_almost_equal(t.length, 4)
		nt.assert_almost_equal(t.srcidxs, ([1, 1, 1, 1]))
		nt.assert_almost_equal(t.dstidxs, ([2, 3, 4, 5]))
		nt.assert_almost_equal(t.weights, ([5., 6.666666666666667, 8.333333333333334, 10.0]))
		# 4 to 1
		t = compute_twit_single_dimension(2, 5, 1, 1, 5.0, 10.0)
		nt.assert_almost_equal(t.length, 4)
		nt.assert_almost_equal(t.srcidxs, ([2, 3, 4, 5]))
		nt.assert_almost_equal(t.dstidxs, ([1, 1, 1, 1]))
		nt.assert_almost_equal(t.weights, ([1.25, 1.66666667, 2.08333333, 2.5]))
		# 1 to 4 reverse W
		t = compute_twit_single_dimension(1, 1, 2, 5, 10.0, 5.0)
		nt.assert_almost_equal(t.length, 4)
		nt.assert_almost_equal(t.srcidxs, ([1, 1, 1, 1]))
		nt.assert_almost_equal(t.dstidxs, ([2, 3, 4, 5]))
		nt.assert_almost_equal(t.weights, ([10.0, 8.333333333333334, 6.666666666666667, 5.]))
		# 4 to 1 rev W
		t = compute_twit_single_dimension(2, 5, 1, 1, 10.0, 5.0)
		nt.assert_almost_equal(t.length, 4)
		nt.assert_almost_equal(t.srcidxs, ([2, 3, 4, 5]))
		nt.assert_almost_equal(t.dstidxs, ([1, 1, 1, 1]))
		nt.assert_almost_equal(t.weights, ([2.5, 2.08333333, 1.66666667, 1.25]))

		# 3 to 5
		t = compute_twit_single_dimension(1, 3, 0, 4, 5.0, 10.0)
		nt.assert_almost_equal(t.length, 7)
		nt.assert_almost_equal(t.srcidxs, ([1, 1, 2, 2, 2, 3, 3]))
		nt.assert_almost_equal(t.dstidxs, ([0, 1, 1, 2, 3, 3, 4]))
		nt.assert_almost_equal(t.weights, ([5., 3.571428571428571, 2.678571428571429, 7.5, 3.7500000000000004, 5.0, 10.0]))
		# 5 to 3
		t = compute_twit_single_dimension(1, 4, 0, 2, 5.0, 10.0)
		nt.assert_almost_equal(t.length, 6)
		nt.assert_almost_equal(t.srcidxs, ([1, 2, 2, 3, 3, 4]))
		nt.assert_almost_equal(t.dstidxs, ([0, 0, 1, 1, 2, 2]))
		nt.assert_almost_equal(t.weights, ([3.75, 1.66666667, 3.33333333, 4.16666667, 2.08333333, 7.5]))
		# 5 to 2
		t = compute_twit_single_dimension(0, 4, 0, 1, 1.0, 1.0)
		nt.assert_almost_equal(t.length, 8)
		nt.assert_almost_equal(t.srcidxs, ([0, 1, 2, 3, 1, 2, 3, 4]))
		nt.assert_almost_equal(t.dstidxs, ([0, 0, 0, 0, 1, 1, 1, 1]))
		nt.assert_almost_equal(t.weights, ([0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.4]))
		pass

	def test_FA_compute_twit_multi_dimension(self):
		t: twit_multi_axis = compute_twit_multi_dimension(2, np.array([0, 3, 0, 4, 0, 4, 0, 1], dtype=np.int64), np.array([0.0, 1.0, 1.0, 1.0], dtype=np.double))
		self.assertEqual(2, t.length)
		self.assertEqual(8, t.axs[0].length)
		nt.assert_array_almost_equal(t.axs[0].srcidxs, [0, 0, 1, 1, 2, 2, 3, 3])
		nt.assert_array_almost_equal(t.axs[0].dstidxs, [0, 1, 1, 2, 2, 3, 3, 4])
		nt.assert_array_almost_equal(t.axs[0].weights, [0.0, 0.0625, 0.1875, 0.25, 0.25, 0.5625, 0.1875, 1.0])
		self.assertEqual(8, t.axs[1].length)
		nt.assert_array_almost_equal(t.axs[1].srcidxs, [0, 1, 2, 3, 1, 2, 3, 4])
		nt.assert_array_almost_equal(t.axs[1].dstidxs, [0, 0, 0, 0, 1, 1, 1, 1])
		nt.assert_array_almost_equal(t.axs[1].weights, [0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.4])

		pass

	def test_HA_apply_twit(self):
		t1 = np.ones((5))
		t2 = np.ones((5))
		# Here params are n_dims, twit_int_array (quads), twit_double_array (pairs), src, dst, preclear src
		# The quads are src_start, src_end, dst_start, dst_end.  The pairs are weight_start, weight_end.
		t: twit_multi_axis = compute_twit_multi_dimension(1, np.array([0, 4, 0, 4], dtype=np.int64),  np.array([1.0, 1.0], dtype=np.float64))
		apply_twit(t, t1, t2, 1)
		a = [1., 1., 1., 1., 1.]
		npt.assert_array_almost_equal(t2, a)

		t1 = np.ones((5))
		t2 = np.ones((4))
		t: twit_multi_axis = compute_twit_multi_dimension(1, np.array([0, 4, 0, 3], dtype=np.int64),  np.array([1.0, 1.0], dtype=np.float64))
		apply_twit(t, t1, t2, 1)
		a = [1., 1., 1., 1.]
		npt.assert_array_almost_equal(t2, a)

		t1 = np.ones((5))
		t2 = np.ones((4))
		t: twit_multi_axis = compute_twit_multi_dimension(1, np.array([0, 4, 0, 3], dtype=np.int64),  np.array([0.0, 1.0], dtype=np.float64))
		apply_twit(t, t1, t2, 1)
		a = [0.05, 0.35, 0.65, 0.95]
		npt.assert_array_almost_equal(t2, a)
		
		t1 = np.ones((5))
		t1[1] = 0.5
		t2 = np.ones((4))
		t: twit_multi_axis = compute_twit_multi_dimension(1, np.array([0, 4, 0, 3], dtype=np.int64),  np.array([0.0, 1.0], dtype=np.float64))
		apply_twit(t, t1, t2, 1)
		a = [0.025, 0.275, 0.65, 0.95]
		npt.assert_array_almost_equal(t2, a)
		
		t1 = np.ones((5, 3))
		t1[1, 1] = 0.5
		t2 = np.ones((4, 5))
		t: twit_multi_axis = compute_twit_multi_dimension(2, np.array([0, 4, 0, 3, 0, 2, 0, 4], dtype=np.int64),  np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64))
		apply_twit(t, t1, t2, 1)
		a = [[1. , 0.95714286, 0.9 , 0.95714286, 1.],
			 [1. , 0.87142857, 0.7 , 0.87142857, 1.],
			 [1., 1., 1., 1., 1.],
			 [1., 1., 1., 1., 1.]]
		npt.assert_array_almost_equal(t2, a)

		t1 = np.ones((5, 3))
		t1[1, 1] = 0.5
		t2 = np.ones((4, 5))
		t: twit_multi_axis = compute_twit_multi_dimension(2, np.array([0, 4, 0, 3, 0, 2, 0, 4], dtype=np.int64),  np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64))
		# No preclear here.
		apply_twit(t, t1, t2, 0)
		a = [[1. + 1. , 0.95714286 + 1., 0.9 + 1. , 0.95714286 + 1., 1. + 1.],
			 [1. + 1. , 0.87142857 + 1., 0.7 + 1., 0.87142857 + 1., 1. + 1.],
			 [1. + 1., 1. + 1., 1. + 1., 1. + 1., 1. + 1.],
			 [1. + 1., 1. + 1., 1. + 1., 1. + 1., 1. + 1.]]
		npt.assert_array_almost_equal(t2, a)
		
		# ------------------------------------------
		# Test same tensors and twit but in different phases of caching.
		t1 = np.ones((5, 3))
		t1[1, 1] = 0.5
		t2 = np.ones((4, 5))
		t: twit_multi_axis = compute_twit_multi_dimension(2, np.array([0, 4, 0, 3, 0, 2, 0, 4], dtype=np.int64),  np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float64))
		apply_twit(t, t1, t2, 1)
		a = [[0.05, 0.03928571, 0.025, 0.03928571, 0.05],
			 [0.35, 0.31785714, 0.275, 0.31785714, 0.35],
			 [0.65, 0.65, 0.65, 0.65, 0.65],
			 [0.95, 0.95, 0.95, 0.95, 0.95]]
		npt.assert_array_almost_equal(t2, a)
		
		# Same test again, over same tensors.
		twt = compute_twit_multi_dimension(2, np.array([0, 4, 0, 3, 0, 2, 0, 4], dtype=np.int64),  np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float64))
		apply_twit(twt, t1, t2, 1)
		npt.assert_array_almost_equal(t2, a)

		# And test reuse of twt on differnet tensor instances.
		t1 = np.ones((5, 3))
		t1[1, 1] = 0.5
		t2 = np.ones((4, 5))
		apply_twit(twt, t1, t2, 1)
		npt.assert_array_almost_equal(t2, a)
		# ------------------------------------------
		
		# 3D
		t1 = np.ones((5, 4, 3)) 
		t1[3, 2, 1] = 0.5
		t2 = np.ones((3, 5, 1))
		# print("-----------------------------------------------------")
		twt: twit_multi_axis = compute_twit_multi_dimension(3, np.array([1, 3, 0, 2, 2, 3, 0, 1, 0, 2, 0, 0], dtype=np.int64),  np.array([0.9, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float64))
		apply_twit(twt, t1, t2, 1)
		
		a = [[[0.9], [0.9], [1.], [1.], [1.]], 
			 [[0.7], [0.7], [1.], [1.], [1.]],
			 [[0.41666667], [0.5], [1.], [1.], [1.]]]
		npt.assert_array_almost_equal(t2, a)
		
		#t1 = np.ones((5, 4, 3))
		#t1[1, 1, 1] = 0.5
		#t2 = np.ones((3, 5, 1))
		## Reverse outermost axis of source
		#t = twit(([(3,1), (0,2), (0.9, 0.5)], [(2, 3), (0,1), (1.0, 1.0)], [(0,2), (0,0), (1.0, 1.0)]))
		#apply_twit(t1, t2, twt=t, preclear=True)
		##print(t2)
		#a = [[[0.9], [0.9], [1.], [1.], [1.]],
		#     [[0.7], [0.7], [1.], [1.], [1.]],
		#     [[0.5], [0.5], [1.], [1.], [1.]]]
		#npt.assert_array_almost_equal(t2, a)
		#
		#t2 = np.ones((3, 5, 1))
		#c = make_twit_cache(t)
		#apply_twit(t1, t2, cache=c, preclear=True)
		#npt.assert_array_almost_equal(t2, a)
		#apply_twit(t1, t2, cache=c, preclear=True)
		#npt.assert_array_almost_equal(t2, a)
		## You can't have both cache and twt given.
		#with self.assertRaises(AttributeError):
		#    apply_twit(t1, t2, cache=c, twt=t, preclear=True)
		#
		## And now the big overall test.
		#t1 = np.ones((5, 4, 3))
		#t1[3, 2, 1] = 0.5
		#t2 = np.ones((3, 5, 1))
		#tensor_transfer(t1, t2, preclear=True)
		#a = [[[1.], [1.], [1.], [1.], [1.]],
		#    [[1.], [1.], [0.97685185], [0.97115385], [1.]],
		#     [[1.], [1.], [0.9691358], [0.96153846], [1.]]]
		#npt.assert_array_almost_equal(t2, a)
		#t2 = np.ones((3, 5, 1))
		#tensor_transfer(t1, t2, preclear=False)
		#a = [[[2.], [2.], [2.], [2.], [2.]],
		#    [[2.], [2.], [1.97685185], [1.97115385], [2.]],
		#    [[2.], [2.], [1.9691358], [1.96153846], [2.]]]
		#npt.assert_array_almost_equal(t2, a)
		#t2 = np.ones((3, 5, 1))
		#tensor_transfer(t1, t2, preclear=True, weight_axis=1, weight_range=(0.0, 1.0))
		#a = [[[0.], [0.25], [0.5], [0.75], [1.]],
		#          [[0.], [0.25], [0.48842593], [0.72836538], [1.]],
		#          [[0.], [0.25], [0.4845679],  [0.72115385], [1.]]]
		#npt.assert_array_almost_equal(t2, a)
		#
		#t1 = np.ones((5, 4))
		#t1[3, 2] = 0.5
		#t2 = np.ones((3, 5, 2))
		#tensor_transfer(t1, t2, preclear = True, weight_axis = 1, weight_range = (0.0, 1.0))
		#a = [[[0., 0.], [0.25, 0.25], [0.5, 0.5], [0.6875, 0.625], [1., 1.]],
		#     [[0., 0.], [0.25, 0.25], [0.5, 0.5], [0.6875, 0.625], [1., 1.]],
		#     [[0., 0.], [0.25, 0.25], [0.5, 0.5], [0.6875, 0.625], [1., 1.]]]
		#npt.assert_array_almost_equal(t2, a)
		#
		#with self.assertRaises(AttributeError): 
		#    t1 = np.ones((5, 4))
		#    t1[3, 2] = 0.5
		#    # Zero length shape here raises assert.
		#    t2 = np.zeros(())
		#    tensor_transfer(t1, t2, preclear=True)
		#    # Lol, simply the average value of t1.  t1 is 19 ones, and a single
		#    # 0.5
		#    # so 19.5/20 is 0.975
		#    a = [0.975]
		#    npt.assert_array_almost_equal(t2, a)
		pass


 
if __name__ == '__main__':
	unittest.main(verbosity=2)

