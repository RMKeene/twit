#include <Python.h>
#include <Windows.h>
#include <cmath>

#include <numpy/arrayobject.h>

struct range_series {
	INT64 length;
	INT64* idxs;
	double* values;
};

struct twit_single_axis {
	INT64 length;
	INT64* srcidxs;
	INT64* dstidxs;
	double* weights;
};

// Must match the python code values.
const int TWIT_FAN_SAME = 0;
const int TWIT_FAN_IN = 1;
const int TWIT_FAN_OUT = 2;


PyObject* generate_twit_list_impl(PyObject*, PyObject* o);
PyObject* twitc_interp_impl(PyObject*, PyObject* args);
PyObject* find_range_series_multipliers_impl(PyObject*, PyObject* args);
PyObject* outside_range_impl(PyObject*, PyObject* args);
PyObject* compute_twit_single_dimension_impl(PyObject*, PyObject* args);

static PyMethodDef twitc_methods[] = {

	{ "generate_twit_list", (PyCFunction)generate_twit_list_impl, METH_O, "Generate the twit list of inter tensor links as a cache." },
	{ "twit_interp", (PyCFunction)twitc_interp_impl, METH_VARARGS, "Interpolate a float value along an integer range with index." },
	{ "find_range_series_multipliers", (PyCFunction)find_range_series_multipliers_impl, METH_VARARGS, "Generate two lists of int index and float fractional value used for single axis interpolation calc." },
	{ "outside_range", (PyCFunction)outside_range_impl, METH_VARARGS, "True if idx is not in the start to end range, inclusive.  Start does not have to be less than end." },
	{ "compute_twit_single_dimension", (PyCFunction)compute_twit_single_dimension_impl, METH_VARARGS, "Compute the source idx, destination idx, weights lists for a single dimension twit." },
	{ nullptr, nullptr, 0, nullptr }
};

static PyModuleDef twitc_module = {
	PyModuleDef_HEAD_INIT,
	"twitc",
	"Provides some twit functions, but faster.",
	0,
	twitc_methods
};

PyMODINIT_FUNC PyInit_twitc() {
	//printf("Init TWITC\n");
	import_array();
	return PyModule_Create(&twitc_module);
}

PyObject* generate_twit_list_impl(PyObject*, PyObject* o) {
	//rintf("TWITC generate_twit_list_impl\n");
	PyErr_Clear();
	double x = PyFloat_AsDouble(o);
	PyObject* ret = PyFloat_FromDouble(x);
	return (ret);
}

double _twit_interp(INT64 range_start, INT64 range_end, double value_start, double value_end, INT64 idx) {
	int rspan = range_end - range_start;
	if (rspan == 0) {
		return value_start;
	}
	return value_start + (value_end - value_start) * (idx - range_start) / rspan;
}

/// Returns either +1 or -1, if x is 0 then returns +1
inline int twit_sign(int x) {
	return (x >= 0) - (x < 0);
}

bool _outside_range(int start, int end, int idx) {
	///True if idx is not between start and end inclusive.
	//printf("_outside_range: start %d, end %d, idx %d\n", start, end, idx);
	if (start <= end) {
		return idx < start || idx > end;
	}
	return idx < end || idx > start;
}

PyObject* outside_range_impl(PyObject*, PyObject* args) {
	int start;
	int end;
	int idx;
	PyArg_ParseTuple(args, "iii", &start, &end, &idx);

	bool b = _outside_range(start, end, idx);
	if (b) {
		Py_INCREF(Py_True);
		return Py_True;
	}

	Py_INCREF(Py_False);
	return Py_False;
}

PyObject* twitc_interp_impl(PyObject*, PyObject* args) {
	PyErr_Clear();
	INT64 range_start;
	INT64 range_end;
	double value_start;
	double value_end;
	INT64 idx;
	PyArg_ParseTuple(args, "LLddL", &range_start, &range_end, &value_start, &value_end, &idx);
	return PyFloat_FromDouble(_twit_interp(range_start, range_end, value_start, value_end, idx));
}

/// Returns a range_series instance.  You are responsible to free it.
range_series* _find_range_series_multipliers(INT64 narrow_range_start, INT64 narrow_range_end, INT64 wide_range_start, INT64 wide_range_end, INT64 narrow_idx) {
	//printf("TWITC - _find_range_series_multipliers(%lld, %lld, %lld, %lld, %lld)\n", narrow_range_start, narrow_range_end, wide_range_start, wide_range_end, narrow_idx);
	if (narrow_idx < min(narrow_range_start, narrow_range_end) || narrow_idx > max(narrow_range_start, narrow_range_end)) {
		PyErr_SetString(PyExc_Exception, "find_range_series_multipliers: narrow_idx is out of range.  Must be in the narrow_range (inclusive).");
		return NULL;
	}
	// Force narrow and wide ranges to be in order.At this low level it does
	// not matter which order we sequence the return values.
	if (narrow_range_start > narrow_range_end) {
		int t = narrow_range_start;
		narrow_range_start = narrow_range_end;
		narrow_range_end = t;
	}
	if (wide_range_start > wide_range_end) {
		int t = wide_range_start;
		wide_range_start = wide_range_end;
		wide_range_end = t;
	}

	if (narrow_range_start < 0 || wide_range_start < 0) {
		PyErr_SetString(PyExc_Exception, "find_range_series_multipliers: Negative range indicies.");
		return NULL;
	}

	//printf("TWITC - _find_range_series_multipliers, math\n");
	INT64 narrow_span = narrow_range_end - narrow_range_start + 1;
	INT64 wide_span = wide_range_end - wide_range_start + 1;
	if (narrow_span >= wide_span) {
		PyErr_SetString(PyExc_Exception, "find_range_series_multipliers: Wide range must be wider than narrow_range.");
		return NULL;
	}
	int wspan = wide_span - 1;

	// Generate the fractional values.
	range_series* ret = (range_series*)malloc(sizeof(range_series));
	ret->idxs = NULL;
	ret->length = 0;
	ret->values = NULL;

	double sum = 0.0;
	INT64 narrow_relidx = narrow_idx - narrow_range_start;
	INT64 narrow_div = narrow_span - 1;
	// Count how many elements so we can allocate idxs and values arrays.
	INT64 element_count = 0;
	if (narrow_div > 0) { // Many to Many
		double narrow_idx_frac_low = max(0.0, (narrow_relidx - 1) / (double)narrow_div);
		double narrow_idx_frac = narrow_relidx / (double)narrow_div;
		double narrow_idx_frac_hi = min(1.0, (narrow_relidx + 1) / (double)narrow_div);

		INT64 wide_idx_low = max(0, INT64(narrow_idx_frac_low * wspan));
		double wide_idx_mid = narrow_idx_frac * wspan;
		INT64 wide_idx_hi = min(wspan, INT64(narrow_idx_frac_hi * wspan));
		double wide_half_span = wspan / (double)narrow_div;

		for (INT64 i = (INT64)wide_idx_low; i <= (INT64)wide_idx_hi; i++) {
			double frac = 1.0 - abs((i - wide_idx_mid) / wide_half_span);
			if (frac > 0.0) {
				element_count++;
			}
		}

		// Per C standard malloc aligns to 16 bytes (a double) on 64 bit systems.
		ret->idxs = (INT64*)PyMem_Malloc(sizeof(INT64) * element_count);
		ret->values = (double*)PyMem_Malloc(sizeof(double) * element_count);
		ret->length = element_count;

		INT64 N = 0;
		for (INT64 i = (INT64)wide_idx_low; i <= (INT64)wide_idx_hi; i++) {
			double frac = 1.0 - abs((i - wide_idx_mid) / wide_half_span);
			if (frac > 0.0) {
				ret->idxs[N] = i + wide_range_start;
				ret->values[N] = frac;
				sum += frac;
				N++;
			}
		}
	}
	else // One to Many
	{
		// Count how large the arrays have to be to hold the results.
		for (INT64 i = (INT64)wide_range_start; i <= (INT64)wide_range_end; i++) {
			element_count++;
		}

		ret->idxs = (INT64*)PyMem_Malloc(sizeof(INT64) * element_count);
		ret->values = (double*)PyMem_Malloc(sizeof(double) * element_count);
		ret->length = element_count;

		double frac = 1.0 / wide_span;
		INT64 N = 0;
		for (INT64 i = (INT64)wide_range_start; i <= (INT64)wide_range_end; i++) {
			ret->idxs[N] = i;
			ret->values[N] = frac;
			sum += frac;
			N++;
		}
	}

	// Normalize so sum is 1.0
	for (INT64 i = 0; i < element_count; i++) {
		ret->values[i] /= sum;
	}

	// Weights of ret will always sum to 1.0 (They are normalized)
	return ret;
}

PyObject* find_range_series_multipliers_impl(PyObject*, PyObject* args) {
	//printf("TWITC find_range_series_multipliers\n");
	PyErr_Clear();
	INT64 narrow_range_start;
	INT64 narrow_range_end;
	INT64 wide_range_start;
	INT64 wide_range_end;
	INT64 narrow_idx;
	PyArg_ParseTuple(args, "LLLLL", &narrow_range_start, &narrow_range_end, &wide_range_start, &wide_range_end, &narrow_idx);
	range_series* ptr = _find_range_series_multipliers(narrow_range_start, narrow_range_end, wide_range_start, wide_range_end, narrow_idx);
	if (ptr == NULL) {
		return NULL;
	}
	npy_intp dims[1];
	dims[0] = ptr->length;
	PyObject* idxs = PyArray_SimpleNewFromData(1, dims, NPY_INT64, (void*)ptr->idxs);
	PyObject* values = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void*)ptr->values);

	PyObject* rslt = PyTuple_New(2);
	PyTuple_SetItem(rslt, 0, idxs);
	PyTuple_SetItem(rslt, 1, values);
	Py_INCREF(rslt);
	return rslt;
}

twit_single_axis* _compute_twit_single_dimension(INT64 src_start, INT64 src_end, INT64 dst_start, INT64 dst_end, double w_start, double w_end) {
	// printf("TWITC _compute_twit_single_dimension(ranges: src %lld, %lld, dst %lld, %lld, weight %f %f\n", src_start, src_end, dst_start, dst_end, w_start, w_end);
	// The + 1 here is because our range is inclusive.
	INT64 input_span = abs(src_start - src_end) + 1;
	INT64 output_span = abs(dst_start - dst_end) + 1;
	// Weight span is signed and the simple difference.
	double weight_span = w_end - w_start;
	int fan = 0;
	if (input_span == output_span) {
		fan = TWIT_FAN_SAME;
	}
	else if (input_span > output_span) {
		fan = TWIT_FAN_IN;
	}
	else {
		fan = TWIT_FAN_OUT;
	}

	INT64 src_idx = src_start;
	INT64 dst_idx = dst_start;

	INT64 src_inc = twit_sign(src_end - src_start);
	INT64 dst_inc = twit_sign(dst_end - dst_start);

	// So.... The smallest length result set for a single axis twit iterator is the larger of the src or dst spans.  The largest 
	// is 2x that and the worst case. Even so this is not large if we over allocate some. A 200 x 200 x 3 image
	// or neural Concept Map is 120000 values of float about 500k bytes.  The worst case twit 
	// single axis iterators will then be 2 x (200 +_ 200 + 3) x 8 indixies and weights are (200 + 200 + 3) x 8 os about
	// 48k bytes.
	twit_single_axis* ret = (twit_single_axis*)PyMem_Malloc(sizeof(twit_single_axis));
	INT64 sz = max(input_span, output_span);
	if (input_span != output_span && input_span != 1 && output_span != 1) {
		sz *= 2;
	}

	ret->srcidxs = (INT64*)PyMem_Malloc(sz * sizeof(INT64));
	ret->dstidxs = (INT64*)PyMem_Malloc(sz * sizeof(INT64));
	ret->weights = (double*)PyMem_Malloc(sz * sizeof(double));

	// Now generate the values and normalize
	if (fan == TWIT_FAN_SAME) {
		INT64 dsti = dst_start;
		sz = 0;
		for (INT64 srci = src_start; srci != src_end + src_inc; srci += src_inc, dsti += dst_inc, sz++) {
			ret->srcidxs[sz] = srci;
			ret->dstidxs[sz] = dsti;
			ret->weights[sz] = _twit_interp(src_start, src_end, w_start, w_end, srci);
		}
		// Normalization not needed for one to one.
	}
	else if (fan == TWIT_FAN_IN) {
		INT64 srci = src_start;
		sz = 0;
		// Sums for normalizing, by index in span
		double* sums = new double[output_span];
		for (INT64 i = 0; i < output_span; i++) sums[i] = 0.0;

		for (INT64 dsti = dst_start; dsti != dst_end + dst_inc; dsti += dst_inc, srci += src_inc) {
			range_series* splits = _find_range_series_multipliers(dst_start, dst_end, src_start, src_end, dsti);
			for (INT64 spi = 0; spi < splits->length; spi++) {
				INT64 k = ret->srcidxs[sz] = splits->idxs[spi];
				ret->dstidxs[sz] = dsti;
				ret->weights[sz] = splits->values[spi];
				sums[(dsti - dst_start) * dst_inc] += ret->weights[sz];
				sz++;
			}
		}

		// Normalize so each each source sums to 1.0 and also do the interpolation of weights across the span.
		for (INT64 i = 0; i < sz; i++) {
			INT64 k = ret->dstidxs[i];
			ret->weights[i] *= _twit_interp(src_start, src_end, w_start, w_end, ret->srcidxs[i]) / sums[(k - dst_start) * dst_inc];
		}
		delete[] sums;

	}
	else {
		// Fan out
		INT64 dsti = dst_start;
		sz = 0;
		// Sums for normalizing, by index in span
		double* sums = new double[output_span];
		for (INT64 i = 0; i < output_span; i++) sums[i] = 0.0;

		for (INT64 srci = src_start; srci != src_end + src_inc; srci += src_inc, dsti += dst_inc) {
			range_series* splits = _find_range_series_multipliers(src_start, src_end, dst_start, dst_end, srci);
			for (INT64 spi = 0; spi < splits->length; spi++) {
				ret->srcidxs[sz] = srci;
				INT64 k = ret->dstidxs[sz] = splits->idxs[spi];
				ret->weights[sz] = splits->values[spi];
				sums[k * dst_inc - dst_start] += ret->weights[sz];
				sz++;
			}
		}

		// Normalize so each each destination sums to 1.0 and also multiply in the interpolated weights across the weight range.
		for (INT64 i = 0; i < sz; i++) {
			INT64 k = ret->dstidxs[i];
			ret->weights[i] *= _twit_interp(dst_start, dst_end, w_start, w_end, ret->dstidxs[i]) / sums[k * dst_inc - dst_start];
		}
		delete[] sums;
	}

	ret->length = sz;
	return ret;
}

PyObject* compute_twit_single_dimension_impl(PyObject*, PyObject* args) {
	//printf("TWITC compute_twit_single_dimension\n");
	PyErr_Clear();
	INT64 src_start;
	INT64 src_end;
	INT64 dst_start;
	INT64 dst_end;
	double w_start;
	double w_end;

	PyArg_ParseTuple(args, "LLLLdd", &src_start, &src_end, &dst_start, &dst_end, &w_start, &w_end);

	twit_single_axis* ptr = _compute_twit_single_dimension(src_start, src_end, dst_start, dst_end, w_start, w_end);

	npy_intp dims[1];
	dims[0] = ptr->length;
	PyObject* srcidxs = PyArray_SimpleNewFromData(1, dims, NPY_INT64, (void*)ptr->srcidxs);
	PyObject* dstidxs = PyArray_SimpleNewFromData(1, dims, NPY_INT64, (void*)ptr->dstidxs);
	PyObject* weights = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void*)ptr->weights);

	PyObject* rslt = PyTuple_New(3);
	PyTuple_SetItem(rslt, 0, srcidxs);
	PyTuple_SetItem(rslt, 1, dstidxs);
	PyTuple_SetItem(rslt, 2, weights);
	Py_INCREF(rslt);
	return rslt;

}

// Do a twit transfer from t1 to t2 by twit.
// t1, t2, and twit all have the same number of dimensions, n_dims.
// t1 is a block of doubles.  t1_dims is first the count of dimensions, then the dimensions, python order so higher dimensions first.
// t2 and t2_dims the same.  t1 is src, t2 is dst.
// twit_i is the integer start and ends of ranges in quads, t1_start, t1_end, t2_start, t2_end and then repeating for each dimension 
// in python order.  twit_w is pairs for dimension weights, w_start, w_end repeating in python order for the dimensions.
void _apply_twit(INT32 n_dims, double* t1, INT32* t1_dims, double* t2, INT32* t2dims, INT32* twit_i, double* twit_w, INT32 preclear) {
}
