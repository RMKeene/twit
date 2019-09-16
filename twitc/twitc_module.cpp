#include <Python.h>
#include <Windows.h>
#include <cmath>

#include <numpy/arrayobject.h>

PyObject* generate_twit_list_impl(PyObject*, PyObject* o);
PyObject* twitc_interp_impl(PyObject*, PyObject* args);
PyObject* find_range_series_multipliers(PyObject*, PyObject* args);
PyObject* outside_range_impl(PyObject*, PyObject* args);

static PyMethodDef twitc_methods[] = {

	{ "generate_twit_list", (PyCFunction)generate_twit_list_impl, METH_O, "Generate the twit list of inter tensor links as a cache." },
	{ "twit_interp", (PyCFunction)twitc_interp_impl, METH_VARARGS, "Interpolate a float value along an integer range with index." },
	{ "find_range_series_multipliers", (PyCFunction)find_range_series_multipliers, METH_VARARGS, "Generate two lists of int index and float fractional value used for single axis interpolation calc." },
	{ "outside_range", (PyCFunction)outside_range_impl, METH_VARARGS, "True if idx is not in the start to end range, inclusive.  Start does not have to be less than end." },
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
	printf("Init TWITC\n");
	import_array();
	return PyModule_Create(&twitc_module);
}

PyObject* generate_twit_list_impl(PyObject*, PyObject* o) {
	printf("TWITC generate_twit_list_impl\n");
	PyErr_Clear();
	double x = PyFloat_AsDouble(o);
	PyObject* ret = PyFloat_FromDouble(x);
	return (ret);
}

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

double _twit_interp(INT64 range_start, INT64 range_end, double value_start, double value_end, INT64 idx) {
	int rspan = range_end - range_start;
	if (rspan == 0) {
		return value_start;
	}
	return value_start + (value_end - value_start) * (idx - range_start) / rspan;
}

bool _outside_range(int start, int end, int idx) {
	///True if idx is not between start and end inclusive.
	printf("_outside_range: start %d, end %d, idx %d\n", start, end, idx);
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
	printf("TWITC - _find_range_series_multipliers(%lld, %lld, %lld, %lld, %lld)\n", narrow_range_start, narrow_range_end, wide_range_start, wide_range_end, narrow_idx);
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

	printf("TWITC - _find_range_series_multipliers, math\n");
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

PyObject* find_range_series_multipliers(PyObject*, PyObject* args) {
	printf("TWITC find_range_series_multipliers\n");
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

twit_single_axis * compute_twit_single_dimension(INT64 src_start, INT64 src_end, INT64 dst_start, INT64 dst_end, double w_start, double w_end) {

}