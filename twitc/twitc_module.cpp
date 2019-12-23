#include <Python.h>
#include <Windows.h>
#include <cmath>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>

#include <numpy/arrayobject.h>

const char const* TWITC_VERSION = "TWITC Version 1.0";

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

struct twit_multi_axis {
	INT64 length;
	twit_single_axis** axs;
};

class twit_processing_context {
public:
	twit_multi_axis const* const twit;
	double const* const t1;
	INT64 const* const t1_dims;
	double* const t2;
	INT64 const* const t2_dims;
	const INT64 preclear;

	twit_processing_context(twit_multi_axis const* const twit,
		double const* const t1, INT64 const* const t1_dims,
		double* const t2, INT64 const* const t2_dims,
		const INT64 preclear) :
		twit(twit), t1(t1), t1_dims(t1_dims), t2(t2), t2_dims(t2_dims), preclear(preclear) {}
};

struct twit_thread_bundle {
public:
	std::mutex* mtx;
	// 0 means not processing, 1 means busy.
	std::condition_variable* mutex_cvar;
	int mutex_count;
	std::thread* thread = NULL;
	twit_processing_context ctx;
};

int twit_thread_count = 1;
int twit_thread_mask = 1;
// Which axis we are processing along multithreaded.
int twit_threading_axis = 0;
// Ensure we are thread safe and only process on overall twit at a time.
std::recursive_mutex twit_mt_mutex;
twit_thread_bundle** twit_thread_bundles;

// N is the index in the mtxs and threads arrays.
void twit_thread_loop(int N);
void twit_thread_kickoff_all_loops();
void twit_thread_wait_for_all_loops_to_finish();
void twit_apply_MT_sequencer(twit_processing_context* ctx);

// Must match the python code values.
const int TWIT_FAN_SAME = 0;
const int TWIT_FAN_IN = 1;
const int TWIT_FAN_OUT = 2;

inline INT64 twit_sign(INT64 x);
double _twit_interp(INT64 range_start, INT64 range_end, double value_start, double value_end, INT64 idx);
bool _outside_range(INT64 start, INT64 end, INT64 idx);
range_series* _find_range_series_multipliers(INT64 narrow_range_start, INT64 narrow_range_end, INT64 wide_range_start, INT64 wide_range_end, INT64 narrow_idx);
twit_single_axis* _compute_twit_single_dimension(INT64 src_start, INT64 src_end, INT64 dst_start, INT64 dst_end, double w_start, double w_end);
twit_multi_axis* _compute_twit_multi_dimension(INT64 n_dims, INT64 const* const twit_i, double const* const twit_w);

void _apply_twit(twit_multi_axis const* const twit, double const* const t1, INT64 const* const t1_dims, double* const t2, INT64 const* const t2_dims, const INT64 preclear);
void _apply_twit_single_thread(twit_processing_context* ctx);
void _apply_twit_multi_thread(twit_processing_context* ctx);

void _make_and_apply_twit(INT64 n_dims, double* t1, INT64* t1_dims, double* t2, INT64* t2dims, INT64* twit_i, double* twit_w, INT64 preclear);
void twit_multi_axis_destructor(PyObject* obj);


PyObject* twitc_version_impl(PyObject*, PyObject* args);
PyObject* twit_set_thread_count_impl(PyObject*, PyObject* args);
PyObject* twitc_interp_impl(PyObject*, PyObject* args);
PyObject* find_range_series_multipliers_impl(PyObject*, PyObject* args);
PyObject* outside_range_impl(PyObject*, PyObject* args);
PyObject* compute_twit_single_dimension_impl(PyObject*, PyObject* args);
PyObject* compute_twit_multi_dimension_impl(PyObject*, PyObject* args);
PyObject* apply_twit_impl(PyObject*, PyObject* args);
PyObject* make_and_apply_twit_impl(PyObject*, PyObject* args);
PyObject* unpack_twit_multi_axis_impl(PyObject*, PyObject* args);
PyObject* pack_twit_multi_axis_impl(PyObject*, PyObject* args);

static PyMethodDef twitc_methods[] = {

	{ "twitc_version", (PyCFunction)twitc_version_impl, METH_VARARGS, "Return the version of TWITC.  Used for testing if the .lib or .dll is linking in ok." },
	{ "set_thread_count", (PyCFunction)twit_set_thread_count_impl, METH_VARARGS, "Set how many threads to use for twit processing.  Must be one of 1, 2, 4, 8, 16, 32, 64, 128, 256. Only can be called once." },
	{ "twit_interp", (PyCFunction)twitc_interp_impl, METH_VARARGS, "Interpolate a float value along an integer range with index." },
	{ "find_range_series_multipliers", (PyCFunction)find_range_series_multipliers_impl, METH_VARARGS, "Generate two lists of int index and float fractional value used for single axis interpolation calc." },
	{ "outside_range", (PyCFunction)outside_range_impl, METH_VARARGS, "True if idx is not in the start to end range, inclusive.  Start does not have to be less than end." },
	{ "compute_twit_single_dimension", (PyCFunction)compute_twit_single_dimension_impl, METH_VARARGS, "Compute the source idx, destination idx, weights lists for a single dimension twit." },
	{ "compute_twit_multi_dimension", (PyCFunction)compute_twit_multi_dimension_impl, METH_VARARGS, "Compute the source idx, destination idx, weights lists for all dimensions of twit."},
	{ "apply_twit", (PyCFunction)apply_twit_impl, METH_VARARGS, "Apply multidimensional transfer." },
	{ "make_and_apply_twit", (PyCFunction)make_and_apply_twit_impl, METH_VARARGS, "Make twit cace and then apply multidimensional transfer." },
	{ "unpack_twit_multi_axis", (PyCFunction)unpack_twit_multi_axis_impl, METH_VARARGS, "Convert twit opaque handle to python tuples and arrays." },
	{ "pack_twit_multi_axis", (PyCFunction)pack_twit_multi_axis_impl, METH_VARARGS, "Convert twit tuples and arrays to opaque handle." },
	{ nullptr, nullptr, 0, nullptr }
};

std::mutex twit_logging_mutex;
FILE* twit_log_file_ptr = NULL;
bool twit_logging_on = true;

void twit_log(const char* format, ...) {
	if (twit_logging_on == false) {
		return;
	}
	std::lock_guard<std::mutex> guard(twit_logging_mutex);
	if (twit_log_file_ptr == NULL) {
		twit_log_file_ptr = fopen("K:/twit/twit_log.txt", "w");
	}
	va_list args;
	va_start(args, format);
	vfprintf(twit_log_file_ptr, format, args);
	vprintf(format, args);
	va_end(args);
	fflush(twit_log_file_ptr);
}

static PyModuleDef twitc_module = {
	PyModuleDef_HEAD_INIT,
	"twitc",
	"Provides some twit functions, but faster.",
	0,
	twitc_methods
};

PyMODINIT_FUNC PyInit_twitc() {
	//twit_log("Init TWITC\n");
	import_array();
	return PyModule_Create(&twitc_module);
}

inline INT64 twit_set_thread_count(INT64 tc) {
	if (twit_thread_bundles != NULL) {
		twit_log("Invalid TWIT thread call.  twit_set_thread_count may only be called once, usually at startup.\n");
		return 0;
	}

	twit_log("twit_set_thread_count %d\n", (int)tc);
	int m = 0x001;
	if (tc == 1) {
		m = 0x001;
	}
	else if (tc == 2) {
		m = 0x001;
	}
	else if (tc == 4) {
		m = 0x003;
	}
	else if (tc == 8) {
		m = 0x007;
	}
	else if (tc == 16) {
		m = 0x00F;
	}
	else if (tc == 32) {
		m = 0x01F;
	}
	else if (tc == 64) {
		m = 0x03F;
	}
	else if (tc == 128) {
		m = 0x07F;
	}
	else if (tc == 256) {
		m = 0x0FF;
	}
	else {
		twit_log("Invalid TWIT thread count.  Must be between 1 and 255 and a power of 2.  E.g. 1,2,4,8,16 etc.  Not changed.\n");
		return 0;
	}

	twit_log("twit_set_thread_count mask is 0x%X\n", m);

	twit_thread_count = (int)tc;
	twit_thread_mask = m;
	twit_thread_bundles = (twit_thread_bundle**)PyMem_Malloc(sizeof(twit_thread_bundle*) * twit_thread_count);
	for (int i = 0; i < twit_thread_count; i++) {
		twit_log("twit_set_thread_count setup %d\n", i);
		twit_thread_bundles[i] = (twit_thread_bundle*)PyMem_Malloc(sizeof(twit_thread_bundle));
		twit_thread_bundles[i]->mtx = new std::mutex();
		twit_thread_bundles[i]->mutex_count = 0;
		twit_thread_bundles[i]->mutex_cvar = new std::condition_variable();
		twit_thread_bundles[i]->thread = new std::thread(twit_thread_loop, i);
		memset(&twit_thread_bundles[i]->ctx, 0, sizeof(twit_processing_context));
	}
	twit_log("twit_set_thread_count done\n");
	return 1;
}

/// Returns either +1 or -1, if x is 0 then returns +1
inline INT64 twit_sign(const INT64 x) {
	return (INT64)(x >= 0l) - (INT64)(x < 0l);
}

void print_spaces(const int spaces) {
	for (int i = 0; i < spaces; i++) {
		twit_log(" ");
	}
}

void print_range_series(const range_series* t, const int spaces, const char* preamble) {
	print_spaces(spaces);
	twit_log(preamble);
	if (!t) {
		twit_log("NULL\n");
		return;
	}
	if (t->length == 0) {
		twit_log("length 0\n");
		return;
	}
	if (t->length == 1) {
		twit_log("length 1: idx %lld, value %f\n", t->idxs[0], t->values[0]);
		return;
	}
	twit_log("length %lld\n", t->length);
	for (INT64 i = 0; i < t->length; i++) {
		print_spaces(spaces + 4);
		twit_log("%lld: idx %lld, value %f\n", i, t->idxs[i], t->values[i]);
	}
}

void print_twit_single_axis(const twit_single_axis* t, const int spaces, const char* preamble) {
	print_spaces(spaces);
	twit_log(preamble);
	if (!t) {
		twit_log("NULL\n");
		return;
	}
	if (t->length == 0) {
		twit_log("length 0\n");
		return;
	}
	if (t->length == 1) {
		twit_log("length 1: srcidx %lld, dstidx %lld, weight %f\n", t->srcidxs[0], t->dstidxs[0], t->weights[0]);
		return;
	}
	twit_log("length %lld\n", t->length);
	for (INT64 i = 0; i < t->length; i++) {
		print_spaces(spaces + 4);
		twit_log("%lld: srcidx %lld, dstidx %lld, weight %f\n", i, t->srcidxs[i], t->dstidxs[i], t->weights[i]);
	}
}

void print_twit_multi_axis(const twit_multi_axis* t, const int spaces, const char* preamble) {
	char buf[64];
	print_spaces(spaces);
	twit_log(preamble);
	if (!t) {
		twit_log("NULL\n");
		return;
	}

	if (t->length == 0) {
		twit_log("length 0\n");
		return;
	}

	twit_log("length %lld\n", t->length);
	for (INT64 i = 0; i < t->length; i++) {
		sprintf_s(buf, sizeof(buf), "%lld: ", i);
		print_twit_single_axis(t->axs[i], spaces + 4, buf);
	}
}

void free_range_series(range_series* p)
{
	//twit_log("\nfree_range_series\n");
	if (!p) {
		twit_log("free_range_series: NULL\n");
		return;
	}
	PyMem_Free(p->idxs);
	PyMem_Free(p->values);
	PyMem_Free(p);
}

void free_twit_single_axis(twit_single_axis* p) {
	//twit_log("\nfree_twit_single_axis\n");
	if (!p) {
		twit_log("free_twit_single_axis: NULL\n");
		return;
	}
	PyMem_Free(p->dstidxs);
	PyMem_Free(p->srcidxs);
	PyMem_Free(p->weights);
	PyMem_Free(p);
}

void free_twit_multi_axis(twit_multi_axis* p) {
	//twit_log("\nfree_twit_multi_axis\n");
	if (!p) {
		twit_log("free_twit_multi_axis: NULL\n");
		return;
	}
	for (int i = 0; i < p->length; i++) {
		free_twit_single_axis(p->axs[i]);
	}
	PyMem_Free(p->axs);
	PyMem_Free(p);
}

double _twit_interp(INT64 range_start, INT64 range_end, double value_start, double value_end, INT64 idx) {
	INT64 rspan = range_end - range_start;
	if (rspan == 0) {
		return value_start;
	}
	return value_start + (value_end - value_start) * (idx - range_start) / rspan;
}

bool _outside_range(const INT64 start, const INT64 end, const INT64 idx) {
	///True if idx is not between start and end inclusive.
	//twit_log("_outside_range: start %d, end %d, idx %d\n", start, end, idx);
	if (start <= end) {
		return idx < start || idx > end;
	}
	return idx < end || idx > start;
}

PyObject* outside_range_impl(PyObject*, PyObject* args) {
	INT64 start;
	INT64 end;
	INT64 idx;
	PyArg_ParseTuple(args, "LLL", &start, &end, &idx);

	bool b = _outside_range(start, end, idx);
	if (b) {
		Py_INCREF(Py_True);
		return Py_True;
	}

	Py_INCREF(Py_False);
	return Py_False;
}

PyObject* twit_set_thread_count_impl(PyObject*, PyObject* args) {
	INT64 n;
	PyArg_ParseTuple(args, "L", &n);

	INT64 b = twit_set_thread_count(n);
	if (b) {
		Py_INCREF(Py_True);
		return Py_True;
	}

	Py_INCREF(Py_False);
	return Py_False;
}


PyObject* twitc_version_impl(PyObject*, PyObject* args) {
	return Py_BuildValue("s", TWITC_VERSION);;
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
range_series* _find_range_series_multipliers(INT64 narrow_range_start, INT64 narrow_range_end, INT64 wide_range_start, INT64 wide_range_end, const INT64 narrow_idx) {
	//twit_log("TWITC - _find_range_series_multipliers(%lld, %lld, %lld, %lld, %lld)\n", narrow_range_start, narrow_range_end, wide_range_start, wide_range_end, narrow_idx);
	if (narrow_idx < min(narrow_range_start, narrow_range_end) || narrow_idx > max(narrow_range_start, narrow_range_end)) {
		PyErr_SetString(PyExc_Exception, "find_range_series_multipliers: narrow_idx is out of range.  Must be in the narrow_range (inclusive).");
		return NULL;
	}
	// Force narrow and wide ranges to be in order. At this low level it does
	// not matter which order we sequence the return values.
	if (narrow_range_start > narrow_range_end) {
		INT64 t = narrow_range_start;
		narrow_range_start = narrow_range_end;
		narrow_range_end = t;
	}
	if (wide_range_start > wide_range_end) {
		INT64 t = wide_range_start;
		wide_range_start = wide_range_end;
		wide_range_end = t;
	}

	if (narrow_range_start < 0 || wide_range_start < 0) {
		PyErr_SetString(PyExc_Exception, "find_range_series_multipliers: Negative range indicies.");
		return NULL;
	}

	//twit_log("TWITC - _find_range_series_multipliers, math\n");
	INT64 narrow_span = narrow_range_end - narrow_range_start + 1;
	INT64 wide_span = wide_range_end - wide_range_start + 1;
	if (narrow_span >= wide_span) {
		PyErr_SetString(PyExc_Exception, "find_range_series_multipliers: Wide range must be wider than narrow_range.");
		return NULL;
	}
	INT64 wspan = wide_span - 1;

	// Generate the fractional values.
	range_series* ret = (range_series*)PyMem_Malloc(sizeof(range_series));
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
		element_count = wide_range_end - wide_range_start + 1;

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
	//print_range_series(ret, 0, "range series:");
	return ret;
}

PyObject* find_range_series_multipliers_impl(PyObject*, PyObject* args) {
	//twit_log("TWITC find_range_series_multipliers\n");
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

	// Very difficult to get the ownership sematics correct to make the PyArrays and return them without memory corruption.
	// So I give up, just make the arrays, and copy the data into them, no ownership issues at all. :-(
	npy_intp dims[1];
	dims[0] = ptr->length;
	PyObject* idxs = PyArray_SimpleNew(1, dims, NPY_INT64);
	PyObject* values = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

	npy_intp pp[] = { 0 };
	INT64* ip = (INT64*)PyArray_GetPtr((PyArrayObject*)idxs, pp);
	double* vp = (double*)PyArray_GetPtr((PyArrayObject*)values, pp);
	for (int i = 0; i < ptr->length; i++) {
		ip[i] = ptr->idxs[i];
		vp[i] = ptr->values[i];
	}

	free_range_series(ptr);

	PyObject* rslt = PyTuple_New(2);
	PyTuple_SetItem(rslt, 0, idxs);
	PyTuple_SetItem(rslt, 1, values);
	return rslt;
}

twit_single_axis* _compute_twit_single_dimension(const INT64 src_start, const INT64 src_end, const INT64 dst_start, const INT64 dst_end, const double w_start, const double w_end) {
	//twit_log("TWITC _compute_twit_single_dimension(ranges: src %lld, %lld, dst %lld, %lld, weight %f %f\n", src_start, src_end, dst_start, dst_end, w_start, w_end);
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
			free_range_series(splits);
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
		double* sums = (double*)PyMem_Malloc(sizeof(double) * output_span);
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
			free_range_series(splits);
		}

		// Normalize so each each destination sums to 1.0 and also multiply in the interpolated weights across the weight range.
		for (INT64 i = 0; i < sz; i++) {
			INT64 k = ret->dstidxs[i];
			ret->weights[i] *= _twit_interp(dst_start, dst_end, w_start, w_end, ret->dstidxs[i]) / sums[k * dst_inc - dst_start];
		}
		PyMem_Free(sums);
	}

	ret->length = sz;
	//print_twit_single_axis(ret, 0, "TWIT Single Ax: ");
	return ret;
}

PyObject* compute_twit_single_dimension_impl(PyObject*, PyObject* args) {
	//twit_log("TWITC compute_twit_single_dimension\n");
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
	PyObject* srcidxs = PyArray_SimpleNew(1, dims, NPY_INT64);
	PyObject* dstidxs = PyArray_SimpleNew(1, dims, NPY_INT64);
	PyObject* weights = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

	npy_intp pp[] = { 0 };
	INT64* sip = (INT64*)PyArray_GetPtr((PyArrayObject*)srcidxs, pp);
	INT64* dip = (INT64*)PyArray_GetPtr((PyArrayObject*)dstidxs, pp);
	double* wp = (double*)PyArray_GetPtr((PyArrayObject*)weights, pp);
	for (int i = 0; i < ptr->length; i++) {
		sip[i] = ptr->srcidxs[i];
		dip[i] = ptr->dstidxs[i];
		wp[i] = ptr->weights[i];
	}

	free_twit_single_axis(ptr);

	PyObject* rslt = PyTuple_New(3);
	PyTuple_SetItem(rslt, 0, srcidxs);
	PyTuple_SetItem(rslt, 1, dstidxs);
	PyTuple_SetItem(rslt, 2, weights);
	return rslt;

}

// n_dims is the number of dimensions in the source and destination arrays.  They have to be the same count of dimensions each.
// Possibly some higher dimensions are size 1.
// twit_i is the integer start and ends of ranges in quads, t1_start, t1_end, t2_start, t2_end and then repeating for each dimension 
// in python order.  twit_w is pairs for dimension weights, w_start, w_end repeating in python order for the dimensions.
twit_multi_axis* _compute_twit_multi_dimension(const INT64 n_dims, INT64 const* const twit_i, double const* const twit_w) {
	// Generate axis series'
	twit_multi_axis* twit = new twit_multi_axis;
	twit->length = n_dims;
	twit->axs = (twit_single_axis**)PyMem_Malloc(n_dims * sizeof(twit_single_axis*));
	//twit_log("_compute_twit_multi_dimension: n_dims %lld\n", n_dims);

	for (INT64 i = 0; i < n_dims; i++) {
		//twit_log(" _compute_twit_multi_dimension: %lld\n", i);
		// This points to t1_start, t1_end, t2_start, t2_end
		INT64 const* const twit_ii = twit_i + i * 4LL;
		// This points to w_start, w_end
		double const* const twit_wi = twit_w + i * 2LL;
		twit->axs[i] = _compute_twit_single_dimension(twit_ii[0], twit_ii[1], twit_ii[2], twit_ii[3], twit_wi[0], twit_wi[1]);
	}

	//print_twit_multi_axis(twit, 0, "TWIT: ");
	return(twit);
}

// Do a twit transfer from t1 (source) to t2 (destionation) by twit.
// t1, t2, and twit all have the same number of dimensions, n_dims.
// t1_dims and t2_dims are how long the diemsions are.
// t1 is a block of doubles.  t1_dims is first the count of dimensions, then the dimensions, python order so higher dimensions first.
// t2 and t2_dims the same.  t1 is src, t2 is dst.
// twit_i is the integer start and ends of ranges in quads, t1_start, t1_end, t2_start, t2_end and then repeating for each dimension 
// in python order.  twit_w is pairs for dimension weights, w_start, w_end repeating in python order for the dimensions.
void _make_and_apply_twit(const INT64 n_dims, double const* const t1, INT64 const* const t1_dims, double* const t2, INT64 const* const t2_dims, INT64 const* const twit_i, double const* const twit_w, const INT64 preclear) {
	// Generate axis series'
	twit_multi_axis* twit = _compute_twit_multi_dimension(n_dims, twit_i, twit_w);

	// Apply from t1 to t2
	_apply_twit(twit, t1, t1_dims, t2, t2_dims, preclear);

	free_twit_multi_axis(twit);
}

/// Apply twit to the tensors.  
///
/// Is thread safe.  You can either call in single thread mode (twit_set_thread_count is 0 or 1) and call from multiple threads
/// as long as you can ensure the destination is not used by multiple threads simultaneouslym OR
/// You can set twit_set_thread_count to more than 1 and this will multithread at the dimension level. You can then call this method from multiple threads
/// and even if the destinations are shared, it will lock correctly and thread correctly.
void _apply_twit(twit_multi_axis const* const twit, double const* const t1, INT64 const* const t1_dims, double* const t2, INT64 const* const t2_dims, const INT64 preclear) {
	bool dbg = false;
	if (twit_thread_count <= 1) {
		twit_processing_context ctx(twit, t1, t1_dims, t2, t2_dims, preclear);
		_apply_twit_single_thread(&ctx);
	}
	else {
		// Only allow one context at a time.
		if (dbg) twit_log("_apply_twit: Locking\n");
		std::lock_guard<std::recursive_mutex> guard(twit_mt_mutex);
		if (dbg) twit_log("_apply_twit: lock ok\n");
		//std::this_thread::sleep_for(std::chrono::milliseconds(2000));

		twit_processing_context ctx(twit, t1, t1_dims, t2, t2_dims, preclear);
		if (dbg) twit_log("_apply_twit: made ctx\n");
		//std::this_thread::sleep_for(std::chrono::milliseconds(2000));

		if (dbg) twit_log("_apply_twit: make _apply_twit_multi_thread\n");
		//std::this_thread::sleep_for(std::chrono::milliseconds(2000));

		_apply_twit_multi_thread(&ctx);
		if (dbg) twit_log("_apply_twit: done\n");
		//std::this_thread::sleep_for(std::chrono::milliseconds(2000));
	}

}

// Find which axis would be best for iteration with the twit_thread_count.
// Favor longer axis, and outer most loop axis. So a rectangular image of 200 x 200 x 3
// with 4 threads would favor the outermost 200
// 3 x 200 x 200 with 4 threads would not want to iterate the 3 with 4 threads, and instead would do the middle 200.
// A cube 400 x 400 x 400 would favor the outermost (first) 400
//
// Will throw if twit_thread_count <= 1
//
// Returns -1 if multithreadding is a bad idea. E.g. all axis are less than the thread count.
float axis_quality(int total_axis_count, int axis_index, int axis_length) {
	bool dbg = false;
	assert(twit_thread_count > 1);
	if (axis_length < twit_thread_count * 3) {
		if (dbg) twit_log("Axis %d of %d. Axis length %d, too short, skipped\n", axis_index, total_axis_count, axis_length);
		return -1;
	}

	float aq = pow((float)axis_length, 1.1f) + 100.0 * pow((float)(total_axis_count - axis_index), 2.0f);
	if (dbg) twit_log("Axis %d of %d. Axis length %d, quality %f\n", axis_index, total_axis_count, axis_length, aq);
	return aq;
}

int find_best_threading_axis(twit_multi_axis const* const twit, INT64 const* const dims) {
	assert(twit->length > 1);
	int besti = -1;
	float bestaq = -1.0f;

	for (int i = 0; i < twit->length; i++) {
		float aq = axis_quality((int)twit->length, i, (int)dims[i]);
		if (aq > 0 && aq > bestaq) {
			bestaq = aq;
			besti = i;
		}
	}
	return besti;
}

void copy_ctx_to_threads(twit_processing_context* ctx) {
	bool dbg = false;
	if (dbg) twit_log("copy_ctx_to_threads:\n");
	for (int i = 0; i < twit_thread_count; i++) {
		memcpy(&twit_thread_bundles[i]->ctx, ctx, sizeof(twit_processing_context));
	}
}
void clear_ctx_from_threads() {
	// twit_log("clear_ctx_from_threads\n");
	for (int i = 0; i < twit_thread_count; i++) {
		// twit_log("clear_ctx_from_threads: i=%d\n", i);
		memset(&twit_thread_bundles[i]->ctx, 0, sizeof(twit_processing_context));
	}
	// twit_log("clear_ctx_from_threads: done\n");
}

void _apply_twit_multi_thread(twit_processing_context* ctx) {
	bool dbg = false;
	if (dbg) twit_log("_apply_twit_multi_thread\n");

	twit_threading_axis = find_best_threading_axis(ctx->twit, ctx->t2_dims);
	if (dbg) twit_log("_apply_twit_multi_thread: Best axis is %d\n", twit_threading_axis);

	if (twit_threading_axis == -1) {
		_apply_twit_single_thread(ctx);
	}

	if (twit_threading_axis == 0) {
		twit_apply_MT_sequencer(ctx);
	}
	else if (twit_threading_axis == 1) {
		twit_apply_MT_sequencer(ctx);
	}
	else if (twit_threading_axis == 2) {
		twit_apply_MT_sequencer(ctx);
	}
	else if (twit_threading_axis == 3) {
		twit_apply_MT_sequencer(ctx);
	}
	else if (twit_threading_axis == 4) {
		twit_apply_MT_sequencer(ctx);
	}
	else if (twit_threading_axis == 5) {
		twit_apply_MT_sequencer(ctx);
	}
	else if (twit_threading_axis == 6) {
		twit_apply_MT_sequencer(ctx);
	}
	else {
		// Punt, not threaded for that use case yet.
		twit_log("Twit apply: Unimplemented multithread strategy: Falling back to single thread.");
		_apply_twit_single_thread(ctx);
	}

	clear_ctx_from_threads();
}

void _apply_twit_single_thread(twit_processing_context* ctx) {
	bool dbg = false;
	if (dbg) twit_log("TWIT  _apply_twit  ");
	if (ctx->twit == NULL) throw 1;
	// Src
	if (ctx->t1 == NULL) throw 2;
	// Dst
	if (ctx->t2 == NULL) throw 3;

	// Speed up some references.
	twit_multi_axis const* const twit = ctx->twit;
	double const* const t1 = ctx->t1;
	INT64 const* const t1_dims = ctx->t1_dims;
	double* const t2 = ctx->t2;
	INT64 const* const t2_dims = ctx->t2_dims;
	const INT64 preclear = ctx->preclear;

	// Fast constants. This entire method tries top save every cpu cycle possible.
	// Premature optimization is the root of all evil, yada yada yada.
	const INT64 L = twit->length;
	if (dbg) twit_log("L = %lld\n", L);

	if (L <= 0) throw 4;

	const INT64 L0 = twit->axs[0]->length;
	// These three are the source indicies, dset indicies, and weight triples along a given axis.
	// Generated by compute_twit_single_dimension()
	const INT64* srcidxs0 = twit->axs[0]->srcidxs;
	const INT64* dstidxs0 = twit->axs[0]->dstidxs;
	const double* ws0 = twit->axs[0]->weights;

	if (L == 1) {
		if (dbg) twit_log("_apply_twit  1D\n");
		if (preclear) {
			//twit_log("preclear\n");
			// TODO This preclear may set pixels to 0.0 more than once.
			// Could have a more efficient version that uses the dimension span
			// and directly sets the values, not from srcidxs0[N]
			for (INT64 i0 = 0; i0 < L0; i0++) {
				t2[srcidxs0[i0]] = 0.0;
			}
		}
		if (dbg) twit_log("Update src\n");
		for (INT64 i0 = 0; i0 < L0; i0++) {
			t2[dstidxs0[i0]] += t1[srcidxs0[i0]] * ws0[i0];
		}
		return;
	}
	else { // L >= 2
		const INT64* srcidxs1 = twit->axs[1]->srcidxs;
		const INT64* dstidxs1 = twit->axs[1]->dstidxs;
		const double* ws1 = twit->axs[1]->weights;
		const INT64 L1 = twit->axs[1]->length;

		if (L == 2) {
			if (dbg) twit_log("_apply_twit  2D\n");
			// This is how far a single incrmenet in the next higher axis advances along the source
			// or destination ndarrays.
			const INT64 srcadvance0 = t1_dims[1];
			const INT64 dstadvance0 = t2_dims[1];

			// Note: Dimensions are innermost last in the lists!
			// So dim[0] (first dim) changes the slowest and dim[L - 1] (last dim) changes the fastest.

			if (preclear) {
				for (INT64 i0 = 0; i0 < L0; i0++) {
					const INT64 doff0 = dstadvance0 * dstidxs0[i0];
					for (INT64 i1 = 0; i1 < L1; i1++) {
						t2[dstidxs1[i1] + doff0] = 0.0;
					}
				}
			}
			for (INT64 i0 = 0; i0 < L0; i0++) {
				const INT64 soff0 = srcadvance0 * srcidxs0[i0];
				const INT64 doff0 = dstadvance0 * dstidxs0[i0];
				const double w0 = ws0[i0];
				for (INT64 i1 = 0; i1 < L1; i1++) {
					t2[dstidxs1[i1] + doff0] += t1[srcidxs1[i1] + soff0] * w0 * ws1[i1];
				}
			}
			return;
		}
		else {
			const INT64* srcidxs2 = twit->axs[2]->srcidxs;
			const INT64* dstidxs2 = twit->axs[2]->dstidxs;
			const double* ws2 = twit->axs[2]->weights;
			const INT64 L2 = twit->axs[2]->length;

			if (L == 3) {
				if (dbg) twit_log("_apply_twit  3D\n");
				const INT64 srcadvance1 = t1_dims[2];
				const INT64 dstadvance1 = t2_dims[2];
				const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
				const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
				if (preclear) {
					if (dbg) twit_log("  preclear\n");
					for (INT64 i0 = 0; i0 < L0; i0++) {
						const INT64 doff0 = dstadvance0 * dstidxs0[i0];
						for (INT64 i1 = 0; i1 < L1; i1++) {
							const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
							for (INT64 i2 = 0; i2 < L2; i2++) {
								t2[dstidxs2[i2] + doff1] = 0.0;
							}
						}
					}
				}
				for (INT64 i0 = 0; i0 < L0; i0++) {
					if (dbg) twit_log("  i0 %lld\n", i0);
					const INT64 soff0 = srcadvance0 * srcidxs0[i0];
					const INT64 doff0 = dstadvance0 * dstidxs0[i0];
					const double w0 = ws0[i0];
					for (INT64 i1 = 0; i1 < L1; i1++) {
						if (i1 == 0 || i1 == L1 - 1) {
							if (dbg) twit_log("    i1 %lld\n", i1);
						}
						const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
						const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
						const double w1 = ws1[i1] * w0;
						for (INT64 i2 = 0; i2 < L2; i2++) {
							if (i2 == 0 || i2 == L2 - 1) {
								//twit_log("      i2 %lld\n", i2);
								//twit_log("L %lld, %lld %lld  i %lld %lld %lld\n", L0, L1, L2, i0, i1, i2);
							}
							t2[dstidxs2[i2] + doff1] += t1[srcidxs2[i2] + soff1] * w1 * ws2[i2];
						}
					}
				}
				if (dbg) twit_log("  return ==========================================\n");
				return;
			}
			else {
				const INT64* srcidxs3 = twit->axs[3]->srcidxs;
				const INT64* dstidxs3 = twit->axs[3]->dstidxs;
				const double* ws3 = twit->axs[3]->weights;
				const INT64 L3 = twit->axs[3]->length;
				if (L == 4) {
					if (dbg) twit_log("_apply_twit  4D\n");
					const INT64 srcadvance2 = t1_dims[3];
					const INT64 dstadvance2 = t2_dims[3];
					const INT64 srcadvance1 = t1_dims[2] * srcadvance2;
					const INT64 dstadvance1 = t2_dims[2] * dstadvance2;
					const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
					const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
					if (preclear) {
						for (INT64 i0 = 0; i0 < L0; i0++) {
							const INT64 doff0 = dstadvance0 * dstidxs0[i0];
							for (INT64 i1 = 0; i1 < L1; i1++) {
								const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
								for (INT64 i2 = 0; i2 < L2; i2++) {
									const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
									for (INT64 i3 = 0; i3 < L3; i3++) {
										t2[dstidxs3[i3] + doff2] = 0.0;
									}
								}
							}
						}
					}
					for (INT64 i0 = 0; i0 < L0; i0++) {
						const INT64 soff0 = srcadvance0 * srcidxs0[i0];
						const INT64 doff0 = dstadvance0 * dstidxs0[i0];
						const double w0 = ws0[i0];
						for (INT64 i1 = 0; i1 < L1; i1++) {
							const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
							const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
							const double w1 = ws1[i1] * w0;
							for (INT64 i2 = 0; i2 < L2; i2++) {
								const INT64 soff2 = soff1 + srcadvance2 * srcidxs2[i2];
								const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
								const double w2 = ws2[i2] * w1;
								for (INT64 i3 = 0; i3 < L3; i3++) {
									t2[dstidxs3[i3] + doff2] += t1[srcidxs3[i3] + soff2] * w2 * ws3[i3];
								}
							}
						}
					}
					return;
				}
				else {
					const INT64* srcidxs4 = twit->axs[4]->srcidxs;
					const INT64* dstidxs4 = twit->axs[4]->dstidxs;
					const double* ws4 = twit->axs[4]->weights;
					const INT64 L4 = twit->axs[4]->length;
					if (L == 5) {
						if (dbg) twit_log("_apply_twit  5D\n");
						const INT64 srcadvance3 = t1_dims[4];
						const INT64 dstadvance3 = t2_dims[4];
						const INT64 srcadvance2 = t1_dims[3] * srcadvance3;
						const INT64 dstadvance2 = t2_dims[3] * dstadvance3;
						const INT64 srcadvance1 = t1_dims[2] * srcadvance2;
						const INT64 dstadvance1 = t2_dims[2] * dstadvance2;
						const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
						const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
						if (preclear) {
							for (INT64 i0 = 0; i0 < L0; i0++) {
								const INT64 doff0 = dstadvance0 * dstidxs0[i0];
								for (INT64 i1 = 0; i1 < L1; i1++) {
									const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
									for (INT64 i2 = 0; i2 < L2; i2++) {
										const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
										for (INT64 i3 = 0; i3 < L3; i3++) {
											const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
											for (INT64 i4 = 0; i4 < L4; i4++) {
												t2[dstidxs4[i4] + doff3] = 0.0;
											}
										}
									}
								}
							}
						}
						for (INT64 i0 = 0; i0 < L0; i0++) {
							const INT64 soff0 = srcadvance0 * srcidxs0[i0];
							const INT64 doff0 = dstadvance0 * dstidxs0[i0];
							const double w0 = ws0[i0];
							for (INT64 i1 = 0; i1 < L1; i1++) {
								const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
								const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
								const double w1 = ws1[i1] * w0;
								for (INT64 i2 = 0; i2 < L2; i2++) {
									const INT64 soff2 = soff1 + srcadvance2 * srcidxs2[i2];
									const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
									const double w2 = ws2[i2] * w1;
									for (INT64 i3 = 0; i3 < L3; i3++) {
										const INT64 soff3 = soff2 + srcadvance3 * srcidxs3[i3];
										const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
										const double w3 = ws3[i3] * w2;
										for (INT64 i4 = 0; i4 < L4; i4++) {
											t2[dstidxs4[i4] + doff3] += t1[srcidxs4[i4] + soff3] * w3 * ws4[i4];
										}
									}
								}
							}
						}
						return;
					}
					else {
						const INT64* srcidxs5 = twit->axs[5]->srcidxs;
						const INT64* dstidxs5 = twit->axs[5]->dstidxs;
						const double* ws5 = twit->axs[5]->weights;
						const INT64 L5 = twit->axs[5]->length;
						if (L == 6) {
							if (dbg) twit_log("_apply_twit  6D\n");
							const INT64 srcadvance4 = t1_dims[5];
							const INT64 dstadvance4 = t2_dims[5];
							const INT64 srcadvance3 = t1_dims[4] * srcadvance4;
							const INT64 dstadvance3 = t2_dims[4] * dstadvance4;
							const INT64 srcadvance2 = t1_dims[3] * srcadvance3;
							const INT64 dstadvance2 = t2_dims[3] * dstadvance3;
							const INT64 srcadvance1 = t1_dims[2] * srcadvance2;
							const INT64 dstadvance1 = t2_dims[2] * dstadvance2;
							const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
							const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
							if (preclear) {
								for (INT64 i0 = 0; i0 < L0; i0++) {
									const INT64 doff0 = dstadvance0 * dstidxs0[i0];
									for (INT64 i1 = 0; i1 < L1; i1++) {
										const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
										for (INT64 i2 = 0; i2 < L2; i2++) {
											const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
											for (INT64 i3 = 0; i3 < L3; i3++) {
												const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
												for (INT64 i4 = 0; i4 < L4; i4++) {
													const INT64 doff4 = doff3 + dstadvance4 * dstidxs4[i4];
													for (INT64 i5 = 0; i5 < L5; i5++) {
														t2[dstidxs5[i5] + doff4] = 0.0;
													}
												}
											}
										}
									}
								}
							}
							for (INT64 i0 = 0; i0 < L0; i0++) {
								const INT64 soff0 = srcadvance0 * srcidxs0[i0];
								const INT64 doff0 = dstadvance0 * dstidxs0[i0];
								const double w0 = ws0[i0];
								for (INT64 i1 = 0; i1 < L1; i1++) {
									const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
									const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
									const double w1 = ws1[i1] * w0;
									for (INT64 i2 = 0; i2 < L2; i2++) {
										const INT64 soff2 = soff1 + srcadvance2 * srcidxs2[i2];
										const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
										const double w2 = ws2[i2] * w1;
										for (INT64 i3 = 0; i3 < L3; i3++) {
											const INT64 soff3 = soff2 + srcadvance3 * srcidxs3[i3];
											const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
											const double w3 = ws3[i3] * w2;
											for (INT64 i4 = 0; i4 < L4; i4++) {
												const INT64 soff4 = soff3 + srcadvance4 * srcidxs4[i4];
												const INT64 doff4 = doff3 + dstadvance4 * dstidxs4[i4];
												const double w4 = ws4[i4] * w3;
												for (INT64 i5 = 0; i5 < L5; i5++) {
													t2[dstidxs5[i5] + doff4] += t1[srcidxs5[i5] + soff4] * w4 * ws5[i5];
												}
											}
										}
									}
								}
							}
							return;
						}
						else {
							// Tsk tsk tsk, unimplemented number of dimensions.
							// They're all custom for each supported dimension count.
							// May implement a slower generic size handler for large numbers of dimensions?
							twit_log("_apply_twit  UNSUPPORTED TWIT Dimensions count.  Max is 6 Dimensions.\n");
							throw 42;
						}
					}
				}
			}
		}
	}
}

// Apply twit multithreadded, axis 0 is threaded.
void _apply_twit_MT_axis_0(twit_processing_context* ctx, int N) {
	bool dbg = false;
	if (dbg) twit_log("TWIT  _apply_twit_MT_axis_0  ");
	if (ctx->twit == NULL) {
		if (dbg) twit_log("TWIT  _apply_twit_MT_axis_0  NULL twit");
		throw 1;
	}
	// Src
	if (ctx->t1 == NULL) {
		if (dbg) twit_log("TWIT  _apply_twit_MT_axis_0  NULL t1");
		throw 2;
	}
	// Dst
	if (ctx->t2 == NULL) {
		if (dbg) twit_log("TWIT  _apply_twit_MT_axis_0  NULL t2");
		throw 3;
	}

	// Speed up some references.
	twit_multi_axis const* const twit = ctx->twit;
	double const* const t1 = ctx->t1;
	INT64 const* const t1_dims = ctx->t1_dims;
	double* const t2 = ctx->t2;
	INT64 const* const t2_dims = ctx->t2_dims;
	const INT64 preclear = ctx->preclear;


	// Fast constants. This entire method tries to save every cpu cycle possible.
	// Premature optimization is the root of all evil, yada yada yada.
	const INT64 L = twit->length;
	if (dbg) twit_log("L = %lld\n", L);

	if (L <= 0) throw 4;

	const INT64 L0 = twit->axs[0]->length;
	// These three are the source indicies, dset indicies, and weight triples along a given axis.
	// Generated by compute_twit_single_dimension()
	const INT64* srcidxs0 = twit->axs[0]->srcidxs;
	const INT64* dstidxs0 = twit->axs[0]->dstidxs;
	const double* ws0 = twit->axs[0]->weights;

	if (L == 1) {
		if (dbg) twit_log("_apply_twit  1D\n");
		if (preclear) {
			//twit_log("preclear\n");
			// TODO This preclear may set pixels to 0.0 more than once.
			// Could have a more efficient version that uses the dimension span
			// and directly sets the values, not from srcidxs0[N]
			for (INT64 i0 = 0; i0 < L0; i0++) {
				if ((srcidxs0[i0] & twit_thread_mask) == N) {
					t2[srcidxs0[i0]] = 0.0;
				}
			}
		}
		if (dbg) twit_log("Update src\n");
		for (INT64 i0 = 0; i0 < L0; i0++) {
			if ((srcidxs0[i0] & twit_thread_mask) == N) {
				t2[dstidxs0[i0]] += t1[srcidxs0[i0]] * ws0[i0];
			}
		}
		return;
	}
	else { // L >= 2
		const INT64* srcidxs1 = twit->axs[1]->srcidxs;
		const INT64* dstidxs1 = twit->axs[1]->dstidxs;
		const double* ws1 = twit->axs[1]->weights;
		const INT64 L1 = twit->axs[1]->length;

		if (L == 2) {
			if (dbg) twit_log("_apply_twit  2D\n");
			// This is how far a single incrmenet in the next higher axis advances along the source
			// or destination ndarrays.
			const INT64 srcadvance0 = t1_dims[1];
			const INT64 dstadvance0 = t2_dims[1];

			// Note: Dimensions are innermost last in the lists!
			// So dim[0] (first dim) changes the slowest and dim[L - 1] (last dim) changes the fastest.

			if (preclear) {
				for (INT64 i0 = 0; i0 < L0; i0++) {
					if ((srcidxs0[i0] & twit_thread_mask) == N) {
						const INT64 doff0 = dstadvance0 * dstidxs0[i0];
						for (INT64 i1 = 0; i1 < L1; i1++) {
							t2[dstidxs1[i1] + doff0] = 0.0;
						}
					}
				}
			}
			for (INT64 i0 = 0; i0 < L0; i0++) {
				if ((srcidxs0[i0] & twit_thread_mask) == N) {
					const INT64 soff0 = srcadvance0 * srcidxs0[i0];
					const INT64 doff0 = dstadvance0 * dstidxs0[i0];
					const double w0 = ws0[i0];
					for (INT64 i1 = 0; i1 < L1; i1++) {
						t2[dstidxs1[i1] + doff0] += t1[srcidxs1[i1] + soff0] * w0 * ws1[i1];
					}
				}
			}
			return;
		}
		else {
			const INT64* srcidxs2 = twit->axs[2]->srcidxs;
			const INT64* dstidxs2 = twit->axs[2]->dstidxs;
			const double* ws2 = twit->axs[2]->weights;
			const INT64 L2 = twit->axs[2]->length;

			if (L == 3) {
				if (dbg) twit_log("_apply_twit  3D\n");
				const INT64 srcadvance1 = t1_dims[2];
				const INT64 dstadvance1 = t2_dims[2];
				const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
				const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
				if (preclear) {
					if (dbg) twit_log("  preclear\n");
					for (INT64 i0 = 0; i0 < L0; i0++) {
						if ((srcidxs0[i0] & twit_thread_mask) == N) {
							const INT64 doff0 = dstadvance0 * dstidxs0[i0];
							for (INT64 i1 = 0; i1 < L1; i1++) {
								const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
								for (INT64 i2 = 0; i2 < L2; i2++) {
									t2[dstidxs2[i2] + doff1] = 0.0;
								}
							}
						}
					}
				}
				for (INT64 i0 = 0; i0 < L0; i0++) {
					if ((srcidxs0[i0] & twit_thread_mask) == N) {
						if (dbg) twit_log("  i0 %lld\n", i0);
						const INT64 soff0 = srcadvance0 * srcidxs0[i0];
						const INT64 doff0 = dstadvance0 * dstidxs0[i0];
						const double w0 = ws0[i0];
						for (INT64 i1 = 0; i1 < L1; i1++) {
							if (i1 == 0 || i1 == L1 - 1) {
								if (dbg) twit_log("    i1 %lld\n", i1);
							}
							const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
							const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
							const double w1 = ws1[i1] * w0;
							for (INT64 i2 = 0; i2 < L2; i2++) {
								if (i2 == 0 || i2 == L2 - 1) {
									//twit_log("      i2 %lld\n", i2);
									//twit_log("L %lld, %lld %lld  i %lld %lld %lld\n", L0, L1, L2, i0, i1, i2);
								}
								t2[dstidxs2[i2] + doff1] += t1[srcidxs2[i2] + soff1] * w1 * ws2[i2];
							}
						}
					}
				}
				if (dbg) twit_log("  return ==========================================\n");
				return;
			}
			else {
				const INT64* srcidxs3 = twit->axs[3]->srcidxs;
				const INT64* dstidxs3 = twit->axs[3]->dstidxs;
				const double* ws3 = twit->axs[3]->weights;
				const INT64 L3 = twit->axs[3]->length;
				if (L == 4) {
					if (dbg) twit_log("_apply_twit  4D\n");
					const INT64 srcadvance2 = t1_dims[3];
					const INT64 dstadvance2 = t2_dims[3];
					const INT64 srcadvance1 = t1_dims[2] * srcadvance2;
					const INT64 dstadvance1 = t2_dims[2] * dstadvance2;
					const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
					const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
					if (preclear) {
						for (INT64 i0 = 0; i0 < L0; i0++) {
							if ((srcidxs0[i0] & twit_thread_mask) == N) {
								const INT64 doff0 = dstadvance0 * dstidxs0[i0];
								for (INT64 i1 = 0; i1 < L1; i1++) {
									const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
									for (INT64 i2 = 0; i2 < L2; i2++) {
										const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
										for (INT64 i3 = 0; i3 < L3; i3++) {
											t2[dstidxs3[i3] + doff2] = 0.0;
										}
									}
								}
							}
						}
					}
					for (INT64 i0 = 0; i0 < L0; i0++) {
						if ((srcidxs0[i0] & twit_thread_mask) == N) {
							const INT64 soff0 = srcadvance0 * srcidxs0[i0];
							const INT64 doff0 = dstadvance0 * dstidxs0[i0];
							const double w0 = ws0[i0];
							for (INT64 i1 = 0; i1 < L1; i1++) {
								const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
								const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
								const double w1 = ws1[i1] * w0;
								for (INT64 i2 = 0; i2 < L2; i2++) {
									const INT64 soff2 = soff1 + srcadvance2 * srcidxs2[i2];
									const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
									const double w2 = ws2[i2] * w1;
									for (INT64 i3 = 0; i3 < L3; i3++) {
										t2[dstidxs3[i3] + doff2] += t1[srcidxs3[i3] + soff2] * w2 * ws3[i3];
									}
								}
							}
						}
					}
					return;
				}
				else {
					const INT64* srcidxs4 = twit->axs[4]->srcidxs;
					const INT64* dstidxs4 = twit->axs[4]->dstidxs;
					const double* ws4 = twit->axs[4]->weights;
					const INT64 L4 = twit->axs[4]->length;
					if (L == 5) {
						if (dbg) twit_log("_apply_twit  5D\n");
						const INT64 srcadvance3 = t1_dims[4];
						const INT64 dstadvance3 = t2_dims[4];
						const INT64 srcadvance2 = t1_dims[3] * srcadvance3;
						const INT64 dstadvance2 = t2_dims[3] * dstadvance3;
						const INT64 srcadvance1 = t1_dims[2] * srcadvance2;
						const INT64 dstadvance1 = t2_dims[2] * dstadvance2;
						const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
						const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
						if (preclear) {
							for (INT64 i0 = 0; i0 < L0; i0++) {
								if ((srcidxs0[i0] & twit_thread_mask) == N) {
									const INT64 doff0 = dstadvance0 * dstidxs0[i0];
									for (INT64 i1 = 0; i1 < L1; i1++) {
										const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
										for (INT64 i2 = 0; i2 < L2; i2++) {
											const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
											for (INT64 i3 = 0; i3 < L3; i3++) {
												const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
												for (INT64 i4 = 0; i4 < L4; i4++) {
													t2[dstidxs4[i4] + doff3] = 0.0;
												}
											}
										}
									}
								}
							}
						}
						for (INT64 i0 = 0; i0 < L0; i0++) {
							if ((srcidxs0[i0] & twit_thread_mask) == N) {
								const INT64 soff0 = srcadvance0 * srcidxs0[i0];
								const INT64 doff0 = dstadvance0 * dstidxs0[i0];
								const double w0 = ws0[i0];
								for (INT64 i1 = 0; i1 < L1; i1++) {
									const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
									const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
									const double w1 = ws1[i1] * w0;
									for (INT64 i2 = 0; i2 < L2; i2++) {
										const INT64 soff2 = soff1 + srcadvance2 * srcidxs2[i2];
										const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
										const double w2 = ws2[i2] * w1;
										for (INT64 i3 = 0; i3 < L3; i3++) {
											const INT64 soff3 = soff2 + srcadvance3 * srcidxs3[i3];
											const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
											const double w3 = ws3[i3] * w2;
											for (INT64 i4 = 0; i4 < L4; i4++) {
												t2[dstidxs4[i4] + doff3] += t1[srcidxs4[i4] + soff3] * w3 * ws4[i4];
											}
										}
									}
								}
							}
						}
						return;
					}
					else {
						const INT64* srcidxs5 = twit->axs[5]->srcidxs;
						const INT64* dstidxs5 = twit->axs[5]->dstidxs;
						const double* ws5 = twit->axs[5]->weights;
						const INT64 L5 = twit->axs[5]->length;
						if (L == 6) {
							if (dbg) twit_log("_apply_twit  6D\n");
							const INT64 srcadvance4 = t1_dims[5];
							const INT64 dstadvance4 = t2_dims[5];
							const INT64 srcadvance3 = t1_dims[4] * srcadvance4;
							const INT64 dstadvance3 = t2_dims[4] * dstadvance4;
							const INT64 srcadvance2 = t1_dims[3] * srcadvance3;
							const INT64 dstadvance2 = t2_dims[3] * dstadvance3;
							const INT64 srcadvance1 = t1_dims[2] * srcadvance2;
							const INT64 dstadvance1 = t2_dims[2] * dstadvance2;
							const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
							const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
							if (preclear) {
								for (INT64 i0 = 0; i0 < L0; i0++) {
									if ((srcidxs0[i0] & twit_thread_mask) == N) {
										const INT64 doff0 = dstadvance0 * dstidxs0[i0];
										for (INT64 i1 = 0; i1 < L1; i1++) {
											const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
											for (INT64 i2 = 0; i2 < L2; i2++) {
												const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
												for (INT64 i3 = 0; i3 < L3; i3++) {
													const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
													for (INT64 i4 = 0; i4 < L4; i4++) {
														const INT64 doff4 = doff3 + dstadvance4 * dstidxs4[i4];
														for (INT64 i5 = 0; i5 < L5; i5++) {
															t2[dstidxs5[i5] + doff4] = 0.0;
														}
													}
												}
											}
										}
									}
								}
							}
							for (INT64 i0 = 0; i0 < L0; i0++) {
								if ((srcidxs0[i0] & twit_thread_mask) == N) {
									const INT64 soff0 = srcadvance0 * srcidxs0[i0];
									const INT64 doff0 = dstadvance0 * dstidxs0[i0];
									const double w0 = ws0[i0];
									for (INT64 i1 = 0; i1 < L1; i1++) {
										const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
										const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
										const double w1 = ws1[i1] * w0;
										for (INT64 i2 = 0; i2 < L2; i2++) {
											const INT64 soff2 = soff1 + srcadvance2 * srcidxs2[i2];
											const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
											const double w2 = ws2[i2] * w1;
											for (INT64 i3 = 0; i3 < L3; i3++) {
												const INT64 soff3 = soff2 + srcadvance3 * srcidxs3[i3];
												const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
												const double w3 = ws3[i3] * w2;
												for (INT64 i4 = 0; i4 < L4; i4++) {
													const INT64 soff4 = soff3 + srcadvance4 * srcidxs4[i4];
													const INT64 doff4 = doff3 + dstadvance4 * dstidxs4[i4];
													const double w4 = ws4[i4] * w3;
													for (INT64 i5 = 0; i5 < L5; i5++) {
														t2[dstidxs5[i5] + doff4] += t1[srcidxs5[i5] + soff4] * w4 * ws5[i5];
													}
												}
											}
										}
									}
								}
							}
							return;
						}
						else {
							// Tsk tsk tsk, unimplemented number of dimensions.
							// They're all custom for each supported dimension count.
							// May implement a slower generic size handler for large numbers of dimensions?
							twit_log("_apply_twit  UNSUPPORTED TWIT Dimensions count.  Max is 6 Dimensions.\n");
							throw 42;
						}
					}
				}
			}
		}
	}
}

void _apply_twit_MT_axis_1(twit_processing_context* ctx, int N) {
	bool dbg = false;
	if (dbg) twit_log("TWIT  _apply_twit_MT_axis_1  ");
	if (ctx->twit == NULL) {
		if (dbg) twit_log("TWIT  _apply_twit_MT_axis_1  NULL twit");
		throw 1;
	}
	// Src
	if (ctx->t1 == NULL) {
		if (dbg) twit_log("TWIT  _apply_twit_MT_axis_1  NULL t1");
		throw 2;
	}
	// Dst
	if (ctx->t2 == NULL) {
		if (dbg) twit_log("TWIT  _apply_twit_MT_axis_1  NULL t2");
		throw 3;
	}

	// Speed up some references.
	twit_multi_axis const* const twit = ctx->twit;
	double const* const t1 = ctx->t1;
	INT64 const* const t1_dims = ctx->t1_dims;
	double* const t2 = ctx->t2;
	INT64 const* const t2_dims = ctx->t2_dims;
	const INT64 preclear = ctx->preclear;


	// Fast constants. This entire method tries to save every cpu cycle possible.
	// Premature optimization is the root of all evil, yada yada yada.
	const INT64 L = twit->length;
	if (dbg) twit_log("L = %lld\n", L);

	if (L <= 1) throw 4;

	const INT64 L0 = twit->axs[0]->length;
	// These three are the source indicies, dset indicies, and weight triples along a given axis.
	// Generated by compute_twit_single_dimension()
	const INT64* srcidxs0 = twit->axs[0]->srcidxs;
	const INT64* dstidxs0 = twit->axs[0]->dstidxs;
	const double* ws0 = twit->axs[0]->weights;

	const INT64* srcidxs1 = twit->axs[1]->srcidxs;
	const INT64* dstidxs1 = twit->axs[1]->dstidxs;
	const double* ws1 = twit->axs[1]->weights;
	const INT64 L1 = twit->axs[1]->length;

	if (L == 2) {
		if (dbg) twit_log("_apply_twit  2D\n");
		// This is how far a single incrmenet in the next higher axis advances along the source
		// or destination ndarrays.
		const INT64 srcadvance0 = t1_dims[1];
		const INT64 dstadvance0 = t2_dims[1];

		// Note: Dimensions are innermost last in the lists!
		// So dim[0] (first dim) changes the slowest and dim[L - 1] (last dim) changes the fastest.

		if (preclear) {
			for (INT64 i0 = 0; i0 < L0; i0++) {
				const INT64 doff0 = dstadvance0 * dstidxs0[i0];
				for (INT64 i1 = 0; i1 < L1; i1++) {
					if ((srcidxs1[i1] & twit_thread_mask) == N) {
						t2[dstidxs1[i1] + doff0] = 0.0;
					}
				}
			}
		}
		for (INT64 i0 = 0; i0 < L0; i0++) {
			const INT64 soff0 = srcadvance0 * srcidxs0[i0];
			const INT64 doff0 = dstadvance0 * dstidxs0[i0];
			const double w0 = ws0[i0];
			for (INT64 i1 = 0; i1 < L1; i1++) {
				if ((srcidxs1[i1] & twit_thread_mask) == N) {
					t2[dstidxs1[i1] + doff0] += t1[srcidxs1[i1] + soff0] * w0 * ws1[i1];

				}
			}
		}
		return;
	}
	else {
		const INT64* srcidxs2 = twit->axs[2]->srcidxs;
		const INT64* dstidxs2 = twit->axs[2]->dstidxs;
		const double* ws2 = twit->axs[2]->weights;
		const INT64 L2 = twit->axs[2]->length;

		if (L == 3) {
			if (dbg) twit_log("_apply_twit  3D\n");
			const INT64 srcadvance1 = t1_dims[2];
			const INT64 dstadvance1 = t2_dims[2];
			const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
			const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
			if (preclear) {
				if (dbg) twit_log("  preclear\n");
				for (INT64 i0 = 0; i0 < L0; i0++) {
					const INT64 doff0 = dstadvance0 * dstidxs0[i0];
					for (INT64 i1 = 0; i1 < L1; i1++) {
						if ((srcidxs1[i1] & twit_thread_mask) == N) {
							const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
							for (INT64 i2 = 0; i2 < L2; i2++) {
								t2[dstidxs2[i2] + doff1] = 0.0;
							}
						}
					}
				}
			}
			for (INT64 i0 = 0; i0 < L0; i0++) {
				if (dbg) twit_log("  i0 %lld\n", i0);
				const INT64 soff0 = srcadvance0 * srcidxs0[i0];
				const INT64 doff0 = dstadvance0 * dstidxs0[i0];
				const double w0 = ws0[i0];
				for (INT64 i1 = 0; i1 < L1; i1++) {
					if ((srcidxs1[i1] & twit_thread_mask) == N) {
						if (i1 == 0 || i1 == L1 - 1) {
							if (dbg) twit_log("    i1 %lld\n", i1);
						}
						const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
						const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
						const double w1 = ws1[i1] * w0;
						for (INT64 i2 = 0; i2 < L2; i2++) {
							if (i2 == 0 || i2 == L2 - 1) {
								//twit_log("      i2 %lld\n", i2);
								//twit_log("L %lld, %lld %lld  i %lld %lld %lld\n", L0, L1, L2, i0, i1, i2);
							}
							t2[dstidxs2[i2] + doff1] += t1[srcidxs2[i2] + soff1] * w1 * ws2[i2];
						}
					}
				}
			}
			if (dbg) twit_log("  return ==========================================\n");
			return;
		}
		else {
			const INT64* srcidxs3 = twit->axs[3]->srcidxs;
			const INT64* dstidxs3 = twit->axs[3]->dstidxs;
			const double* ws3 = twit->axs[3]->weights;
			const INT64 L3 = twit->axs[3]->length;
			if (L == 4) {
				if (dbg) twit_log("_apply_twit  4D\n");
				const INT64 srcadvance2 = t1_dims[3];
				const INT64 dstadvance2 = t2_dims[3];
				const INT64 srcadvance1 = t1_dims[2] * srcadvance2;
				const INT64 dstadvance1 = t2_dims[2] * dstadvance2;
				const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
				const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
				if (preclear) {
					for (INT64 i0 = 0; i0 < L0; i0++) {
						const INT64 doff0 = dstadvance0 * dstidxs0[i0];
						for (INT64 i1 = 0; i1 < L1; i1++) {
							if ((srcidxs1[i1] & twit_thread_mask) == N) {
								const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
								for (INT64 i2 = 0; i2 < L2; i2++) {
									const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
									for (INT64 i3 = 0; i3 < L3; i3++) {
										t2[dstidxs3[i3] + doff2] = 0.0;
									}
								}
							}
						}
					}
				}
				for (INT64 i0 = 0; i0 < L0; i0++) {
					const INT64 soff0 = srcadvance0 * srcidxs0[i0];
					const INT64 doff0 = dstadvance0 * dstidxs0[i0];
					const double w0 = ws0[i0];
					for (INT64 i1 = 0; i1 < L1; i1++) {
						if ((srcidxs1[i1] & twit_thread_mask) == N) {
							const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
							const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
							const double w1 = ws1[i1] * w0;
							for (INT64 i2 = 0; i2 < L2; i2++) {
								const INT64 soff2 = soff1 + srcadvance2 * srcidxs2[i2];
								const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
								const double w2 = ws2[i2] * w1;
								for (INT64 i3 = 0; i3 < L3; i3++) {
									t2[dstidxs3[i3] + doff2] += t1[srcidxs3[i3] + soff2] * w2 * ws3[i3];
								}
							}
						}
					}
				}
				return;
			}
			else {
				const INT64* srcidxs4 = twit->axs[4]->srcidxs;
				const INT64* dstidxs4 = twit->axs[4]->dstidxs;
				const double* ws4 = twit->axs[4]->weights;
				const INT64 L4 = twit->axs[4]->length;
				if (L == 5) {
					if (dbg) twit_log("_apply_twit  5D\n");
					const INT64 srcadvance3 = t1_dims[4];
					const INT64 dstadvance3 = t2_dims[4];
					const INT64 srcadvance2 = t1_dims[3] * srcadvance3;
					const INT64 dstadvance2 = t2_dims[3] * dstadvance3;
					const INT64 srcadvance1 = t1_dims[2] * srcadvance2;
					const INT64 dstadvance1 = t2_dims[2] * dstadvance2;
					const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
					const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
					if (preclear) {
						for (INT64 i0 = 0; i0 < L0; i0++) {
							const INT64 doff0 = dstadvance0 * dstidxs0[i0];
							for (INT64 i1 = 0; i1 < L1; i1++) {
								if ((srcidxs1[i1] & twit_thread_mask) == N) {
									const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
									for (INT64 i2 = 0; i2 < L2; i2++) {
										const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
										for (INT64 i3 = 0; i3 < L3; i3++) {
											const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
											for (INT64 i4 = 0; i4 < L4; i4++) {
												t2[dstidxs4[i4] + doff3] = 0.0;
											}
										}
									}
								}
							}
						}
					}
					for (INT64 i0 = 0; i0 < L0; i0++) {
						const INT64 soff0 = srcadvance0 * srcidxs0[i0];
						const INT64 doff0 = dstadvance0 * dstidxs0[i0];
						const double w0 = ws0[i0];
						for (INT64 i1 = 0; i1 < L1; i1++) {
							if ((srcidxs1[i1] & twit_thread_mask) == N) {
								const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
								const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
								const double w1 = ws1[i1] * w0;
								for (INT64 i2 = 0; i2 < L2; i2++) {
									const INT64 soff2 = soff1 + srcadvance2 * srcidxs2[i2];
									const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
									const double w2 = ws2[i2] * w1;
									for (INT64 i3 = 0; i3 < L3; i3++) {
										const INT64 soff3 = soff2 + srcadvance3 * srcidxs3[i3];
										const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
										const double w3 = ws3[i3] * w2;
										for (INT64 i4 = 0; i4 < L4; i4++) {
											t2[dstidxs4[i4] + doff3] += t1[srcidxs4[i4] + soff3] * w3 * ws4[i4];
										}
									}
								}
							}
						}
					}
					return;
				}
				else {
					const INT64* srcidxs5 = twit->axs[5]->srcidxs;
					const INT64* dstidxs5 = twit->axs[5]->dstidxs;
					const double* ws5 = twit->axs[5]->weights;
					const INT64 L5 = twit->axs[5]->length;
					if (L == 6) {
						if (dbg) twit_log("_apply_twit  6D\n");
						const INT64 srcadvance4 = t1_dims[5];
						const INT64 dstadvance4 = t2_dims[5];
						const INT64 srcadvance3 = t1_dims[4] * srcadvance4;
						const INT64 dstadvance3 = t2_dims[4] * dstadvance4;
						const INT64 srcadvance2 = t1_dims[3] * srcadvance3;
						const INT64 dstadvance2 = t2_dims[3] * dstadvance3;
						const INT64 srcadvance1 = t1_dims[2] * srcadvance2;
						const INT64 dstadvance1 = t2_dims[2] * dstadvance2;
						const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
						const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
						if (preclear) {
							for (INT64 i0 = 0; i0 < L0; i0++) {
								const INT64 doff0 = dstadvance0 * dstidxs0[i0];
								for (INT64 i1 = 0; i1 < L1; i1++) {
									if ((srcidxs1[i1] & twit_thread_mask) == N) {
										const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
										for (INT64 i2 = 0; i2 < L2; i2++) {
											const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
											for (INT64 i3 = 0; i3 < L3; i3++) {
												const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
												for (INT64 i4 = 0; i4 < L4; i4++) {
													const INT64 doff4 = doff3 + dstadvance4 * dstidxs4[i4];
													for (INT64 i5 = 0; i5 < L5; i5++) {
														t2[dstidxs5[i5] + doff4] = 0.0;
													}
												}
											}
										}
									}
								}
							}
						}
						for (INT64 i0 = 0; i0 < L0; i0++) {
							const INT64 soff0 = srcadvance0 * srcidxs0[i0];
							const INT64 doff0 = dstadvance0 * dstidxs0[i0];
							const double w0 = ws0[i0];
							for (INT64 i1 = 0; i1 < L1; i1++) {
								if ((srcidxs1[i1] & twit_thread_mask) == N) {
									const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
									const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
									const double w1 = ws1[i1] * w0;
									for (INT64 i2 = 0; i2 < L2; i2++) {
										const INT64 soff2 = soff1 + srcadvance2 * srcidxs2[i2];
										const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
										const double w2 = ws2[i2] * w1;
										for (INT64 i3 = 0; i3 < L3; i3++) {
											const INT64 soff3 = soff2 + srcadvance3 * srcidxs3[i3];
											const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
											const double w3 = ws3[i3] * w2;
											for (INT64 i4 = 0; i4 < L4; i4++) {
												const INT64 soff4 = soff3 + srcadvance4 * srcidxs4[i4];
												const INT64 doff4 = doff3 + dstadvance4 * dstidxs4[i4];
												const double w4 = ws4[i4] * w3;
												for (INT64 i5 = 0; i5 < L5; i5++) {
													t2[dstidxs5[i5] + doff4] += t1[srcidxs5[i5] + soff4] * w4 * ws5[i5];
												}
											}
										}
									}
								}
							}
						}
						return;
					}
					else {
						// Tsk tsk tsk, unimplemented number of dimensions.
						// They're all custom for each supported dimension count.
						// May implement a slower generic size handler for large numbers of dimensions?
						twit_log("_apply_twit  UNSUPPORTED TWIT Dimensions count.  Max is 6 Dimensions.\n");
						throw 42;
					}
				}
			}
		}
	}
}

void _apply_twit_MT_axis_2(twit_processing_context* ctx, int N) {
	bool dbg = false;
	if (dbg) twit_log("TWIT  _apply_twit_MT_axis_2  ");
	if (ctx->twit == NULL) {
		if (dbg) twit_log("TWIT  _apply_twit_MT_axis_2  NULL twit");
		throw 1;
	}
	// Src
	if (ctx->t1 == NULL) {
		if (dbg) twit_log("TWIT  _apply_twit_MT_axis_2  NULL t1");
		throw 2;
	}
	// Dst
	if (ctx->t2 == NULL) {
		if (dbg) twit_log("TWIT  _apply_twit_MT_axis_2  NULL t2");
		throw 3;
	}

	// Speed up some references.
	twit_multi_axis const* const twit = ctx->twit;
	double const* const t1 = ctx->t1;
	INT64 const* const t1_dims = ctx->t1_dims;
	double* const t2 = ctx->t2;
	INT64 const* const t2_dims = ctx->t2_dims;
	const INT64 preclear = ctx->preclear;


	// Fast constants. This entire method tries to save every cpu cycle possible.
	// Premature optimization is the root of all evil, yada yada yada.
	const INT64 L = twit->length;
	if (dbg) twit_log("L = %lld\n", L);

	if (L <= 2) throw 4;

	const INT64 L0 = twit->axs[0]->length;
	// These three are the source indicies, dset indicies, and weight triples along a given axis.
	// Generated by compute_twit_single_dimension()
	const INT64* srcidxs0 = twit->axs[0]->srcidxs;
	const INT64* dstidxs0 = twit->axs[0]->dstidxs;
	const double* ws0 = twit->axs[0]->weights;

	const INT64* srcidxs1 = twit->axs[1]->srcidxs;
	const INT64* dstidxs1 = twit->axs[1]->dstidxs;
	const double* ws1 = twit->axs[1]->weights;
	const INT64 L1 = twit->axs[1]->length;

	const INT64* srcidxs2 = twit->axs[2]->srcidxs;
	const INT64* dstidxs2 = twit->axs[2]->dstidxs;
	const double* ws2 = twit->axs[2]->weights;
	const INT64 L2 = twit->axs[2]->length;

	if (L == 3) {
		if (dbg) twit_log("_apply_twit  3D\n");
		const INT64 srcadvance1 = t1_dims[2];
		const INT64 dstadvance1 = t2_dims[2];
		const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
		const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
		if (preclear) {
			if (dbg) twit_log("  preclear\n");
			for (INT64 i0 = 0; i0 < L0; i0++) {
				const INT64 doff0 = dstadvance0 * dstidxs0[i0];
				for (INT64 i1 = 0; i1 < L1; i1++) {
					const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
					for (INT64 i2 = 0; i2 < L2; i2++) {
						if ((srcidxs2[i2] & twit_thread_mask) == N) {
							t2[dstidxs2[i2] + doff1] = 0.0;
						}
					}
				}
			}
		}
		for (INT64 i0 = 0; i0 < L0; i0++) {
			if (dbg) twit_log("  i0 %lld\n", i0);
			const INT64 soff0 = srcadvance0 * srcidxs0[i0];
			const INT64 doff0 = dstadvance0 * dstidxs0[i0];
			const double w0 = ws0[i0];
			for (INT64 i1 = 0; i1 < L1; i1++) {
				const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
				const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
				const double w1 = ws1[i1] * w0;
				for (INT64 i2 = 0; i2 < L2; i2++) {
					if ((srcidxs2[i2] & twit_thread_mask) == N) {
						t2[dstidxs2[i2] + doff1] += t1[srcidxs2[i2] + soff1] * w1 * ws2[i2];
					}
				}
			}
		}
		if (dbg) twit_log("  return ==========================================\n");
		return;
	}
	else {
		const INT64* srcidxs3 = twit->axs[3]->srcidxs;
		const INT64* dstidxs3 = twit->axs[3]->dstidxs;
		const double* ws3 = twit->axs[3]->weights;
		const INT64 L3 = twit->axs[3]->length;
		if (L == 4) {
			if (dbg) twit_log("_apply_twit  4D\n");
			const INT64 srcadvance2 = t1_dims[3];
			const INT64 dstadvance2 = t2_dims[3];
			const INT64 srcadvance1 = t1_dims[2] * srcadvance2;
			const INT64 dstadvance1 = t2_dims[2] * dstadvance2;
			const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
			const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
			if (preclear) {
				for (INT64 i0 = 0; i0 < L0; i0++) {
					const INT64 doff0 = dstadvance0 * dstidxs0[i0];
					for (INT64 i1 = 0; i1 < L1; i1++) {
						const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
						for (INT64 i2 = 0; i2 < L2; i2++) {
							if ((srcidxs2[i2] & twit_thread_mask) == N) {
								const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
								for (INT64 i3 = 0; i3 < L3; i3++) {
									t2[dstidxs3[i3] + doff2] = 0.0;
								}
							}
						}
					}
				}
			}
			for (INT64 i0 = 0; i0 < L0; i0++) {
				const INT64 soff0 = srcadvance0 * srcidxs0[i0];
				const INT64 doff0 = dstadvance0 * dstidxs0[i0];
				const double w0 = ws0[i0];
				for (INT64 i1 = 0; i1 < L1; i1++) {
					const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
					const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
					const double w1 = ws1[i1] * w0;
					for (INT64 i2 = 0; i2 < L2; i2++) {
						if ((srcidxs2[i2] & twit_thread_mask) == N) {
							const INT64 soff2 = soff1 + srcadvance2 * srcidxs2[i2];
							const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
							const double w2 = ws2[i2] * w1;
							for (INT64 i3 = 0; i3 < L3; i3++) {
								t2[dstidxs3[i3] + doff2] += t1[srcidxs3[i3] + soff2] * w2 * ws3[i3];
							}
						}
					}
				}
			}
			return;
		}
		else {
			const INT64* srcidxs4 = twit->axs[4]->srcidxs;
			const INT64* dstidxs4 = twit->axs[4]->dstidxs;
			const double* ws4 = twit->axs[4]->weights;
			const INT64 L4 = twit->axs[4]->length;
			if (L == 5) {
				if (dbg) twit_log("_apply_twit  5D\n");
				const INT64 srcadvance3 = t1_dims[4];
				const INT64 dstadvance3 = t2_dims[4];
				const INT64 srcadvance2 = t1_dims[3] * srcadvance3;
				const INT64 dstadvance2 = t2_dims[3] * dstadvance3;
				const INT64 srcadvance1 = t1_dims[2] * srcadvance2;
				const INT64 dstadvance1 = t2_dims[2] * dstadvance2;
				const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
				const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
				if (preclear) {
					for (INT64 i0 = 0; i0 < L0; i0++) {
						const INT64 doff0 = dstadvance0 * dstidxs0[i0];
						for (INT64 i1 = 0; i1 < L1; i1++) {
							const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
							for (INT64 i2 = 0; i2 < L2; i2++) {
								if ((srcidxs2[i2] & twit_thread_mask) == N) {
									const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
									for (INT64 i3 = 0; i3 < L3; i3++) {
										const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
										for (INT64 i4 = 0; i4 < L4; i4++) {
											t2[dstidxs4[i4] + doff3] = 0.0;
										}
									}
								}
							}
						}
					}
				}
				for (INT64 i0 = 0; i0 < L0; i0++) {
					const INT64 soff0 = srcadvance0 * srcidxs0[i0];
					const INT64 doff0 = dstadvance0 * dstidxs0[i0];
					const double w0 = ws0[i0];
					for (INT64 i1 = 0; i1 < L1; i1++) {
						const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
						const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
						const double w1 = ws1[i1] * w0;
						for (INT64 i2 = 0; i2 < L2; i2++) {
							if ((srcidxs2[i2] & twit_thread_mask) == N) {
								const INT64 soff2 = soff1 + srcadvance2 * srcidxs2[i2];
								const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
								const double w2 = ws2[i2] * w1;
								for (INT64 i3 = 0; i3 < L3; i3++) {
									const INT64 soff3 = soff2 + srcadvance3 * srcidxs3[i3];
									const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
									const double w3 = ws3[i3] * w2;
									for (INT64 i4 = 0; i4 < L4; i4++) {
										t2[dstidxs4[i4] + doff3] += t1[srcidxs4[i4] + soff3] * w3 * ws4[i4];
									}
								}
							}
						}
					}
				}
				return;
			}
			else {
				const INT64* srcidxs5 = twit->axs[5]->srcidxs;
				const INT64* dstidxs5 = twit->axs[5]->dstidxs;
				const double* ws5 = twit->axs[5]->weights;
				const INT64 L5 = twit->axs[5]->length;
				if (L == 6) {
					if (dbg) twit_log("_apply_twit  6D\n");
					const INT64 srcadvance4 = t1_dims[5];
					const INT64 dstadvance4 = t2_dims[5];
					const INT64 srcadvance3 = t1_dims[4] * srcadvance4;
					const INT64 dstadvance3 = t2_dims[4] * dstadvance4;
					const INT64 srcadvance2 = t1_dims[3] * srcadvance3;
					const INT64 dstadvance2 = t2_dims[3] * dstadvance3;
					const INT64 srcadvance1 = t1_dims[2] * srcadvance2;
					const INT64 dstadvance1 = t2_dims[2] * dstadvance2;
					const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
					const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
					if (preclear) {
						for (INT64 i0 = 0; i0 < L0; i0++) {
							const INT64 doff0 = dstadvance0 * dstidxs0[i0];
							for (INT64 i1 = 0; i1 < L1; i1++) {
								const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
								for (INT64 i2 = 0; i2 < L2; i2++) {
									if ((srcidxs2[i2] & twit_thread_mask) == N) {
										const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
										for (INT64 i3 = 0; i3 < L3; i3++) {
											const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
											for (INT64 i4 = 0; i4 < L4; i4++) {
												const INT64 doff4 = doff3 + dstadvance4 * dstidxs4[i4];
												for (INT64 i5 = 0; i5 < L5; i5++) {
													t2[dstidxs5[i5] + doff4] = 0.0;
												}
											}
										}
									}
								}
							}
						}
					}
					for (INT64 i0 = 0; i0 < L0; i0++) {
						const INT64 soff0 = srcadvance0 * srcidxs0[i0];
						const INT64 doff0 = dstadvance0 * dstidxs0[i0];
						const double w0 = ws0[i0];
						for (INT64 i1 = 0; i1 < L1; i1++) {
							const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
							const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
							const double w1 = ws1[i1] * w0;
							for (INT64 i2 = 0; i2 < L2; i2++) {
								if ((srcidxs2[i2] & twit_thread_mask) == N) {
									const INT64 soff2 = soff1 + srcadvance2 * srcidxs2[i2];
									const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
									const double w2 = ws2[i2] * w1;
									for (INT64 i3 = 0; i3 < L3; i3++) {
										const INT64 soff3 = soff2 + srcadvance3 * srcidxs3[i3];
										const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
										const double w3 = ws3[i3] * w2;
										for (INT64 i4 = 0; i4 < L4; i4++) {
											const INT64 soff4 = soff3 + srcadvance4 * srcidxs4[i4];
											const INT64 doff4 = doff3 + dstadvance4 * dstidxs4[i4];
											const double w4 = ws4[i4] * w3;
											for (INT64 i5 = 0; i5 < L5; i5++) {
												t2[dstidxs5[i5] + doff4] += t1[srcidxs5[i5] + soff4] * w4 * ws5[i5];
											}
										}
									}
								}
							}
						}
					}
					return;
				}
				else {
					// Tsk tsk tsk, unimplemented number of dimensions.
					// They're all custom for each supported dimension count.
					// May implement a slower generic size handler for large numbers of dimensions?
					twit_log("_apply_twit  UNSUPPORTED TWIT Dimensions count.  Max is 6 Dimensions.\n");
					throw 42;
				}
			}
		}
	}
}

void _apply_twit_MT_axis_3(twit_processing_context* ctx, int N) {
	bool dbg = false;
	if (dbg) twit_log("TWIT  _apply_twit_MT_axis_3  ");
	if (ctx->twit == NULL) {
		if (dbg) twit_log("TWIT  _apply_twit_MT_axis_3  NULL twit");
		throw 1;
	}
	// Src
	if (ctx->t1 == NULL) {
		if (dbg) twit_log("TWIT  _apply_twit_MT_axis_3  NULL t1");
		throw 2;
	}
	// Dst
	if (ctx->t2 == NULL) {
		if (dbg) twit_log("TWIT  _apply_twit_MT_axis_3  NULL t2");
		throw 3;
	}

	// Speed up some references.
	twit_multi_axis const* const twit = ctx->twit;
	double const* const t1 = ctx->t1;
	INT64 const* const t1_dims = ctx->t1_dims;
	double* const t2 = ctx->t2;
	INT64 const* const t2_dims = ctx->t2_dims;
	const INT64 preclear = ctx->preclear;


	// Fast constants. This entire method tries to save every cpu cycle possible.
	// Premature optimization is the root of all evil, yada yada yada.
	const INT64 L = twit->length;
	if (dbg) twit_log("L = %lld\n", L);

	if (L <= 3) throw 4;

	const INT64 L0 = twit->axs[0]->length;
	// These three are the source indicies, dset indicies, and weight triples along a given axis.
	// Generated by compute_twit_single_dimension()
	const INT64* srcidxs0 = twit->axs[0]->srcidxs;
	const INT64* dstidxs0 = twit->axs[0]->dstidxs;
	const double* ws0 = twit->axs[0]->weights;

	const INT64* srcidxs1 = twit->axs[1]->srcidxs;
	const INT64* dstidxs1 = twit->axs[1]->dstidxs;
	const double* ws1 = twit->axs[1]->weights;
	const INT64 L1 = twit->axs[1]->length;

	const INT64* srcidxs2 = twit->axs[2]->srcidxs;
	const INT64* dstidxs2 = twit->axs[2]->dstidxs;
	const double* ws2 = twit->axs[2]->weights;
	const INT64 L2 = twit->axs[2]->length;

	const INT64* srcidxs3 = twit->axs[3]->srcidxs;
	const INT64* dstidxs3 = twit->axs[3]->dstidxs;
	const double* ws3 = twit->axs[3]->weights;
	const INT64 L3 = twit->axs[3]->length;
	if (L == 4) {
		if (dbg) twit_log("_apply_twit  4D\n");
		const INT64 srcadvance2 = t1_dims[3];
		const INT64 dstadvance2 = t2_dims[3];
		const INT64 srcadvance1 = t1_dims[2] * srcadvance2;
		const INT64 dstadvance1 = t2_dims[2] * dstadvance2;
		const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
		const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
		if (preclear) {
			for (INT64 i0 = 0; i0 < L0; i0++) {
				const INT64 doff0 = dstadvance0 * dstidxs0[i0];
				for (INT64 i1 = 0; i1 < L1; i1++) {
					const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
					for (INT64 i2 = 0; i2 < L2; i2++) {
						const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
						for (INT64 i3 = 0; i3 < L3; i3++) {
							if ((srcidxs3[i3] & twit_thread_mask) == N) {
								t2[dstidxs3[i3] + doff2] = 0.0;
							}
						}
					}
				}
			}
		}
		for (INT64 i0 = 0; i0 < L0; i0++) {
			const INT64 soff0 = srcadvance0 * srcidxs0[i0];
			const INT64 doff0 = dstadvance0 * dstidxs0[i0];
			const double w0 = ws0[i0];
			for (INT64 i1 = 0; i1 < L1; i1++) {
				const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
				const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
				const double w1 = ws1[i1] * w0;
				for (INT64 i2 = 0; i2 < L2; i2++) {
					const INT64 soff2 = soff1 + srcadvance2 * srcidxs2[i2];
					const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
					const double w2 = ws2[i2] * w1;
					for (INT64 i3 = 0; i3 < L3; i3++) {
						if ((srcidxs3[i3] & twit_thread_mask) == N) {
							t2[dstidxs3[i3] + doff2] += t1[srcidxs3[i3] + soff2] * w2 * ws3[i3];
						}
					}
				}
			}
		}
		return;
	}
	else {
		const INT64* srcidxs4 = twit->axs[4]->srcidxs;
		const INT64* dstidxs4 = twit->axs[4]->dstidxs;
		const double* ws4 = twit->axs[4]->weights;
		const INT64 L4 = twit->axs[4]->length;
		if (L == 5) {
			if (dbg) twit_log("_apply_twit  5D\n");
			const INT64 srcadvance3 = t1_dims[4];
			const INT64 dstadvance3 = t2_dims[4];
			const INT64 srcadvance2 = t1_dims[3] * srcadvance3;
			const INT64 dstadvance2 = t2_dims[3] * dstadvance3;
			const INT64 srcadvance1 = t1_dims[2] * srcadvance2;
			const INT64 dstadvance1 = t2_dims[2] * dstadvance2;
			const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
			const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
			if (preclear) {
				for (INT64 i0 = 0; i0 < L0; i0++) {
					const INT64 doff0 = dstadvance0 * dstidxs0[i0];
					for (INT64 i1 = 0; i1 < L1; i1++) {
						const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
						for (INT64 i2 = 0; i2 < L2; i2++) {
							const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
							for (INT64 i3 = 0; i3 < L3; i3++) {
								if ((srcidxs3[i3] & twit_thread_mask) == N) {
									const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
									for (INT64 i4 = 0; i4 < L4; i4++) {
										t2[dstidxs4[i4] + doff3] = 0.0;
									}
								}
							}
						}
					}
				}
			}
			for (INT64 i0 = 0; i0 < L0; i0++) {
				const INT64 soff0 = srcadvance0 * srcidxs0[i0];
				const INT64 doff0 = dstadvance0 * dstidxs0[i0];
				const double w0 = ws0[i0];
				for (INT64 i1 = 0; i1 < L1; i1++) {
					const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
					const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
					const double w1 = ws1[i1] * w0;
					for (INT64 i2 = 0; i2 < L2; i2++) {
						const INT64 soff2 = soff1 + srcadvance2 * srcidxs2[i2];
						const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
						const double w2 = ws2[i2] * w1;
						for (INT64 i3 = 0; i3 < L3; i3++) {
							if ((srcidxs3[i3] & twit_thread_mask) == N) {
								const INT64 soff3 = soff2 + srcadvance3 * srcidxs3[i3];
								const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
								const double w3 = ws3[i3] * w2;
								for (INT64 i4 = 0; i4 < L4; i4++) {
									t2[dstidxs4[i4] + doff3] += t1[srcidxs4[i4] + soff3] * w3 * ws4[i4];
								}
							}
						}
					}
				}
			}
			return;
		}
		else {
			const INT64* srcidxs5 = twit->axs[5]->srcidxs;
			const INT64* dstidxs5 = twit->axs[5]->dstidxs;
			const double* ws5 = twit->axs[5]->weights;
			const INT64 L5 = twit->axs[5]->length;
			if (L == 6) {
				if (dbg) twit_log("_apply_twit  6D\n");
				const INT64 srcadvance4 = t1_dims[5];
				const INT64 dstadvance4 = t2_dims[5];
				const INT64 srcadvance3 = t1_dims[4] * srcadvance4;
				const INT64 dstadvance3 = t2_dims[4] * dstadvance4;
				const INT64 srcadvance2 = t1_dims[3] * srcadvance3;
				const INT64 dstadvance2 = t2_dims[3] * dstadvance3;
				const INT64 srcadvance1 = t1_dims[2] * srcadvance2;
				const INT64 dstadvance1 = t2_dims[2] * dstadvance2;
				const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
				const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
				if (preclear) {
					for (INT64 i0 = 0; i0 < L0; i0++) {
						const INT64 doff0 = dstadvance0 * dstidxs0[i0];
						for (INT64 i1 = 0; i1 < L1; i1++) {
							const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
							for (INT64 i2 = 0; i2 < L2; i2++) {
								const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
								for (INT64 i3 = 0; i3 < L3; i3++) {
									if ((srcidxs3[i3] & twit_thread_mask) == N) {
										const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
										for (INT64 i4 = 0; i4 < L4; i4++) {
											const INT64 doff4 = doff3 + dstadvance4 * dstidxs4[i4];
											for (INT64 i5 = 0; i5 < L5; i5++) {
												t2[dstidxs5[i5] + doff4] = 0.0;
											}
										}
									}
								}
							}
						}
					}
				}
				for (INT64 i0 = 0; i0 < L0; i0++) {
					const INT64 soff0 = srcadvance0 * srcidxs0[i0];
					const INT64 doff0 = dstadvance0 * dstidxs0[i0];
					const double w0 = ws0[i0];
					for (INT64 i1 = 0; i1 < L1; i1++) {
						const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
						const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
						const double w1 = ws1[i1] * w0;
						for (INT64 i2 = 0; i2 < L2; i2++) {
							const INT64 soff2 = soff1 + srcadvance2 * srcidxs2[i2];
							const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
							const double w2 = ws2[i2] * w1;
							for (INT64 i3 = 0; i3 < L3; i3++) {
								if ((srcidxs3[i3] & twit_thread_mask) == N) {
									const INT64 soff3 = soff2 + srcadvance3 * srcidxs3[i3];
									const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
									const double w3 = ws3[i3] * w2;
									for (INT64 i4 = 0; i4 < L4; i4++) {
										const INT64 soff4 = soff3 + srcadvance4 * srcidxs4[i4];
										const INT64 doff4 = doff3 + dstadvance4 * dstidxs4[i4];
										const double w4 = ws4[i4] * w3;
										for (INT64 i5 = 0; i5 < L5; i5++) {
											t2[dstidxs5[i5] + doff4] += t1[srcidxs5[i5] + soff4] * w4 * ws5[i5];
										}
									}
								}
							}
						}
					}
				}
				return;
			}
			else {
				// Tsk tsk tsk, unimplemented number of dimensions.
				// They're all custom for each supported dimension count.
				// May implement a slower generic size handler for large numbers of dimensions?
				twit_log("_apply_twit  UNSUPPORTED TWIT Dimensions count.  Max is 6 Dimensions.\n");
				throw 42;
			}
		}
	}
}

void _apply_twit_MT_axis_4(twit_processing_context* ctx, int N) {
	bool dbg = false;
	if (dbg) twit_log("TWIT  _apply_twit_MT_axis_4  ");
	if (ctx->twit == NULL) {
		if (dbg) twit_log("TWIT  _apply_twit_MT_axis_4  NULL twit");
		throw 1;
	}
	// Src
	if (ctx->t1 == NULL) {
		if (dbg) twit_log("TWIT  _apply_twit_MT_axis_4  NULL t1");
		throw 2;
	}
	// Dst
	if (ctx->t2 == NULL) {
		if (dbg) twit_log("TWIT  _apply_twit_MT_axis_4  NULL t2");
		throw 3;
	}

	// Speed up some references.
	twit_multi_axis const* const twit = ctx->twit;
	double const* const t1 = ctx->t1;
	INT64 const* const t1_dims = ctx->t1_dims;
	double* const t2 = ctx->t2;
	INT64 const* const t2_dims = ctx->t2_dims;
	const INT64 preclear = ctx->preclear;

	// Fast constants. This entire method tries to save every cpu cycle possible.
	// Premature optimization is the root of all evil, yada yada yada.
	const INT64 L = twit->length;
	if (dbg) twit_log("L = %lld\n", L);

	if (L <= 4) throw 4;

	const INT64 L0 = twit->axs[0]->length;
	// These three are the source indicies, dset indicies, and weight triples along a given axis.
	// Generated by compute_twit_single_dimension()
	const INT64* srcidxs0 = twit->axs[0]->srcidxs;
	const INT64* dstidxs0 = twit->axs[0]->dstidxs;
	const double* ws0 = twit->axs[0]->weights;


	const INT64* srcidxs1 = twit->axs[1]->srcidxs;
	const INT64* dstidxs1 = twit->axs[1]->dstidxs;
	const double* ws1 = twit->axs[1]->weights;
	const INT64 L1 = twit->axs[1]->length;

	const INT64* srcidxs2 = twit->axs[2]->srcidxs;
	const INT64* dstidxs2 = twit->axs[2]->dstidxs;
	const double* ws2 = twit->axs[2]->weights;
	const INT64 L2 = twit->axs[2]->length;

	const INT64* srcidxs3 = twit->axs[3]->srcidxs;
	const INT64* dstidxs3 = twit->axs[3]->dstidxs;
	const double* ws3 = twit->axs[3]->weights;
	const INT64 L3 = twit->axs[3]->length;

	const INT64* srcidxs4 = twit->axs[4]->srcidxs;
	const INT64* dstidxs4 = twit->axs[4]->dstidxs;
	const double* ws4 = twit->axs[4]->weights;
	const INT64 L4 = twit->axs[4]->length;
	if (L == 5) {
		if (dbg) twit_log("_apply_twit  5D\n");
		const INT64 srcadvance3 = t1_dims[4];
		const INT64 dstadvance3 = t2_dims[4];
		const INT64 srcadvance2 = t1_dims[3] * srcadvance3;
		const INT64 dstadvance2 = t2_dims[3] * dstadvance3;
		const INT64 srcadvance1 = t1_dims[2] * srcadvance2;
		const INT64 dstadvance1 = t2_dims[2] * dstadvance2;
		const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
		const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
		if (preclear) {
			for (INT64 i0 = 0; i0 < L0; i0++) {
				const INT64 doff0 = dstadvance0 * dstidxs0[i0];
				for (INT64 i1 = 0; i1 < L1; i1++) {
					const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
					for (INT64 i2 = 0; i2 < L2; i2++) {
						const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
						for (INT64 i3 = 0; i3 < L3; i3++) {
							const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
							for (INT64 i4 = 0; i4 < L4; i4++) {
								if ((srcidxs4[i4] & twit_thread_mask) == N) {
									t2[dstidxs4[i4] + doff3] = 0.0;
								}
							}
						}
					}
				}
			}
		}
		for (INT64 i0 = 0; i0 < L0; i0++) {
			const INT64 soff0 = srcadvance0 * srcidxs0[i0];
			const INT64 doff0 = dstadvance0 * dstidxs0[i0];
			const double w0 = ws0[i0];
			for (INT64 i1 = 0; i1 < L1; i1++) {
				const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
				const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
				const double w1 = ws1[i1] * w0;
				for (INT64 i2 = 0; i2 < L2; i2++) {
					const INT64 soff2 = soff1 + srcadvance2 * srcidxs2[i2];
					const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
					const double w2 = ws2[i2] * w1;
					for (INT64 i3 = 0; i3 < L3; i3++) {
						const INT64 soff3 = soff2 + srcadvance3 * srcidxs3[i3];
						const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
						const double w3 = ws3[i3] * w2;
						for (INT64 i4 = 0; i4 < L4; i4++) {
							if ((srcidxs4[i4] & twit_thread_mask) == N) {
								t2[dstidxs4[i4] + doff3] += t1[srcidxs4[i4] + soff3] * w3 * ws4[i4];
							}
						}
					}
				}
			}
		}
		return;
	}
	else {
		const INT64* srcidxs5 = twit->axs[5]->srcidxs;
		const INT64* dstidxs5 = twit->axs[5]->dstidxs;
		const double* ws5 = twit->axs[5]->weights;
		const INT64 L5 = twit->axs[5]->length;
		if (L == 6) {
			if (dbg) twit_log("_apply_twit  6D\n");
			const INT64 srcadvance4 = t1_dims[5];
			const INT64 dstadvance4 = t2_dims[5];
			const INT64 srcadvance3 = t1_dims[4] * srcadvance4;
			const INT64 dstadvance3 = t2_dims[4] * dstadvance4;
			const INT64 srcadvance2 = t1_dims[3] * srcadvance3;
			const INT64 dstadvance2 = t2_dims[3] * dstadvance3;
			const INT64 srcadvance1 = t1_dims[2] * srcadvance2;
			const INT64 dstadvance1 = t2_dims[2] * dstadvance2;
			const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
			const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
			if (preclear) {
				for (INT64 i0 = 0; i0 < L0; i0++) {
					const INT64 doff0 = dstadvance0 * dstidxs0[i0];
					for (INT64 i1 = 0; i1 < L1; i1++) {
						const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
						for (INT64 i2 = 0; i2 < L2; i2++) {
							const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
							for (INT64 i3 = 0; i3 < L3; i3++) {
								const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
								for (INT64 i4 = 0; i4 < L4; i4++) {
									if ((srcidxs4[i4] & twit_thread_mask) == N) {
										const INT64 doff4 = doff3 + dstadvance4 * dstidxs4[i4];
										for (INT64 i5 = 0; i5 < L5; i5++) {
											t2[dstidxs5[i5] + doff4] = 0.0;
										}
									}
								}
							}
						}
					}
				}
			}
			for (INT64 i0 = 0; i0 < L0; i0++) {
				const INT64 soff0 = srcadvance0 * srcidxs0[i0];
				const INT64 doff0 = dstadvance0 * dstidxs0[i0];
				const double w0 = ws0[i0];
				for (INT64 i1 = 0; i1 < L1; i1++) {
					const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
					const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
					const double w1 = ws1[i1] * w0;
					for (INT64 i2 = 0; i2 < L2; i2++) {
						const INT64 soff2 = soff1 + srcadvance2 * srcidxs2[i2];
						const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
						const double w2 = ws2[i2] * w1;
						for (INT64 i3 = 0; i3 < L3; i3++) {
							const INT64 soff3 = soff2 + srcadvance3 * srcidxs3[i3];
							const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
							const double w3 = ws3[i3] * w2;
							for (INT64 i4 = 0; i4 < L4; i4++) {
								if ((srcidxs4[i4] & twit_thread_mask) == N) {
									const INT64 soff4 = soff3 + srcadvance4 * srcidxs4[i4];
									const INT64 doff4 = doff3 + dstadvance4 * dstidxs4[i4];
									const double w4 = ws4[i4] * w3;
									for (INT64 i5 = 0; i5 < L5; i5++) {
										t2[dstidxs5[i5] + doff4] += t1[srcidxs5[i5] + soff4] * w4 * ws5[i5];
									}
								}
							}
						}
					}
				}
			}
			return;
		}
		else {
			// Tsk tsk tsk, unimplemented number of dimensions.
			// They're all custom for each supported dimension count.
			// May implement a slower generic size handler for large numbers of dimensions?
			twit_log("_apply_twit  UNSUPPORTED TWIT Dimensions count.  Max is 6 Dimensions.\n");
			throw 42;
		}
	}
}

void _apply_twit_MT_axis_5(twit_processing_context* ctx, int N) {
	bool dbg = false;
	if (dbg) twit_log("TWIT  _apply_twit_MT_axis_5  ");
	if (ctx->twit == NULL) {
		if (dbg) twit_log("TWIT  _apply_twit_MT_axis_5  NULL twit");
		throw 1;
	}
	// Src
	if (ctx->t1 == NULL) {
		if (dbg) twit_log("TWIT  _apply_twit_MT_axis_5  NULL t1");
		throw 2;
	}
	// Dst
	if (ctx->t2 == NULL) {
		if (dbg) twit_log("TWIT  _apply_twit_MT_axis_5  NULL t2");
		throw 3;
	}

	// Speed up some references.
	twit_multi_axis const* const twit = ctx->twit;
	double const* const t1 = ctx->t1;
	INT64 const* const t1_dims = ctx->t1_dims;
	double* const t2 = ctx->t2;
	INT64 const* const t2_dims = ctx->t2_dims;
	const INT64 preclear = ctx->preclear;


	// Fast constants. This entire method tries to save every cpu cycle possible.
	// Premature optimization is the root of all evil, yada yada yada.
	const INT64 L = twit->length;
	if (dbg) twit_log("L = %lld\n", L);

	if (L <= 5) throw 4;

	const INT64 L0 = twit->axs[0]->length;
	// These three are the source indicies, dset indicies, and weight triples along a given axis.
	// Generated by compute_twit_single_dimension()
	const INT64* srcidxs0 = twit->axs[0]->srcidxs;
	const INT64* dstidxs0 = twit->axs[0]->dstidxs;
	const double* ws0 = twit->axs[0]->weights;


	const INT64* srcidxs1 = twit->axs[1]->srcidxs;
	const INT64* dstidxs1 = twit->axs[1]->dstidxs;
	const double* ws1 = twit->axs[1]->weights;
	const INT64 L1 = twit->axs[1]->length;


	const INT64* srcidxs2 = twit->axs[2]->srcidxs;
	const INT64* dstidxs2 = twit->axs[2]->dstidxs;
	const double* ws2 = twit->axs[2]->weights;
	const INT64 L2 = twit->axs[2]->length;


	const INT64* srcidxs3 = twit->axs[3]->srcidxs;
	const INT64* dstidxs3 = twit->axs[3]->dstidxs;
	const double* ws3 = twit->axs[3]->weights;
	const INT64 L3 = twit->axs[3]->length;

	const INT64* srcidxs4 = twit->axs[4]->srcidxs;
	const INT64* dstidxs4 = twit->axs[4]->dstidxs;
	const double* ws4 = twit->axs[4]->weights;
	const INT64 L4 = twit->axs[4]->length;

	const INT64* srcidxs5 = twit->axs[5]->srcidxs;
	const INT64* dstidxs5 = twit->axs[5]->dstidxs;
	const double* ws5 = twit->axs[5]->weights;
	const INT64 L5 = twit->axs[5]->length;
	if (L == 6) {
		if (dbg) twit_log("_apply_twit  6D\n");
		const INT64 srcadvance4 = t1_dims[5];
		const INT64 dstadvance4 = t2_dims[5];
		const INT64 srcadvance3 = t1_dims[4] * srcadvance4;
		const INT64 dstadvance3 = t2_dims[4] * dstadvance4;
		const INT64 srcadvance2 = t1_dims[3] * srcadvance3;
		const INT64 dstadvance2 = t2_dims[3] * dstadvance3;
		const INT64 srcadvance1 = t1_dims[2] * srcadvance2;
		const INT64 dstadvance1 = t2_dims[2] * dstadvance2;
		const INT64 srcadvance0 = t1_dims[1] * srcadvance1;
		const INT64 dstadvance0 = t2_dims[1] * dstadvance1;
		if (preclear) {
			for (INT64 i0 = 0; i0 < L0; i0++) {
				const INT64 doff0 = dstadvance0 * dstidxs0[i0];
				for (INT64 i1 = 0; i1 < L1; i1++) {
					const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
					for (INT64 i2 = 0; i2 < L2; i2++) {
						const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
						for (INT64 i3 = 0; i3 < L3; i3++) {
							const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
							for (INT64 i4 = 0; i4 < L4; i4++) {
								const INT64 doff4 = doff3 + dstadvance4 * dstidxs4[i4];
								for (INT64 i5 = 0; i5 < L5; i5++) {
									if ((srcidxs5[i5] & twit_thread_mask) == N) {
										t2[dstidxs5[i5] + doff4] = 0.0;
									}
								}
							}
						}
					}
				}
			}
		}
		for (INT64 i0 = 0; i0 < L0; i0++) {
			const INT64 soff0 = srcadvance0 * srcidxs0[i0];
			const INT64 doff0 = dstadvance0 * dstidxs0[i0];
			const double w0 = ws0[i0];
			for (INT64 i1 = 0; i1 < L1; i1++) {
				const INT64 soff1 = soff0 + srcadvance1 * srcidxs1[i1];
				const INT64 doff1 = doff0 + dstadvance1 * dstidxs1[i1];
				const double w1 = ws1[i1] * w0;
				for (INT64 i2 = 0; i2 < L2; i2++) {
					const INT64 soff2 = soff1 + srcadvance2 * srcidxs2[i2];
					const INT64 doff2 = doff1 + dstadvance2 * dstidxs2[i2];
					const double w2 = ws2[i2] * w1;
					for (INT64 i3 = 0; i3 < L3; i3++) {
						const INT64 soff3 = soff2 + srcadvance3 * srcidxs3[i3];
						const INT64 doff3 = doff2 + dstadvance3 * dstidxs3[i3];
						const double w3 = ws3[i3] * w2;
						for (INT64 i4 = 0; i4 < L4; i4++) {
							const INT64 soff4 = soff3 + srcadvance4 * srcidxs4[i4];
							const INT64 doff4 = doff3 + dstadvance4 * dstidxs4[i4];
							const double w4 = ws4[i4] * w3;
							for (INT64 i5 = 0; i5 < L5; i5++) {
								if ((srcidxs5[i5] & twit_thread_mask) == N) {
									t2[dstidxs5[i5] + doff4] += t1[srcidxs5[i5] + soff4] * w4 * ws5[i5];
								}
							}
						}
					}
				}
			}
		}
		return;
	}
	else {
		// Tsk tsk tsk, unimplemented number of dimensions.
		// They're all custom for each supported dimension count.
		// May implement a slower generic size handler for large numbers of dimensions?
		twit_log("_apply_twit  UNSUPPORTED TWIT Dimensions count.  Max is 6 Dimensions.\n");
		throw 42;
	}
}

void twit_multi_axis_destructor(PyObject* obj) {
	//twit_log("twit_multi_axis_destructor\n");
	twit_multi_axis* ptr = (twit_multi_axis*)PyCapsule_GetPointer(obj, "twit_multi_axis");
	free_twit_multi_axis(ptr);
}

PyObject* compute_twit_multi_dimension_impl(PyObject*, PyObject* args) {
	//twit_log("TWITC compute_twit_multi_dimension\n");
	PyErr_Clear();
	// How many dimentsion t1 and t2 will have.
	INT64 n_dims;
	// src_start_i, src_end_i, dst_start_i. dst_end_i, ..... in quads.
	PyArrayObject* int_array;
	INT64* twit_i;
	// start_w, end_w, .... in pairs.
	PyArrayObject* double_array;
	double* twit_w;

	//twit_log("TWITC parse args...\n");
	PyArg_ParseTuple(args, "LO!O!", &n_dims, &PyArray_Type, &int_array, &PyArray_Type, &double_array);
	//twit_log("n_dims %lld\n", n_dims);

	twit_i = (INT64*)PyArray_DATA(int_array);
	twit_w = (double*)PyArray_DATA(double_array);

	//twit_log("Input params, twit_i:\n");
	//for (INT64 i = 0; i < n_dims; i++) {
	//	twit_log("  %lld %lld,  %lld %lld,  %lf %lf\n", twit_i[i * 4 + 0], twit_i[i * 4 + 1], twit_i[i * 4 + 2], twit_i[i * 4 + 3], twit_w[i * 2 + 0], twit_w[i * 2 + 1]);
	//}
	twit_multi_axis* twit = _compute_twit_multi_dimension(n_dims, twit_i, twit_w);

	//twit_log("Result is: %lld\n", ptr->length);
	//for (INT64 i = 0; i < ptr->length; i++) {
	//	twit_log("  ax %lld\n", i);
	//	twit_single_axis* ax = ptr->axs[i];
	//	for (INT64 k = 0; k < ax->length; k++) {
	//		twit_log("     src %lld   dst %lld   w %lf\n", ax->srcidxs[k], ax->dstidxs[k], ax->weights[k]);
	//	}
	//}

	PyObject* rslt = PyCapsule_New(twit, "twit_multi_axis", twit_multi_axis_destructor);
	return rslt;
}

PyObject* unpack_twit_multi_axis_impl(PyObject*, PyObject* args) {
	//twit_log("unpack_twit_multi_axis\n");
	PyObject* capobj;
	PyArg_ParseTuple(args, "O!", &PyCapsule_Type, &capobj);

	twit_multi_axis* twit = (twit_multi_axis*)PyCapsule_GetPointer(capobj, "twit_multi_axis");
	//twit_log("unpack_twit_multi_axis: ptr is 0x%p  length is %lld\n", twit, twit->length);

	PyObject* rslt = PyTuple_New(twit->length + 1);
	PyTuple_SetItem(rslt, 0, PyLong_FromLongLong(twit->length));
	for (INT64 i = 0; i < twit->length; i++) {
		twit_single_axis* ax = twit->axs[i];
		//twit_log("ax length is %lld\n", ax->length);
		PyObject* axres = PyTuple_New(4);
		PyTuple_SetItem(axres, 0, PyLong_FromLongLong(ax->length));

		npy_intp pp[] = { 0 };
		npy_intp dims[1];
		dims[0] = ax->length;
		PyObject* srcidxs = PyArray_SimpleNew(1, dims, NPY_INT64);
		INT64* sip = (INT64*)PyArray_GetPtr((PyArrayObject*)srcidxs, pp);
		for (int k = 0; k < ax->length; k++) {
			sip[k] = ax->srcidxs[k];
		}
		PyTuple_SetItem(axres, 1, srcidxs);

		PyObject* dstidxs = PyArray_SimpleNew(1, dims, NPY_INT64);
		INT64* dip = (INT64*)PyArray_GetPtr((PyArrayObject*)dstidxs, pp);
		for (int k = 0; k < ax->length; k++) {
			dip[k] = ax->dstidxs[k];
		}
		PyTuple_SetItem(axres, 2, dstidxs);

		PyObject* weights = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
		double* wp = (double*)PyArray_GetPtr((PyArrayObject*)weights, pp);
		for (int k = 0; k < ax->length; k++) {
			wp[k] = ax->weights[k];
		}
		PyTuple_SetItem(axres, 3, weights);

		PyTuple_SetItem(rslt, i + 1, axres);
	}

	return rslt;
}

PyObject* pack_twit_multi_axis_impl(PyObject*, PyObject* args) {
	PyErr_Clear();

	PyObject* toptuple;
	PyArg_ParseTuple(args, "O!", &PyTuple_Type, &toptuple);

	twit_multi_axis* twit = (twit_multi_axis*)PyMem_Malloc(sizeof(twit_multi_axis));
	twit->length = PyLong_AsLongLong(PyTuple_GetItem(toptuple, 0));
	twit->axs = (twit_single_axis**)PyMem_Malloc(twit->length * sizeof(twit_single_axis*));

	for (Py_ssize_t i = 0; i < twit->length; i++) {
		PyObject* subtup = PyTuple_GetItem(toptuple, i + 1);
		twit_single_axis* ax = (twit_single_axis*)PyMem_Malloc(sizeof(twit_single_axis));
		const INT64 L = PyLong_AsLongLong(PyTuple_GetItem(subtup, 0));
		twit->axs[i] = ax;
		ax->length = L;
		ax->srcidxs = (INT64*)PyMem_Malloc(L * sizeof(INT64));
		ax->dstidxs = (INT64*)PyMem_Malloc(L * sizeof(INT64));
		ax->weights = (double*)PyMem_Malloc(L * sizeof(double));
		PyObject* srcarray = PyTuple_GetItem(subtup, 1);
		PyObject* dstarray = PyTuple_GetItem(subtup, 2);
		PyObject* warray = PyTuple_GetItem(subtup, 3);
		for (INT64 k = 0; k < L; k++) {
			ax->srcidxs[k] = *(INT64*)PyArray_GETPTR1((PyArrayObject*)srcarray, k);
			ax->dstidxs[k] = *(INT64*)PyArray_GETPTR1((PyArrayObject*)dstarray, k);
			ax->weights[k] = *(double*)PyArray_GETPTR1((PyArrayObject*)warray, k);
		}
	}

	PyObject* rslt = PyCapsule_New(twit, "twit_multi_axis", twit_multi_axis_destructor);
	return rslt;
}

PyObject* apply_twit_impl(PyObject*, PyObject* args) {
	//twit_log("TWITC apply_twit_impl\n");
	PyObject* capobj;
	PyObject* srcarrayobj;
	PyObject* dstarrayobj;
	INT64 preclear;
	PyArg_ParseTuple(args, "O!O!O!L", &PyCapsule_Type, &capobj, &PyArray_Type, &srcarrayobj, &PyArray_Type, &dstarrayobj, &preclear);

	twit_multi_axis* twit = (twit_multi_axis*)PyCapsule_GetPointer(capobj, "twit_multi_axis");
	//twit_log("apply_twit_impl: ptr is 0x%p  length is %lld\n", twit, twit->length);

	// So far we assume the src and dst arrays are packed and contiguous.  TODO - Add check for that and force to C array.

	_apply_twit(twit, (double*)PyArray_DATA(srcarrayobj), PyArray_SHAPE((PyArrayObject*)srcarrayobj), (double*)PyArray_DATA(dstarrayobj), PyArray_SHAPE((PyArrayObject*)dstarrayobj), preclear);

	Py_IncRef(Py_True);
	return Py_True;
}

/// Arguments ar
/// make_and_apply_twit(N_Dims, twit_integers_array, twit_double_array, src_ndarray, dst_ndarray, preclear)
PyObject* make_and_apply_twit_impl(PyObject*, PyObject* args) {
	//twit_log("TWITC make_and_apply_twit_impl\n");
	PyErr_Clear();
	// How many dimentsion t1 and t2 will have.
	INT64 n_dims;
	INT64 preclear;
	// src_start_i, src_end_i, dst_start_i. dst_end_i, ..... in quads.
	PyArrayObject* int_array;
	INT64* twit_i;
	// start_w, end_w, .... in pairs.
	PyArrayObject* double_array;
	double* twit_w;
	PyObject* srcarrayobj;
	PyObject* dstarrayobj;
	double* src;
	double* dst;


	//twit_log("TWITC parse args...\n");
	PyArg_ParseTuple(args, "LO!O!O!O!L", &n_dims, &PyArray_Type, &int_array, &PyArray_Type, &double_array, &PyArray_Type, &srcarrayobj, &PyArray_Type, &dstarrayobj, &preclear);
	//twit_log("n_dims %lld  preclear=%lld\n", n_dims, preclear);

	twit_i = (INT64*)PyArray_DATA(int_array);
	twit_w = (double*)PyArray_DATA(double_array);
	src = (double*)PyArray_DATA(srcarrayobj);
	dst = (double*)PyArray_DATA(dstarrayobj);

	twit_multi_axis* twit = _compute_twit_multi_dimension(n_dims, twit_i, twit_w);
	//twit_log("make_and_apply_twit_impl: ptr is 0x%p  length is %lld\n", twit, twit->length);

	// So far we assume the src and dst arrays are packed and contiguous.  TODO - Add check for that and force to C array.

	_apply_twit(twit, src, PyArray_SHAPE((PyArrayObject*)srcarrayobj), dst, PyArray_SHAPE((PyArrayObject*)dstarrayobj), preclear);

	Py_IncRef(Py_True);
	return Py_True;

}

void twit_apply_MT_sequencer(twit_processing_context* ctx) {
	bool dbg = false;
	if (dbg) twit_log("twit_apply_MT_sequencer: Locking.\n");
	std::lock_guard<std::recursive_mutex> guard(twit_mt_mutex);
	if (dbg) twit_log("twit_apply_MT_sequencer: Locked ok. Copy ctx.\n");
	copy_ctx_to_threads(ctx);
	if (dbg) twit_log("twit_apply_MT_sequencer: Kickoff all.\n");
	twit_thread_kickoff_all_loops();
	if (dbg) twit_log("twit_apply_MT_sequencer: Wait all.\n");
	twit_thread_wait_for_all_loops_to_finish();
	if (dbg) twit_log("twit_apply_MT_sequencer: Done.\n");
}

void twit_apply_MT(int N) {
	bool dbg = false;
	if (dbg) twit_log("Thread %d start\n", N);

	// How did we get here if it is single threadded?????
	assert(twit_thread_count > 1);
	assert(N >= 0);
	assert(N < twit_thread_count);

	twit_thread_bundle* tb = twit_thread_bundles[N];
	twit_processing_context* ctx = &tb->ctx;
	assert(ctx->t1 != NULL);

	switch (twit_threading_axis) {
	case 0:
		_apply_twit_MT_axis_0(ctx, N);
		break;
	case 1:
		_apply_twit_MT_axis_1(ctx, N);
		break;
	case 2:
		_apply_twit_MT_axis_2(ctx, N);
		break;
	case 3:
		_apply_twit_MT_axis_3(ctx, N);
		break;
	case 4:
		_apply_twit_MT_axis_4(ctx, N);
		break;
	case 5:
		_apply_twit_MT_axis_5(ctx, N);
		break;
	default:
		twit_log("Invalid threadding axis.\n");
		throw 201;
	}

	if (dbg) twit_log("Thread %d end\n", N);
}

// This loop is what each thread run in for MT processing if enabled.
// It waits for the mutex, then processes it's interleaved destination rows of data.
// N is the thread index in mtxs and threads arrays of pointers.
// Used by _twit_apply_MT
void twit_thread_loop(int N) {
	// The outer control loop in the main thread will set twit_thread_mutex_counts[N] to 1
	// and then notify_one()
	// If the mutex is destroyed then exit multithreading.
	while (twit_thread_bundles[N]->mtx != NULL) {
		// Wait for twit_thread_mutex_counts[N] to increment to 1
		{
			std::unique_lock<std::mutex> lock(*(twit_thread_bundles[N]->mtx));
			while (twit_thread_bundles[N]->mutex_count == 0) {
				twit_thread_bundles[N]->mutex_cvar->wait(lock);
			}
		}
		// Apply twit and set twit_thread_mutex_counts[N] back to 0
		{
			twit_apply_MT(N);
			// Signal to main thread I'm done processing.
			std::unique_lock<std::mutex> lock(*(twit_thread_bundles[N]->mtx));
			twit_thread_bundles[N]->mutex_count = 0;
			twit_thread_bundles[N]->mutex_cvar->notify_one();
		}
	}
}

void twit_thread_kickoff_all_loops() {
	bool dbg = false;
	for (int i = 0; i < twit_thread_count; i++) {
		{
			if (dbg) twit_log("twit_thread_kickoff_all_loops: wait lock %d\n", i);
			std::unique_lock<std::mutex> lock(*(twit_thread_bundles[i]->mtx));
			twit_thread_bundles[i]->mutex_count = 1;
			if (dbg) twit_log("twit_thread_kickoff_all_loops: notify_one, %d\n", i);
			twit_thread_bundles[i]->mutex_cvar->notify_one();
		}
	}
}

void twit_thread_wait_for_all_loops_to_finish() {
	bool dbg = false;
	for (int i = 0; i < twit_thread_count; i++) {
		{
			if (dbg) twit_log("twit_thread_wait_for_all_loops_to_finish: wait lock %d\n", i);
			std::unique_lock<std::mutex> lock(*(twit_thread_bundles[i]->mtx));
			while (twit_thread_bundles[i]->mutex_count == 1) {
				// While waiting the function wait automatically releases the lock
				// and re-acquires it when signaled to via notify_one().
				twit_thread_bundles[i]->mutex_cvar->wait(lock);
				if (dbg) twit_log("twit_thread_kickoff_all_loops: wait done, %d\n", i);
			}
		}
	}
}