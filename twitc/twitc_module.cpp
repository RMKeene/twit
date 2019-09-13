#include <Python.h>
#include <Windows.h>
#include <cmath>

const double e = 2.7182818284590452353602874713527;

double sinh_impl(double x) {
	return (1 - pow(e, (-2 * x))) / (2 * pow(e, -x));
}

double cosh_impl(double x) {
	return (1 + pow(e, (-2 * x))) / (2 * pow(e, -x));
}

PyObject* generate_twit_list_impl(PyObject*, PyObject* o) {
	printf("TWITC Debug 1\n");
	double x = PyFloat_AsDouble(o);
	printf("TWITC Debug 2 passed in %f\n", x);
	PyObject * ret = PyFloat_FromDouble(42.0);
	printf("TWITC Debug 4\n");
	return (ret);
}

static PyMethodDef twitc_methods[] = {

	{ "generate_twit_list", (PyCFunction)generate_twit_list_impl, METH_O, "Generate the twit list of inter tensor links as a cache." },
	{ nullptr, nullptr, 0, nullptr }
};

static PyModuleDef twitc_module = {
	PyModuleDef_HEAD_INIT,
	"twitc",
	"Provides some functions, but faster",
	0,
	twitc_methods
};

PyMODINIT_FUNC PyInit_twitc() {
	printf("Init TWITC\n");
	return PyModule_Create(&twitc_module);
}

