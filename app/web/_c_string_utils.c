/* C extension for faster string processing — production-grade Lumina desktop */
#include <Python.h>
#include <string.h>

/* Fast lowercase conversion (used in response processing) */
static PyObject* cstr_lower(PyObject* self, PyObject* args) {
    const char* input;
    if (!PyArg_ParseTuple(args, "s", &input)) return NULL;
    char* result = malloc(strlen(input) + 1);
    if (!result) return PyErr_NoMemory();
    for (size_t i = 0; i < strlen(input); i++) {
        result[i] = (input[i] >= 'A' && input[i] <= 'Z') ? (input[i] + 32) : input[i];
    }
    result[strlen(input)] = '\0';
    PyObject* py_str = PyUnicode_FromString(result);
    free(result);
    return py_str;
}

/* Quick length check (production: avoids Python overhead for small strings) */
static PyObject* cstr_len_fast(PyObject* self, PyObject* args) {
    const char* input;
    if (!PyArg_ParseTuple(args, "s", &input)) return NULL;
    return PyLong_FromSize_t(strlen(input));
}

static PyMethodDef CStrMethods[] = {
    {"lower_fast", cstr_lower, METH_VARARGS, "Fast lowercase (C-level, no Python loop)"},
    {"len_fast", cstr_len_fast, METH_VARARGS, "Fast string length check"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cstr_module = {
    PyModuleDef_HEAD_INIT,
    "_c_string_utils",
    "C-accelerated string processing for Lumina desktop",
    -1,
    CStrMethods
};

PyMODINIT_FUNC PyInit__c_string_utils(void) {
    return PyModule_Create(&cstr_module);
}
