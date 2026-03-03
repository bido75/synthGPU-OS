/*
 * SynthGPU CUDA Shim — C-to-Python Bridge
 * =========================================
 * Embeds the CPython interpreter and calls bridge_api.py for
 * operations that benefit from going through the WarpScheduler.
 *
 * compile: gcc ... $(python3-config --includes --ldflags) ...
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "synthgpu_cuda.h"

/* Python.h must come before any system headers that may conflict */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

static int    _bridge_ready  = 0;
static PyObject *_bridge_mod = NULL;

/* ── Initialise bridge ───────────────────────────────────────────── */

int synthgpu_bridge_init(void) {
    if (_bridge_ready) return 0;

    if (!Py_IsInitialized()) {
        Py_Initialize();
    }

    /* Add cuda_shim package root to sys.path */
    PyObject *sys_path = PySys_GetObject("path");
    if (sys_path) {
        /* Find the project root relative to this library */
        const char *env_root = getenv("SYNTHGPU_ROOT");
        const char *root = env_root ? env_root : ".";
        PyList_Append(sys_path, PyUnicode_FromString(root));
    }

    _bridge_mod = PyImport_ImportModule("cuda_shim.kernels.bridge_api");
    if (!_bridge_mod) {
        PyErr_Print();
        fprintf(stderr, "[SynthGPU] WARNING: Python bridge not available. "
                        "Falling back to pure C compute.\n");
        return -1;
    }

    _bridge_ready = 1;
    fprintf(stderr, "[SynthGPU] Python bridge ready (warp telemetry active)\n");
    return 0;
}

/* ── Helper — call a Python function in bridge_api ──────────────── */

static PyObject *_call(const char *fn, PyObject *args) {
    if (!_bridge_ready || !_bridge_mod) return NULL;
    PyObject *func = PyObject_GetAttrString(_bridge_mod, fn);
    if (!func) { PyErr_Clear(); return NULL; }
    PyObject *result = PyObject_CallObject(func, args);
    Py_DECREF(func);
    if (!result) { PyErr_Print(); PyErr_Clear(); }
    return result;
}

/* ── Telemetry warp recording ────────────────────────────────────── */

static long  _warps_fallback = 0;
static float _throughput_fallback = 0.0f;

long synthgpu_get_warps_executed(void) {
    return _warps_fallback;
}

float synthgpu_get_warp_throughput(void) {
    return _throughput_fallback;
}

/* ── bridge_sgemm ────────────────────────────────────────────────── */

void bridge_sgemm(const float *A, const float *B, float *C,
                  int m, int n, int k,
                  float alpha, float beta,
                  int trans_a, int trans_b)
{
    /* Fast path: pure C via OpenBLAS (already called in cublas.c).
     * Only call Python bridge if telemetry recording is needed.
     * The warp recording happens here; computation was already done. */
    if (_bridge_ready) {
        PyGILState_STATE gstate = PyGILState_Ensure();

        /* Estimate warp count: 1 warp per 32x32 tile */
        int warps = (m / 32 + 1) * (n / 32 + 1);
        PyObject *args = Py_BuildValue("(if)", warps, 1.0f);
        PyObject *r = _call("_scheduler", NULL);
        if (r) {
            PyObject *method = PyObject_GetAttrString(r, "record_external_warps");
            if (method) {
                PyObject *call_args = Py_BuildValue("(if)", warps, 1.0f);
                PyObject *ret = PyObject_CallObject(method, call_args);
                Py_XDECREF(ret);
                Py_DECREF(call_args);
                Py_DECREF(method);
            }
            Py_DECREF(r);
        }
        Py_XDECREF(args);

        PyGILState_Release(gstate);
    }
    _warps_fallback += (m / 32 + 1) * (n / 32 + 1);
}

/* ── bridge_dgemm ────────────────────────────────────────────────── */

void bridge_dgemm(const double *A, const double *B, double *C,
                  int m, int n, int k,
                  double alpha, double beta,
                  int trans_a, int trans_b)
{
    /* Double precision — just record warps */
    bridge_sgemm(NULL, NULL, NULL, m, n, k,
                 (float)alpha, (float)beta, trans_a, trans_b);
    (void)A; (void)B; (void)C;
}

/* ── bridge_softmax / bridge_relu / bridge_layer_norm ───────────── */

void bridge_softmax(const float *input, float *output, int rows, int cols) {
    _warps_fallback += rows;
    (void)input; (void)output;
}

void bridge_relu(const float *input, float *output, int n) {
    _warps_fallback += (n / 32) + 1;
    (void)input; (void)output;
}

void bridge_layer_norm(const float *input, const float *gamma,
                       const float *beta, float *output,
                       int rows, int cols, float eps)
{
    _warps_fallback += rows * 2;
    (void)input; (void)gamma; (void)beta; (void)output; (void)eps;
}
