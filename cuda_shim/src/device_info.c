/*
 * SynthGPU CUDA Shim — Device Info Functions
 * ===========================================
 * Implements cudaGetDeviceCount, cudaGetDeviceProperties, etc.
 * These are the functions that make PyTorch detect SynthGPU as a
 * CUDA-capable device on any CPU-only machine.
 */

#define _GNU_SOURCE
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "synthgpu_cuda.h"

/* ── Error codes (local convenience — real types in headers) ─── */
#define cudaSuccess              0
#define cudaErrorInvalidDevice  101
#define cudaErrorNoDevice       100

/* ── Internal helpers ──────────────────────────────────────────── */

static int _current_device = 0;
cudaError_t _last_error = cudaSuccess;

#ifdef _WIN32
#include <windows.h>
static int get_cpu_cores(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
}
#else
static int get_cpu_cores(void) {
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return 4;
    int cores = 0;
    char line[256];
    while (fgets(line, sizeof(line), f))
        if (strncmp(line, "processor", 9) == 0) cores++;
    fclose(f);
    return cores > 0 ? cores : 4;
}
#endif

/* ── Public helpers called by shim.c / cublas.c ─────────────────── */

int synthgpu_compute_units(void) {
    return get_cpu_cores();
}

const char *synthgpu_device_name(void) {
    return "SynthGPU Virtual Accelerator v0.3";
}

/* ── CUDA Runtime API ────────────────────────────────────────────── */

cudaError_t cudaGetDeviceCount(int *count) {
    if (!count) {
        _last_error = cudaErrorInvalidValue;
        return cudaErrorInvalidValue;
    }
    *count = 1;
    return cudaSuccess;
}

cudaError_t cudaGetDevice(int *device) {
    if (!device) {
        _last_error = cudaErrorInvalidValue;
        return cudaErrorInvalidValue;
    }
    *device = _current_device;
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
    if (device != 0) {
        _last_error = cudaErrorInvalidDevice;
        return cudaErrorInvalidDevice;
    }
    _current_device = device;
    return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
    if (!prop) {
        _last_error = cudaErrorInvalidValue;
        return cudaErrorInvalidValue;
    }
    if (device != 0) {
        _last_error = cudaErrorInvalidDevice;
        return cudaErrorInvalidDevice;
    }

    memset(prop, 0, sizeof(*prop));
    strncpy(prop->name, synthgpu_device_name(), sizeof(prop->name) - 1);
    prop->totalGlobalMem           = synthgpu_vram_total_bytes();
    prop->sharedMemPerBlock        = 49152;
    prop->regsPerBlock             = 65536;
    prop->warpSize                 = 32;
    prop->maxThreadsPerBlock       = 1024;
    prop->maxThreadsDim[0]         = 1024;
    prop->maxThreadsDim[1]         = 1024;
    prop->maxThreadsDim[2]         = 64;
    prop->maxGridSize[0]           = 2147483647;
    prop->maxGridSize[1]           = 65535;
    prop->maxGridSize[2]           = 65535;
    prop->clockRate                = 1700000;
    prop->totalConstMem            = 65536;
    prop->major                    = 8;
    prop->minor                    = 0;
    prop->multiProcessorCount      = synthgpu_compute_units();
    prop->l2CacheSize              = 4194304;
    prop->memoryClockRate          = 9001000;
    prop->memoryBusWidth           = 256;
    prop->concurrentKernels        = 1;
    prop->computeMode              = cudaComputeModeDefault;
    prop->unifiedAddressing        = 1;
    prop->canMapHostMemory         = 1;

    fprintf(stderr,
        "[SynthGPU] cudaGetDeviceProperties → \"%s\" (%zuMB vRAM, %d CUs)\n",
        prop->name,
        prop->totalGlobalMem / 1024 / 1024,
        prop->multiProcessorCount);

    return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(int *value, int attr, int device) {
    if (!value) {
        _last_error = cudaErrorInvalidValue;
        return cudaErrorInvalidValue;
    }
    if (device != 0) {
        _last_error = cudaErrorInvalidDevice;
        return cudaErrorInvalidDevice;
    }
    switch (attr) {
        case 1:  *value = 1024;         break; /* MaxThreadsPerBlock */
        case 10: *value = 32;           break; /* WarpSize */
        case 16: *value = get_cpu_cores(); break; /* MultiProcessorCount */
        case 75: *value = 8;            break; /* ComputeCapabilityMajor */
        case 76: *value = 0;            break; /* ComputeCapabilityMinor */
        default: *value = 0;
    }
    return cudaSuccess;
}

cudaError_t cudaRuntimeGetVersion(int *v) {
    if (!v) { _last_error = cudaErrorInvalidValue; return cudaErrorInvalidValue; }
    *v = 12020;
    return cudaSuccess;
}
cudaError_t cudaDriverGetVersion(int *v) {
    if (!v) { _last_error = cudaErrorInvalidValue; return cudaErrorInvalidValue; }
    *v = 12020;
    return cudaSuccess;
}
cudaError_t cudaDeviceSynchronize(void) { return cudaSuccess; }
cudaError_t cudaDeviceReset(void)       { return cudaSuccess; }

/* ── Error handling ──────────────────────────────────────────────── */
cudaError_t cudaGetLastError(void) {
    cudaError_t e = _last_error;
    _last_error = cudaSuccess;
    return e;
}

cudaError_t cudaPeekAtLastError(void) { return _last_error; }

const char *cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case 0:   return "no error";
        case 1:   return "invalid argument";
        case 2:   return "out of memory";
        case 3:   return "initialization error";
        case 100: return "no CUDA-capable device is detected";
        case 101: return "invalid device ordinal";
        default:  return "unknown error";
    }
}

const char *cudaGetErrorName(cudaError_t error) {
    switch (error) {
        case 0:   return "cudaSuccess";
        case 1:   return "cudaErrorInvalidValue";
        case 2:   return "cudaErrorMemoryAllocation";
        case 100: return "cudaErrorNoDevice";
        case 101: return "cudaErrorInvalidDevice";
        default:  return "cudaErrorUnknown";
    }
}
