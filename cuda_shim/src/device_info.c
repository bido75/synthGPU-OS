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

#ifdef _WIN32
#include <windows.h>
static int get_cpu_cores(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
}
static size_t get_available_ram(void) {
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    return (size_t)(ms.ullAvailPhys * 40 / 100);
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
static size_t get_available_ram(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return (size_t)4 * 1024 * 1024 * 1024;
    long long kb = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "MemAvailable:", 13) == 0) {
            sscanf(line, "MemAvailable: %lld kB", &kb);
            break;
        }
    }
    fclose(f);
    return (size_t)(kb * 1024LL * 40 / 100);
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

int cudaGetDeviceCount(int *count) {
    if (!count) return cudaErrorInvalidDevice;
    *count = 1;
    return cudaSuccess;
}

int cudaGetDevice(int *device) {
    if (!device) return cudaErrorInvalidDevice;
    *device = _current_device;
    return cudaSuccess;
}

int cudaSetDevice(int device) {
    if (device != 0) return cudaErrorInvalidDevice;
    _current_device = device;
    return cudaSuccess;
}

int cudaGetDeviceProperties(void *prop_raw, int device) {
    if (!prop_raw || device != 0) return cudaErrorInvalidDevice;

    /* Zero the entire struct — safer than field-by-field on unknown layout */
    memset(prop_raw, 0, 1024);

    /* Access named fields via char pointer arithmetic using offsets from
     * the CUDA SDK struct layout.  We use a flexible cast approach that
     * is safe on both 32- and 64-bit platforms. */

    typedef struct {
        char   name[256];
        size_t totalGlobalMem;
        size_t sharedMemPerBlock;
        int    regsPerBlock;
        int    warpSize;
        size_t memPitch;
        int    maxThreadsPerBlock;
        int    maxThreadsDim[3];
        int    maxGridSize[3];
        int    clockRate;
        size_t totalConstMem;
        int    major;
        int    minor;
        size_t textureAlignment;
        int    deviceOverlap;
        int    multiProcessorCount;
        int    kernelExecTimeoutEnabled;
        int    integrated;
        int    canMapHostMemory;
        int    computeMode;
        int    maxTexture1D;
        int    maxTexture2D[2];
        int    maxTexture3D[3];
        int    concurrentKernels;
        int    ECCEnabled;
        int    pciBusID;
        int    pciDeviceID;
        int    pciDomainID;
        int    tccDriver;
        int    asyncEngineCount;
        int    unifiedAddressing;
        int    memoryClockRate;
        int    memoryBusWidth;
        size_t l2CacheSize;
        int    maxThreadsPerMultiProcessor;
    } Props;

    Props *p = (Props *)prop_raw;

    strncpy(p->name, synthgpu_device_name(), 255);
    p->totalGlobalMem           = get_available_ram();
    p->sharedMemPerBlock        = 49152;
    p->regsPerBlock             = 65536;
    p->warpSize                 = 32;
    p->maxThreadsPerBlock       = 1024;
    p->maxThreadsDim[0]         = 1024;
    p->maxThreadsDim[1]         = 1024;
    p->maxThreadsDim[2]         = 64;
    p->maxGridSize[0]           = 2147483647;
    p->maxGridSize[1]           = 65535;
    p->maxGridSize[2]           = 65535;
    p->clockRate                = 1700000;
    p->totalConstMem            = 65536;
    p->major                    = 8;
    p->minor                    = 0;
    p->multiProcessorCount      = get_cpu_cores();
    p->l2CacheSize              = 4194304;
    p->memoryClockRate          = 9001000;
    p->memoryBusWidth           = 256;
    p->concurrentKernels        = 1;
    p->unifiedAddressing        = 1;
    p->canMapHostMemory         = 1;

    fprintf(stderr,
        "[SynthGPU] cudaGetDeviceProperties → \"%s\" (%zuMB vRAM, %d CUs)\n",
        p->name,
        p->totalGlobalMem / 1024 / 1024,
        p->multiProcessorCount);

    return cudaSuccess;
}

int cudaDeviceGetAttribute(int *value, int attr, int device) {
    if (!value || device != 0) return cudaErrorInvalidDevice;
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

int cudaRuntimeGetVersion(int *v) { if (v) *v = 12020; return cudaSuccess; }
int cudaDriverGetVersion(int *v)  { if (v) *v = 12020; return cudaSuccess; }
int cudaDeviceSynchronize(void)   { return cudaSuccess; }
int cudaDeviceReset(void)         { return cudaSuccess; }

/* ── Error handling ──────────────────────────────────────────────── */
static int _last_error = 0;

int cudaGetLastError(void) {
    int e = _last_error;
    _last_error = cudaSuccess;
    return e;
}

int cudaPeekAtLastError(void) { return _last_error; }

const char *cudaGetErrorString(int error) {
    switch (error) {
        case 0:   return "no error";
        case 1:   return "invalid argument";
        case 2:   return "out of memory";
        case 100: return "no CUDA-capable device is detected";
        case 101: return "invalid device ordinal";
        default:  return "unknown error";
    }
}

const char *cudaGetErrorName(int error) {
    switch (error) {
        case 0:   return "cudaSuccess";
        case 1:   return "cudaErrorInvalidValue";
        case 2:   return "cudaErrorMemoryAllocation";
        case 100: return "cudaErrorNoDevice";
        case 101: return "cudaErrorInvalidDevice";
        default:  return "cudaErrorUnknown";
    }
}
