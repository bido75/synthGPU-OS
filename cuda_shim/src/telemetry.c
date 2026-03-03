/*
 * SynthGPU CUDA Shim — Telemetry Reporter
 * =========================================
 * Polls the Python WarpScheduler via the bridge and makes
 * stats available to the C layer. Not required for compute;
 * purely for dashboard integration.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include "synthgpu_cuda.h"

/*
 * These are lightweight wrappers that the dashboard REST endpoint
 * (or any future C caller) can call without going through Python.
 * The actual counters live in python_bridge.c.
 */

void synthgpu_print_stats(void) {
    fprintf(stderr,
        "[SynthGPU] Stats: warps=%ld  vram_used=%zuMB / %zuMB\n",
        synthgpu_get_warps_executed(),
        synthgpu_vram_used_bytes()  / 1024 / 1024,
        synthgpu_vram_total_bytes() / 1024 / 1024);
}
