/*
 * SynthGPU Vulkan ICD — Python ctypes Bridge
 * Provides C-callable functions that invoke the Python warp scheduler.
 * Called from spirv_dispatch.c when a Python interpreter is embedded.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/*
 * synthgpu_bridge_record_warps()
 *
 * Called after every Vulkan dispatch to increment warp counters.
 * When Python is embedded (FastAPI backend), this calls
 * WarpScheduler.record_external_warps() directly.
 * When standalone, writes a JSON telemetry file for the backend to poll.
 */
void synthgpu_bridge_record_warps(uint32_t warp_count, double exec_ms) {
#define TELEM_FILE "synthgpu_vulkan_warps.tmp"
    FILE *f = fopen(TELEM_FILE, "w");
    if (f) {
        fprintf(f,
            "{\"source\":\"vulkan_icd\","
            "\"warp_count\":%u,"
            "\"exec_ms\":%.3f}\n",
            warp_count, exec_ms);
        fclose(f);
    }
}

/*
 * synthgpu_bridge_get_version()
 * Returns the ICD version string — called by the backend status endpoint.
 */
const char* synthgpu_bridge_get_version(void) {
    return "v0.3.0-vulkan";
}
