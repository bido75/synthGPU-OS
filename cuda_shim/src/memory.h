/*
 * SynthGPU CUDA Shim — Virtual VRAM Allocator Header
 * ====================================================
 * Declares the internal memory management API used by shim.c.
 * All "device" pointers are system RAM — no GPU involved.
 */

#ifndef SYNTHGPU_MEMORY_H
#define SYNTHGPU_MEMORY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialise the virtual VRAM pool (call once at startup) */
void   synthgpu_vram_init(void);

/* Allocate size bytes from the virtual VRAM pool.
 * Returns a 256-byte-aligned pointer, or NULL on OOM. */
void  *synthgpu_alloc(size_t size);

/* Release a pointer previously returned by synthgpu_alloc(). */
void   synthgpu_free(void *ptr);

/* Convert a "device pointer" to a host pointer.
 * In SynthGPU these are identical — device IS system RAM. */
void  *synthgpu_d2h_ptr(const void *device_ptr);

/* Query pool sizes */
size_t synthgpu_vram_total_bytes(void);
size_t synthgpu_vram_used_bytes(void);

/* Query the number of CPU compute units (used by device props) */
int    synthgpu_compute_units(void);

#ifdef __cplusplus
}
#endif

#endif /* SYNTHGPU_MEMORY_H */
