/*
 * SynthGPU Vulkan ICD — SPIR-V Dispatch Header
 * Declares the dispatch context and entry point used by queue.c and spirv_dispatch.c
 */
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <vulkan/vulkan.h>

/* Maximum storage buffer bindings per dispatch */
#define SYNTHGPU_MAX_BINDINGS 32

typedef struct SynthGPU_BufferBinding {
    void        *ptr;     /* Base pointer into mapped host memory */
    size_t       size;    /* Range in bytes */
    uint32_t     set;     /* Descriptor set index */
    uint32_t     binding; /* Binding index within set */
} SynthGPU_BufferBinding;

typedef struct SynthGPU_DispatchContext {
    /* SPIR-V bytecode */
    const uint32_t *spirv_code;
    size_t          spirv_word_count;

    /* Workgroup dimensions from SPIR-V LocalSize decoration */
    uint32_t  local_size_x;
    uint32_t  local_size_y;
    uint32_t  local_size_z;

    /* Dispatch dimensions from vkCmdDispatch */
    uint32_t  group_count_x;
    uint32_t  group_count_y;
    uint32_t  group_count_z;

    /* Bound storage buffers */
    SynthGPU_BufferBinding bindings[SYNTHGPU_MAX_BINDINGS];
    uint32_t               binding_count;
} SynthGPU_DispatchContext;

/* Entry point — called from queue.c execute_dispatch() */
VkResult synthgpu_spirv_dispatch(const SynthGPU_DispatchContext *ctx);
