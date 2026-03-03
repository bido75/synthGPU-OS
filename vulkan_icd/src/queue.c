/*
 * SynthGPU Vulkan ICD — Queue Submission + Execution Engine
 * Executes recorded command buffers synchronously on the CPU.
 * vkCmdDispatch routes SPIR-V workgroups through the SynthGPU warp scheduler.
 */
#include "synthgpu_vk.h"
#include "spirv_dispatch.h"
#include <time.h>

/* ── Dispatch Execution ─────────────────────────────────────────────── */

static VkResult execute_dispatch(SynthGPU_Cmd_T *cmd) {
    SynthGPU_Pipeline_T      *pipeline  = cmd->dispatch.pipeline;
    SynthGPU_DescriptorSet_T **desc_sets = cmd->dispatch.descriptor_sets;
    uint32_t desc_count = cmd->dispatch.desc_set_count;
    uint32_t group_x    = cmd->dispatch.group_count_x;
    uint32_t group_y    = cmd->dispatch.group_count_y;
    uint32_t group_z    = cmd->dispatch.group_count_z;

    if (!pipeline || !pipeline->spirv_code) return VK_SUCCESS;

    uint32_t total_groups = group_x * group_y * group_z;
    if (total_groups == 0) return VK_SUCCESS;

    SynthGPU_DispatchContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.spirv_code       = pipeline->spirv_code;
    ctx.spirv_word_count = pipeline->spirv_word_count;
    ctx.local_size_x     = pipeline->local_size_x ? pipeline->local_size_x : 1;
    ctx.local_size_y     = pipeline->local_size_y ? pipeline->local_size_y : 1;
    ctx.local_size_z     = pipeline->local_size_z ? pipeline->local_size_z : 1;
    ctx.group_count_x    = group_x;
    ctx.group_count_y    = group_y;
    ctx.group_count_z    = group_z;

    /* Bind storage buffers from all active descriptor sets */
    for (uint32_t s = 0; s < desc_count && s < 4; s++) {
        if (!desc_sets[s]) continue;
        for (uint32_t b = 0; b < 16; b++) {
            SynthGPU_Buffer_T *buf = desc_sets[s]->bound_buffers[b];
            if (!buf || !buf->bound_memory) continue;

            uint8_t *base = (uint8_t*)buf->bound_memory->ptr
                          + buf->bind_offset
                          + desc_sets[s]->bound_offsets[b];
            SynthGPU_BufferBinding *bnd = &ctx.bindings[ctx.binding_count];
            bnd->ptr     = base;
            bnd->size    = (size_t)desc_sets[s]->bound_ranges[b];
            bnd->set     = s;
            bnd->binding = b;
            ctx.binding_count++;
            if (ctx.binding_count >= SYNTHGPU_MAX_BINDINGS) goto done_bindings;
        }
    }
done_bindings:;

#ifdef _WIN32
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);
#else
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
#endif

    VkResult result = synthgpu_spirv_dispatch(&ctx);

#ifdef _WIN32
    QueryPerformanceCounter(&t1);
    double exec_ms = (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / freq.QuadPart;
#else
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double exec_ms = (t1.tv_sec - t0.tv_sec) * 1000.0
                   + (t1.tv_nsec - t0.tv_nsec) / 1e6;
#endif

    fprintf(stderr,
        "[SynthGPU Vulkan] Dispatch(%u,%u,%u) = %u groups — %.2fms\n",
        group_x, group_y, group_z, total_groups, exec_ms);

    return result;
}

static VkResult execute_copy_buffer(SynthGPU_Cmd_T *cmd) {
    SynthGPU_Buffer_T *src = cmd->copy_buffer.src;
    SynthGPU_Buffer_T *dst = cmd->copy_buffer.dst;
    if (!src->bound_memory || !dst->bound_memory) return VK_SUCCESS;

    uint8_t *src_ptr = (uint8_t*)src->bound_memory->ptr
                     + src->bind_offset + cmd->copy_buffer.src_offset;
    uint8_t *dst_ptr = (uint8_t*)dst->bound_memory->ptr
                     + dst->bind_offset + cmd->copy_buffer.dst_offset;
    memcpy(dst_ptr, src_ptr, (size_t)cmd->copy_buffer.size);
    return VK_SUCCESS;
}

static VkResult execute_fill_buffer(SynthGPU_Cmd_T *cmd) {
    SynthGPU_Buffer_T *dst = cmd->fill_buffer.dst;
    if (!dst->bound_memory) return VK_SUCCESS;

    uint8_t *ptr = (uint8_t*)dst->bound_memory->ptr
                 + dst->bind_offset + cmd->fill_buffer.offset;
    size_t sz = (cmd->fill_buffer.size == VK_WHOLE_SIZE)
                ? (size_t)(dst->size - cmd->fill_buffer.offset)
                : (size_t)cmd->fill_buffer.size;

    uint32_t val = cmd->fill_buffer.data;
    for (size_t i = 0; i + 4 <= sz; i += 4)
        memcpy(ptr + i, &val, 4);
    return VK_SUCCESS;
}

static VkResult execute_command_buffer(SynthGPU_CommandBuffer_T *cb) {
    SynthGPU_Cmd_T *cmd = cb->cmd_head;
    while (cmd) {
        VkResult r = VK_SUCCESS;
        switch (cmd->type) {
            case SYNTHGPU_CMD_DISPATCH:         r = execute_dispatch(cmd);     break;
            case SYNTHGPU_CMD_COPY_BUFFER:      r = execute_copy_buffer(cmd);  break;
            case SYNTHGPU_CMD_FILL_BUFFER:      r = execute_fill_buffer(cmd);  break;
            case SYNTHGPU_CMD_PIPELINE_BARRIER:                                break;
        }
        if (r != VK_SUCCESS) return r;
        cmd = cmd->next;
    }
    return VK_SUCCESS;
}

/* ── Queue Submit ────────────────────────────────────────────────────── */

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_QueueSubmit(
        VkQueue queue, uint32_t submitCount,
        const VkSubmitInfo *pSubmits, VkFence fence) {
    (void)queue;
    for (uint32_t i = 0; i < submitCount; i++) {
        for (uint32_t j = 0; j < pSubmits[i].commandBufferCount; j++) {
            SynthGPU_CommandBuffer_T *cb =
                (SynthGPU_CommandBuffer_T*)pSubmits[i].pCommandBuffers[j];
            VkResult r = execute_command_buffer(cb);
            if (r != VK_SUCCESS) return r;
        }
    }
    /* Signal fence — execution is synchronous */
    if (fence != VK_NULL_HANDLE) {
        SynthGPU_Fence_T *f = (SynthGPU_Fence_T*)(uintptr_t)fence;
        f->signaled = 1;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_QueueSubmit2(
        VkQueue queue, uint32_t submitCount,
        const VkSubmitInfo2 *pSubmits, VkFence fence) {
    (void)queue;
    for (uint32_t i = 0; i < submitCount; i++) {
        for (uint32_t j = 0; j < pSubmits[i].commandBufferInfoCount; j++) {
            SynthGPU_CommandBuffer_T *cb =
                (SynthGPU_CommandBuffer_T*)
                pSubmits[i].pCommandBufferInfos[j].commandBuffer;
            VkResult r = execute_command_buffer(cb);
            if (r != VK_SUCCESS) return r;
        }
    }
    if (fence != VK_NULL_HANDLE) {
        SynthGPU_Fence_T *f = (SynthGPU_Fence_T*)(uintptr_t)fence;
        f->signaled = 1;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_QueueWaitIdle(VkQueue queue) {
    (void)queue;
    return VK_SUCCESS;
}

/* ── Sync Objects ────────────────────────────────────────────────────── */

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateFence(
        VkDevice device, const VkFenceCreateInfo *pCreateInfo,
        const VkAllocationCallbacks *pAllocator, VkFence *pFence) {
    (void)device; (void)pAllocator;
    SynthGPU_Fence_T *f = (SynthGPU_Fence_T*)SYNTHGPU_ALLOC(sizeof(*f));
    if (!f) return VK_ERROR_OUT_OF_HOST_MEMORY;
    f->signaled = (pCreateInfo->flags & VK_FENCE_CREATE_SIGNALED_BIT) ? 1 : 0;
    *pFence = (VkFence)(uintptr_t)f;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyFence(
        VkDevice device, VkFence fence,
        const VkAllocationCallbacks *pAllocator) {
    (void)device; (void)pAllocator;
    if (fence != VK_NULL_HANDLE) SYNTHGPU_FREE((void*)(uintptr_t)fence);
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_WaitForFences(
        VkDevice device, uint32_t fenceCount, const VkFence *pFences,
        VkBool32 waitAll, uint64_t timeout) {
    (void)device; (void)waitAll; (void)timeout;
    /* Synchronous execution: fences are always signaled by the time we return */
    (void)fenceCount; (void)pFences;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_ResetFences(
        VkDevice device, uint32_t fenceCount, const VkFence *pFences) {
    (void)device;
    for (uint32_t i = 0; i < fenceCount; i++) {
        SynthGPU_Fence_T *f = (SynthGPU_Fence_T*)(uintptr_t)pFences[i];
        if (f) f->signaled = 0;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_GetFenceStatus(VkDevice device, VkFence fence) {
    (void)device;
    SynthGPU_Fence_T *f = (SynthGPU_Fence_T*)(uintptr_t)fence;
    return f->signaled ? VK_SUCCESS : VK_NOT_READY;
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateSemaphore(
        VkDevice device, const VkSemaphoreCreateInfo *pCreateInfo,
        const VkAllocationCallbacks *pAllocator, VkSemaphore *pSemaphore) {
    (void)device; (void)pCreateInfo; (void)pAllocator;
    SynthGPU_Semaphore_T *s = (SynthGPU_Semaphore_T*)SYNTHGPU_ALLOC(sizeof(*s));
    if (!s) return VK_ERROR_OUT_OF_HOST_MEMORY;
    *pSemaphore = (VkSemaphore)(uintptr_t)s;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_DestroySemaphore(
        VkDevice device, VkSemaphore semaphore,
        const VkAllocationCallbacks *pAllocator) {
    (void)device; (void)pAllocator;
    if (semaphore != VK_NULL_HANDLE) SYNTHGPU_FREE((void*)(uintptr_t)semaphore);
}
