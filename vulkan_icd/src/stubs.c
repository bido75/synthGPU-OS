/*
 * SynthGPU Vulkan ICD — Stub implementations
 * No-op functions required by the dispatch table for a compute-only ICD.
 * None of these are exercised by compute workloads.
 */
#include "synthgpu_vk.h"

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_FreeDescriptorSets(
        VkDevice device, VkDescriptorPool descriptorPool,
        uint32_t descriptorSetCount, const VkDescriptorSet *pDescriptorSets) {
    (void)device; (void)descriptorPool;
    for (uint32_t i = 0; i < descriptorSetCount; i++) {
        if (pDescriptorSets[i] != VK_NULL_HANDLE)
            SYNTHGPU_FREE((void*)(uintptr_t)pDescriptorSets[i]);
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_ResetCommandPool(
        VkDevice device, VkCommandPool commandPool,
        VkCommandPoolResetFlags flags) {
    (void)device; (void)commandPool; (void)flags;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_CmdDispatchIndirect(
        VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset) {
    /* Indirect dispatch: read group counts from buffer.
       For MVP: no-op (requires buffer readback before dispatch). */
    (void)commandBuffer; (void)buffer; (void)offset;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_CmdUpdateBuffer(
        VkCommandBuffer commandBuffer, VkBuffer dstBuffer,
        VkDeviceSize dstOffset, VkDeviceSize dataSize, const void *pData) {
    /* Inline buffer update — equivalent to a small memcpy */
    SynthGPU_Buffer_T *buf = (SynthGPU_Buffer_T*)(uintptr_t)dstBuffer;
    if (buf && buf->bound_memory && buf->bound_memory->ptr) {
        uint8_t *dst = (uint8_t*)buf->bound_memory->ptr
                     + buf->bind_offset + dstOffset;
        memcpy(dst, pData, (size_t)dataSize);
    }
    (void)commandBuffer;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_CmdPipelineBarrier2(
        VkCommandBuffer commandBuffer, const VkDependencyInfo *pDependencyInfo) {
    /* CPU execution is sequentially consistent — barriers are no-ops */
    (void)commandBuffer; (void)pDependencyInfo;
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_GetPipelineCacheData(
        VkDevice device, VkPipelineCache pipelineCache,
        size_t *pDataSize, void *pData) {
    (void)device; (void)pipelineCache;
    /* No cache data to return */
    *pDataSize = 0;
    (void)pData;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_MergePipelineCaches(
        VkDevice device, VkPipelineCache dstCache,
        uint32_t srcCacheCount, const VkPipelineCache *pSrcCaches) {
    (void)device; (void)dstCache; (void)srcCacheCount; (void)pSrcCaches;
    return VK_SUCCESS;
}
