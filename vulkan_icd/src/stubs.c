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

#include <string.h>
#include <stdint.h>

VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceFormatProperties(
    VkPhysicalDevice pd, VkFormat fmt, VkFormatProperties *p) {
    (void)pd; (void)fmt; memset(p, 0, sizeof(*p)); }

VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceFormatProperties2(
    VkPhysicalDevice pd, VkFormat fmt, VkFormatProperties2 *p) {
    (void)pd; (void)fmt; memset(p, 0, sizeof(*p));
    p->sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2; }

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_GetPhysicalDeviceImageFormatProperties(
    VkPhysicalDevice pd, VkFormat f, VkImageType t, VkImageTiling ti,
    VkImageUsageFlags u, VkImageCreateFlags c, VkImageFormatProperties *p) {
    (void)pd;(void)f;(void)t;(void)ti;(void)u;(void)c;
    memset(p, 0, sizeof(*p)); return VK_ERROR_FORMAT_NOT_SUPPORTED; }

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_GetPhysicalDeviceImageFormatProperties2(
    VkPhysicalDevice pd, const VkPhysicalDeviceImageFormatInfo2 *info,
    VkImageFormatProperties2 *p) {
    (void)pd;(void)info; memset(p, 0, sizeof(*p));
    p->sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
    return VK_ERROR_FORMAT_NOT_SUPPORTED; }

VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceSparseImageFormatProperties(
    VkPhysicalDevice pd, VkFormat f, VkImageType t, VkSampleCountFlagBits s,
    VkImageUsageFlags u, VkImageTiling ti, uint32_t *n,
    VkSparseImageFormatProperties *p) {
    (void)pd;(void)f;(void)t;(void)s;(void)u;(void)ti;(void)p;
    if(n) *n = 0; }

VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceSparseImageFormatProperties2(
    VkPhysicalDevice pd, const VkPhysicalDeviceSparseImageFormatInfo2 *i,
    uint32_t *n, VkSparseImageFormatProperties2 *p) {
    (void)pd;(void)i;(void)p; if(n) *n = 0; }

VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceExternalBufferProperties(
    VkPhysicalDevice pd, const VkPhysicalDeviceExternalBufferInfo *i,
    VkExternalBufferProperties *p) {
    (void)pd;(void)i; memset(p, 0, sizeof(*p));
    p->sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES; }

VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceExternalSemaphoreProperties(
    VkPhysicalDevice pd, const VkPhysicalDeviceExternalSemaphoreInfo *i,
    VkExternalSemaphoreProperties *p) {
    (void)pd;(void)i; memset(p, 0, sizeof(*p));
    p->sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES; }

VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceExternalFenceProperties(
    VkPhysicalDevice pd, const VkPhysicalDeviceExternalFenceInfo *i,
    VkExternalFenceProperties *p) {
    (void)pd;(void)i; memset(p, 0, sizeof(*p));
    p->sType = VK_STRUCTURE_TYPE_EXTERNAL_FENCE_PROPERTIES; }

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateEvent(
    VkDevice d, const VkEventCreateInfo *ci,
    const VkAllocationCallbacks *a, VkEvent *e) {
    (void)d;(void)ci;(void)a;
    *e = (VkEvent)(uintptr_t)0xDEAD0001; return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyEvent(
    VkDevice d, VkEvent e, const VkAllocationCallbacks *a) {
    (void)d;(void)e;(void)a; }
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_GetEventStatus(VkDevice d, VkEvent e) {
    (void)d;(void)e; return VK_EVENT_SET; }
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_SetEvent(VkDevice d, VkEvent e) {
    (void)d;(void)e; return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_ResetEvent(VkDevice d, VkEvent e) {
    (void)d;(void)e; return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL synthgpu_CmdSetEvent(
    VkCommandBuffer cb, VkEvent e, VkPipelineStageFlags f) {
    (void)cb;(void)e;(void)f; }
VKAPI_ATTR void VKAPI_CALL synthgpu_CmdResetEvent(
    VkCommandBuffer cb, VkEvent e, VkPipelineStageFlags f) {
    (void)cb;(void)e;(void)f; }
VKAPI_ATTR void VKAPI_CALL synthgpu_CmdWaitEvents(
    VkCommandBuffer cb, uint32_t ec, const VkEvent *ev,
    VkPipelineStageFlags sf, VkPipelineStageFlags df,
    uint32_t mc, const VkMemoryBarrier *mb,
    uint32_t bc, const VkBufferMemoryBarrier *bb,
    uint32_t ic, const VkImageMemoryBarrier *ib) {
    (void)cb;(void)ec;(void)ev;(void)sf;(void)df;
    (void)mc;(void)mb;(void)bc;(void)bb;(void)ic;(void)ib; }

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateQueryPool(
    VkDevice d, const VkQueryPoolCreateInfo *ci,
    const VkAllocationCallbacks *a, VkQueryPool *p) {
    (void)d;(void)ci;(void)a;
    *p = (VkQueryPool)(uintptr_t)0xDEAD0002; return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyQueryPool(
    VkDevice d, VkQueryPool p, const VkAllocationCallbacks *a) {
    (void)d;(void)p;(void)a; }
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_GetQueryPoolResults(
    VkDevice d, VkQueryPool p, uint32_t fi, uint32_t qc,
    size_t ds, void *dat, VkDeviceSize st, VkQueryResultFlags f) {
    (void)d;(void)p;(void)fi;(void)qc;(void)ds;(void)dat;(void)st;(void)f;
    return VK_NOT_READY; }
VKAPI_ATTR void VKAPI_CALL synthgpu_CmdBeginQuery(
    VkCommandBuffer cb, VkQueryPool p, uint32_t q, VkQueryControlFlags f) {
    (void)cb;(void)p;(void)q;(void)f; }
VKAPI_ATTR void VKAPI_CALL synthgpu_CmdEndQuery(
    VkCommandBuffer cb, VkQueryPool p, uint32_t q) {
    (void)cb;(void)p;(void)q; }
VKAPI_ATTR void VKAPI_CALL synthgpu_CmdResetQueryPool(
    VkCommandBuffer cb, VkQueryPool p, uint32_t fi, uint32_t qc) {
    (void)cb;(void)p;(void)fi;(void)qc; }
VKAPI_ATTR void VKAPI_CALL synthgpu_CmdWriteTimestamp(
    VkCommandBuffer cb, VkPipelineStageFlagBits s, VkQueryPool p, uint32_t q) {
    (void)cb;(void)s;(void)p;(void)q; }
VKAPI_ATTR void VKAPI_CALL synthgpu_CmdWriteTimestamp2(
    VkCommandBuffer cb, VkPipelineStageFlags2 s, VkQueryPool p, uint32_t q) {
    (void)cb;(void)s;(void)p;(void)q; }
VKAPI_ATTR void VKAPI_CALL synthgpu_CmdCopyQueryPoolResults(
    VkCommandBuffer cb, VkQueryPool p, uint32_t fi, uint32_t qc,
    VkBuffer b, VkDeviceSize off, VkDeviceSize st, VkQueryResultFlags f) {
    (void)cb;(void)p;(void)fi;(void)qc;(void)b;(void)off;(void)st;(void)f; }

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_QueueBindSparse(
    VkQueue q, uint32_t bc, const VkBindSparseInfo *bi, VkFence f) {
    (void)q;(void)bc;(void)bi;(void)f; return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL synthgpu_GetDeviceMemoryCommitment(
    VkDevice d, VkDeviceMemory m, VkDeviceSize *s) {
    (void)d;(void)m; if(s) *s = 0; }

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_GetSemaphoreCounterValue(
    VkDevice d, VkSemaphore s, uint64_t *v) {
    (void)d;(void)s; if(v) *v = 0; return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_WaitSemaphores(
    VkDevice d, const VkSemaphoreWaitInfo *wi, uint64_t t) {
    (void)d;(void)wi;(void)t; return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_SignalSemaphore(
    VkDevice d, const VkSemaphoreSignalInfo *si) {
    (void)d;(void)si; return VK_SUCCESS; }

VKAPI_ATTR void VKAPI_CALL synthgpu_CmdDispatchBase(
    VkCommandBuffer cb, uint32_t bx, uint32_t by, uint32_t bz,
    uint32_t gx, uint32_t gy, uint32_t gz) {
    (void)bx;(void)by;(void)bz;
    synthgpu_CmdDispatch(cb, gx, gy, gz); }
