/*
 * SynthGPU Vulkan ICD — Command Buffers
 * Records vkCmd* calls as a linked list; executed at vkQueueSubmit time.
 */
#include "synthgpu_vk.h"

/* ── Command Pool ────────────────────────────────────────────────────── */

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateCommandPool(
        VkDevice device, const VkCommandPoolCreateInfo *pCreateInfo,
        const VkAllocationCallbacks *pAllocator, VkCommandPool *pCommandPool) {
    (void)pAllocator;
    SynthGPU_CommandPool_T *pool =
        (SynthGPU_CommandPool_T*)SYNTHGPU_ALLOC(sizeof(*pool));
    if (!pool) return VK_ERROR_OUT_OF_HOST_MEMORY;
    pool->device             = (SynthGPU_Device_T*)device;
    pool->queue_family_index = pCreateInfo->queueFamilyIndex;
    *pCommandPool = (VkCommandPool)(uintptr_t)pool;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyCommandPool(
        VkDevice device, VkCommandPool commandPool,
        const VkAllocationCallbacks *pAllocator) {
    (void)device; (void)pAllocator;
    if (commandPool != VK_NULL_HANDLE)
        SYNTHGPU_FREE((void*)(uintptr_t)commandPool);
}

/* ── Command Buffers ─────────────────────────────────────────────────── */

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_AllocateCommandBuffers(
        VkDevice device, const VkCommandBufferAllocateInfo *pAllocateInfo,
        VkCommandBuffer *pCommandBuffers) {
    for (uint32_t i = 0; i < pAllocateInfo->commandBufferCount; i++) {
        SynthGPU_CommandBuffer_T *cb =
            (SynthGPU_CommandBuffer_T*)SYNTHGPU_ALLOC(sizeof(*cb));
        if (!cb) return VK_ERROR_OUT_OF_HOST_MEMORY;
        SET_LOADER_MAGIC(cb);
        cb->device = (SynthGPU_Device_T*)device;
        pCommandBuffers[i] = (VkCommandBuffer)cb;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_FreeCommandBuffers(
        VkDevice device, VkCommandPool commandPool,
        uint32_t commandBufferCount, const VkCommandBuffer *pCommandBuffers) {
    (void)device; (void)commandPool;
    for (uint32_t i = 0; i < commandBufferCount; i++) {
        if (!pCommandBuffers[i]) continue;
        SynthGPU_CommandBuffer_T *cb = (SynthGPU_CommandBuffer_T*)pCommandBuffers[i];
        /* Free command list */
        SynthGPU_Cmd_T *cmd = cb->cmd_head;
        while (cmd) {
            SynthGPU_Cmd_T *next = cmd->next;
            SYNTHGPU_FREE(cmd);
            cmd = next;
        }
        SYNTHGPU_FREE(cb);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_BeginCommandBuffer(
        VkCommandBuffer commandBuffer,
        const VkCommandBufferBeginInfo *pBeginInfo) {
    (void)pBeginInfo;
    SynthGPU_CommandBuffer_T *cb = (SynthGPU_CommandBuffer_T*)commandBuffer;
    /* Clear existing commands */
    SynthGPU_Cmd_T *cmd = cb->cmd_head;
    while (cmd) {
        SynthGPU_Cmd_T *next = cmd->next;
        SYNTHGPU_FREE(cmd);
        cmd = next;
    }
    cb->cmd_head        = NULL;
    cb->cmd_tail        = NULL;
    cb->cmd_count       = 0;
    cb->bound_pipeline  = NULL;
    cb->bound_desc_count = 0;
    memset(cb->bound_desc_sets, 0, sizeof(cb->bound_desc_sets));
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_EndCommandBuffer(VkCommandBuffer commandBuffer) {
    (void)commandBuffer;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_ResetCommandBuffer(
        VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags) {
    (void)flags;
    return synthgpu_BeginCommandBuffer(commandBuffer, NULL);
}

/* ── Helper: append command to buffer ───────────────────────────────── */
static SynthGPU_Cmd_T *cmd_append(SynthGPU_CommandBuffer_T *cb) {
    SynthGPU_Cmd_T *cmd = (SynthGPU_Cmd_T*)SYNTHGPU_ALLOC(sizeof(*cmd));
    if (!cmd) return NULL;
    if (cb->cmd_tail)
        cb->cmd_tail->next = cmd;
    else
        cb->cmd_head = cmd;
    cb->cmd_tail = cmd;
    cb->cmd_count++;
    return cmd;
}

/* ── Command Recording ───────────────────────────────────────────────── */

VKAPI_ATTR void VKAPI_CALL synthgpu_CmdBindPipeline(
        VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
        VkPipeline pipeline) {
    (void)pipelineBindPoint;
    SynthGPU_CommandBuffer_T *cb = (SynthGPU_CommandBuffer_T*)commandBuffer;
    cb->bound_pipeline = (SynthGPU_Pipeline_T*)(uintptr_t)pipeline;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_CmdBindDescriptorSets(
        VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
        VkPipelineLayout layout, uint32_t firstSet, uint32_t descriptorSetCount,
        const VkDescriptorSet *pDescriptorSets,
        uint32_t dynamicOffsetCount, const uint32_t *pDynamicOffsets) {
    (void)pipelineBindPoint; (void)layout;
    (void)dynamicOffsetCount; (void)pDynamicOffsets;
    SynthGPU_CommandBuffer_T *cb = (SynthGPU_CommandBuffer_T*)commandBuffer;
    for (uint32_t i = 0; i < descriptorSetCount; i++) {
        uint32_t slot = firstSet + i;
        if (slot < 4) {
            cb->bound_desc_sets[slot] =
                (SynthGPU_DescriptorSet_T*)(uintptr_t)pDescriptorSets[i];
        }
    }
    cb->bound_desc_count = firstSet + descriptorSetCount;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_CmdDispatch(
        VkCommandBuffer commandBuffer,
        uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) {
    SynthGPU_CommandBuffer_T *cb = (SynthGPU_CommandBuffer_T*)commandBuffer;
    SynthGPU_Cmd_T *cmd = cmd_append(cb);
    if (!cmd) return;

    cmd->type                       = SYNTHGPU_CMD_DISPATCH;
    cmd->dispatch.pipeline          = cb->bound_pipeline;
    cmd->dispatch.group_count_x     = groupCountX;
    cmd->dispatch.group_count_y     = groupCountY;
    cmd->dispatch.group_count_z     = groupCountZ;
    cmd->dispatch.desc_set_count    = cb->bound_desc_count;
    memcpy(cmd->dispatch.descriptor_sets, cb->bound_desc_sets,
           sizeof(cb->bound_desc_sets));
}

VKAPI_ATTR void VKAPI_CALL synthgpu_CmdCopyBuffer(
        VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer,
        uint32_t regionCount, const VkBufferCopy *pRegions) {
    SynthGPU_CommandBuffer_T *cb = (SynthGPU_CommandBuffer_T*)commandBuffer;
    for (uint32_t i = 0; i < regionCount; i++) {
        SynthGPU_Cmd_T *cmd = cmd_append(cb);
        if (!cmd) return;
        cmd->type                   = SYNTHGPU_CMD_COPY_BUFFER;
        cmd->copy_buffer.src        = (SynthGPU_Buffer_T*)(uintptr_t)srcBuffer;
        cmd->copy_buffer.dst        = (SynthGPU_Buffer_T*)(uintptr_t)dstBuffer;
        cmd->copy_buffer.src_offset = pRegions[i].srcOffset;
        cmd->copy_buffer.dst_offset = pRegions[i].dstOffset;
        cmd->copy_buffer.size       = pRegions[i].size;
    }
}

VKAPI_ATTR void VKAPI_CALL synthgpu_CmdFillBuffer(
        VkCommandBuffer commandBuffer, VkBuffer dstBuffer,
        VkDeviceSize dstOffset, VkDeviceSize size, uint32_t data) {
    SynthGPU_CommandBuffer_T *cb = (SynthGPU_CommandBuffer_T*)commandBuffer;
    SynthGPU_Cmd_T *cmd = cmd_append(cb);
    if (!cmd) return;
    cmd->type                  = SYNTHGPU_CMD_FILL_BUFFER;
    cmd->fill_buffer.dst       = (SynthGPU_Buffer_T*)(uintptr_t)dstBuffer;
    cmd->fill_buffer.offset    = dstOffset;
    cmd->fill_buffer.size      = size;
    cmd->fill_buffer.data      = data;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_CmdPipelineBarrier(
        VkCommandBuffer commandBuffer,
        VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask,
        VkDependencyFlags dependencyFlags,
        uint32_t memoryBarrierCount, const VkMemoryBarrier *pMemoryBarriers,
        uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier *pBufferMemoryBarriers,
        uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier *pImageMemoryBarriers) {
    /* CPU execution is sequentially consistent — barriers are no-ops */
    (void)commandBuffer;
    (void)srcStageMask; (void)dstStageMask; (void)dependencyFlags;
    (void)memoryBarrierCount; (void)pMemoryBarriers;
    (void)bufferMemoryBarrierCount; (void)pBufferMemoryBarriers;
    (void)imageMemoryBarrierCount; (void)pImageMemoryBarriers;
}
