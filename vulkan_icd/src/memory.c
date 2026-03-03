/*
 * SynthGPU Vulkan ICD — Memory Management
 * vkAllocateMemory backed by malloc (system RAM = our virtual VRAM).
 * All allocations are host-visible + host-coherent + device-local.
 */
#include "synthgpu_vk.h"

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_AllocateMemory(
        VkDevice device, const VkMemoryAllocateInfo *pAllocateInfo,
        const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMemory) {
    (void)pAllocator;
    SynthGPU_Device_T *dev = (SynthGPU_Device_T*)device;

    /* Check VRAM limit BEFORE allocating */
    if (dev->vram_allocated + pAllocateInfo->allocationSize > dev->vram_pool_size)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;

    SynthGPU_DeviceMemory_T *mem =
        (SynthGPU_DeviceMemory_T*)SYNTHGPU_ALLOC(sizeof(*mem));
    if (!mem) return VK_ERROR_OUT_OF_HOST_MEMORY;

    mem->size              = (size_t)pAllocateInfo->allocationSize;
    mem->memory_type_index = pAllocateInfo->memoryTypeIndex;
    mem->mapped            = 0;
    mem->ptr               = calloc(1, mem->size);
    if (!mem->ptr) {
        SYNTHGPU_FREE(mem);
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    /* Track allocated bytes against virtual VRAM budget */
    dev->vram_allocated += pAllocateInfo->allocationSize;

    *pMemory = (VkDeviceMemory)(uintptr_t)mem;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_FreeMemory(
        VkDevice device, VkDeviceMemory memory,
        const VkAllocationCallbacks *pAllocator) {
    (void)pAllocator;
    if (memory == VK_NULL_HANDLE) return;
    SynthGPU_Device_T *dev = (SynthGPU_Device_T*)device;
    SynthGPU_DeviceMemory_T *mem = (SynthGPU_DeviceMemory_T*)(uintptr_t)memory;

    /* Decrement VRAM counter */
    if (dev->vram_allocated >= mem->size)
        dev->vram_allocated -= mem->size;
    else
        dev->vram_allocated = 0;

    if (mem->ptr) free(mem->ptr);
    SYNTHGPU_FREE(mem);
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_MapMemory(
        VkDevice device, VkDeviceMemory memory,
        VkDeviceSize offset, VkDeviceSize size,
        VkMemoryMapFlags flags, void **ppData) {
    (void)device; (void)size; (void)flags;
    SynthGPU_DeviceMemory_T *mem = (SynthGPU_DeviceMemory_T*)(uintptr_t)memory;
    *ppData = (uint8_t*)mem->ptr + offset;
    mem->mapped = 1;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_UnmapMemory(VkDevice device, VkDeviceMemory memory) {
    (void)device;
    SynthGPU_DeviceMemory_T *mem = (SynthGPU_DeviceMemory_T*)(uintptr_t)memory;
    mem->mapped = 0;
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_FlushMappedMemoryRanges(
        VkDevice device, uint32_t memoryRangeCount,
        const VkMappedMemoryRange *pMemoryRanges) {
    /* Host-coherent: no explicit flush needed */
    (void)device; (void)memoryRangeCount; (void)pMemoryRanges;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_InvalidateMappedMemoryRanges(
        VkDevice device, uint32_t memoryRangeCount,
        const VkMappedMemoryRange *pMemoryRanges) {
    /* Host-coherent: always valid */
    (void)device; (void)memoryRangeCount; (void)pMemoryRanges;
    return VK_SUCCESS;
}

/* ── Buffers ─────────────────────────────────────────────────────────── */

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateBuffer(
        VkDevice device, const VkBufferCreateInfo *pCreateInfo,
        const VkAllocationCallbacks *pAllocator, VkBuffer *pBuffer) {
    (void)device; (void)pAllocator;

    SynthGPU_Buffer_T *buf = (SynthGPU_Buffer_T*)SYNTHGPU_ALLOC(sizeof(*buf));
    if (!buf) return VK_ERROR_OUT_OF_HOST_MEMORY;

    buf->size         = pCreateInfo->size;
    buf->usage        = pCreateInfo->usage;
    buf->bound_memory = NULL;
    buf->bind_offset  = 0;

    *pBuffer = (VkBuffer)(uintptr_t)buf;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyBuffer(
        VkDevice device, VkBuffer buffer,
        const VkAllocationCallbacks *pAllocator) {
    (void)device; (void)pAllocator;
    if (buffer != VK_NULL_HANDLE)
        SYNTHGPU_FREE((void*)(uintptr_t)buffer);
}

VKAPI_ATTR void VKAPI_CALL synthgpu_GetBufferMemoryRequirements(
        VkDevice device, VkBuffer buffer, VkMemoryRequirements *pMemRequirements) {
    (void)device;
    SynthGPU_Buffer_T *buf = (SynthGPU_Buffer_T*)(uintptr_t)buffer;
    pMemRequirements->size           = buf->size;
    pMemRequirements->alignment      = 64;
    pMemRequirements->memoryTypeBits = 1; /* Only type 0 */
}

VKAPI_ATTR void VKAPI_CALL synthgpu_GetBufferMemoryRequirements2(
        VkDevice device, const VkBufferMemoryRequirementsInfo2 *pInfo,
        VkMemoryRequirements2 *pMemRequirements) {
    synthgpu_GetBufferMemoryRequirements(device, pInfo->buffer,
                                          &pMemRequirements->memoryRequirements);
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_BindBufferMemory(
        VkDevice device, VkBuffer buffer, VkDeviceMemory memory,
        VkDeviceSize memoryOffset) {
    (void)device;
    SynthGPU_Buffer_T *buf = (SynthGPU_Buffer_T*)(uintptr_t)buffer;
    buf->bound_memory = (SynthGPU_DeviceMemory_T*)(uintptr_t)memory;
    buf->bind_offset  = memoryOffset;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_BindBufferMemory2(
        VkDevice device, uint32_t bindInfoCount,
        const VkBindBufferMemoryInfo *pBindInfos) {
    for (uint32_t i = 0; i < bindInfoCount; i++) {
        VkResult r = synthgpu_BindBufferMemory(
            device, pBindInfos[i].buffer,
            pBindInfos[i].memory, pBindInfos[i].memoryOffset);
        if (r != VK_SUCCESS) return r;
    }
    return VK_SUCCESS;
}
