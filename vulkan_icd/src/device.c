/*
 * SynthGPU Vulkan ICD — Logical Device + Queue
 * Handles vkCreateDevice, vkGetDeviceQueue, and device-level dispatch.
 */
#include "synthgpu_vk.h"

/* Up to 4 compute queues backed by CPU threads */
static SynthGPU_Queue_T g_queues[4];

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateDevice(
        VkPhysicalDevice physicalDevice,
        const VkDeviceCreateInfo *pCreateInfo,
        const VkAllocationCallbacks *pAllocator,
        VkDevice *pDevice) {
    (void)pCreateInfo; (void)pAllocator;

    SynthGPU_Device_T *dev = (SynthGPU_Device_T*)SYNTHGPU_ALLOC(sizeof(*dev));
    if (!dev) return VK_ERROR_OUT_OF_HOST_MEMORY;

    SET_LOADER_MAGIC(dev);
    dev->physical_device = (SynthGPU_PhysicalDevice_T*)physicalDevice;

    /* Pre-allocate virtual VRAM pool from system RAM */
    dev->vram_pool_size = SYNTHGPU_VRAM_BYTES;
    dev->vram_pool      = malloc(dev->vram_pool_size);
    if (!dev->vram_pool) {
        /* Fallback: smaller pool if RAM is tight */
        dev->vram_pool_size = 64ULL * 1024 * 1024;
        dev->vram_pool      = malloc(dev->vram_pool_size);
    }
    if (!dev->vram_pool) {
        SYNTHGPU_FREE(dev);
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    dev->vram_allocated = 0;

    /* Initialize queues */
    for (int i = 0; i < 4; i++) {
        SET_LOADER_MAGIC(&g_queues[i]);
        g_queues[i].device       = dev;
        g_queues[i].family_index = 0;
        g_queues[i].queue_index  = (uint32_t)i;
    }

    *pDevice = (VkDevice)dev;
    fprintf(stderr, "[SynthGPU Vulkan] Logical device created (%zuMB vRAM pool)\n",
            dev->vram_pool_size / (1024 * 1024));
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyDevice(
        VkDevice device, const VkAllocationCallbacks *pAllocator) {
    (void)pAllocator;
    SynthGPU_Device_T *dev = (SynthGPU_Device_T*)device;
    if (dev) {
        if (dev->vram_pool) free(dev->vram_pool);
        SYNTHGPU_FREE(dev);
    }
}

VKAPI_ATTR void VKAPI_CALL synthgpu_GetDeviceQueue(
        VkDevice device, uint32_t queueFamilyIndex,
        uint32_t queueIndex, VkQueue *pQueue) {
    (void)device;
    if (queueFamilyIndex == 0 && queueIndex < 4)
        *pQueue = (VkQueue)&g_queues[queueIndex];
    else
        *pQueue = VK_NULL_HANDLE;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_GetDeviceQueue2(
        VkDevice device, const VkDeviceQueueInfo2 *pQueueInfo, VkQueue *pQueue) {
    synthgpu_GetDeviceQueue(device, pQueueInfo->queueFamilyIndex,
                             pQueueInfo->queueIndex, pQueue);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL synthgpu_GetDeviceProcAddr(
        VkDevice device, const char *pName) {
    (void)device;
    /* Re-use the same instance proc dispatch table */
    extern PFN_vkVoidFunction get_instance_proc(const char *name);
    return NULL; /* Resolved via vk_icdGetInstanceProcAddr */
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_DeviceWaitIdle(VkDevice device) {
    (void)device;
    return VK_SUCCESS;
}
