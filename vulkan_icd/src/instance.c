/*
 * SynthGPU Vulkan ICD — Instance Management
 * Handles vkCreateInstance, vkDestroyInstance, and instance-level queries.
 */
#include "synthgpu_vk.h"

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateInstance(
        const VkInstanceCreateInfo *pCreateInfo,
        const VkAllocationCallbacks *pAllocator,
        VkInstance *pInstance) {
    (void)pAllocator;

    SynthGPU_Instance_T *inst = (SynthGPU_Instance_T*)SYNTHGPU_ALLOC(sizeof(*inst));
    if (!inst) return VK_ERROR_OUT_OF_HOST_MEMORY;

    SET_LOADER_MAGIC(inst);
    inst->debug_enabled = 0;
    inst->api_version = VK_API_VERSION_1_3;

    if (pCreateInfo->pApplicationInfo)
        inst->api_version = pCreateInfo->pApplicationInfo->apiVersion;

    *pInstance = (VkInstance)inst;
    fprintf(stderr, "[SynthGPU Vulkan] Instance created (API %u.%u.%u)\n",
        VK_API_VERSION_MAJOR(inst->api_version),
        VK_API_VERSION_MINOR(inst->api_version),
        VK_API_VERSION_PATCH(inst->api_version));
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyInstance(
        VkInstance instance, const VkAllocationCallbacks *pAllocator) {
    (void)pAllocator;
    SYNTHGPU_FREE(instance);
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_EnumerateInstanceVersion(uint32_t *pApiVersion) {
    *pApiVersion = VK_MAKE_API_VERSION(0, 1, 3, 0);
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_EnumerateInstanceExtensionProperties(
        const char *pLayerName, uint32_t *pPropertyCount,
        VkExtensionProperties *pProperties) {
    (void)pLayerName;
    /* No instance extensions beyond core 1.3 */
    *pPropertyCount = 0;
    (void)pProperties;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_EnumerateInstanceLayerProperties(
        uint32_t *pPropertyCount, VkLayerProperties *pProperties) {
    *pPropertyCount = 0;
    (void)pProperties;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_EnumerateDeviceLayerProperties(
        VkPhysicalDevice physicalDevice, uint32_t *pPropertyCount,
        VkLayerProperties *pProperties) {
    (void)physicalDevice;
    *pPropertyCount = 0;
    (void)pProperties;
    return VK_SUCCESS;
}
