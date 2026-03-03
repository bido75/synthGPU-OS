/*
 * SynthGPU Vulkan ICD — Physical Device
 * Defines what vulkaninfo displays: device name, limits, memory, queue families.
 */
#include "synthgpu_vk.h"

static SynthGPU_PhysicalDevice_T g_physical_device;
static int g_physical_device_initialized = 0;

static void init_physical_device(SynthGPU_Instance_T *instance) {
    if (g_physical_device_initialized) return;
    memset(&g_physical_device, 0, sizeof(g_physical_device));
    SET_LOADER_MAGIC(&g_physical_device);
    g_physical_device.instance = instance;

    /* ── Device Properties (what vulkaninfo displays) ── */
    VkPhysicalDeviceProperties *p = &g_physical_device.props;
    p->apiVersion    = VK_MAKE_API_VERSION(0, 1, 3, 0);
    p->driverVersion = VK_MAKE_API_VERSION(0, 0, 3, 0);

    /* Device identification — these exact strings appear in vulkaninfo */
    strncpy(p->deviceName,
            "SynthGPU Virtual Accelerator v0.3",
            VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1);
    p->deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1] = '\0';

    p->vendorID  = 0x5347;   /* 'SG' — SynthGPU vendor identifier */
    p->deviceID  = 0x0003;
    p->deviceType = VK_PHYSICAL_DEVICE_TYPE_OTHER;

    /* Pipeline cache UUID — unique per build */
    memset(p->pipelineCacheUUID, 0, VK_UUID_SIZE);
    p->pipelineCacheUUID[0] = 'S';
    p->pipelineCacheUUID[1] = 'G';
    p->pipelineCacheUUID[2] = 0x03;

    /* Limits — conservative values for a CPU-backed compute device */
    p->limits.maxImageDimension2D                        = 4096;
    p->limits.maxImageDimension3D                        = 256;
    p->limits.maxUniformBufferRange                      = 65536;
    p->limits.maxStorageBufferRange                      = (uint32_t)SYNTHGPU_VRAM_BYTES;
    p->limits.maxPushConstantsSize                       = 128;
    p->limits.maxMemoryAllocationCount                   = 1024;
    p->limits.maxSamplerAllocationCount                  = 256;
    p->limits.maxBoundDescriptorSets                     = 4;
    p->limits.maxPerStageDescriptorStorageBuffers        = 8;
    p->limits.maxDescriptorSetStorageBuffers             = 32;
    p->limits.maxComputeSharedMemorySize                 = 32768;
    p->limits.maxComputeWorkGroupCount[0]                = 65535;
    p->limits.maxComputeWorkGroupCount[1]                = 65535;
    p->limits.maxComputeWorkGroupCount[2]                = 65535;
    p->limits.maxComputeWorkGroupInvocations             = 1024;
    p->limits.maxComputeWorkGroupSize[0]                 = 1024;
    p->limits.maxComputeWorkGroupSize[1]                 = 1024;
    p->limits.maxComputeWorkGroupSize[2]                 = 64;
    p->limits.minMemoryMapAlignment                      = 64;
    p->limits.minStorageBufferOffsetAlignment            = 64;
    p->limits.minUniformBufferOffsetAlignment            = 256;
    p->limits.maxDrawIndexedIndexValue                   = UINT32_MAX;
    p->limits.timestampComputeAndGraphics                = VK_FALSE;
    p->limits.maxDescriptorSetUniformBuffers             = 12;
    p->limits.maxDescriptorSetStorageBuffersDynamic      = 4;
    p->limits.maxPerStageResources                       = 128;
    p->limits.maxVertexInputAttributes                   = 0;
    p->limits.maxVertexInputBindings                     = 0;
    p->limits.maxFramebufferWidth                        = 0;
    p->limits.maxFramebufferHeight                       = 0;
    p->limits.maxColorAttachments                        = 0;

    /* ── Memory Properties ── */
    VkPhysicalDeviceMemoryProperties *m = &g_physical_device.mem_props;
    m->memoryTypeCount = 1;
    m->memoryTypes[0].propertyFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT  |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    m->memoryTypes[0].heapIndex = 0;
    m->memoryHeapCount = 1;
    m->memoryHeaps[0].size  = SYNTHGPU_VRAM_BYTES;
    m->memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;

    /* ── Features — compute-focused ── */
    VkPhysicalDeviceFeatures *f = &g_physical_device.features;
    f->shaderInt64                             = VK_TRUE;
    f->shaderFloat64                           = VK_TRUE;
    f->shaderInt16                             = VK_TRUE;
    f->robustBufferAccess                      = VK_TRUE;
    f->fragmentStoresAndAtomics                = VK_FALSE;
    f->vertexPipelineStoresAndAtomics          = VK_FALSE;
    f->shaderStorageBufferArrayDynamicIndexing = VK_TRUE;
    f->shaderUniformBufferArrayDynamicIndexing = VK_TRUE;

    g_physical_device_initialized = 1;
    fprintf(stderr, "[SynthGPU Vulkan] Physical device initialized: %s\n", p->deviceName);
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_EnumeratePhysicalDevices(
        VkInstance instance, uint32_t *pPhysicalDeviceCount,
        VkPhysicalDevice *pPhysicalDevices) {
    SynthGPU_Instance_T *inst = (SynthGPU_Instance_T*)instance;
    init_physical_device(inst);

    if (!pPhysicalDevices) {
        *pPhysicalDeviceCount = 1;
        return VK_SUCCESS;
    }
    if (*pPhysicalDeviceCount < 1) {
        *pPhysicalDeviceCount = 1;
        return VK_INCOMPLETE;
    }
    pPhysicalDevices[0] = (VkPhysicalDevice)&g_physical_device;
    *pPhysicalDeviceCount = 1;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceProperties(
        VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties *pProperties) {
    *pProperties = ((SynthGPU_PhysicalDevice_T*)physicalDevice)->props;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceProperties2(
        VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2 *pProperties) {
    pProperties->properties = ((SynthGPU_PhysicalDevice_T*)physicalDevice)->props;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceFeatures(
        VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures *pFeatures) {
    *pFeatures = ((SynthGPU_PhysicalDevice_T*)physicalDevice)->features;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceFeatures2(
        VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2 *pFeatures) {
    pFeatures->features = ((SynthGPU_PhysicalDevice_T*)physicalDevice)->features;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceMemoryProperties(
        VkPhysicalDevice physicalDevice,
        VkPhysicalDeviceMemoryProperties *pMemProperties) {
    *pMemProperties = ((SynthGPU_PhysicalDevice_T*)physicalDevice)->mem_props;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceMemoryProperties2(
        VkPhysicalDevice physicalDevice,
        VkPhysicalDeviceMemoryProperties2 *pMemProperties) {
    pMemProperties->memoryProperties =
        ((SynthGPU_PhysicalDevice_T*)physicalDevice)->mem_props;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceQueueFamilyProperties(
        VkPhysicalDevice physicalDevice, uint32_t *pQueueFamilyPropertyCount,
        VkQueueFamilyProperties *pQueueFamilyProperties) {
    (void)physicalDevice;
    if (!pQueueFamilyProperties) {
        *pQueueFamilyPropertyCount = 1;
        return;
    }
    if (*pQueueFamilyPropertyCount < 1) return;
    /* One queue family: compute + transfer only (no graphics, no presentation) */
    pQueueFamilyProperties[0].queueFlags =
        VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    pQueueFamilyProperties[0].queueCount = 4;
    pQueueFamilyProperties[0].timestampValidBits = 0;
    pQueueFamilyProperties[0].minImageTransferGranularity = (VkExtent3D){1,1,1};
    *pQueueFamilyPropertyCount = 1;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceQueueFamilyProperties2(
        VkPhysicalDevice physicalDevice, uint32_t *pQueueFamilyPropertyCount,
        VkQueueFamilyProperties2 *pQueueFamilyProperties) {
    if (!pQueueFamilyProperties) { *pQueueFamilyPropertyCount = 1; return; }
    VkQueueFamilyProperties props;
    uint32_t count = 1;
    synthgpu_GetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, &props);
    pQueueFamilyProperties[0].queueFamilyProperties = props;
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_EnumerateDeviceExtensionProperties(
        VkPhysicalDevice physicalDevice, const char *pLayerName,
        uint32_t *pPropertyCount, VkExtensionProperties *pProperties) {
    (void)physicalDevice; (void)pLayerName;
    static const VkExtensionProperties exts[] = {
        {"VK_KHR_storage_buffer_storage_class", 1},
        {"VK_KHR_variable_pointers",            1},
        {"VK_KHR_shader_float16_int8",          1},
        {"VK_KHR_16bit_storage",                1},
        {"VK_EXT_memory_budget",                1},
    };
    uint32_t count = (uint32_t)(sizeof(exts) / sizeof(exts[0]));
    if (!pProperties) { *pPropertyCount = count; return VK_SUCCESS; }
    uint32_t copy = (*pPropertyCount < count) ? *pPropertyCount : count;
    memcpy(pProperties, exts, copy * sizeof(VkExtensionProperties));
    *pPropertyCount = copy;
    return (copy < count) ? VK_INCOMPLETE : VK_SUCCESS;
}
