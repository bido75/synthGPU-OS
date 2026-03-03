/*
 * SynthGPU Vulkan ICD — Entry Point
 * The Vulkan Loader calls vk_icdGetInstanceProcAddr to discover all functions.
 * We implement a dispatch table mapping every Vulkan function name to our impl.
 */
#include "synthgpu_vk.h"

/* ── Forward declarations ────────────────────────────────────────────── */

/* Instance */
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateInstance(
    const VkInstanceCreateInfo*, const VkAllocationCallbacks*, VkInstance*);
VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyInstance(
    VkInstance, const VkAllocationCallbacks*);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_EnumerateInstanceExtensionProperties(
    const char*, uint32_t*, VkExtensionProperties*);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_EnumerateInstanceLayerProperties(
    uint32_t*, VkLayerProperties*);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_EnumerateInstanceVersion(uint32_t*);

/* Physical device */
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_EnumeratePhysicalDevices(
    VkInstance, uint32_t*, VkPhysicalDevice*);
VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceProperties(
    VkPhysicalDevice, VkPhysicalDeviceProperties*);
VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceProperties2(
    VkPhysicalDevice, VkPhysicalDeviceProperties2*);
VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceFeatures(
    VkPhysicalDevice, VkPhysicalDeviceFeatures*);
VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceFeatures2(
    VkPhysicalDevice, VkPhysicalDeviceFeatures2*);
VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceMemoryProperties(
    VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*);
VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceMemoryProperties2(
    VkPhysicalDevice, VkPhysicalDeviceMemoryProperties2*);
VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceQueueFamilyProperties(
    VkPhysicalDevice, uint32_t*, VkQueueFamilyProperties*);
VKAPI_ATTR void VKAPI_CALL synthgpu_GetPhysicalDeviceQueueFamilyProperties2(
    VkPhysicalDevice, uint32_t*, VkQueueFamilyProperties2*);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_EnumerateDeviceExtensionProperties(
    VkPhysicalDevice, const char*, uint32_t*, VkExtensionProperties*);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_EnumerateDeviceLayerProperties(
    VkPhysicalDevice, uint32_t*, VkLayerProperties*);

/* Logical device */
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateDevice(
    VkPhysicalDevice, const VkDeviceCreateInfo*,
    const VkAllocationCallbacks*, VkDevice*);
VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyDevice(
    VkDevice, const VkAllocationCallbacks*);
VKAPI_ATTR void VKAPI_CALL synthgpu_GetDeviceQueue(
    VkDevice, uint32_t, uint32_t, VkQueue*);
VKAPI_ATTR void VKAPI_CALL synthgpu_GetDeviceQueue2(
    VkDevice, const VkDeviceQueueInfo2*, VkQueue*);
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL synthgpu_GetDeviceProcAddr(
    VkDevice, const char*);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_DeviceWaitIdle(VkDevice);

/* Memory */
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_AllocateMemory(
    VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*, VkDeviceMemory*);
VKAPI_ATTR void VKAPI_CALL synthgpu_FreeMemory(
    VkDevice, VkDeviceMemory, const VkAllocationCallbacks*);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_MapMemory(
    VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags, void**);
VKAPI_ATTR void VKAPI_CALL synthgpu_UnmapMemory(VkDevice, VkDeviceMemory);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_FlushMappedMemoryRanges(
    VkDevice, uint32_t, const VkMappedMemoryRange*);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_InvalidateMappedMemoryRanges(
    VkDevice, uint32_t, const VkMappedMemoryRange*);

/* Buffers */
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateBuffer(
    VkDevice, const VkBufferCreateInfo*, const VkAllocationCallbacks*, VkBuffer*);
VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyBuffer(
    VkDevice, VkBuffer, const VkAllocationCallbacks*);
VKAPI_ATTR void VKAPI_CALL synthgpu_GetBufferMemoryRequirements(
    VkDevice, VkBuffer, VkMemoryRequirements*);
VKAPI_ATTR void VKAPI_CALL synthgpu_GetBufferMemoryRequirements2(
    VkDevice, const VkBufferMemoryRequirementsInfo2*, VkMemoryRequirements2*);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_BindBufferMemory(
    VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_BindBufferMemory2(
    VkDevice, uint32_t, const VkBindBufferMemoryInfo*);

/* Shaders & Pipelines */
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateShaderModule(
    VkDevice, const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*, VkShaderModule*);
VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyShaderModule(
    VkDevice, VkShaderModule, const VkAllocationCallbacks*);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreatePipelineLayout(
    VkDevice, const VkPipelineLayoutCreateInfo*, const VkAllocationCallbacks*, VkPipelineLayout*);
VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyPipelineLayout(
    VkDevice, VkPipelineLayout, const VkAllocationCallbacks*);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateComputePipelines(
    VkDevice, VkPipelineCache, uint32_t,
    const VkComputePipelineCreateInfo*, const VkAllocationCallbacks*, VkPipeline*);
VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyPipeline(
    VkDevice, VkPipeline, const VkAllocationCallbacks*);

/* Descriptors */
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateDescriptorSetLayout(
    VkDevice, const VkDescriptorSetLayoutCreateInfo*,
    const VkAllocationCallbacks*, VkDescriptorSetLayout*);
VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyDescriptorSetLayout(
    VkDevice, VkDescriptorSetLayout, const VkAllocationCallbacks*);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateDescriptorPool(
    VkDevice, const VkDescriptorPoolCreateInfo*, const VkAllocationCallbacks*, VkDescriptorPool*);
VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyDescriptorPool(
    VkDevice, VkDescriptorPool, const VkAllocationCallbacks*);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_AllocateDescriptorSets(
    VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet*);
VKAPI_ATTR void VKAPI_CALL synthgpu_UpdateDescriptorSets(
    VkDevice, uint32_t, const VkWriteDescriptorSet*, uint32_t, const VkCopyDescriptorSet*);

/* Commands */
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateCommandPool(
    VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*, VkCommandPool*);
VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyCommandPool(
    VkDevice, VkCommandPool, const VkAllocationCallbacks*);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_AllocateCommandBuffers(
    VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer*);
VKAPI_ATTR void VKAPI_CALL synthgpu_FreeCommandBuffers(
    VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer*);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_BeginCommandBuffer(
    VkCommandBuffer, const VkCommandBufferBeginInfo*);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_EndCommandBuffer(VkCommandBuffer);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_ResetCommandBuffer(
    VkCommandBuffer, VkCommandBufferResetFlags);
VKAPI_ATTR void VKAPI_CALL synthgpu_CmdBindPipeline(
    VkCommandBuffer, VkPipelineBindPoint, VkPipeline);
VKAPI_ATTR void VKAPI_CALL synthgpu_CmdBindDescriptorSets(
    VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout,
    uint32_t, uint32_t, const VkDescriptorSet*,
    uint32_t, const uint32_t*);
VKAPI_ATTR void VKAPI_CALL synthgpu_CmdDispatch(
    VkCommandBuffer, uint32_t, uint32_t, uint32_t);
VKAPI_ATTR void VKAPI_CALL synthgpu_CmdCopyBuffer(
    VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const VkBufferCopy*);
VKAPI_ATTR void VKAPI_CALL synthgpu_CmdPipelineBarrier(
    VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags,
    VkDependencyFlags, uint32_t, const VkMemoryBarrier*,
    uint32_t, const VkBufferMemoryBarrier*,
    uint32_t, const VkImageMemoryBarrier*);
VKAPI_ATTR void VKAPI_CALL synthgpu_CmdFillBuffer(
    VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, uint32_t);

/* Queue submission */
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_QueueSubmit(
    VkQueue, uint32_t, const VkSubmitInfo*, VkFence);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_QueueSubmit2(
    VkQueue, uint32_t, const VkSubmitInfo2*, VkFence);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_QueueWaitIdle(VkQueue);

/* Sync */
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateFence(
    VkDevice, const VkFenceCreateInfo*, const VkAllocationCallbacks*, VkFence*);
VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyFence(
    VkDevice, VkFence, const VkAllocationCallbacks*);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_WaitForFences(
    VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_ResetFences(VkDevice, uint32_t, const VkFence*);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_GetFenceStatus(VkDevice, VkFence);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateSemaphore(
    VkDevice, const VkSemaphoreCreateInfo*, const VkAllocationCallbacks*, VkSemaphore*);
VKAPI_ATTR void VKAPI_CALL synthgpu_DestroySemaphore(
    VkDevice, VkSemaphore, const VkAllocationCallbacks*);

/* Pipeline cache */
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreatePipelineCache(
    VkDevice, const VkPipelineCacheCreateInfo*, const VkAllocationCallbacks*, VkPipelineCache*);
VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyPipelineCache(
    VkDevice, VkPipelineCache, const VkAllocationCallbacks*);

/* Additional stubs for dispatch table completeness */
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_FreeDescriptorSets(
    VkDevice, VkDescriptorPool, uint32_t, const VkDescriptorSet*);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_ResetCommandPool(
    VkDevice, VkCommandPool, VkCommandPoolResetFlags);
VKAPI_ATTR void VKAPI_CALL synthgpu_CmdDispatchIndirect(
    VkCommandBuffer, VkBuffer, VkDeviceSize);
VKAPI_ATTR void VKAPI_CALL synthgpu_CmdUpdateBuffer(
    VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, const void*);
VKAPI_ATTR void VKAPI_CALL synthgpu_CmdPipelineBarrier2(
    VkCommandBuffer, const VkDependencyInfo*);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_GetPipelineCacheData(
    VkDevice, VkPipelineCache, size_t*, void*);
VKAPI_ATTR VkResult VKAPI_CALL synthgpu_MergePipelineCaches(
    VkDevice, VkPipelineCache, uint32_t, const VkPipelineCache*);

/* ── Dispatch Table ─────────────────────────────────────────────────── */

static PFN_vkVoidFunction get_instance_proc(const char *name) {
    #define PROC(fn) if (strcmp(name, #fn) == 0) return (PFN_vkVoidFunction)synthgpu_##fn;

    PROC(CreateInstance)
    PROC(DestroyInstance)
    PROC(EnumerateInstanceExtensionProperties)
    PROC(EnumerateInstanceLayerProperties)
    PROC(EnumerateInstanceVersion)
    PROC(EnumeratePhysicalDevices)
    PROC(GetPhysicalDeviceProperties)
    PROC(GetPhysicalDeviceProperties2)
    PROC(GetPhysicalDeviceFeatures)
    PROC(GetPhysicalDeviceFeatures2)
    PROC(GetPhysicalDeviceMemoryProperties)
    PROC(GetPhysicalDeviceMemoryProperties2)
    PROC(GetPhysicalDeviceQueueFamilyProperties)
    PROC(GetPhysicalDeviceQueueFamilyProperties2)
    PROC(EnumerateDeviceExtensionProperties)
    PROC(EnumerateDeviceLayerProperties)
    PROC(CreateDevice)
    PROC(DestroyDevice)
    PROC(GetDeviceQueue)
    PROC(GetDeviceQueue2)
    PROC(GetDeviceProcAddr)
    PROC(DeviceWaitIdle)
    PROC(AllocateMemory)
    PROC(FreeMemory)
    PROC(MapMemory)
    PROC(UnmapMemory)
    PROC(FlushMappedMemoryRanges)
    PROC(InvalidateMappedMemoryRanges)
    PROC(CreateBuffer)
    PROC(DestroyBuffer)
    PROC(GetBufferMemoryRequirements)
    PROC(GetBufferMemoryRequirements2)
    PROC(BindBufferMemory)
    PROC(BindBufferMemory2)
    PROC(CreateShaderModule)
    PROC(DestroyShaderModule)
    PROC(CreatePipelineLayout)
    PROC(DestroyPipelineLayout)
    PROC(CreateComputePipelines)
    PROC(DestroyPipeline)
    PROC(CreateDescriptorSetLayout)
    PROC(DestroyDescriptorSetLayout)
    PROC(CreateDescriptorPool)
    PROC(DestroyDescriptorPool)
    PROC(AllocateDescriptorSets)
    PROC(FreeDescriptorSets)
    PROC(UpdateDescriptorSets)
    PROC(CreateCommandPool)
    PROC(DestroyCommandPool)
    PROC(AllocateCommandBuffers)
    PROC(FreeCommandBuffers)
    PROC(BeginCommandBuffer)
    PROC(EndCommandBuffer)
    PROC(ResetCommandBuffer)
    PROC(ResetCommandPool)
    PROC(CmdBindPipeline)
    PROC(CmdBindDescriptorSets)
    PROC(CmdDispatch)
    PROC(CmdDispatchIndirect)
    PROC(CmdCopyBuffer)
    PROC(CmdFillBuffer)
    PROC(CmdUpdateBuffer)
    PROC(CmdPipelineBarrier)
    PROC(CmdPipelineBarrier2)
    PROC(QueueSubmit)
    PROC(QueueSubmit2)
    PROC(QueueWaitIdle)
    PROC(DeviceWaitIdle)
    PROC(CreateFence)
    PROC(DestroyFence)
    PROC(WaitForFences)
    PROC(ResetFences)
    PROC(GetFenceStatus)
    PROC(CreateSemaphore)
    PROC(DestroySemaphore)
    PROC(CreatePipelineCache)
    PROC(DestroyPipelineCache)
    PROC(GetPipelineCacheData)
    PROC(MergePipelineCaches)

    #undef PROC
    return NULL;
}

SYNTHGPU_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance instance, const char *pName) {
    (void)instance;
    return get_instance_proc(pName);
}

SYNTHGPU_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetPhysicalDeviceProcAddr(VkInstance instance, const char *pName) {
    (void)instance;
    return get_instance_proc(pName);
}
