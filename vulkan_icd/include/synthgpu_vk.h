/*
 * SynthGPU Vulkan ICD — Internal Header
 * Defines device state, memory management, and dispatch tables
 */
#pragma once

#ifdef _WIN32
  #define VK_USE_PLATFORM_WIN32_KHR
  #define SYNTHGPU_EXPORT __declspec(dllexport)
#else
  #define VK_USE_PLATFORM_XLIB_KHR
  #define SYNTHGPU_EXPORT __attribute__((visibility("default")))
#endif

#include <vulkan/vulkan.h>
#include <vulkan/vk_icd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Version ─────────────────────────────────────────────────────────── */
#define SYNTHGPU_ICD_VERSION_MAJOR 0
#define SYNTHGPU_ICD_VERSION_MINOR 3
#define SYNTHGPU_ICD_VERSION_PATCH 0
#define SYNTHGPU_DEVICE_NAME       "SynthGPU Virtual Accelerator v0.3"
#define SYNTHGPU_VENDOR_ID         0x5347  /* 'SG' in hex */
#define SYNTHGPU_DEVICE_ID         0x0003
#define SYNTHGPU_VRAM_BYTES        (128ULL * 1024 * 1024)  /* 128MB virtual */

/* ── Internal Object Types ───────────────────────────────────────────── */

typedef struct SynthGPU_Instance_T {
    VK_LOADER_DATA  loader_data;   /* MUST be first — Vulkan Loader requirement */
    VkAllocationCallbacks alloc;
    uint32_t        api_version;
    int             debug_enabled;
} SynthGPU_Instance_T;

typedef struct SynthGPU_PhysicalDevice_T {
    VK_LOADER_DATA       loader_data;  /* MUST be first */
    SynthGPU_Instance_T *instance;
    VkPhysicalDeviceProperties    props;
    VkPhysicalDeviceMemoryProperties mem_props;
    VkPhysicalDeviceFeatures      features;
} SynthGPU_PhysicalDevice_T;

typedef struct SynthGPU_Device_T {
    VK_LOADER_DATA              loader_data;  /* MUST be first */
    SynthGPU_PhysicalDevice_T  *physical_device;
    VkAllocationCallbacks       alloc;
    /* Memory heap: system RAM backing our virtual VRAM */
    void    *vram_pool;
    size_t   vram_pool_size;
    size_t   vram_allocated;
} SynthGPU_Device_T;

typedef struct SynthGPU_Queue_T {
    VK_LOADER_DATA      loader_data;
    SynthGPU_Device_T  *device;
    uint32_t            family_index;
    uint32_t            queue_index;
} SynthGPU_Queue_T;

typedef struct SynthGPU_DeviceMemory_T {
    void    *ptr;         /* Actual system RAM pointer */
    size_t   size;
    uint32_t memory_type_index;
    int      mapped;
} SynthGPU_DeviceMemory_T;

typedef struct SynthGPU_Buffer_T {
    VkDeviceSize        size;
    VkBufferUsageFlags  usage;
    SynthGPU_DeviceMemory_T *bound_memory;
    VkDeviceSize        bind_offset;
} SynthGPU_Buffer_T;

/* SPIR-V shader module */
typedef struct SynthGPU_ShaderModule_T {
    uint32_t *spirv_code;
    size_t    spirv_word_count;
} SynthGPU_ShaderModule_T;

/* Compute pipeline — stores SPIR-V for dispatch-time execution */
typedef struct SynthGPU_Pipeline_T {
    uint32_t *spirv_code;
    size_t    spirv_word_count;
    char      entry_point[64];
    /* Workgroup dimensions extracted from SPIR-V LocalSize decoration */
    uint32_t  local_size_x;
    uint32_t  local_size_y;
    uint32_t  local_size_z;
} SynthGPU_Pipeline_T;

typedef struct SynthGPU_DescriptorSetLayout_T {
    uint32_t binding_count;
    VkDescriptorSetLayoutBinding *bindings;
} SynthGPU_DescriptorSetLayout_T;

typedef struct SynthGPU_DescriptorSet_T {
    SynthGPU_DescriptorSetLayout_T *layout;
    /* Bound buffer per binding slot */
    SynthGPU_Buffer_T  *bound_buffers[16];
    VkDeviceSize        bound_offsets[16];
    VkDeviceSize        bound_ranges[16];
} SynthGPU_DescriptorSet_T;

typedef struct SynthGPU_DescriptorPool_T {
    SynthGPU_Device_T *device;
    uint32_t           max_sets;
    uint32_t           allocated;
} SynthGPU_DescriptorPool_T;

typedef struct SynthGPU_PipelineLayout_T {
    uint32_t set_layout_count;
} SynthGPU_PipelineLayout_T;

typedef struct SynthGPU_PipelineCache_T {
    uint32_t dummy;
} SynthGPU_PipelineCache_T;

/* Command types recorded into command buffers */
typedef enum SynthGPU_CmdType {
    SYNTHGPU_CMD_DISPATCH         = 1,
    SYNTHGPU_CMD_COPY_BUFFER      = 2,
    SYNTHGPU_CMD_PIPELINE_BARRIER = 3,
    SYNTHGPU_CMD_FILL_BUFFER      = 4,
} SynthGPU_CmdType;

typedef struct SynthGPU_Cmd_T {
    SynthGPU_CmdType type;
    union {
        struct {
            SynthGPU_Pipeline_T      *pipeline;
            SynthGPU_DescriptorSet_T *descriptor_sets[4];
            uint32_t                  desc_set_count;
            uint32_t                  group_count_x;
            uint32_t                  group_count_y;
            uint32_t                  group_count_z;
        } dispatch;
        struct {
            SynthGPU_Buffer_T *src;
            SynthGPU_Buffer_T *dst;
            VkDeviceSize       src_offset;
            VkDeviceSize       dst_offset;
            VkDeviceSize       size;
        } copy_buffer;
        struct {
            SynthGPU_Buffer_T *dst;
            VkDeviceSize       offset;
            VkDeviceSize       size;
            uint32_t           data;
        } fill_buffer;
    };
    struct SynthGPU_Cmd_T *next;
} SynthGPU_Cmd_T;

typedef struct SynthGPU_CommandBuffer_T {
    VK_LOADER_DATA      loader_data;
    SynthGPU_Device_T  *device;
    SynthGPU_Cmd_T     *cmd_head;
    SynthGPU_Cmd_T     *cmd_tail;
    uint32_t            cmd_count;
    SynthGPU_Pipeline_T       *bound_pipeline;
    SynthGPU_DescriptorSet_T  *bound_desc_sets[4];
    uint32_t                   bound_desc_count;
} SynthGPU_CommandBuffer_T;

typedef struct SynthGPU_CommandPool_T {
    SynthGPU_Device_T *device;
    uint32_t           queue_family_index;
} SynthGPU_CommandPool_T;

typedef struct SynthGPU_Fence_T {
    volatile int signaled;
} SynthGPU_Fence_T;

typedef struct SynthGPU_Semaphore_T {
    volatile int signaled;
} SynthGPU_Semaphore_T;

/* ── Helper Macros ───────────────────────────────────────────────────── */
#define SYNTHGPU_ALLOC(size) calloc(1, size)
#define SYNTHGPU_FREE(ptr)   free(ptr)

#define SET_LOADER_MAGIC(obj) \
    set_loader_magic_value((void*)(obj))

/* ICD proc addr export */
SYNTHGPU_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance instance, const char *pName);

SYNTHGPU_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetPhysicalDeviceProcAddr(VkInstance instance, const char *pName);
