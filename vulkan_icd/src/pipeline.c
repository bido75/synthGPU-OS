/*
 * SynthGPU Vulkan ICD — Pipeline + Shader Modules + Descriptors
 * Ingests SPIR-V bytecode, extracts LocalSize, and builds compute pipelines.
 */
#include "synthgpu_vk.h"

#define SPIRV_MAGIC                       0x07230203u
#define SPIRV_OP_EXECUTION_MODE           16u
#define SPIRV_EXECUTION_MODE_LOCAL_SIZE   17u

static void extract_local_size(const uint32_t *code, size_t word_count,
                                 uint32_t *lx, uint32_t *ly, uint32_t *lz) {
    *lx = 1; *ly = 1; *lz = 1;
    if (word_count < 5 || code[0] != SPIRV_MAGIC) return;

    size_t i = 5;
    while (i < word_count) {
        uint32_t w        = code[i];
        uint32_t opcode   = w & 0xFFFFu;
        uint32_t word_len = (w >> 16) & 0xFFFFu;
        if (word_len == 0 || i + word_len > word_count) break;

        if (opcode == SPIRV_OP_EXECUTION_MODE && word_len >= 6) {
            if (code[i + 2] == SPIRV_EXECUTION_MODE_LOCAL_SIZE) {
                *lx = code[i + 3];
                *ly = code[i + 4];
                *lz = code[i + 5];
                return;
            }
        }
        i += word_len;
    }
}

/* ── Shader Modules ──────────────────────────────────────────────────── */

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateShaderModule(
        VkDevice device, const VkShaderModuleCreateInfo *pCreateInfo,
        const VkAllocationCallbacks *pAllocator, VkShaderModule *pShaderModule) {
    (void)device; (void)pAllocator;

    SynthGPU_ShaderModule_T *sm =
        (SynthGPU_ShaderModule_T*)SYNTHGPU_ALLOC(sizeof(*sm));
    if (!sm) return VK_ERROR_OUT_OF_HOST_MEMORY;

    sm->spirv_word_count = pCreateInfo->codeSize / 4;
    sm->spirv_code       = (uint32_t*)malloc(pCreateInfo->codeSize);
    if (!sm->spirv_code) { SYNTHGPU_FREE(sm); return VK_ERROR_OUT_OF_HOST_MEMORY; }
    memcpy(sm->spirv_code, pCreateInfo->pCode, pCreateInfo->codeSize);

    *pShaderModule = (VkShaderModule)(uintptr_t)sm;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyShaderModule(
        VkDevice device, VkShaderModule shaderModule,
        const VkAllocationCallbacks *pAllocator) {
    (void)device; (void)pAllocator;
    if (shaderModule == VK_NULL_HANDLE) return;
    SynthGPU_ShaderModule_T *sm = (SynthGPU_ShaderModule_T*)(uintptr_t)shaderModule;
    if (sm->spirv_code) free(sm->spirv_code);
    SYNTHGPU_FREE(sm);
}

/* ── Pipeline Layout ─────────────────────────────────────────────────── */

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreatePipelineLayout(
        VkDevice device, const VkPipelineLayoutCreateInfo *pCreateInfo,
        const VkAllocationCallbacks *pAllocator, VkPipelineLayout *pPipelineLayout) {
    (void)device; (void)pAllocator;
    SynthGPU_PipelineLayout_T *pl =
        (SynthGPU_PipelineLayout_T*)SYNTHGPU_ALLOC(sizeof(*pl));
    if (!pl) return VK_ERROR_OUT_OF_HOST_MEMORY;
    pl->set_layout_count = pCreateInfo->setLayoutCount;
    *pPipelineLayout = (VkPipelineLayout)(uintptr_t)pl;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyPipelineLayout(
        VkDevice device, VkPipelineLayout pipelineLayout,
        const VkAllocationCallbacks *pAllocator) {
    (void)device; (void)pAllocator;
    if (pipelineLayout != VK_NULL_HANDLE)
        SYNTHGPU_FREE((void*)(uintptr_t)pipelineLayout);
}

/* ── Compute Pipelines ───────────────────────────────────────────────── */

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateComputePipelines(
        VkDevice device, VkPipelineCache pipelineCache,
        uint32_t createInfoCount,
        const VkComputePipelineCreateInfo *pCreateInfos,
        const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines) {
    (void)device; (void)pipelineCache; (void)pAllocator;

    for (uint32_t i = 0; i < createInfoCount; i++) {
        SynthGPU_Pipeline_T *pipeline =
            (SynthGPU_Pipeline_T*)SYNTHGPU_ALLOC(sizeof(*pipeline));
        if (!pipeline) return VK_ERROR_OUT_OF_HOST_MEMORY;

        SynthGPU_ShaderModule_T *sm =
            (SynthGPU_ShaderModule_T*)(uintptr_t)pCreateInfos[i].stage.module;

        /* Deep-copy SPIR-V so pipeline owns its own copy */
        pipeline->spirv_word_count = sm->spirv_word_count;
        pipeline->spirv_code = (uint32_t*)malloc(sm->spirv_word_count * 4);
        if (!pipeline->spirv_code) {
            SYNTHGPU_FREE(pipeline);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        memcpy(pipeline->spirv_code, sm->spirv_code, sm->spirv_word_count * 4);

        /* Entry point name */
        const char *ep = pCreateInfos[i].stage.pName;
        strncpy(pipeline->entry_point, ep ? ep : "main",
                sizeof(pipeline->entry_point) - 1);

        /* Extract LocalSize from SPIR-V */
        extract_local_size(pipeline->spirv_code, pipeline->spirv_word_count,
                           &pipeline->local_size_x,
                           &pipeline->local_size_y,
                           &pipeline->local_size_z);

        pPipelines[i] = (VkPipeline)(uintptr_t)pipeline;
        fprintf(stderr, "[SynthGPU Vulkan] Pipeline created: LocalSize(%u,%u,%u)\n",
                pipeline->local_size_x, pipeline->local_size_y, pipeline->local_size_z);
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyPipeline(
        VkDevice device, VkPipeline pipeline,
        const VkAllocationCallbacks *pAllocator) {
    (void)device; (void)pAllocator;
    if (pipeline == VK_NULL_HANDLE) return;
    SynthGPU_Pipeline_T *p = (SynthGPU_Pipeline_T*)(uintptr_t)pipeline;
    if (p->spirv_code) free(p->spirv_code);
    SYNTHGPU_FREE(p);
}

/* ── Pipeline Cache (stubs) ──────────────────────────────────────────── */

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreatePipelineCache(
        VkDevice device, const VkPipelineCacheCreateInfo *pCreateInfo,
        const VkAllocationCallbacks *pAllocator, VkPipelineCache *pPipelineCache) {
    (void)device; (void)pCreateInfo; (void)pAllocator;
    SynthGPU_PipelineCache_T *pc =
        (SynthGPU_PipelineCache_T*)SYNTHGPU_ALLOC(sizeof(*pc));
    if (!pc) return VK_ERROR_OUT_OF_HOST_MEMORY;
    *pPipelineCache = (VkPipelineCache)(uintptr_t)pc;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyPipelineCache(
        VkDevice device, VkPipelineCache pipelineCache,
        const VkAllocationCallbacks *pAllocator) {
    (void)device; (void)pAllocator;
    if (pipelineCache != VK_NULL_HANDLE)
        SYNTHGPU_FREE((void*)(uintptr_t)pipelineCache);
}

/* ── Descriptor Sets ─────────────────────────────────────────────────── */

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateDescriptorSetLayout(
        VkDevice device, const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
        const VkAllocationCallbacks *pAllocator,
        VkDescriptorSetLayout *pSetLayout) {
    (void)device; (void)pAllocator;

    SynthGPU_DescriptorSetLayout_T *dsl =
        (SynthGPU_DescriptorSetLayout_T*)SYNTHGPU_ALLOC(sizeof(*dsl));
    if (!dsl) return VK_ERROR_OUT_OF_HOST_MEMORY;

    dsl->binding_count = pCreateInfo->bindingCount;
    if (pCreateInfo->bindingCount > 0) {
        size_t sz = pCreateInfo->bindingCount * sizeof(VkDescriptorSetLayoutBinding);
        dsl->bindings = (VkDescriptorSetLayoutBinding*)malloc(sz);
        if (!dsl->bindings) { SYNTHGPU_FREE(dsl); return VK_ERROR_OUT_OF_HOST_MEMORY; }
        memcpy(dsl->bindings, pCreateInfo->pBindings, sz);
    }

    *pSetLayout = (VkDescriptorSetLayout)(uintptr_t)dsl;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyDescriptorSetLayout(
        VkDevice device, VkDescriptorSetLayout descriptorSetLayout,
        const VkAllocationCallbacks *pAllocator) {
    (void)device; (void)pAllocator;
    if (descriptorSetLayout == VK_NULL_HANDLE) return;
    SynthGPU_DescriptorSetLayout_T *dsl =
        (SynthGPU_DescriptorSetLayout_T*)(uintptr_t)descriptorSetLayout;
    if (dsl->bindings) free(dsl->bindings);
    SYNTHGPU_FREE(dsl);
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_CreateDescriptorPool(
        VkDevice device, const VkDescriptorPoolCreateInfo *pCreateInfo,
        const VkAllocationCallbacks *pAllocator, VkDescriptorPool *pDescriptorPool) {
    (void)device; (void)pAllocator;
    SynthGPU_DescriptorPool_T *pool =
        (SynthGPU_DescriptorPool_T*)SYNTHGPU_ALLOC(sizeof(*pool));
    if (!pool) return VK_ERROR_OUT_OF_HOST_MEMORY;
    pool->max_sets  = pCreateInfo->maxSets;
    pool->allocated = 0;
    *pDescriptorPool = (VkDescriptorPool)(uintptr_t)pool;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_DestroyDescriptorPool(
        VkDevice device, VkDescriptorPool descriptorPool,
        const VkAllocationCallbacks *pAllocator) {
    (void)device; (void)pAllocator;
    if (descriptorPool != VK_NULL_HANDLE)
        SYNTHGPU_FREE((void*)(uintptr_t)descriptorPool);
}

VKAPI_ATTR VkResult VKAPI_CALL synthgpu_AllocateDescriptorSets(
        VkDevice device, const VkDescriptorSetAllocateInfo *pAllocateInfo,
        VkDescriptorSet *pDescriptorSets) {
    (void)device;
    for (uint32_t i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
        SynthGPU_DescriptorSet_T *ds =
            (SynthGPU_DescriptorSet_T*)SYNTHGPU_ALLOC(sizeof(*ds));
        if (!ds) return VK_ERROR_OUT_OF_HOST_MEMORY;
        ds->layout = (SynthGPU_DescriptorSetLayout_T*)
                     (uintptr_t)pAllocateInfo->pSetLayouts[i];
        pDescriptorSets[i] = (VkDescriptorSet)(uintptr_t)ds;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL synthgpu_UpdateDescriptorSets(
        VkDevice device, uint32_t descriptorWriteCount,
        const VkWriteDescriptorSet *pDescriptorWrites,
        uint32_t descriptorCopyCount,
        const VkCopyDescriptorSet *pDescriptorCopies) {
    (void)device; (void)descriptorCopyCount; (void)pDescriptorCopies;

    for (uint32_t i = 0; i < descriptorWriteCount; i++) {
        const VkWriteDescriptorSet *w = &pDescriptorWrites[i];
        SynthGPU_DescriptorSet_T *ds =
            (SynthGPU_DescriptorSet_T*)(uintptr_t)w->dstSet;
        if (!ds) continue;

        uint32_t binding = w->dstBinding;
        if (binding >= 16) continue;

        if (w->descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
            w->descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
            w->descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC) {
            for (uint32_t j = 0; j < w->descriptorCount; j++) {
                uint32_t b = binding + j;
                if (b >= 16) break;
                const VkDescriptorBufferInfo *bi = &w->pBufferInfo[j];
                ds->bound_buffers[b] =
                    (SynthGPU_Buffer_T*)(uintptr_t)bi->buffer;
                ds->bound_offsets[b] = bi->offset;
                ds->bound_ranges[b]  =
                    (bi->range == VK_WHOLE_SIZE)
                    ? ((SynthGPU_Buffer_T*)(uintptr_t)bi->buffer)->size
                    : bi->range;
            }
        }
    }
}
