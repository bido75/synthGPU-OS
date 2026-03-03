cmake_minimum_required(VERSION 3.20)
project(SynthGPU_Vulkan_ICD VERSION 0.3.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)

# ── Find dependencies ──────────────────────────────────────────────────
find_package(Vulkan REQUIRED)
find_package(Python3 COMPONENTS Development)

# ── Sources ────────────────────────────────────────────────────────────
set(ICD_SOURCES
    src/icd_main.c
    src/instance.c
    src/physical_device.c
    src/device.c
    src/memory.c
    src/pipeline.c
    src/commands.c
    src/queue.c
    src/spirv_dispatch.c
    bridge/py_bridge.c
)

# ── Shared library (the actual ICD .dll / .so) ─────────────────────────
add_library(synthgpu_vulkan_icd SHARED ${ICD_SOURCES})

target_include_directories(synthgpu_vulkan_icd PRIVATE
    include/
    ${Vulkan_INCLUDE_DIRS}
)

if(Python3_FOUND)
    target_include_directories(synthgpu_vulkan_icd PRIVATE ${Python3_INCLUDE_DIRS})
    target_link_libraries(synthgpu_vulkan_icd PRIVATE ${Python3_LIBRARIES})
    target_compile_definitions(synthgpu_vulkan_icd PRIVATE SYNTHGPU_HAS_PYTHON=1)
endif()

# ── Platform-specific settings ─────────────────────────────────────────
if(WIN32)
    target_compile_definitions(synthgpu_vulkan_icd PRIVATE
        VK_USE_PLATFORM_WIN32_KHR
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _CRT_SECURE_NO_WARNINGS
    )
    set_target_properties(synthgpu_vulkan_icd PROPERTIES
        OUTPUT_NAME "synthgpu_vulkan_icd"
        PREFIX      ""
        SUFFIX      ".dll"
    )
    target_link_libraries(synthgpu_vulkan_icd PRIVATE Vulkan::Vulkan)
else()
    target_compile_definitions(synthgpu_vulkan_icd PRIVATE
        VK_USE_PLATFORM_XLIB_KHR
    )
    set_target_properties(synthgpu_vulkan_icd PROPERTIES
        OUTPUT_NAME  "synthgpu_vulkan_icd"
        PREFIX       "lib"
        SUFFIX       ".so"
        C_VISIBILITY_PRESET hidden
    )
    target_compile_options(synthgpu_vulkan_icd PRIVATE -fvisibility=hidden)
    target_link_libraries(synthgpu_vulkan_icd PRIVATE pthread dl)
endif()

# ── Installation ───────────────────────────────────────────────────────
if(WIN32)
    install(TARGETS synthgpu_vulkan_icd DESTINATION bin)
    install(FILES manifests/synthgpu_icd_win64.json DESTINATION bin)
else()
    install(TARGETS synthgpu_vulkan_icd DESTINATION /usr/local/lib/synthgpu)
    install(FILES manifests/synthgpu_icd_linux.json
            DESTINATION /etc/vulkan/icd.d
            RENAME synthgpu_icd.json)
endif()

# ── Custom test targets ────────────────────────────────────────────────
find_program(PYTHON_EXEC python3 python)
if(PYTHON_EXEC)
    add_custom_target(test_enumeration
        COMMAND ${PYTHON_EXEC} ${CMAKE_SOURCE_DIR}/tests/test_enumeration.py
        DEPENDS synthgpu_vulkan_icd
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running Vulkan enumeration test"
    )
    add_custom_target(test_compute
        COMMAND ${PYTHON_EXEC} ${CMAKE_SOURCE_DIR}/tests/test_compute.py
        DEPENDS synthgpu_vulkan_icd
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running Vulkan compute test"
    )
endif()

message(STATUS "SynthGPU Vulkan ICD v0.3.0 — configured")
message(STATUS "  Vulkan headers: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "  Python found:   ${Python3_FOUND}")
if(WIN32)
    message(STATUS "  Output: synthgpu_vulkan_icd.dll")
    message(STATUS "  Install: scripts/install_windows.bat")
else()
    message(STATUS "  Output: libsynthgpu_vulkan_icd.so")
    message(STATUS "  Install: sudo scripts/install_linux.sh")
endif()
