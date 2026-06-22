cmake_minimum_required(VERSION 3.15)
project(synthgpu_cuda VERSION 0.3.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ── Output name ──────────────────────────────────────────────────────
# Produces libsynthgpu_cuda.so (Linux/macOS) or SynthGPUCUDA.dll (Windows)
set(SHIM_TARGET synthgpu_cuda)

# ── Source files ─────────────────────────────────────────────────────
set(SHIM_SOURCES
    src/shim.c
    src/memory.c
    src/stream.c
    src/event.c
    src/bridge.c
    src/telemetry.c
    src/device_info.c
    src/cublas.c
    src/cudnn.c
    src/cuda_streams.c
    src/virtual_vram.c
    src/python_bridge.c
)

# ── Shared library ───────────────────────────────────────────────────
add_library(${SHIM_TARGET} SHARED ${SHIM_SOURCES})

target_include_directories(${SHIM_TARGET}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(UNIX AND NOT APPLE)
    add_executable(synthgpu_cuda_demo tests/demo_client.c)
    target_include_directories(synthgpu_cuda_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(synthgpu_cuda_demo PRIVATE -Wall -Wextra -Werror)
endif()

# ── Platform flags ───────────────────────────────────────────────────
if(UNIX AND NOT APPLE)
    target_compile_options(${SHIM_TARGET} PRIVATE -O3 -march=native -fPIC -Wall -Wextra)
    target_link_options(${SHIM_TARGET} PRIVATE -Wl,--no-undefined)
elseif(APPLE)
    target_compile_options(${SHIM_TARGET} PRIVATE -O3 -fPIC -Wall -Wextra)
elseif(WIN32)
    target_compile_options(${SHIM_TARGET} PRIVATE /O2 /W3)
    target_compile_definitions(${SHIM_TARGET} PRIVATE _CRT_SECURE_NO_WARNINGS WIN32_LEAN_AND_MEAN)
endif()

# ── Python (embedding) ───────────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
target_include_directories(${SHIM_TARGET} PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(${SHIM_TARGET} PRIVATE ${Python3_LIBRARIES})
message(STATUS "Python3 include: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 libs:    ${Python3_LIBRARIES}")

# ── OpenBLAS ─────────────────────────────────────────────────────────
find_package(BLAS QUIET)
if(BLAS_FOUND)
    target_link_libraries(${SHIM_TARGET} PRIVATE ${BLAS_LIBRARIES})
    message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
else()
    # Fallback: try to find OpenBLAS directly
    find_library(OPENBLAS_LIB
        NAMES openblas blas
        HINTS
            /usr/lib
            /usr/lib/x86_64-linux-gnu
            /usr/local/lib
            /opt/OpenBLAS/lib
            $ENV{OPENBLAS_ROOT}/lib
    )
    if(OPENBLAS_LIB)
        target_link_libraries(${SHIM_TARGET} PRIVATE ${OPENBLAS_LIB})
        message(STATUS "OpenBLAS found at: ${OPENBLAS_LIB}")
    else()
        message(WARNING "OpenBLAS not found. Matrix ops will be stub-only.")
        target_compile_definitions(${SHIM_TARGET} PRIVATE SYNTHGPU_NO_BLAS)
    endif()
endif()

# ── System libraries ─────────────────────────────────────────────────
if(UNIX)
    target_link_libraries(${SHIM_TARGET} PRIVATE dl m pthread)
endif()

# ── Installation ─────────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS ${SHIM_TARGET}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
if(TARGET synthgpu_cuda_demo)
    install(TARGETS synthgpu_cuda_demo RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/synthgpu)

# ── Summary ──────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "SynthGPU CUDA Shim v${PROJECT_VERSION}")
message(STATUS "  Sources : ${SHIM_SOURCES}")
message(STATUS "  Output  : lib${SHIM_TARGET}.so / ${SHIM_TARGET}.dll")
message(STATUS "")
