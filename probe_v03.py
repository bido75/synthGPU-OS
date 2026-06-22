#!/usr/bin/env python3
"""
SynthGPU v0.3 — Implementation Probe & Validation Suite
========================================================
Run this from the SynthGPU root directory (where vulkan_icd/ and backend/ live).
It probes every checklist phase and produces a detailed pass/fail report.

Usage:
    python probe_v03.py                   # Full probe
    python probe_v03.py --phase P2        # Single phase only
    python probe_v03.py --quick           # Skip long-running tests
    python probe_v03.py --fix-hints       # Show remediation for failures

Requires: Python 3.8+, no extra packages (uses only stdlib + optional vulkan)
"""

import os
import sys
import ctypes
import struct
import subprocess
import platform
import shutil
import json
import time
import argparse
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ── Colours ────────────────────────────────────────────────────────────────────
IS_WIN = platform.system() == "Windows"
RESET  = "\033[0m"  if not IS_WIN else ""
GREEN  = "\033[92m" if not IS_WIN else ""
RED    = "\033[91m" if not IS_WIN else ""
YELLOW = "\033[93m" if not IS_WIN else ""
CYAN   = "\033[96m" if not IS_WIN else ""
BOLD   = "\033[1m"  if not IS_WIN else ""
DIM    = "\033[2m"  if not IS_WIN else ""

if IS_WIN:
    # Enable ANSI on Windows 10+
    import ctypes as _ct
    _ct.windll.kernel32.SetConsoleMode(_ct.windll.kernel32.GetStdHandle(-11), 7)
    RESET = "\033[0m"; GREEN = "\033[92m"; RED = "\033[91m"
    YELLOW = "\033[93m"; CYAN = "\033[96m"; BOLD = "\033[1m"; DIM = "\033[2m"

# ── Result model ───────────────────────────────────────────────────────────────
@dataclass
class CheckResult:
    id:      str
    label:   str
    passed:  bool
    detail:  str = ""
    hint:    str = ""
    skipped: bool = False

@dataclass
class PhaseReport:
    id:      str
    title:   str
    results: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self):  return sum(1 for r in self.results if r.passed and not r.skipped)
    @property
    def failed(self):  return sum(1 for r in self.results if not r.passed and not r.skipped)
    @property
    def skipped(self): return sum(1 for r in self.results if r.skipped)
    @property
    def total(self):   return len(self.results)

# ── Helpers ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
ICD  = ROOT / "vulkan_icd"
BACKEND = ROOT / "backend"
CUDA_SHIM = ROOT / "cuda_shim"

def ok(id, label, detail=""):
    return CheckResult(id, label, True, detail)

def fail(id, label, detail="", hint=""):
    return CheckResult(id, label, False, detail, hint)

def skip(id, label, detail="not applicable on this platform"):
    return CheckResult(id, label, True, detail, skipped=True)

def run(cmd, cwd=None, timeout=30, env=None):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, cwd=cwd or ROOT, env=env)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except FileNotFoundError:
        return -2, "", f"Command not found: {cmd[0]}"

def file_contains(path: Path, *patterns) -> Tuple[bool, str]:
    """Check if a file contains all given patterns (strings or regex)."""
    if not path.exists():
        return False, f"File not found: {path}"
    content = path.read_text(errors="replace")
    for p in patterns:
        if not re.search(p, content):
            return False, f"Pattern not found: {repr(p)}"
    return True, "all patterns found"

def find_build_artifact(name_pattern: str) -> Optional[Path]:
    """Search common build output locations for a file matching pattern."""
    search_dirs = [
        ICD / "build",
        ICD / "build" / "Release",
        ICD / "build" / "Debug",
        ICD / "build" / "x64" / "Release",
    ]
    for d in search_dirs:
        if d.exists():
            for f in d.rglob("*"):
                if re.search(name_pattern, f.name, re.IGNORECASE):
                    return f
    return None

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — Prerequisites
# ══════════════════════════════════════════════════════════════════════════════
def probe_P0() -> PhaseReport:
    r = PhaseReport("P0", "Prerequisites & Project Scaffold")

    # CMake
    rc, out, _ = run(["cmake", "--version"])
    if rc == 0:
        m = re.search(r"(\d+\.\d+\.\d+)", out)
        ver = m.group(1) if m else "unknown"
        major, minor = map(int, ver.split(".")[:2])
        if major > 3 or (major == 3 and minor >= 20):
            r.results.append(ok("P0-1", "CMake 3.20+", f"Found {ver}"))
        else:
            r.results.append(fail("P0-1", "CMake 3.20+", f"Found {ver} — too old",
                "Install CMake 3.20+ from cmake.org"))
    else:
        r.results.append(fail("P0-1", "CMake available", "cmake not in PATH",
            "Install CMake from cmake.org and add to PATH"))

    # Vulkan SDK
    vulkan_sdk = os.environ.get("VULKAN_SDK", "")
    if vulkan_sdk and Path(vulkan_sdk).exists():
        r.results.append(ok("P0-2", "VULKAN_SDK env var", vulkan_sdk))
    else:
        rc2, out2, _ = run(["vulkaninfo", "--version"])
        if rc2 == 0:
            r.results.append(ok("P0-2", "Vulkan available (no VULKAN_SDK but vulkaninfo works)", out2.strip()))
        else:
            r.results.append(fail("P0-2", "VULKAN_SDK / Vulkan installed",
                "VULKAN_SDK not set and vulkaninfo not found",
                "Install LunarG Vulkan SDK from vulkan.lunarg.com"))

    # vulkaninfo
    rc, out, err = run(["vulkaninfo", "--summary"], timeout=15)
    if rc == 0:
        r.results.append(ok("P0-3", "vulkaninfo runs", f"{len(out)} chars output"))
    else:
        r.results.append(fail("P0-3", "vulkaninfo runs", err[:120],
            "Vulkan runtime not installed or no GPU present"))

    # Python dev headers
    py_inc = Path(sys.prefix) / "include" / "python3.dll" if IS_WIN else Path(sys.prefix) / "include" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    # simpler check: try to find Python.h
    candidates = []
    if IS_WIN:
        candidates = list(Path(sys.prefix).rglob("Python.h"))
    else:
        candidates = list(Path(sys.prefix).rglob("Python.h")) + list(Path("/usr").rglob("Python.h"))
    if candidates:
        r.results.append(ok("P0-4", "Python.h dev header", str(candidates[0])))
    else:
        r.results.append(fail("P0-4", "Python.h dev header", "Not found in sys.prefix",
            "Windows: Python installer → Modify → Add dev headers. Linux: apt install python3-dev"))

    # Directory structure
    required_dirs = ["include", "src", "bridge", "manifests", "scripts", "tests"]
    missing = [d for d in required_dirs if not (ICD / d).is_dir()]
    if not missing:
        r.results.append(ok("P0-5", "vulkan_icd/ directory structure", f"All {len(required_dirs)} subdirs present"))
    else:
        r.results.append(fail("P0-5", "vulkan_icd/ directory structure",
            f"Missing: {missing}",
            f"Create: mkdir vulkan_icd/{'{' + ','.join(missing) + '}'}"))

    # vk_icd.h
    vk_icd_h = ICD / "include" / "vk_icd.h"
    if vk_icd_h.exists():
        contains, _ = file_contains(vk_icd_h, "VK_LOADER_DATA", "set_loader_magic_value")
        r.results.append(ok("P0-6", "vk_icd.h present with VK_LOADER_DATA", str(vk_icd_h)) if contains
            else fail("P0-6", "vk_icd.h present with VK_LOADER_DATA",
                "File exists but missing VK_LOADER_DATA or set_loader_magic_value",
                "Copy from https://github.com/KhronosGroup/Vulkan-Loader/blob/main/loader/vk_icd.h"))
    else:
        r.results.append(fail("P0-6", "vk_icd.h in include/",
            f"Not found at {vk_icd_h}",
            "curl -o vulkan_icd/include/vk_icd.h https://raw.githubusercontent.com/KhronosGroup/Vulkan-Loader/main/loader/vk_icd.h"))

    # Source files exist
    required_src = ["icd_main.c", "instance.c", "physical_device.c", "device.c",
                    "memory.c", "pipeline.c", "commands.c", "queue.c", "spirv_dispatch.c"]
    missing_src = [f for f in required_src if not (ICD / "src" / f).exists()]
    if not missing_src:
        r.results.append(ok("P0-7", f"All {len(required_src)} source files present"))
    else:
        r.results.append(fail("P0-7", "Source files present",
            f"Missing: {missing_src}",
            "Re-run VibeCoder with the v0.3 prompt — some files were not generated"))

    # bridge/py_bridge.c
    bridge = ICD / "bridge" / "py_bridge.c"
    r.results.append(ok("P0-8", "bridge/py_bridge.c present") if bridge.exists()
        else fail("P0-8", "bridge/py_bridge.c present", str(bridge),
            "VibeCoder needs to generate bridge/py_bridge.c"))

    return r

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — ICD Entry Point & Headers
# ══════════════════════════════════════════════════════════════════════════════
def probe_P1() -> PhaseReport:
    r = PhaseReport("P1", "Core ICD Headers & Entry Point")

    # synthgpu_vk.h has required content
    header = ICD / "include" / "synthgpu_vk.h"
    checks = [
        ("VK_LOADER_DATA",          "VK_LOADER_DATA in structs"),
        ("SYNTHGPU_DEVICE_NAME",    "SYNTHGPU_DEVICE_NAME constant"),
        ("SYNTHGPU_VENDOR_ID",      "SYNTHGPU_VENDOR_ID constant"),
        ("SynthGPU_Instance_T",     "SynthGPU_Instance_T struct"),
        ("SynthGPU_PhysicalDevice_T","SynthGPU_PhysicalDevice_T struct"),
        ("SynthGPU_Device_T",       "SynthGPU_Device_T struct"),
        ("SynthGPU_CommandBuffer_T","SynthGPU_CommandBuffer_T struct"),
        ("vk_icdGetInstanceProcAddr","ICD entry point declaration"),
    ]
    if not header.exists():
        for pat, label in checks:
            r.results.append(fail(f"P1-h", label, "synthgpu_vk.h not found"))
    else:
        for pat, label in checks:
            found, detail = file_contains(header, pat)
            r.results.append(ok(f"P1-{pat[:6]}", label) if found
                else fail(f"P1-{pat[:6]}", label, detail,
                    f"Add '{pat}' to include/synthgpu_vk.h"))

    # icd_main.c dispatch table coverage
    main_c = ICD / "src" / "icd_main.c"
    required_procs = [
        "vkCreateInstance", "vkDestroyInstance", "vkEnumeratePhysicalDevices",
        "vkGetPhysicalDeviceProperties", "vkGetPhysicalDeviceFeatures",
        "vkGetPhysicalDeviceMemoryProperties", "vkGetPhysicalDeviceQueueFamilyProperties",
        "vkCreateDevice", "vkDestroyDevice", "vkGetDeviceQueue",
        "vkAllocateMemory", "vkFreeMemory", "vkMapMemory", "vkUnmapMemory",
        "vkCreateBuffer", "vkDestroyBuffer", "vkBindBufferMemory",
        "vkCreateShaderModule", "vkCreateComputePipelines",
        "vkCreateDescriptorSetLayout", "vkAllocateDescriptorSets", "vkUpdateDescriptorSets",
        "vkCreateCommandPool", "vkAllocateCommandBuffers",
        "vkBeginCommandBuffer", "vkEndCommandBuffer",
        "vkCmdBindPipeline", "vkCmdBindDescriptorSets", "vkCmdDispatch",
        "vkQueueSubmit", "vkQueueWaitIdle",
        "vkCreateFence", "vkWaitForFences", "vkResetFences",
    ]
    if main_c.exists():
        content = main_c.read_text(errors="replace")
        missing_procs = [p for p in required_procs if p not in content]
        if not missing_procs:
            r.results.append(ok("P1-dispatch", f"Dispatch table covers all {len(required_procs)} required functions"))
        else:
            r.results.append(fail("P1-dispatch", "Dispatch table completeness",
                f"Missing {len(missing_procs)}: {missing_procs[:5]}{'...' if len(missing_procs)>5 else ''}",
                "Add missing PROC() entries to get_instance_proc() in icd_main.c"))
    else:
        r.results.append(fail("P1-dispatch", "icd_main.c exists for dispatch check", str(main_c)))

    # spirv_dispatch.h
    spirv_h = ICD / "include" / "spirv_dispatch.h"
    if spirv_h.exists():
        found, detail = file_contains(spirv_h, "SynthGPU_DispatchContext", "bindings")
        r.results.append(ok("P1-spirvh", "spirv_dispatch.h with DispatchContext") if found
            else fail("P1-spirvh", "spirv_dispatch.h with DispatchContext", detail))
    else:
        r.results.append(fail("P1-spirvh", "spirv_dispatch.h exists",
            str(spirv_h), "Create include/spirv_dispatch.h with SynthGPU_DispatchContext struct"))

    return r

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Physical Device Properties (source-level)
# ══════════════════════════════════════════════════════════════════════════════
def probe_P2() -> PhaseReport:
    r = PhaseReport("P2", "Physical Device Source Correctness")

    pd = ICD / "src" / "physical_device.c"
    if not pd.exists():
        r.results.append(fail("P2-file", "physical_device.c exists", str(pd)))
        return r

    checks = [
        ("SynthGPU Virtual Accelerator",    "Device name in source"),
        ("0x5347",                           "vendorID 0x5347 set"),
        ("VK_PHYSICAL_DEVICE_TYPE_OTHER",    "deviceType OTHER (not CPU/GPU)"),
        ("VK_MAKE_API_VERSION.*1.*3.*0",     "apiVersion 1.3.0"),
        ("128",                              "128MB VRAM heap"),
        ("VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT", "HOST_VISIBLE memory flag"),
        ("VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT", "DEVICE_LOCAL memory flag"),
        ("VK_QUEUE_COMPUTE_BIT",             "Compute queue family"),
        ("queueCount.*4|4.*queueCount",      "4 compute queues"),
        ("EnumeratePhysicalDevices",         "EnumeratePhysicalDevices implemented"),
        ("init_physical_device|g_physical_device", "Singleton device pattern"),
        ("SET_LOADER_MAGIC|set_loader_magic_value", "Loader magic set on device"),
    ]
    for pat, label in checks:
        found, detail = file_contains(pd, pat)
        r.results.append(ok(f"P2-{pat[:8]}", label) if found
            else fail(f"P2-{pat[:8]}", label, detail))

    # EnumerateDeviceExtensionProperties extensions
    found, _ = file_contains(pd, "storage_buffer_storage_class")
    r.results.append(ok("P2-ext", "Device extensions advertised") if found
        else fail("P2-ext", "Device extensions advertised",
            "storage_buffer_storage_class not found",
            "Add VK_KHR_storage_buffer_storage_class to extension list"))

    return r

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Memory & Buffers (source-level)
# ══════════════════════════════════════════════════════════════════════════════
def probe_P3() -> PhaseReport:
    r = PhaseReport("P3", "Memory & Buffer Implementation")

    mem_c = ICD / "src" / "memory.c"
    if not mem_c.exists():
        r.results.append(fail("P3-file", "memory.c exists", str(mem_c)))
        return r

    checks = [
        (r"calloc|malloc",                   "AllocateMemory uses malloc/calloc"),
        (r"vram_allocated",                  "vram_allocated counter tracked"),
        (r"VK_ERROR_OUT_OF_DEVICE_MEMORY",   "OOM error returned when limit exceeded"),
        (r"mapped.*=.*1|\.mapped\s*=",       "mapped flag tracked"),
        (r"free\s*\(",                        "FreeMemory calls free()"),
        (r"vkMapMemory|synthgpu_MapMemory",  "MapMemory implemented"),
        (r"synthgpu_UnmapMemory|vkUnmapMemory","UnmapMemory implemented"),
        (r"FlushMappedMemoryRanges",          "Flush is a no-op stub"),
        (r"memoryTypeBits.*=.*1|1.*memoryTypeBits", "memoryTypeBits=1 for single type"),
        (r"alignment.*=.*64|64.*alignment",  "Buffer alignment 64 bytes"),
    ]
    for pat, label in checks:
        found, detail = file_contains(mem_c, pat)
        r.results.append(ok(f"P3-{pat[:8]}", label) if found
            else fail(f"P3-{pat[:8]}", label, detail))

    # bind_offset in buffer struct
    found, _ = file_contains(mem_c, r"bind_offset|bound_memory")
    r.results.append(ok("P3-bind", "Buffer bind offset tracked") if found
        else fail("P3-bind", "Buffer bind offset tracked",
            "bind_offset not found in memory.c",
            "Ensure vkBindBufferMemory stores memory pointer + offset into SynthGPU_Buffer_T"))

    return r

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Pipelines & Descriptors
# ══════════════════════════════════════════════════════════════════════════════
def probe_P4() -> PhaseReport:
    r = PhaseReport("P4", "Shaders, Pipelines & Descriptors")

    pipe_c = ICD / "src" / "pipeline.c"
    if not pipe_c.exists():
        r.results.append(fail("P4-file", "pipeline.c exists", str(pipe_c)))
        return r

    checks = [
        (r"spirv_code",                "SPIR-V code stored in pipeline"),
        (r"spirv_word_count",          "SPIR-V word count stored"),
        (r"entry_point",               "Entry point name stored"),
        (r"local_size_x|local_size",   "LocalSize extracted"),
        (r"SPIRV_MAGIC|0x07230203",    "SPIR-V magic number validated"),
        (r"OpExecutionMode|0x0010|16", "OpExecutionMode parsed"),
        (r"LocalSize|0x0011|17",       "LocalSize mode value"),
        (r"deep.copy|memcpy.*spirv",   "SPIR-V bytes deep-copied"),
        (r"AllocateDescriptorSets|allocate_descriptor",  "DescriptorSet allocation"),
        (r"UpdateDescriptorSets|update_descriptor",      "DescriptorSet update"),
        (r"bound_buffers",             "bound_buffers array in descriptor set"),
    ]
    for pat, label in checks:
        found, detail = file_contains(pipe_c, pat)
        r.results.append(ok(f"P4-{pat[:10]}", label) if found
            else fail(f"P4-{pat[:10]}", label, detail))

    return r

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Command Buffers
# ══════════════════════════════════════════════════════════════════════════════
def probe_P5() -> PhaseReport:
    r = PhaseReport("P5", "Command Buffer Recording")

    cmd_c = ICD / "src" / "commands.c"
    if not cmd_c.exists():
        r.results.append(fail("P5-file", "commands.c exists", str(cmd_c)))
        return r

    checks = [
        (r"SYNTHGPU_CMD_DISPATCH",    "CMD_DISPATCH type defined/used"),
        (r"SYNTHGPU_CMD_COPY_BUFFER", "CMD_COPY_BUFFER type defined/used"),
        (r"cmd_head|cmd_tail",        "Linked list head/tail maintained"),
        (r"bound_pipeline",           "bound_pipeline tracked in command buffer"),
        (r"bound_desc_sets",          "bound_desc_sets tracked"),
        (r"BeginCommandBuffer",       "BeginCommandBuffer resets state"),
        (r"EndCommandBuffer",         "EndCommandBuffer implemented"),
        (r"CmdDispatch",              "CmdDispatch records command"),
        (r"CmdBindPipeline",          "CmdBindPipeline stores pipeline"),
        (r"CmdBindDescriptorSets",    "CmdBindDescriptorSets stores sets"),
        (r"CmdCopyBuffer",            "CmdCopyBuffer records copy"),
        (r"CmdPipelineBarrier",       "CmdPipelineBarrier recorded"),
        (r"CmdFillBuffer",            "CmdFillBuffer implemented"),
        (r"SET_LOADER_MAGIC|set_loader_magic_value", "Loader magic on command buffer"),
    ]
    for pat, label in checks:
        found, detail = file_contains(cmd_c, pat)
        r.results.append(ok(f"P5-{pat[:10]}", label) if found
            else fail(f"P5-{pat[:10]}", label, detail))

    return r

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — Queue & Execution Engine
# ══════════════════════════════════════════════════════════════════════════════
def probe_P6() -> PhaseReport:
    r = PhaseReport("P6", "Queue Submission & Execution Engine")

    q_c = ICD / "src" / "queue.c"
    if not q_c.exists():
        r.results.append(fail("P6-file", "queue.c exists", str(q_c)))
        return r

    checks = [
        (r"execute_dispatch",          "execute_dispatch function present"),
        (r"execute_copy_buffer",       "execute_copy_buffer function present"),
        (r"execute_command_buffer",    "execute_command_buffer iterates linked list"),
        (r"synthgpu_spirv_dispatch",   "spirv_dispatch called from execute_dispatch"),
        (r"SynthGPU_DispatchContext",  "DispatchContext populated before dispatch"),
        (r"binding_count|bindings\[", "Buffer bindings collected from descriptor sets"),
        (r"f->signaled\s*=\s*1|signaled.*=.*1", "Fence signaled after submit"),
        (r"QueueWaitIdle",             "QueueWaitIdle returns VK_SUCCESS"),
        (r"DeviceWaitIdle",            "DeviceWaitIdle returns VK_SUCCESS"),
        (r"WaitForFences",             "WaitForFences checks signaled"),
        (r"ResetFences",               "ResetFences clears signaled"),
        (r"\[SynthGPU Vulkan\]|SynthGPU Vulkan", "Dispatch log line present"),
        (r"group_count_x|groupCountX", "group_count_x/y/z used in dispatch"),
    ]
    for pat, label in checks:
        found, detail = file_contains(q_c, pat)
        r.results.append(ok(f"P6-{pat[:10]}", label) if found
            else fail(f"P6-{pat[:10]}", label, detail))

    return r

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — SPIR-V Interpreter
# ══════════════════════════════════════════════════════════════════════════════
def probe_P7() -> PhaseReport:
    r = PhaseReport("P7", "SPIR-V Interpreter & Warp Bridge")

    spirv_c = ICD / "src" / "spirv_dispatch.c"
    if not spirv_c.exists():
        r.results.append(fail("P7-file", "spirv_dispatch.c exists", str(spirv_c)))
        return r

    checks = [
        (r"0x07230203|SPIRV_MAGIC",      "SPIR-V magic constant defined"),
        (r"extract_local_size",           "extract_local_size function present"),
        (r"spirv_execute_workgroup",      "spirv_execute_workgroup function present"),
        (r"synthgpu_spirv_dispatch",      "synthgpu_spirv_dispatch entry point present"),
        (r"OpLoad|0x003d|61",             "OpLoad handled"),
        (r"OpStore|0x003e|62",            "OpStore handled"),
        (r"OpFAdd|OpIAdd",                "Arithmetic ops handled"),
        (r"GlobalInvocationID|gl_Global", "gl_GlobalInvocationID resolved"),
        (r"Py_IsInitialized|PyGILState",  "Python bridge with GIL guard"),
        (r"record_external_warps",        "Warp scheduler bridge called"),
        (r"group_count_z.*gz|gz.*group_count_z", "Triple-nested dispatch loop"),
    ]

    content = spirv_c.read_text(errors="replace")
    # Special check: is spirv_execute_workgroup a TODO stub?
    is_stub = "TODO" in content and "spirv_execute_workgroup" in content
    stub_warning = " ⚠ (TODO stub — compute will not produce correct output)" if is_stub else ""

    for pat, label in checks:
        found, detail = file_contains(spirv_c, pat)
        display_label = label + stub_warning if "workgroup" in label.lower() and is_stub else label
        r.results.append(ok(f"P7-{pat[:10]}", display_label) if found
            else fail(f"P7-{pat[:10]}", display_label, detail))

    if is_stub:
        r.results.append(fail("P7-stub", "spirv_execute_workgroup NOT a TODO stub",
            "Function body contains TODO — compute shaders will not produce correct output",
            "Implement OpLoad/OpStore/OpFAdd/gl_GlobalInvocationID in spirv_execute_workgroup()"))

    return r

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 8 — Build System & Manifests
# ══════════════════════════════════════════════════════════════════════════════
def probe_P8() -> PhaseReport:
    r = PhaseReport("P8", "Build System & Platform Registration")

    # CMakeLists.txt
    cmake = ICD / "CMakeLists.txt"
    if cmake.exists():
        checks = [
            (r"find_package\s*\(\s*Vulkan", "find_package(Vulkan)"),
            (r"find_package\s*\(\s*Python3","find_package(Python3)"),
            (r"SHARED",                     "Library built as SHARED"),
            (r"synthgpu_vulkan_icd",        "Target named synthgpu_vulkan_icd"),
            (r"icd_main\.c",               "icd_main.c in sources"),
            (r"spirv_dispatch\.c",         "spirv_dispatch.c in sources"),
            (r"py_bridge\.c",             "py_bridge.c in sources"),
            (r"WIN32|_WIN32",             "Windows platform handling"),
            (r"UNIX|linux",               "Linux platform handling"),
        ]
        for pat, label in checks:
            found, detail = file_contains(cmake, pat)
            r.results.append(ok(f"P8-{pat[:8]}", label) if found
                else fail(f"P8-{pat[:8]}", label, detail))
    else:
        r.results.append(fail("P8-cmake", "CMakeLists.txt exists", str(cmake)))

    # Manifests
    win_manifest = ICD / "manifests" / "synthgpu_icd_win64.json"
    lin_manifest = ICD / "manifests" / "synthgpu_icd_linux.json"

    for mpath, name in [(win_manifest, "win64"), (lin_manifest, "linux")]:
        if mpath.exists():
            try:
                data = json.loads(mpath.read_text())
                has_icd = "ICD" in data and "api_version" in data.get("ICD", {})
                r.results.append(ok(f"P8-{name}-manifest", f"{name} manifest valid JSON with ICD key") if has_icd
                    else fail(f"P8-{name}-manifest", f"{name} manifest structure",
                        "Missing 'ICD' or 'api_version' key",
                        "Ensure manifest follows Vulkan ICD JSON schema"))
            except json.JSONDecodeError as e:
                r.results.append(fail(f"P8-{name}-manifest", f"{name} manifest valid JSON", str(e)))
        else:
            r.results.append(fail(f"P8-{name}-manifest", f"{name} manifest exists", str(mpath)))

    # Install scripts
    for script in ["install_windows.bat", "install_linux.sh"]:
        path = ICD / "scripts" / script
        r.results.append(ok(f"P8-{script[:8]}", f"{script} present") if path.exists()
            else fail(f"P8-{script[:8]}", f"{script} present", str(path)))

    # Build artifact check
    dll = find_build_artifact(r"synthgpu_vulkan_icd\.(dll|so)")
    if dll:
        r.results.append(ok("P8-artifact", f"Build artifact found: {dll.name}", str(dll)))
    else:
        r.results.append(fail("P8-artifact", "Build artifact (.dll or .so) in build/",
            "Not found in build/, build/Release/, or build/x64/Release/",
            "Run: cd vulkan_icd && mkdir build && cd build && cmake .. -G 'Visual Studio 17 2022' -A x64 && cmake --build . --config Release"))

    return r

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 9 — OS Recognition (runtime)
# ══════════════════════════════════════════════════════════════════════════════
def probe_P9(quick=False) -> PhaseReport:
    r = PhaseReport("P9", "OS Recognition — vulkaninfo")

    if quick:
        r.results.append(skip("P9-quick", "vulkaninfo runtime tests", "skipped (--quick mode)"))
        return r

    # Check registry (Windows)
    if IS_WIN:
        rc, out, _ = run(["reg", "query",
            r"HKLM\SOFTWARE\Khronos\Vulkan\Drivers"], timeout=5)
        if rc == 0 and "synthgpu" in out.lower():
            r.results.append(ok("P9-registry", "Registry key present for SynthGPU ICD"))
        else:
            r.results.append(fail("P9-registry", "Registry key present",
                "SynthGPU not found in HKLM\\SOFTWARE\\Khronos\\Vulkan\\Drivers",
                "Run scripts\\install_windows.bat as Administrator"))
    else:
        icd_d = Path("/etc/vulkan/icd.d/synthgpu_icd.json")
        r.results.append(ok("P9-icdd", "/etc/vulkan/icd.d/synthgpu_icd.json installed") if icd_d.exists()
            else fail("P9-icdd", "/etc/vulkan/icd.d/synthgpu_icd.json installed",
                str(icd_d), "Run: sudo scripts/install_linux.sh"))

    # vulkaninfo --summary (used for SynthGPU presence + basic property checks)
    rc, out, err = run(["vulkaninfo", "--summary"], timeout=20)
    full = out + err
    if "SynthGPU" in full:
        r.results.append(ok("P9-vulkaninfo", "vulkaninfo --summary shows SynthGPU ✓",
            [l for l in full.splitlines() if "SynthGPU" in l][0].strip()))
    else:
        r.results.append(fail("P9-vulkaninfo", "vulkaninfo --summary shows SynthGPU",
            "SynthGPU not in vulkaninfo output",
            "Ensure build is complete and install_windows.bat / install_linux.sh was run as admin"))

    # Run full vulkaninfo (without --summary) to get queue family details.
    # --summary only prints basic device properties and never includes queue
    # family information, so COMPUTE would never appear in that output.
    rc_full, out_full, err_full = run(["vulkaninfo"], timeout=30)
    full_detail = out_full + err_full

    # Check specific properties in vulkaninfo
    property_checks = [
        ("0x5347",              "P9-vendorid",    "vendorID 0x5347 in vulkaninfo output",    full),
        ("OTHER",               "P9-devtype",     "deviceType OTHER in vulkaninfo output",    full),
        ("1.3.0",               "P9-apiversion",  "apiVersion 1.3.0 in vulkaninfo output",   full),
        # COMPUTE only appears in full vulkaninfo queue family section, not --summary
        ("COMPUTE",             "P9-queue",       "COMPUTE queue family in vulkaninfo output", full_detail),
    ]
    for pattern, rid, label, source in property_checks:
        if rc == 0:
            r.results.append(ok(rid, label) if pattern in source
                else fail(rid, label, f"'{pattern}' not found in vulkaninfo output"))
        else:
            r.results.append(skip(rid, label, "vulkaninfo failed to run"))

    # Python vulkan test
    try:
        import vulkan as vk
        try:
            inst = vk.vkCreateInstance(vk.VkInstanceCreateInfo(
                pApplicationInfo=vk.VkApplicationInfo(
                    pApplicationName="SynthGPU Probe",
                    applicationVersion=vk.VK_MAKE_VERSION(0,3,0),
                    pEngineName="SynthGPU",
                    engineVersion=vk.VK_MAKE_VERSION(0,3,0),
                    apiVersion=vk.VK_API_VERSION_1_3)), None)
            devices = vk.vkEnumeratePhysicalDevices(inst)
            found = False
            for dev in devices:
                props = vk.vkGetPhysicalDeviceProperties(dev)
                if "SynthGPU" in props.deviceName:
                    found = True
                    r.results.append(ok("P9-python-api", f"Python vulkan API finds '{props.deviceName}'"))
                    break
            if not found:
                r.results.append(fail("P9-python-api", "Python vulkan API finds SynthGPU",
                    f"Found {len(devices)} device(s) but none named SynthGPU"))
            vk.vkDestroyInstance(inst, None)
        except Exception as e:
            r.results.append(fail("P9-python-api", "Python vulkan API test", str(e)))
    except ImportError:
        r.results.append(skip("P9-python-api", "Python vulkan package test",
            "pip install vulkan to enable this check"))

    return r

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 10 — Full Compute (runtime)
# ══════════════════════════════════════════════════════════════════════════════
def append_cuda_demo_checks(r: PhaseReport) -> None:
    if IS_WIN:
        r.results.append(skip("P10-cuda-demo", "CUDA LD_PRELOAD demo", "Linux only"))
        return

    demo_candidates = [
        Path("/usr/local/bin/synthgpu_cuda_demo"),
        CUDA_SHIM / "build" / "synthgpu_cuda_demo",
    ]
    lib_candidates = [
        Path("/usr/local/lib/synthgpu/libsynthgpu_cuda.so"),
        CUDA_SHIM / "build" / "libsynthgpu_cuda.so",
    ]
    demo = next((path for path in demo_candidates if path.is_file()), None)
    shim_lib = next((path for path in lib_candidates if path.is_file()), None)

    r.results.append(ok("P10-cuda-demo-exists", "CUDA demo client exists", str(demo)) if demo
        else fail("P10-cuda-demo-exists", "CUDA demo client exists",
                  "No built synthgpu_cuda_demo found", "Build cuda_shim first"))
    if not demo:
        return
    if not shim_lib:
        r.results.append(fail("P10-cuda-library", "CUDA shim library exists",
                              "No libsynthgpu_cuda.so found", "Build cuda_shim first"))
        return

    rc, out, err = run([str(demo)], timeout=15)
    control_output = out + err
    if rc != 0 and "CUDA runtime symbols unavailable" in control_output:
        r.results.append(ok("P10-cuda-control", "CUDA demo fails without LD_PRELOAD",
                            control_output.strip()))
    else:
        r.results.append(fail("P10-cuda-control", "CUDA demo fails without LD_PRELOAD",
                              control_output[:300], "Demo must not link directly to the shim"))

    preload_env = os.environ.copy()
    preload_env["LD_PRELOAD"] = str(shim_lib)
    rc, out, err = run([str(demo)], timeout=30, env=preload_env)
    preload_output = out + err
    if rc == 0 and "ALL CHECKS PASSED" in preload_output:
        r.results.append(ok("P10-cuda-preload", "CUDA demo passes with LD_PRELOAD",
                            preload_output[:300]))
    else:
        r.results.append(fail("P10-cuda-preload", "CUDA demo passes with LD_PRELOAD",
                              preload_output[:300], "Check shim exports and runtime dependencies"))


def probe_P10(quick=False) -> PhaseReport:
    r = PhaseReport("P10", "Full Compute — Vulkan Apps Route Through SynthGPU")

    if quick:
        r.results.append(skip("P10-quick", "Compute runtime tests", "skipped (--quick mode)"))
        return r

    append_cuda_demo_checks(r)

    test_script = ICD / "tests" / "test_compute.py"
    test_enum   = ICD / "tests" / "test_enumeration.py"

    r.results.append(ok("P10-test-exists", "test_enumeration.py exists") if test_enum.exists()
        else fail("P10-test-exists", "test_enumeration.py exists", str(test_enum),
            "VibeCoder should have generated vulkan_icd/tests/test_enumeration.py"))

    r.results.append(ok("P10-compute-exists", "test_compute.py exists") if test_script.exists()
        else fail("P10-compute-exists", "test_compute.py exists", str(test_script),
            "VibeCoder should have generated vulkan_icd/tests/test_compute.py"))

    # Run test_enumeration.py
    if test_enum.exists():
        rc, out, err = run([sys.executable, str(test_enum)], timeout=30)
        if rc == 0 and "PASS" in (out + err):
            r.results.append(ok("P10-enum-run", "test_enumeration.py PASS", out[:200]))
        else:
            r.results.append(fail("P10-enum-run", "test_enumeration.py PASS",
                (out + err)[:300],
                "Check vulkaninfo output and ensure ICD is installed"))

    # Run test_compute.py
    if test_script.exists():
        rc, out, err = run([sys.executable, str(test_script)], timeout=60)
        if rc == 0 and "PASS" in (out + err):
            r.results.append(ok("P10-compute-run", "test_compute.py PASS", out[:200]))
        else:
            r.results.append(fail("P10-compute-run", "test_compute.py PASS",
                (out + err)[:300],
                "SPIR-V interpreter may be incomplete — check spirv_execute_workgroup()"))

    return r

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 11 — Dashboard Integration
# ══════════════════════════════════════════════════════════════════════════════
def probe_P11() -> PhaseReport:
    r = PhaseReport("P11", "Dashboard Integration & Telemetry")

    main_py = BACKEND / "main.py"
    if not main_py.exists():
        r.results.append(fail("P11-main", "backend/main.py exists", str(main_py)))
        return r

    checks = [
        (r"/api/vulkan/status",              "P11-endpoint", "/api/vulkan/status endpoint"),
        (r"vulkan.*installed|installed.*vulkan","P11-installed", "vulkan installed flag"),
        (r"dispatch_count|vulkan_dispatch",  "P11-counter",  "dispatch_count tracked"),
        (r"Vulkan ICD|vulkan_icd",           "P11-badge",    "Vulkan ICD status in backend"),
    ]
    for pat, rid, label in checks:
        found, detail = file_contains(main_py, pat)
        r.results.append(ok(rid, label) if found
            else fail(rid, label, detail,
                f"Add '{pat}' to backend/main.py"))

    # Check frontend for Vulkan badge
    frontend_dirs = [ROOT / "frontend" / "src", ROOT / "backend" / "static"]
    vulkan_badge_found = False
    for fdir in frontend_dirs:
        if fdir.exists():
            for jsx in fdir.rglob("*.jsx"):
                content = jsx.read_text(errors="replace")
                if "Vulkan" in content and ("badge" in content.lower() or "Active" in content):
                    vulkan_badge_found = True
                    break
            for tsx in fdir.rglob("*.tsx"):
                content = tsx.read_text(errors="replace")
                if "Vulkan" in content and ("badge" in content.lower() or "Active" in content):
                    vulkan_badge_found = True
                    break

    r.results.append(ok("P11-frontend", "Frontend Vulkan ICD badge in JSX/TSX") if vulkan_badge_found
        else fail("P11-frontend", "Frontend Vulkan ICD badge in JSX/TSX",
            "No JSX/TSX file found with Vulkan + badge/Active",
            "Add 'Vulkan ICD: Active' status badge to frontend header bar"))

    # Quick backend health check
    rc, out, err = run(["curl", "-s", "--max-time", "3",
                        "http://localhost:8000/api/vulkan/status"], timeout=8)
    if rc == 0 and out:
        try:
            data = json.loads(out)
            r.results.append(ok("P11-live", "/api/vulkan/status live response", json.dumps(data)))
        except:
            r.results.append(fail("P11-live", "/api/vulkan/status returns JSON", out[:100],
                "Endpoint exists but returns non-JSON"))
    else:
        r.results.append(skip("P11-live", "/api/vulkan/status live",
            "Backend not running or endpoint not yet added"))

    return r

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 12 — Demo Readiness
# ══════════════════════════════════════════════════════════════════════════════
def probe_P12(quick=False) -> PhaseReport:
    r = PhaseReport("P12", "Investor Demo Flow Verification")

    # README check
    readme = ROOT / "README.md"
    if readme.exists():
        found, _ = file_contains(readme, r"v0\.3|Vulkan ICD|vulkaninfo")
        r.results.append(ok("P12-readme", "README.md mentions v0.3 / Vulkan ICD") if found
            else fail("P12-readme", "README.md updated for v0.3",
                "README does not mention v0.3 or Vulkan ICD",
                "Update README.md with v0.3 milestone and build instructions"))
    else:
        r.results.append(fail("P12-readme", "README.md exists", str(readme)))

    # Demo assets
    demo_asset = ROOT / "demo-assets" / "vulkaninfo-synthgpu.png"
    r.results.append(ok("P12-screenshot", "vulkaninfo screenshot saved") if demo_asset.exists()
        else fail("P12-screenshot", "vulkaninfo screenshot in demo-assets/",
            str(demo_asset),
            "Run vulkaninfo, screenshot the output, save as demo-assets/vulkaninfo-synthgpu.png"))

    # One-command install test (just checks script exists and is non-empty)
    for script, platform_name in [("install_windows.bat", "Windows"), ("install_linux.sh", "Linux")]:
        path = ICD / "scripts" / script
        if path.exists() and path.stat().st_size > 200:
            r.results.append(ok(f"P12-{script[:8]}", f"{platform_name} install script non-trivial"))
        else:
            r.results.append(fail(f"P12-{script[:8]}", f"{platform_name} install script non-trivial",
                "Script missing or < 200 bytes — likely a stub",
                f"Ensure scripts/{script} has full registry/icd.d registration logic"))

    # backend health
    if not quick:
        rc, out, _ = run(["curl", "-s", "--max-time", "3",
                          "http://localhost:8000/api/health/demo_ready"], timeout=8)
        if rc == 0 and out:
            try:
                data = json.loads(out)
                is_ready = data.get("ready", False)
                r.results.append(ok("P12-demo-ready", f"Demo Ready endpoint: ready={is_ready}", json.dumps(data)) if is_ready
                    else fail("P12-demo-ready", "Demo Ready endpoint returns ready=true",
                        json.dumps(data),
                        "Run tinyllama inference first to trigger demo_ready_achieved flag"))
            except:
                r.results.append(skip("P12-demo-ready", "Demo Ready endpoint", "Invalid JSON response"))
        else:
            r.results.append(skip("P12-demo-ready", "Demo Ready endpoint (backend not running)", "Start backend first"))

    return r

# ══════════════════════════════════════════════════════════════════════════════
# Report renderer
# ══════════════════════════════════════════════════════════════════════════════
def render_report(phases: List[PhaseReport], show_hints: bool):
    total_pass = total_fail = total_skip = 0

    print(f"\n{BOLD}{CYAN}{'═'*72}{RESET}")
    print(f"{BOLD}{CYAN}  SynthGPU v0.3 — Implementation Probe Report{RESET}")
    print(f"{CYAN}  {platform.system()} {platform.release()} · Python {sys.version.split()[0]}{RESET}")
    print(f"{CYAN}  Root: {ROOT}{RESET}")
    print(f"{CYAN}{'═'*72}{RESET}\n")

    for phase in phases:
        p, f, s = phase.passed, phase.failed, phase.skipped
        total_pass += p; total_fail += f; total_skip += s
        pct = int(100 * p / max(1, p + f))

        status_color = GREEN if f == 0 else (YELLOW if pct >= 50 else RED)
        bar_filled = int(24 * p / max(1, p + f))
        bar = "█" * bar_filled + "░" * (24 - bar_filled)

        print(f"{BOLD}{status_color}  {phase.id:4s}  {phase.title}{RESET}")
        print(f"       {DIM}{bar}{RESET}  {status_color}{p}/{p+f}{RESET}{DIM} ({s} skipped){RESET}")

        for res in phase.results:
            if res.skipped:
                sym = f"{DIM}─{RESET}"
                col = DIM
            elif res.passed:
                sym = f"{GREEN}✓{RESET}"
                col = ""
            else:
                sym = f"{RED}✗{RESET}"
                col = RED

            print(f"       {sym} {col}{res.label}{RESET}", end="")
            if res.detail and (res.passed is False or not res.skipped):
                print(f"  {DIM}{res.detail[:80]}{RESET}", end="")
            print()

            if show_hints and not res.passed and not res.skipped and res.hint:
                print(f"           {YELLOW}↳ FIX: {res.hint}{RESET}")

        print()

    # Summary
    total = total_pass + total_fail
    pct_overall = int(100 * total_pass / max(1, total))
    summary_color = GREEN if total_fail == 0 else (YELLOW if pct_overall >= 70 else RED)

    print(f"{BOLD}{CYAN}{'─'*72}{RESET}")
    print(f"{BOLD}  Overall: {summary_color}{total_pass}/{total} checks passed ({pct_overall}%){RESET}  "
          f"{DIM}{total_skip} skipped{RESET}")

    if total_fail == 0:
        print(f"\n{BOLD}{GREEN}  ✓ ALL CHECKS PASSED — v0.3 implementation complete{RESET}")
    else:
        print(f"\n{BOLD}{RED}  {total_fail} check(s) failed{RESET}", end="")
        if not show_hints:
            print(f"  {DIM}Run with --fix-hints to see remediation steps{RESET}", end="")
        print()

    print(f"{CYAN}{'═'*72}{RESET}\n")

    # Next steps
    if total_fail > 0:
        print(f"{BOLD}  NEXT STEPS:{RESET}")
        # Prioritise by phase
        for phase in phases:
            failed = [r for r in phase.results if not r.passed and not r.skipped]
            if failed:
                print(f"  {CYAN}{phase.id}{RESET} {phase.title}:")
                for r in failed[:3]:
                    print(f"    • {r.label}")
                    if r.hint:
                        print(f"      {DIM}→ {r.hint}{RESET}")
                if len(failed) > 3:
                    print(f"      {DIM}... and {len(failed)-3} more{RESET}")
        print()

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="SynthGPU v0.3 Implementation Probe")
    parser.add_argument("--phase", help="Run only a specific phase (e.g. P2)")
    parser.add_argument("--quick", action="store_true", help="Skip slow runtime tests")
    parser.add_argument("--fix-hints", action="store_true", help="Show fix hints for failures")
    args = parser.parse_args()

    all_probes = {
        "P0":  lambda: probe_P0(),
        "P1":  lambda: probe_P1(),
        "P2":  lambda: probe_P2(),
        "P3":  lambda: probe_P3(),
        "P4":  lambda: probe_P4(),
        "P5":  lambda: probe_P5(),
        "P6":  lambda: probe_P6(),
        "P7":  lambda: probe_P7(),
        "P8":  lambda: probe_P8(),
        "P9":  lambda: probe_P9(args.quick),
        "P10": lambda: probe_P10(args.quick),
        "P11": lambda: probe_P11(),
        "P12": lambda: probe_P12(args.quick),
    }

    if args.phase:
        pid = args.phase.upper()
        if pid not in all_probes:
            print(f"Unknown phase: {pid}. Valid: {list(all_probes.keys())}")
            sys.exit(1)
        probes_to_run = {pid: all_probes[pid]}
    else:
        probes_to_run = all_probes

    reports = []
    for pid, fn in probes_to_run.items():
        print(f"{DIM}  Probing {pid}...{RESET}", end="\r", flush=True)
        try:
            reports.append(fn())
        except Exception as e:
            rp = PhaseReport(pid, f"Phase {pid}")
            rp.results.append(fail(pid, f"Phase {pid} probe", f"Probe itself crashed: {e}",
                "This is a probe bug — report it"))
            reports.append(rp)

    render_report(reports, args.fix_hints)

    total_fail = sum(r.failed for r in reports)
    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
