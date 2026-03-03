"""
Test Phase 1: Verify SynthGPU appears as a Vulkan device.
Run after install_windows.bat or install_linux.sh.

Usage:
    python tests/test_enumeration.py
"""
import subprocess
import sys
import os


def test_vulkaninfo():
    """Check vulkaninfo --summary reports SynthGPU as a device."""
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout + result.stderr
    except FileNotFoundError:
        print("SKIP: vulkaninfo not found (install Vulkan SDK)")
        return True
    except subprocess.TimeoutExpired:
        print("FAIL: vulkaninfo timed out")
        return False

    if "SynthGPU" in output:
        lines = [l for l in output.splitlines() if "SynthGPU" in l]
        print("PASS: SynthGPU Virtual Accelerator detected by vulkaninfo")
        for line in lines:
            print(f"  {line.strip()}")
        return True
    else:
        print("FAIL: SynthGPU not found in vulkaninfo output")
        print("Output preview:")
        print(output[:1500])
        return False


def test_vulkan_python():
    """Enumerate Vulkan devices via the vulkan Python binding."""
    try:
        import vulkan as vk
    except ImportError:
        print("SKIP: vulkan Python package not installed (pip install vulkan)")
        return True

    try:
        app_info = vk.VkApplicationInfo(
            pApplicationName="SynthGPU Test",
            applicationVersion=vk.VK_MAKE_VERSION(0, 3, 0),
            pEngineName="SynthGPU",
            engineVersion=vk.VK_MAKE_VERSION(0, 3, 0),
            apiVersion=vk.VK_API_VERSION_1_3,
        )
        create_info = vk.VkInstanceCreateInfo(pApplicationInfo=app_info)
        instance = vk.vkCreateInstance(create_info, None)

        devices = vk.vkEnumeratePhysicalDevices(instance)
        found = False
        for device in devices:
            props = vk.vkGetPhysicalDeviceProperties(device)
            print(f"  Device found: {props.deviceName}")
            if "SynthGPU" in props.deviceName:
                print(f"PASS: Found '{props.deviceName}' via Vulkan Python API")
                found = True

        vk.vkDestroyInstance(instance, None)

        if not found:
            print("FAIL: SynthGPU device not found via Vulkan Python API")
        return found

    except Exception as e:
        print(f"FAIL: Vulkan Python enumeration raised: {e}")
        return False


def test_icd_files_present():
    """Verify the ICD manifest and DLL/so files exist in expected locations."""
    import platform
    system = platform.system()

    if system == "Windows":
        import winreg
        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Khronos\Vulkan\Drivers"
            )
            i = 0
            found = False
            while True:
                try:
                    name, _, _ = winreg.EnumValue(key, i)
                    if "synthgpu" in name.lower():
                        print(f"PASS: Registry entry found: {name}")
                        found = True
                    i += 1
                except OSError:
                    break
            winreg.CloseKey(key)
            if not found:
                print("FAIL: No SynthGPU entry in Vulkan registry")
            return found
        except Exception as e:
            print(f"SKIP: Could not read registry ({e})")
            return True

    elif system == "Linux":
        icd_path = "/etc/vulkan/icd.d/synthgpu_icd.json"
        lib_path = "/usr/local/lib/synthgpu/libsynthgpu_vulkan_icd.so"
        ok = True
        if os.path.exists(icd_path):
            print(f"PASS: ICD manifest present: {icd_path}")
        else:
            print(f"FAIL: ICD manifest not found: {icd_path}")
            ok = False
        if os.path.exists(lib_path):
            print(f"PASS: ICD library present: {lib_path}")
        else:
            print(f"FAIL: ICD library not found: {lib_path}")
            ok = False
        return ok

    else:
        print(f"SKIP: Unsupported platform: {system}")
        return True


if __name__ == "__main__":
    print("=" * 60)
    print("SynthGPU Vulkan ICD — Enumeration Test")
    print("=" * 60)

    results = {
        "icd_files":     test_icd_files_present(),
        "vulkaninfo":    test_vulkaninfo(),
        "vulkan_python": test_vulkan_python(),
    }

    print()
    print("Results:")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status:4s}  {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("All tests passed.")
    else:
        print("Some tests failed. See output above.")
    sys.exit(0 if all_pass else 1)
