#!/bin/bash
sudo rm -f /usr/local/lib/synthgpu/libsynthgpu_vulkan_icd.so
sudo rm -f /etc/vulkan/icd.d/synthgpu_icd.json
sudo ldconfig
echo "[SynthGPU] Uninstalled Vulkan ICD."
