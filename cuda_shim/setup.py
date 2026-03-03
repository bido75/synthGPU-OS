"""
SynthGPU CUDA Shim — Python build / install script
====================================================
Installs the Python kernel layer (cuda_shim package).
The C shared library must be compiled separately via CMake.

  pip install -e .          # editable dev install
  pip install .             # production install
"""

from setuptools import setup, find_packages
import os

# Read long description from the project README if present
_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_here)
_readme_path = os.path.join(_project_root, "README.md")
try:
    with open(_readme_path, encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "SynthGPU CUDA Compatibility Shim"

setup(
    name="synthgpu-cuda-shim",
    version="0.3.0",
    description="CPU-only CUDA compatibility shim for SynthGPU",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SynthGPU Contributors",
    url="https://github.com/OpenVGPU/SynthGPU",
    license="MIT",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
    ],
    entry_points={
        "console_scripts": [
            "synthgpu-shim-info=cuda_shim.kernels.bridge_api:print_info",
        ],
    },
)
