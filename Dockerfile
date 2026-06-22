# =============================================================================
# SynthGPU v0.3 — Multi-Stage Docker Build
# =============================================================================
# syntax=docker/dockerfile:1

# ── Stage 1: C extension builder (CUDA Shim + Vulkan ICD) ────────────────
FROM python:3.11-slim-bookworm AS c-builder

# Retry loop for flaky networks
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN for i in 1 2 3; do \
      apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        pkg-config \
        libvulkan-dev \
        libx11-dev \
        libopenblas-dev \
        libpython3-dev \
        && break || { echo "Attempt $i failed, retrying..."; sleep 5; }; \
    done \
    && rm -rf /var/lib/apt/lists/* \
    && for i in 1 2 3; do \
         pip install --index-url https://pypi.org/simple --retries 5 --timeout 120 --no-cache-dir numpy \
         && break; \
         if [ "$i" -eq 3 ]; then exit 1; fi; \
         echo "pip attempt $i failed, retrying..."; sleep 10; \
       done

WORKDIR /build
RUN mkdir -p /artifacts/vulkan /artifacts/cuda

# Vulkan ICD build (optional — graceful failure)
COPY vulkan_icd/ vulkan_icd/
RUN mkdir -p /artifacts/vulkan && \
    cd vulkan_icd \
    && rm -rf build \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 \
    && cmake --build . --config Release -- -j"$(nproc)" 2>&1 \
    && cp libsynthgpu_vulkan_icd.so /artifacts/vulkan/ 2>/dev/null; \
    echo "[SynthGPU] Vulkan ICD build complete (check artifacts)"

# CUDA Shim build (optional — graceful failure)
COPY cuda_shim/ cuda_shim/
RUN mkdir -p /artifacts/cuda/lib /artifacts/cuda/bin && \
    cd cuda_shim \
    && rm -rf build \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 \
    && cmake --build . --config Release -- -j"$(nproc)" 2>&1 \
    && cp libsynthgpu_cuda.so /artifacts/cuda/lib/ \
    && cp synthgpu_cuda_demo /artifacts/cuda/bin/; \
    echo "[SynthGPU] CUDA Shim build complete (check artifacts)"

# ── Stage 2: Frontend builder ────────────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --prefer-offline || npm ci
COPY frontend/ .
RUN npm run build

# ── Stage 3: Runtime image ───────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="SynthGPU"
LABEL org.opencontainers.image.description="Software-defined Virtual GPU Accelerator"
LABEL org.opencontainers.image.version="0.3.0"

# System dependencies for runtime (no build toolchain needed)
RUN for i in 1 2 3; do \
      apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libvulkan1 \
        mesa-vulkan-drivers \
        vulkan-tools \
        libopenblas0 \
        libgomp1 \
        && break || { echo "Attempt $i failed, retrying..."; sleep 5; }; \
    done \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r synthgpu && useradd -r -g synthgpu -d /opt/synthgpu -s /sbin/nologin synthgpu

WORKDIR /opt/synthgpu

# ── Copy Python backend + cuda_shim ─────────────────────────────────────
COPY backend/ backend/
COPY cuda_shim/ cuda_shim/
COPY probe_v03.py .
COPY requirements.txt .
RUN for i in 1 2 3; do \
      pip install --index-url https://pypi.org/simple --retries 5 --timeout 120 -r requirements.txt \
      && pip install --index-url https://pypi.org/simple --retries 5 --timeout 120 -r backend/requirements.txt \
      && break; \
      if [ "$i" -eq 3 ]; then exit 1; fi; \
      echo "pip attempt $i failed, retrying..."; sleep 10; \
    done && \
    pip cache purge

# ── Copy built frontend ──────────────────────────────────────────────────
COPY --from=frontend-builder /app/frontend/dist/ frontend/dist/

# ── Copy built C extensions (if any) ─────────────────────────────────────
RUN mkdir -p /usr/local/lib/synthgpu
COPY --from=c-builder /artifacts/vulkan/ /usr/local/lib/synthgpu/
COPY --from=c-builder /artifacts/cuda/lib/ /usr/local/lib/synthgpu/
COPY --from=c-builder /artifacts/cuda/bin/ /usr/local/bin/

# ── Install Vulkan ICD manifest ──────────────────────────────────────────
RUN mkdir -p /etc/vulkan/icd.d
COPY vulkan_icd/manifests/synthgpu_icd_linux.json /etc/vulkan/icd.d/synthgpu_icd.json
COPY vulkan_icd/tests/test_telemetry_runtime.py vulkan_icd/tests/test_telemetry_runtime.py

# ── Environment ──────────────────────────────────────────────────────────
ENV PYTHONPATH=/opt/synthgpu/backend:/opt/synthgpu \
    SYNTHGPU_ROOT=/opt/synthgpu \
    SYNTHGPU_DOCKER=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=1 \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_MAX_LOADED_MODELS=1

# ── Healthcheck ──────────────────────────────────────────────────────────
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/device/status || exit 1

EXPOSE 8000

USER synthgpu
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
