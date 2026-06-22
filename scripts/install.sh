#!/usr/bin/env sh
set -eu

PROJECT_NAME="synthgpu"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
UNINSTALL=0
NO_OPEN=0

usage() {
    echo "Usage: $0 [--uninstall] [--no-open]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --uninstall) UNINSTALL=1 ;;
        --no-open) NO_OPEN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker Engine is required. Install it before running this script." >&2
    exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
    echo "Docker Compose v2 is required ('docker compose')." >&2
    exit 1
fi

if [ "$UNINSTALL" -eq 1 ]; then
    cd "$PROJECT_DIR"
    docker compose -p "$PROJECT_NAME" down --rmi local --volumes --remove-orphans
    echo "SynthGPU Docker resources removed. Source files were preserved."
    exit 0
fi

deadline=$(( $(date +%s) + 30 ))
version=""
while [ "$(date +%s)" -lt "$deadline" ]; do
    if command -v timeout >/dev/null 2>&1; then
        version=$(timeout 5 docker version --format '{{.Server.Version}}' 2>/dev/null || true)
    else
        version=$(docker version --format '{{.Server.Version}}' 2>/dev/null || true)
    fi
    [ -n "$version" ] && break
    sleep 3
done
if [ -z "$version" ]; then
    echo "Docker Engine did not respond within 30 seconds." >&2
    exit 1
fi
echo "Docker engine: $version"

cd "$PROJECT_DIR"
docker compose -p "$PROJECT_NAME" up -d --build
docker compose -p "$PROJECT_NAME" ps
echo "SynthGPU is running at http://localhost:8000"

if [ "$NO_OPEN" -eq 0 ]; then
    case "$(uname -s)" in
        Darwin) command -v open >/dev/null 2>&1 && open http://localhost:8000 || true ;;
        Linux) command -v xdg-open >/dev/null 2>&1 && xdg-open http://localhost:8000 >/dev/null 2>&1 || true ;;
    esac
fi
