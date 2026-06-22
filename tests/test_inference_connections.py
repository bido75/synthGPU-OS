import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from synthgpu.inference_proxy import (
    LM_STUDIO_URL,
    OLLAMA_URL,
    _normalize_backend_url,
    _resolve_backend_target,
)


def test_ollama_preset_ignores_stale_custom_url():
    target = _resolve_backend_target("ollama", "http://10.0.0.219:11434")
    assert target["url"] == _normalize_backend_url(OLLAMA_URL)
    assert target["type"] == "ollama"


def test_lmstudio_preset_ignores_stale_custom_url():
    target = _resolve_backend_target("lmstudio", "http://10.0.0.219:11434")
    assert target["url"] == _normalize_backend_url(LM_STUDIO_URL)
    assert target["type"] == "lmstudio"


def test_custom_url_is_normalized_without_changing_host():
    target = _resolve_backend_target("custom", "10.0.0.219:11434/")
    assert target["url"] == "http://10.0.0.219:11434"
    assert target["models_path"] == "/api/tags"


def test_custom_url_rejects_api_paths():
    try:
        _resolve_backend_target("custom", "http://10.0.0.219:11434/api/tags")
    except ValueError as exc:
        assert "must not include an API path" in str(exc)
    else:
        raise AssertionError("Custom URL with API path should be rejected")
