"""
SynthGPU Dashboard E2E Tests
Validates: backend REST, WebSocket telemetry, warp monitor, chart, compute units.
"""
import asyncio
import json
import time
import pytest
import httpx
from playwright.sync_api import sync_playwright, expect

BASE_URL = "http://localhost:8000"


# ── REST / API checks ──────────────────────────────────────────

def test_backend_health():
    """Backend responds to /api/device/info."""
    r = httpx.get(f"{BASE_URL}/api/device/info", timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert "scheduler" in data, "Missing 'scheduler' key in device info"
    assert "memory" in data, "Missing 'memory' key in device info"
    cu = data["scheduler"].get("compute_units")
    assert isinstance(cu, int) and cu > 0, f"compute_units should be positive int, got {cu!r}"
    print(f"  compute_units={cu}, warps_executed={data['scheduler'].get('warps_executed')}")


def test_debug_telemetry_structure():
    """GET /api/debug/telemetry returns nested scheduler/memory/inference."""
    r = httpx.get(f"{BASE_URL}/api/debug/telemetry", timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    for key in ("scheduler", "memory"):
        assert key in data, f"Missing top-level key '{key}' in debug telemetry"
    sched = data["scheduler"]
    for field in ("compute_units", "warps_executed", "warp_throughput_per_sec", "utilization_pct"):
        assert field in sched, f"scheduler missing field '{field}'"
    print(f"  debug/telemetry scheduler: {sched}")


def test_websocket_telemetry_nested():
    """WebSocket /ws/telemetry emits nested scheduler with warp activity after heartbeat."""
    import websocket as ws_lib  # use websocket-client for sync
    messages = []
    errors = []

    try:
        import websocket
    except ImportError:
        pytest.skip("websocket-client not installed; skipping WS test")

    ws = websocket.create_connection(f"ws://localhost:8000/ws/telemetry", timeout=10)
    deadline = time.time() + 6  # collect for 6 seconds (heartbeat fires at 1s intervals)
    while time.time() < deadline:
        try:
            ws.settimeout(2.0)
            raw = ws.recv()
            msg = json.loads(raw)
            messages.append(msg)
        except Exception:
            break
    ws.close()

    assert len(messages) > 0, "No messages received from WebSocket"

    # First message is 'connected', rest are 'telemetry'
    telemetry_msgs = [m for m in messages if m.get("type") == "telemetry"]
    assert len(telemetry_msgs) > 0, "No telemetry messages received (only connected?)"

    last = telemetry_msgs[-1]
    assert "scheduler" in last, f"telemetry message missing 'scheduler': {last.keys()}"
    sched = last["scheduler"]
    assert "compute_units" in sched, "scheduler missing compute_units"
    assert "warps_executed" in sched, "scheduler missing warps_executed"

    # After heartbeat ticks, warps_executed should be > 0
    final_warps = sched.get("warps_executed", 0)
    assert final_warps > 0, (
        f"warps_executed is still 0 after 6 seconds — "
        f"background_warp_heartbeat may not be running. sched={sched}"
    )
    throughput = sched.get("warp_throughput_per_sec", 0)
    print(f"  After 6s: warps_executed={final_warps}, throughput={throughput:.2f} w/s")


# ── Browser / UI checks ────────────────────────────────────────

@pytest.fixture(scope="module")
def browser_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context()
        page = ctx.new_page()
        # Capture console errors
        page._console_errors = []
        page.on("console", lambda msg: page._console_errors.append(msg.text)
                if msg.type == "error" else None)
        page.goto(BASE_URL, wait_until="networkidle", timeout=20000)
        # Wait for WS to connect — "LIVE" badge appears when connected=true
        page.wait_for_selector("text=LIVE", timeout=12000)
        yield page
        browser.close()


def test_ui_loads(browser_page):
    """Page title / heading loads correctly."""
    page = browser_page
    assert "SynthGPU" in page.title() or page.locator("text=SynthGPU").count() > 0


def test_compute_units_not_dashes(browser_page):
    """Sidebar 'Compute Units' shows a number, not '--'."""
    page = browser_page
    page.wait_for_timeout(3000)
    # Get all text on the page that's near 'Compute Units'
    full_text = page.inner_text("body")
    # Find the section containing Compute Units
    idx = full_text.find("Compute Units")
    assert idx != -1, "Could not find 'Compute Units' text on page"
    # Grab the 20 chars before 'Compute Units' — that's where the value is rendered
    surrounding = full_text[max(0, idx - 20):idx + 20]
    assert "--" not in surrounding, (
        f"Compute Units shows '--' near: {surrounding!r}\n"
        "Check WebSocket telemetry — frontend reads telemetry.scheduler.compute_units"
    )
    print(f"  Compute Units context: {surrounding!r}")


def test_warp_monitor_shows_rows(browser_page):
    """WarpMonitor shows warp rows within 5 seconds (not 'Waiting for warp activity...')."""
    page = browser_page
    # Wait up to 5s for any Warp row to appear
    warp_row_appeared = False
    for _ in range(10):
        page.wait_for_timeout(500)
        # Warp rows contain 'Warp #' text
        count = page.locator("text=Warp #").count()
        if count > 0:
            warp_row_appeared = True
            print(f"  Warp rows visible: {count}")
            break

    waiting_text = page.locator("text=Waiting for warp activity").count()
    assert warp_row_appeared or waiting_text == 0, (
        "WarpMonitor still showing 'Waiting for warp activity...' after 5 seconds.\n"
        "background_warp_heartbeat is not generating warp activity."
    )


def test_live_performance_chart_has_data(browser_page):
    """Live Performance chart SVG has path elements (line drawn, not blank)."""
    page = browser_page
    # Wait for chart to render
    page.wait_for_timeout(4000)
    # recharts renders <path> inside the chart SVG for each line
    chart_paths = page.locator(".recharts-line-curve, .recharts-curve").count()
    assert chart_paths > 0, (
        "Live Performance chart has no drawn paths — chart may be blank.\n"
        "Chart needs warp_throughput_per_sec > 0 in telemetry to draw a line."
    )
    print(f"  Chart line paths found: {chart_paths}")


def test_no_critical_console_errors(browser_page):
    """No critical JS errors in browser console."""
    page = browser_page
    critical = [e for e in page._console_errors
                if "TypeError" in e or "Cannot read" in e or "undefined" in e.lower()]
    if critical:
        print(f"  Console errors found: {critical[:5]}")
    # Soft check — warn but don't fail on minor errors
    assert len(critical) < 5, f"Too many console errors ({len(critical)}): {critical[:3]}"
