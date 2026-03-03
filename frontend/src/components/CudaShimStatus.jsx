import { useState, useEffect } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

function StatusBadge({ available }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-semibold ${
        available
          ? "bg-emerald-900/60 text-emerald-300 border border-emerald-700"
          : "bg-red-900/40 text-red-400 border border-red-800"
      }`}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${available ? "bg-emerald-400 animate-pulse" : "bg-red-500"}`}
      />
      {available ? "Active" : "Unavailable"}
    </span>
  );
}

function StatCard({ label, value, unit, color = "text-cyan-300" }) {
  return (
    <div className="bg-gray-900/60 border border-gray-700/50 rounded-lg p-3 flex flex-col gap-1">
      <span className="text-xs text-gray-400 uppercase tracking-wide">{label}</span>
      <span className={`text-lg font-mono font-bold ${color}`}>
        {value}
        {unit && <span className="text-xs text-gray-500 ml-1">{unit}</span>}
      </span>
    </div>
  );
}

export default function CudaShimStatus() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;

    async function fetchStatus() {
      try {
        const res = await fetch(`${API}/api/cuda_shim/status`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (alive) { setStatus(data); setError(null); }
      } catch (e) {
        if (alive) setError(e.message);
      }
    }

    fetchStatus();
    const id = setInterval(fetchStatus, 3000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const vramPct =
    status?.available && status.vram_total_mb > 0
      ? Math.round((status.vram_used_mb / status.vram_total_mb) * 100)
      : 0;

  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-5 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold text-white">CUDA Shim</h2>
          {status?.version && (
            <span className="text-xs text-gray-500">v{status.version}</span>
          )}
        </div>
        <StatusBadge available={!!status?.available} />
      </div>

      {/* Error / unavailable message */}
      {(!status?.available || error) && (
        <div className="text-xs text-red-400 bg-red-900/20 border border-red-800/40 rounded-lg p-3">
          {error
            ? `Connection error: ${error}`
            : status?.message || "CUDA shim not running."}
        </div>
      )}

      {/* Stats grid */}
      {status?.available && (
        <>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            <StatCard
              label="Warps Executed"
              value={status.warps_executed?.toLocaleString() ?? "—"}
            />
            <StatCard
              label="Warp Throughput"
              value={
                status.warp_throughput != null
                  ? status.warp_throughput.toFixed(1)
                  : "—"
              }
              unit="W/ms"
              color="text-purple-300"
            />
            <StatCard
              label="Kernels Dispatched"
              value={status.kernels_dispatched?.toLocaleString() ?? "—"}
              color="text-yellow-300"
            />
            <StatCard
              label="Active Streams"
              value={status.active_streams ?? "—"}
              color="text-blue-300"
            />
            <StatCard
              label="VRAM Used"
              value={`${status.vram_used_mb} / ${status.vram_total_mb}`}
              unit="MB"
              color="text-orange-300"
            />
          </div>

          {/* VRAM bar */}
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-gray-400">
              <span>Virtual VRAM</span>
              <span>{vramPct}%</span>
            </div>
            <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${vramPct}%`,
                  background:
                    vramPct > 80
                      ? "#ef4444"
                      : vramPct > 60
                      ? "#f97316"
                      : "#00d4ff",
                }}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
