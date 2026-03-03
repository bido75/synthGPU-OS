import { useMemo } from 'react';

export default function SystemRAMMonitor({ telemetry }) {
  const ram = telemetry?.system_ram;

  const riskLevel = useMemo(() => {
    if (!ram) return 'normal';
    if (ram.utilization_pct > 85) return 'critical';
    if (ram.utilization_pct > 70) return 'warning';
    return 'normal';
  }, [ram]);

  const maxSafeModelMB = useMemo(() => {
    if (!ram) return 0;
    return Math.round(ram.available_gb * 0.7 * 1024);
  }, [ram]);

  if (!ram) return null;

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold text-gray-400 tracking-wider uppercase">
          System RAM
        </h3>
        {riskLevel === 'critical' && (
          <span className="text-xs text-red-400 bg-red-950 border border-red-900 px-2 py-0.5 rounded animate-pulse">
            Swap risk — inference may be slow
          </span>
        )}
        {riskLevel === 'warning' && (
          <span className="text-xs text-amber-400 bg-amber-950 border border-amber-900 px-2 py-0.5 rounded">
            Memory pressure
          </span>
        )}
        {riskLevel === 'normal' && (
          <span className="text-xs text-green-400 font-mono">
            {ram.available_gb}GB free
          </span>
        )}
      </div>

      <div className="w-full bg-gray-800 rounded-full h-2.5 mb-2">
        <div
          className={`h-2.5 rounded-full transition-all duration-500 ${
            riskLevel === 'critical' ? 'bg-red-500'
            : riskLevel === 'warning' ? 'bg-amber-500'
            : 'bg-cyan-500'
          }`}
          style={{ width: `${Math.min(100, ram.utilization_pct)}%` }}
        />
      </div>

      <div className="flex justify-between text-xs text-gray-500 mb-1">
        <span>Used: <span className="text-gray-300">{ram.used_gb}GB</span></span>
        <span className={riskLevel === 'critical' ? 'text-red-400 font-bold' : 'text-gray-300'}>
          Free: {ram.available_gb}GB
        </span>
        <span>Total: <span className="text-gray-300">{ram.total_gb}GB</span></span>
      </div>

      {ram.swap_active && (
        <div className="flex items-center gap-1 mt-1 text-xs text-amber-400">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse inline-block" />
          Swap active: {ram.swap_used_gb}GB — disk I/O slowing inference
        </div>
      )}

      {riskLevel !== 'normal' && maxSafeModelMB > 0 && (
        <p className="mt-2 text-xs text-gray-500">
          Models under <span className="text-cyan-400">{maxSafeModelMB}MB</span> run
          at full speed.{' '}
          <span className="text-cyan-400">tinyllama:latest (638MB)</span> recommended.
        </p>
      )}
    </div>
  );
}
