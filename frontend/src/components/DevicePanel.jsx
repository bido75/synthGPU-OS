import { useEffect, useState } from 'react'

function formatUptime(seconds) {
  if (!seconds) return '00:00:00'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  return [h, m, s].map(v => v.toString().padStart(2, '0')).join(':')
}

export default function DevicePanel({ telemetry, connected, onNavigateToLLM }) {
  const [pulse, setPulse] = useState(true)

  useEffect(() => {
    const t = setInterval(() => setPulse(p => !p), 1200)
    return () => clearInterval(t)
  }, [])

  const dev = telemetry?.device || {}
  const sched = telemetry?.scheduler || {}
  const mem = telemetry?.memory || {}
  const inference = telemetry?.inference || {}

  const inferenceActive = inference?.active === true
  const activeModel = inference?.active_model || (inferenceActive ? inference?.active_session?.model : null)
  const backendConnected = inference?.backend_status === 'connected'
  const backendName = inference?.backend

  const inferenceStatus = inferenceActive
    ? `Generating: ${activeModel}`
    : backendConnected
    ? `${backendName} ready`
    : 'No backend'

  const inferenceStatusColor = inferenceActive ? '#00d4ff' : backendConnected ? '#10b981' : '#ef4444'

  return (
    <div className="card">
      <div className="card-title">Virtual GPU Device</div>

      <div className="flex items-center gap-3 mb-4">
        <div className="w-3 h-3 rounded-full"
             style={{
               background: connected ? '#10b981' : '#f59e0b',
               boxShadow: connected ? `0 0 ${pulse ? '10px' : '4px'} #10b981` : '0 0 6px #f59e0b',
               transition: 'box-shadow 0.6s ease',
             }} />
        <div>
          <div className="font-bold text-base" style={{ color: '#f1f5f9' }}>
            {dev.name || 'SynthGPU Virtual Accelerator'}
          </div>
          <div style={{ color: '#94a3b8', fontSize: '0.7rem' }}>
            {dev.version || 'v0.2.0-beta'}
          </div>
        </div>
        <span className="badge-no-gpu ml-auto">✗ NO PHYSICAL GPU</span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {[
          { label: 'Compute Units', value: sched.compute_units || '--' },
          { label: 'Warp Size', value: `${sched.warp_size || 32} lanes` },
          { label: 'vRAM Total', value: `${mem.vram_total_mb || '--'} MB` },
          { label: 'Uptime', value: formatUptime(dev.uptime_seconds) },
          { label: 'Platform', value: (dev.platform || 'x86_64').substring(0, 16) || 'x86_64' },
          { label: 'OS', value: (dev.os || 'Unknown').substring(0, 16) },
          { label: 'Ops Executed', value: (dev.ops_executed || 0).toLocaleString() },
          { label: 'Physical GPU', value: '✗ NONE — 100% SW' },
        ].map(({ label, value }) => (
          <div key={label} className="rounded-lg p-2"
               style={{ background: '#0a0a0f', border: '1px solid #2a2a3e' }}>
            <div style={{ color: '#94a3b8', fontSize: '0.6rem', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              {label}
            </div>
            <div style={{ color: '#f1f5f9', fontSize: '0.8rem', fontWeight: 600, marginTop: '2px',
                          fontFamily: 'monospace' }}>
              {value}
            </div>
          </div>
        ))}

        {/* Inference status spanning 2 cols */}
        <div className="col-span-2 rounded-lg p-2 cursor-pointer"
             style={{ background: inferenceActive ? '#00d4ff0a' : backendConnected ? '#10b9810a' : '#ef44440a',
                      border: `1px solid ${inferenceStatusColor}33`,
                      transition: 'all 0.3s' }}
             onClick={onNavigateToLLM}>
          <div style={{ color: '#94a3b8', fontSize: '0.6rem', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
            LLM Inference Status
          </div>
          <div className="flex items-center gap-2 mt-1">
            <div style={{ width: 6, height: 6, borderRadius: '50%', background: inferenceStatusColor,
                          boxShadow: `0 0 ${inferenceActive ? '8px' : '4px'} ${inferenceStatusColor}` }} />
            <span style={{ color: inferenceStatusColor, fontSize: '0.78rem', fontWeight: 600,
                           fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis',
                           whiteSpace: 'nowrap' }}>
              {inferenceStatus}
            </span>
            <span style={{ color: '#94a3b8', fontSize: '0.6rem', marginLeft: 'auto' }}>click →</span>
          </div>
        </div>

        {activeModel && (
          <div className="col-span-2 rounded-lg p-2"
               style={{ background: '#00d4ff0a', border: '1px solid #00d4ff22' }}>
            <div style={{ color: '#94a3b8', fontSize: '0.6rem', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Active Model
            </div>
            <div style={{ color: '#00d4ff', fontSize: '0.78rem', fontWeight: 600, marginTop: '2px',
                          fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap' }}>
              {activeModel}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
