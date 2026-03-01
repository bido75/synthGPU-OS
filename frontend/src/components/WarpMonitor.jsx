import { useEffect, useRef, useState } from 'react'

const WARP_SIZE = 32
const NUM_VISIBLE_WARPS = 5

const OP_LABELS = ['attention', 'matmul', 'gelu', 'softmax', 'layernorm', 'embed']

function WarpRow({ warpId, status, progress, execMs, opLabel, inferenceActive }) {
  const lanes = Array.from({ length: WARP_SIZE }, (_, i) => {
    if (status === 'completed') return 'done'
    if (status === 'running') return i < Math.floor(progress * WARP_SIZE) ? 'active' : 'idle'
    return 'idle'
  })

  const laneColor = inferenceActive ? '#7c3aed' : '#00d4ff'
  const statusColor = status === 'completed' ? '#10b981' : status === 'running' ? laneColor : '#2a2a3e'
  const label = status === 'completed' ? `✓ ${execMs?.toFixed(1) || '?'}ms` :
                status === 'running' ? 'running...' : 'queued'

  return (
    <div className="flex items-center gap-3 py-1">
      <div style={{ color: '#94a3b8', fontSize: '0.65rem', fontFamily: 'monospace', minWidth: '72px' }}>
        Warp #{String(warpId).padStart(4, '0')}
      </div>
      {inferenceActive && opLabel && (
        <div style={{ color: '#a78bfa', fontSize: '0.6rem', fontFamily: 'monospace',
                      minWidth: 72, background: '#7c3aed22', border: '1px solid #7c3aed44',
                      borderRadius: 4, padding: '1px 5px' }}>
          [{opLabel}]
        </div>
      )}
      <div className="flex gap-0.5 flex-1">
        {lanes.map((laneState, i) => (
          <div key={i}
               style={{
                 width: '7px',
                 height: '10px',
                 borderRadius: '1px',
                 background: laneState === 'active' || laneState === 'done' ? laneColor : '#1a1a2e',
                 boxShadow: laneState === 'active' ? `0 0 4px ${laneColor}` : 'none',
                 transition: 'background 0.1s, box-shadow 0.1s',
               }} />
        ))}
      </div>
      <div style={{ color: statusColor, fontSize: '0.65rem', minWidth: '70px', textAlign: 'right' }}>
        {label}
      </div>
    </div>
  )
}

export default function WarpMonitor({ scheduler, inference }) {
  const [warps, setWarps] = useState([])
  const animRef = useRef(null)
  const frameRef = useRef(0)
  const prevExecuted = useRef(0)
  const opIdx = useRef(0)

  const inferenceActive = inference?.active === true
  const activeModel = inference?.active_model || inference?.active_session?.model

  useEffect(() => {
    const executed = scheduler?.warps_executed || 0
    const inFlight = scheduler?.warps_in_flight || 0
    const newWarps = executed - prevExecuted.current
    prevExecuted.current = executed

    setWarps(prev => {
      let next = [...prev]

      for (let i = 0; i < Math.min(newWarps, 3); i++) {
        const id = executed - i
        const existing = next.find(w => w.id === id)
        if (!existing) {
          next.unshift({
            id,
            status: 'completed',
            progress: 1,
            execMs: scheduler?.avg_warp_ms || Math.random() * 8 + 1,
            opLabel: OP_LABELS[opIdx.current % OP_LABELS.length],
          })
          opIdx.current++
        }
      }

      for (let i = 0; i < inFlight; i++) {
        const id = executed + i + 1
        const existing = next.find(w => w.id === id)
        if (!existing) {
          next.push({
            id, status: 'running', progress: Math.random() * 0.7 + 0.1, execMs: null,
            opLabel: OP_LABELS[opIdx.current % OP_LABELS.length],
          })
          opIdx.current++
        } else {
          existing.progress = Math.min(1, existing.progress + 0.15)
          if (existing.progress >= 1) existing.status = 'completed'
        }
      }

      return next.slice(0, NUM_VISIBLE_WARPS)
    })
  }, [scheduler?.warps_executed, scheduler?.warps_in_flight])

  useEffect(() => {
    const interval = setInterval(() => {
      setWarps(prev => prev.map(w =>
        w.status === 'running'
          ? { ...w, progress: Math.min(0.95, w.progress + 0.08) }
          : w
      ))
    }, 150)
    return () => clearInterval(interval)
  }, [])

  const cu = scheduler?.compute_units || 0
  const inFlight = scheduler?.warps_in_flight || 0
  const throughput = scheduler?.warp_throughput_per_sec || 0

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="card-title mb-0">Warp Execution Monitor</div>
          {inferenceActive && activeModel && (
            <div style={{
              background: '#7c3aed22', border: '1px solid #7c3aed66',
              borderRadius: 6, padding: '2px 10px', display: 'flex', alignItems: 'center', gap: 6,
            }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#a78bfa',
                            boxShadow: '0 0 6px #a78bfa', animation: 'blink 1s step-end infinite' }} />
              <span style={{ color: '#a78bfa', fontSize: '0.68rem', fontWeight: 700 }}>
                INFERENCE ACTIVE: {activeModel}
              </span>
            </div>
          )}
        </div>
        <div className="flex items-center gap-4">
          <div style={{ fontSize: '0.75rem' }}>
            <span style={{ color: '#94a3b8' }}>Active Warps: </span>
            <span style={{ color: inferenceActive ? '#a78bfa' : '#00d4ff', fontWeight: 700 }}>{inFlight}</span>
            <span style={{ color: '#94a3b8' }}>/{cu}</span>
          </div>
          <div style={{ fontSize: '0.75rem' }}>
            <span style={{ color: '#94a3b8' }}>Throughput: </span>
            <span style={{ color: inferenceActive ? '#a78bfa' : '#00d4ff', fontWeight: 700 }}>{throughput.toFixed(1)}</span>
            <span style={{ color: '#94a3b8' }}> w/s</span>
          </div>
        </div>
      </div>

      <div className="mb-2" style={{ fontSize: '0.65rem', color: '#94a3b8' }}>
        Each row = 1 synthetic GPU warp (32 parallel threads).
        {inferenceActive
          ? <span style={{ color: '#a78bfa' }}> Purple</span>
          : <span style={{ color: '#00d4ff' }}> Cyan</span>
        } = executing lane.
        {inferenceActive && <span style={{ color: '#a78bfa' }}> LLM inference ops shown per warp.</span>}
      </div>

      <div style={{ border: `1px solid ${inferenceActive ? '#7c3aed44' : '#2a2a3e'}`,
                    borderRadius: '8px', padding: '0.75rem',
                    background: inferenceActive ? '#7c3aed08' : '#0a0a0f', minHeight: '160px',
                    transition: 'all 0.3s' }}>
        {warps.length === 0 ? (
          <div style={{ color: '#94a3b8', fontSize: '0.75rem', textAlign: 'center', padding: '2rem' }}>
            Waiting for warp activity... Run a benchmark to see warp execution.
          </div>
        ) : (
          warps.map(w => (
            <WarpRow key={w.id} warpId={w.id} status={w.status}
                     progress={w.progress} execMs={w.execMs}
                     opLabel={w.opLabel} inferenceActive={inferenceActive} />
          ))
        )}
      </div>

      <div className="flex gap-4 mt-3">
        {[
          { color: inferenceActive ? '#7c3aed' : '#00d4ff', label: 'Active Lane' },
          { color: '#1a1a2e', label: 'Inactive Lane', border: '1px solid #2a2a3e' },
        ].map(({ color, label, border }) => (
          <div key={label} className="flex items-center gap-1.5">
            <div style={{ width: 8, height: 10, background: color, borderRadius: 1, border }} />
            <span style={{ color: '#94a3b8', fontSize: '0.65rem' }}>{label}</span>
          </div>
        ))}
        <div style={{ marginLeft: 'auto', color: '#94a3b8', fontSize: '0.65rem' }}>
          Total: <span style={{ color: '#f1f5f9' }}>{(scheduler?.warps_executed || 0).toLocaleString()} warps</span>
        </div>
      </div>
    </div>
  )
}
