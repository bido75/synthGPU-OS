import { useEffect, useRef } from 'react'

function ArcGauge({ pct, modelMb, kvMb, totalMb, inferenceActive }) {
  const canvasRef = useRef(null)
  const color = pct > 85 ? '#ef4444' : pct > 60 ? '#f59e0b' : '#10b981'

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width
    const H = canvas.height
    const cx = W / 2
    const cy = H * 0.6
    const r = Math.min(W, H) * 0.4

    ctx.clearRect(0, 0, W, H)

    // Background arc
    ctx.beginPath()
    ctx.arc(cx, cy, r, Math.PI * 0.75, Math.PI * 2.25)
    ctx.strokeStyle = '#2a2a3e'
    ctx.lineWidth = 12
    ctx.lineCap = 'round'
    ctx.stroke()

    if (inferenceActive && totalMb > 0 && (modelMb > 0 || kvMb > 0)) {
      // Segmented arc: model weights (purple) + KV cache (cyan)
      const modelPct = modelMb / totalMb
      const kvPct = kvMb / totalMb
      const arcStart = Math.PI * 0.75
      const arcTotal = Math.PI * 1.5

      // Model weights segment (purple)
      if (modelPct > 0) {
        const segEnd = arcStart + arcTotal * modelPct
        ctx.beginPath()
        ctx.arc(cx, cy, r, arcStart, segEnd)
        ctx.strokeStyle = '#7c3aed'
        ctx.lineWidth = 12
        ctx.lineCap = 'butt'
        ctx.stroke()
      }

      // KV cache segment (cyan)
      if (kvPct > 0) {
        const kvStart = arcStart + arcTotal * modelPct
        const kvEnd = kvStart + arcTotal * kvPct
        ctx.beginPath()
        ctx.arc(cx, cy, r, kvStart, kvEnd)
        ctx.strokeStyle = '#00d4ff'
        ctx.lineWidth = 12
        ctx.lineCap = 'round'
        ctx.stroke()

        // Glow at leading edge
        ctx.shadowColor = '#00d4ff'
        ctx.shadowBlur = 10
        ctx.beginPath()
        ctx.arc(cx, cy, r, kvEnd - 0.05, kvEnd)
        ctx.strokeStyle = '#00d4ff'
        ctx.lineWidth = 12
        ctx.stroke()
        ctx.shadowBlur = 0
      }
    } else {
      // Single color arc
      const fillEnd = Math.PI * 0.75 + (Math.PI * 1.5 * (pct / 100))
      const gradient = ctx.createLinearGradient(cx - r, cy, cx + r, cy)
      gradient.addColorStop(0, color + '88')
      gradient.addColorStop(1, color)
      ctx.beginPath()
      ctx.arc(cx, cy, r, Math.PI * 0.75, fillEnd)
      ctx.strokeStyle = gradient
      ctx.lineWidth = 12
      ctx.lineCap = 'round'
      ctx.stroke()

      if (pct > 0) {
        ctx.shadowColor = color
        ctx.shadowBlur = 12
        ctx.beginPath()
        ctx.arc(cx, cy, r, fillEnd - 0.05, fillEnd)
        ctx.strokeStyle = color
        ctx.lineWidth = 12
        ctx.stroke()
        ctx.shadowBlur = 0
      }
    }

    // Percentage text
    ctx.fillStyle = inferenceActive ? '#a78bfa' : color
    ctx.font = `bold ${r * 0.38}px monospace`
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(`${pct.toFixed(1)}%`, cx, cy - r * 0.08)

    // Sub text
    ctx.fillStyle = '#94a3b8'
    ctx.font = `${r * 0.2}px sans-serif`
    ctx.fillText('vRAM Usage', cx, cy + r * 0.28)
  }, [pct, color, modelMb, kvMb, totalMb, inferenceActive])

  return <canvas ref={canvasRef} width={200} height={140} style={{ display: 'block', margin: '0 auto' }} />
}

export default function MemoryGauge({ memory, inference }) {
  const used = memory?.vram_used_mb || 0
  const total = memory?.vram_total_mb || 4096
  const free = Math.max(0, memory?.vram_free_mb ?? total)
  const pct = memory?.utilization_pct || 0
  const allocs = memory?.num_allocations || 0
  const h2d = memory?.h2d_transferred_mb || 0
  const d2h = memory?.d2h_transferred_mb || 0
  const color = pct > 85 ? '#ef4444' : pct > 60 ? '#f59e0b' : '#10b981'

  const inferenceActive = inference?.active === true
  const modelMb = inference?.memory?.model_weights_mb || memory?.model_weights_mb || 0
  const kvMb = inference?.memory?.kv_cache_mb || memory?.kv_cache_mb || 0

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-1">
        <div className="card-title mb-0">Virtual VRAM</div>
        <div style={{ background: '#10b98111', border: '1px solid #10b98133',
                      borderRadius: 6, padding: '2px 8px' }}>
          <span style={{ color: '#10b981', fontSize: '0.62rem', fontWeight: 700 }}>
            Source: System RAM — NOT hard drive
          </span>
        </div>
      </div>

      <ArcGauge pct={pct} modelMb={modelMb} kvMb={kvMb}
                totalMb={total} inferenceActive={inferenceActive} />

      {inferenceActive && (modelMb > 0 || kvMb > 0) && (
        <div className="flex justify-center gap-4 mb-2">
          <div className="flex items-center gap-1.5">
            <div style={{ width: 10, height: 10, borderRadius: 2, background: '#7c3aed' }} />
            <span style={{ color: '#a78bfa', fontSize: '0.65rem' }}>
              Model weights ~{modelMb.toFixed(0)} MB
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <div style={{ width: 10, height: 10, borderRadius: 2, background: '#00d4ff' }} />
            <span style={{ color: '#00d4ff', fontSize: '0.65rem' }}>
              KV cache {kvMb.toFixed(1)} MB
            </span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-2 mt-3">
        {[
          { label: 'Used', value: `${used.toFixed(0)} MB`, color },
          { label: 'Free', value: `${free.toFixed(0)} MB`, color: '#10b981' },
          { label: 'Total', value: `${total.toFixed(0)} MB`, color: '#f1f5f9' },
          { label: 'Allocations', value: allocs, color: '#f1f5f9' },
          { label: 'H→D Transfer', value: `${h2d.toFixed(1)} MB`, color: '#94a3b8' },
          { label: 'D→H Transfer', value: `${d2h.toFixed(1)} MB`, color: '#94a3b8' },
        ].map(({ label, value, color: c }) => (
          <div key={label} className="flex justify-between items-center py-1"
               style={{ borderBottom: '1px solid #2a2a3e' }}>
            <span style={{ color: '#94a3b8', fontSize: '0.7rem' }}>{label}</span>
            <span style={{ color: c, fontSize: '0.75rem', fontWeight: 600, fontFamily: 'monospace' }}>
              {value}
            </span>
          </div>
        ))}
      </div>

      {pct > 85 && (
        <div className="mt-2 p-2 rounded" style={{ background: '#ef444422', border: '1px solid #ef444444' }}>
          <span style={{ color: '#ef4444', fontSize: '0.7rem', fontWeight: 600 }}>
            ⚠ High vRAM usage — approaching limit
          </span>
        </div>
      )}
    </div>
  )
}
