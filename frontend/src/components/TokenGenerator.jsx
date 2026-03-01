import { useState, useEffect, useRef } from 'react'

export default function TokenGenerator() {
  const [running, setRunning] = useState(false)
  const [tokens, setTokens] = useState([])
  const [stats, setStats] = useState(null)
  const [totalMs, setTotalMs] = useState(0)
  const wsRef = useRef(null)
  const startTimeRef = useRef(null)
  const allMsRef = useRef([])

  const NUM_TOKENS = 20

  const connectTokenWS = () => {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/tokens`)
    wsRef.current = ws

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (data.type === 'token') {
        allMsRef.current.push(data.ms)
        setTokens(prev => [...prev, data])
        const avgMs = allMsRef.current.reduce((a, b) => a + b, 0) / allMsRef.current.length
        setStats({
          step: data.step + 1,
          total: data.total_tokens,
          ms: data.ms,
          tps: data.tokens_per_sec,
          avgMs,
          avgTps: 1000 / avgMs,
          pct: data.pct_complete,
        })
        if (data.pct_complete >= 100) {
          setRunning(false)
          setTotalMs(performance.now() - startTimeRef.current)
        }
      }
    }
  }

  const startGeneration = async () => {
    setTokens([])
    setStats(null)
    setTotalMs(0)
    allMsRef.current = []
    setRunning(true)
    startTimeRef.current = performance.now()

    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      connectTokenWS()
      await new Promise(r => setTimeout(r, 300))
    }

    await fetch('/api/generate/tokens', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ num_tokens: NUM_TOKENS, d_model: 256, num_layers: 2 }),
    })
  }

  const stopGeneration = () => {
    setRunning(false)
    wsRef.current?.close()
    wsRef.current = null
  }

  useEffect(() => {
    connectTokenWS()
    return () => wsRef.current?.close()
  }, [])

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <div className="card-title mb-0">LLM Token Generation Demo</div>
        <span className="badge-no-gpu">GPU HARDWARE REQUIRED: ✗ NONE</span>
      </div>

      <div className="mb-3 p-3 rounded-lg" style={{ background: '#0a0a0f', border: '1px solid #2a2a3e' }}>
        <div style={{ color: '#94a3b8', fontSize: '0.75rem' }}>
          Model: <span style={{ color: '#f1f5f9' }}>Mini-Transformer (2 layers, d=256)</span>
          &nbsp;&nbsp;·&nbsp;&nbsp;
          Running entirely on <span style={{ color: '#00d4ff' }}>SynthGPU virtual device</span>
        </div>
      </div>

      <div className="flex gap-3 mb-4">
        <button className="btn btn-primary" disabled={running} onClick={startGeneration}>
          ▶ Start Generation
        </button>
        {running && (
          <button className="btn btn-secondary" onClick={stopGeneration}>
            ■ Stop
          </button>
        )}
      </div>

      {/* Token output area */}
      <div className="p-4 rounded-lg mb-4" style={{
        background: '#0a0a0f', border: '1px solid #2a2a3e',
        minHeight: '80px', fontFamily: 'monospace',
      }}>
        <div className="card-title mb-2">OUTPUT</div>
        <div className="flex flex-wrap gap-1.5">
          {tokens.map((t, i) => (
            <span key={i}
                  className="token-new"
                  style={{
                    background: '#00d4ff22',
                    border: '1px solid #00d4ff44',
                    borderRadius: 4,
                    padding: '2px 6px',
                    color: '#00d4ff',
                    fontSize: '0.75rem',
                    fontWeight: 600,
                  }}>
              [{t.token_id}]
            </span>
          ))}
          {running && (
            <span style={{ color: '#f1f5f9', fontSize: '0.85rem' }} className="blink">█</span>
          )}
        </div>
        {tokens.length === 0 && !running && (
          <span style={{ color: '#94a3b8', fontSize: '0.75rem' }}>
            Press Start Generation to begin...
          </span>
        )}
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-2 gap-2">
          {[
            { label: 'Tokens Generated', value: `${stats.step} / ${stats.total}` },
            { label: 'Current Speed', value: `${stats.tps?.toFixed(1)} tok/sec` },
            { label: 'Avg Latency', value: `${stats.avgMs?.toFixed(0)} ms/token` },
            { label: 'Avg Speed', value: `${stats.avgTps?.toFixed(1)} tok/sec` },
            { label: 'Total Time', value: totalMs > 0 ? `${(totalMs / 1000).toFixed(2)}s` : 'running...' },
            { label: 'Device', value: 'SynthGPU v0.2' },
          ].map(({ label, value }) => (
            <div key={label} className="flex justify-between p-2 rounded"
                 style={{ background: '#0a0a0f', border: '1px solid #2a2a3e' }}>
              <span style={{ color: '#94a3b8', fontSize: '0.7rem' }}>{label}</span>
              <span style={{ color: '#00d4ff', fontSize: '0.75rem', fontWeight: 700,
                             fontFamily: 'monospace' }}>{value}</span>
            </div>
          ))}
        </div>
      )}

      {running && stats && (
        <div className="mt-3">
          <div style={{ height: 6, background: '#2a2a3e', borderRadius: 3, overflow: 'hidden' }}>
            <div style={{
              height: '100%',
              width: `${stats.pct}%`,
              background: 'linear-gradient(90deg, #00d4ff88, #00d4ff)',
              borderRadius: 3,
              transition: 'width 0.3s',
            }} />
          </div>
          <div style={{ color: '#94a3b8', fontSize: '0.65rem', marginTop: 4 }}>
            {stats.pct}% complete
          </div>
        </div>
      )}
    </div>
  )
}
