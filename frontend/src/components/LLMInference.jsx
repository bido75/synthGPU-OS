import { useState, useEffect, useRef, useCallback } from 'react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const PRESET_PROMPTS = [
  { label: 'What is AI?', text: 'What is artificial intelligence? Explain simply.' },
  { label: 'Explain GPUs', text: 'Explain what a GPU does and why it matters for computing.' },
  { label: 'Why AI needs GPUs?', text: 'Why does artificial intelligence need expensive GPU hardware? Explain in detail.' },
  { label: 'SynthGPU pitch', text: 'What would it mean if AI inference could run without any GPU hardware at all?' },
]

const BACKEND_URLS = { ollama: 'http://localhost:11434', lmstudio: 'http://localhost:1234' }

// ── Panel A: Backend Connection Manager ──────────────────────────────────────
function BackendConnector({ inference, onModelSelect, onConnected, systemRam }) {
  const [backendType, setBackendType] = useState('ollama')
  const [customUrl, setCustomUrl] = useState('')
  const [connecting, setConnecting] = useState(false)
  const [connResult, setConnResult] = useState(null)
  const [pullModel, setPullModel] = useState('')
  const [pulling, setPulling] = useState(false)
  const [pullProgress, setPullProgress] = useState(null)

  const connected = inference?.backend_status === 'connected'
  const backendName = inference?.backend
  const models = inference?.available_models || []
  const backendUrl = inference?.backend_url || ''
  const freeRamMB = systemRam?.available_mb || 1500

  const getModelBadge = (model) => {
    const sizeMB = model.size_mb || 0
    if (sizeMB === 0) return null
    if (sizeMB < freeRamMB * 0.7) return { label: 'FAST', color: '#10b981', bg: '#10b98122', tip: 'Fits in RAM' }
    if (sizeMB < freeRamMB) return { label: 'MARGINAL', color: '#f59e0b', bg: '#f59e0b22', tip: 'May use swap' }
    return { label: 'TOO LARGE', color: '#ef4444', bg: '#ef444422', tip: 'Will use disk swap' }
  }

  const testConnection = async () => {
    setConnecting(true)
    setConnResult(null)
    try {
      const url = backendType === 'custom' ? customUrl : BACKEND_URLS[backendType]
      const resp = await fetch('/api/inference/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ backend: backendType, url }),
      })
      const data = await resp.json()
      setConnResult(data)
      if (data.connected && onConnected) onConnected(data)
    } catch (e) {
      setConnResult({ connected: false, error: e.message })
    } finally {
      setConnecting(false)
    }
  }

  const doPull = async () => {
    if (!pullModel.trim()) return
    setPulling(true)
    setPullProgress({ status: 'starting', pct: 0, detail: '' })
    try {
      const resp = await fetch('/api/inference/pull', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: pullModel.trim() }),
      })
      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const lines = decoder.decode(value).split('\n').filter(Boolean)
        for (const line of lines) {
          try {
            const d = JSON.parse(line)
            const total = d.total || 0
            const completed = d.completed || 0
            const pct = total > 0 ? Math.round((completed / total) * 100) : 0
            setPullProgress({ status: d.status, pct, detail: d.digest || '' })
            if (d.status === 'success') {
              setPulling(false)
              setPullProgress(null)
              setPullModel('')
            }
          } catch {}
        }
      }
    } catch (e) {
      setPullProgress({ status: 'error', pct: 0, detail: e.message })
    } finally {
      setPulling(false)
    }
  }

  const statusDot = connected
    ? { color: '#10b981', shadow: '0 0 8px #10b981', label: `${backendName} Connected` }
    : connecting
    ? { color: '#f59e0b', shadow: '0 0 6px #f59e0b', label: 'Connecting...' }
    : { color: '#ef4444', shadow: '0 0 4px #ef444488', label: 'No LLM Backend' }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <div className="card-title mb-0">LLM Backend Connection</div>
        <div className="flex items-center gap-2">
          <div style={{ width: 8, height: 8, borderRadius: '50%',
                        background: statusDot.color, boxShadow: statusDot.shadow,
                        transition: 'all 0.4s' }} />
          <span style={{ fontSize: '0.72rem', fontWeight: 600,
                          color: connected ? '#10b981' : connecting ? '#f59e0b' : '#ef4444' }}>
            {statusDot.label}
          </span>
        </div>
      </div>

      {/* Backend type selector */}
      <div className="flex gap-2 mb-3">
        {[
          { id: 'ollama', label: '● Ollama' },
          { id: 'lmstudio', label: '○ LM Studio' },
          { id: 'custom', label: '○ Custom URL' },
        ].map(b => (
          <button key={b.id}
                  onClick={() => setBackendType(b.id)}
                  className="btn"
                  style={{
                    background: backendType === b.id ? '#00d4ff22' : '#1a1a2e',
                    border: `1px solid ${backendType === b.id ? '#00d4ff66' : '#2a2a3e'}`,
                    color: backendType === b.id ? '#00d4ff' : '#94a3b8',
                    fontSize: '0.75rem', padding: '0.35rem 0.75rem',
                  }}>
            {b.label}
          </button>
        ))}
      </div>

      <div className="flex gap-2 mb-2">
        <input
          value={backendType === 'custom' ? customUrl : (BACKEND_URLS[backendType] || '')}
          onChange={e => backendType === 'custom' && setCustomUrl(e.target.value)}
          readOnly={backendType !== 'custom'}
          style={{
            flex: 1, background: '#0a0a0f', border: '1px solid #2a2a3e',
            borderRadius: 8, padding: '0.45rem 0.75rem', color: '#f1f5f9',
            fontSize: '0.8rem', fontFamily: 'monospace',
          }}
        />
        <button className="btn btn-primary" onClick={testConnection} disabled={connecting}>
          {connecting ? '⏳' : '⚡'} Test Connection
        </button>
      </div>

      {connResult && (
        <div className="mb-3" style={{ fontSize: '0.75rem' }}>
          {connResult.connected ? (
            <div style={{ color: '#10b981', background: '#10b98111',
                          border: '1px solid #10b98133', borderRadius: 8,
                          padding: '0.6rem 0.75rem' }}>
              ✓ {connResult.message || `${connResult.backend} detected · ${connResult.models?.length || 0} models`}
            </div>
          ) : (
            <div style={{ background: '#ef444411', border: '1px solid #ef444433',
                          borderRadius: 8, padding: '0.6rem 0.75rem' }}>
              <div style={{ color: '#ef4444', fontWeight: 600, marginBottom: 4 }}>
                ✗ {connResult.message || connResult.error || 'Connection failed'}
              </div>
              {connResult.fix && (
                <div style={{ marginTop: 6 }}>
                  <span style={{ color: '#94a3b8' }}>Run this command:</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
                    <code style={{ color: '#00d4ff', background: '#00d4ff11',
                                   padding: '2px 8px', borderRadius: 4, fontFamily: 'monospace' }}>
                      {connResult.fix}
                    </code>
                    <button
                      onClick={() => navigator.clipboard.writeText(connResult.fix)}
                      style={{ color: '#94a3b8', background: '#2a2a3e',
                               border: '1px solid #3a3a4e', borderRadius: 4,
                               padding: '1px 8px', fontSize: '0.65rem', cursor: 'pointer' }}
                      onMouseEnter={e => e.currentTarget.style.color = '#00d4ff'}
                      onMouseLeave={e => e.currentTarget.style.color = '#94a3b8'}>
                      Copy
                    </button>
                  </div>
                </div>
              )}
              {connResult.fix_windows && (
                <div style={{ color: '#94a3b8', fontSize: '0.68rem', marginTop: 4 }}>
                  Windows: {connResult.fix_windows}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Model list */}
      {models.length > 0 ? (
        <div>
          <div className="card-title mb-2">Available Models</div>
          <div style={{ border: '1px solid #2a2a3e', borderRadius: 8, overflow: 'hidden' }}>
            {models.map((m, i) => (
              <div key={m.name}
                   className="flex items-center justify-between px-3 py-2"
                   style={{
                     borderBottom: i < models.length - 1 ? '1px solid #1a1a2e' : 'none',
                     background: 'transparent',
                     transition: 'background 0.15s',
                   }}
                   onMouseEnter={e => e.currentTarget.style.background = '#00d4ff0a'}
                   onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <span style={{ color: '#f1f5f9', fontWeight: 600, fontSize: '0.8rem',
                                 overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {m.name}
                  </span>
                  {m.size_mb > 0 && (
                    <span style={{ color: '#94a3b8', fontSize: '0.7rem', whiteSpace: 'nowrap' }}>
                      {m.size_mb >= 1000 ? `${(m.size_mb / 1000).toFixed(1)}GB` : `${m.size_mb}MB`}
                    </span>
                  )}
                  {(() => {
                    const badge = getModelBadge(m)
                    if (!badge) return null
                    return (
                      <span title={badge.tip} style={{
                        fontSize: '0.6rem', fontWeight: 700, padding: '1px 5px', borderRadius: 4,
                        color: badge.color, background: badge.bg, border: `1px solid ${badge.color}55`,
                        whiteSpace: 'nowrap', letterSpacing: '0.04em',
                      }}>{badge.label}</span>
                    )
                  })()}
                </div>
                <button className="btn btn-primary"
                        style={{ fontSize: '0.7rem', padding: '0.25rem 0.65rem', whiteSpace: 'nowrap' }}
                        onClick={() => onModelSelect && onModelSelect(m.name)}>
                  ▶ Run
                </button>
              </div>
            ))}
          </div>
        </div>
      ) : connected ? (
        <div style={{ color: '#94a3b8', fontSize: '0.75rem', padding: '0.75rem',
                      border: '1px dashed #2a2a3e', borderRadius: 8, textAlign: 'center' }}>
          No models found. Pull one below ↓
        </div>
      ) : (
        <div style={{ color: '#94a3b8', fontSize: '0.8rem', padding: '1rem',
                      textAlign: 'center', border: '1px dashed #2a2a3e', borderRadius: 8 }}>
          Start Ollama or LM Studio, then click Test Connection
        </div>
      )}

      {/* Pull model (Ollama only) */}
      {connected && backendName === 'Ollama' && (
        <div className="mt-3">
          <div className="card-title mb-2">+ Pull New Model</div>
          <div className="flex gap-2">
            <input
              value={pullModel}
              onChange={e => setPullModel(e.target.value)}
              placeholder="e.g. llama3.2:1b"
              style={{
                flex: 1, background: '#0a0a0f', border: '1px solid #2a2a3e',
                borderRadius: 8, padding: '0.4rem 0.75rem', color: '#f1f5f9',
                fontSize: '0.8rem', fontFamily: 'monospace',
              }}
            />
            <button className="btn btn-secondary" onClick={doPull} disabled={pulling || !pullModel.trim()}>
              {pulling ? '⏳ Pulling...' : '⬇ Pull'}
            </button>
          </div>
          {pullProgress && (
            <div className="mt-2">
              <div className="flex justify-between mb-1">
                <span style={{ color: '#94a3b8', fontSize: '0.7rem' }}>
                  {pullProgress.status}
                </span>
                <span style={{ color: '#00d4ff', fontSize: '0.7rem' }}>{pullProgress.pct}%</span>
              </div>
              <div style={{ height: 5, background: '#2a2a3e', borderRadius: 3, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${pullProgress.pct}%`,
                              background: 'linear-gradient(90deg, #00d4ff88, #00d4ff)',
                              borderRadius: 3, transition: 'width 0.3s' }} />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}


// ── Panel B: Live Inference Console ──────────────────────────────────────────
function InferenceConsole({ inference, selectedModel, onModelChange }) {
  const [model, setModel] = useState(selectedModel || '')
  const [prompt, setPrompt] = useState('')
  const [output, setOutput] = useState('')
  const [running, setRunning] = useState(false)
  const [metrics, setMetrics] = useState(null)
  const [error, setError] = useState(null)
  const [showRawJson, setShowRawJson] = useState(false)
  const [inferencePhase, setInferencePhase] = useState(null)
  const [preflightResult, setPreflightResult] = useState(null)
  const [elapsedMs, setElapsedMs] = useState(0)
  const elapsedTimerRef = useRef(null)
  const wsRef = useRef(null)
  const typewriteQueue = useRef([])
  const isTyping = useRef(false)
  const outputRef = useRef(null)
  const totalTokensRef = useRef(0)
  const startTimeRef = useRef(0)
  const allTpsRef = useRef([])

  const connected = inference?.backend_status === 'connected'
  const models = inference?.available_models || []

  useEffect(() => {
    if (selectedModel) setModel(selectedModel)
  }, [selectedModel])

  // Preflight check when model changes
  useEffect(() => {
    if (!model) { setPreflightResult(null); return }
    const info = models.find(m => m.name === model)
    if (!info) { setPreflightResult(null); return }
    fetch('/api/inference/preflight', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, size_mb: info.size_mb || 0 }),
    })
      .then(r => r.json())
      .then(d => setPreflightResult(d))
      .catch(() => setPreflightResult(null))
  }, [model])

  // Typewriter effect
  const drainQueue = useCallback(() => {
    if (typewriteQueue.current.length === 0) {
      isTyping.current = false
      return
    }
    isTyping.current = true
    const chars = typewriteQueue.current.splice(0, 2).join('')
    setOutput(prev => prev + chars)
    setTimeout(drainQueue, 15)
  }, [])

  const typewriteToken = useCallback((token) => {
    typewriteQueue.current.push(...token.split(''))
    if (!isTyping.current) drainQueue()
  }, [drainQueue])

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight
    }
  }, [output])

  const connectTokenWS = () => {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/tokens`)
    wsRef.current = ws

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (data.type === 'token' && data.token) {
        setInferencePhase(p => p === 'loading_model' || p === null ? 'generating' : p)
        typewriteToken(data.token)
        totalTokensRef.current += 1
        allTpsRef.current.push(data.tokens_per_sec || 0)
        const avgTps = allTpsRef.current.reduce((a, b) => a + b, 0) / allTpsRef.current.length
        setMetrics({
          totalTokens: totalTokensRef.current,
          currentTps: data.tokens_per_sec || 0,
          avgTps: avgTps.toFixed(1),
          latencyMs: data.token_ms || 0,
          warpsDispatched: (inference?.active_session?.tokens_so_far || 0) * 50,
        })
      } else if (data.type === 'done') {
        setRunning(false)
        setInferencePhase('done')
        clearInterval(elapsedTimerRef.current)
      } else if (data.type === 'error') {
        setError(data.message)
        setRunning(false)
        setInferencePhase(null)
        clearInterval(elapsedTimerRef.current)
      }
    }
    ws.onerror = () => { setError('WebSocket connection failed'); setRunning(false); setInferencePhase(null) }
    return ws
  }

  const startInference = async () => {
    if (!connected) { setError('No backend connected — use the panel above'); return }
    if (!model) { setError('Select a model first'); return }
    if (!prompt.trim()) { setError('Enter a prompt'); return }

    setOutput('')
    setError(null)
    setRunning(true)
    setMetrics(null)
    setInferencePhase('loading_model')
    setElapsedMs(0)
    clearInterval(elapsedTimerRef.current)
    const t0 = Date.now()
    elapsedTimerRef.current = setInterval(() => setElapsedMs(Date.now() - t0), 100)
    totalTokensRef.current = 0
    allTpsRef.current = []
    typewriteQueue.current = []
    isTyping.current = false
    startTimeRef.current = t0

    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      connectTokenWS()
      await new Promise(r => setTimeout(r, 300))
    }

    try {
      const resp = await fetch('/api/inference/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, prompt, stream: true }),
      })
      if (!resp.ok) {
        const err = await resp.json()
        throw new Error(err.error || 'Inference failed')
      }
    } catch (e) {
      setError(e.message)
      setRunning(false)
      setInferencePhase(null)
      clearInterval(elapsedTimerRef.current)
    }
  }

  const stopInference = () => {
    wsRef.current?.close()
    wsRef.current = null
    setRunning(false)
    setInferencePhase(null)
    clearInterval(elapsedTimerRef.current)
    isTyping.current = false
    typewriteQueue.current = []
  }

  useEffect(() => {
    connectTokenWS()
    return () => { wsRef.current?.close(); clearInterval(elapsedTimerRef.current) }
  }, [])

  const rawJson = {
    model, prompt: prompt.slice(0, 60) + '...',
    response: output.slice(0, 100) + '...',
    synthgpu: {
      device: 'SynthGPU Virtual Accelerator',
      warps_executed: (metrics?.warpsDispatched || 0),
      vram_used_mb: inference?.memory?.vram_used_mb || 0,
      kv_cache_mb: inference?.memory?.kv_cache_mb || 0,
      no_physical_gpu: true,
      tokens_per_sec: metrics?.currentTps || 0,
    }
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <div className="card-title mb-0">Live Inference Console</div>
        <span className="badge-no-gpu">GPU HARDWARE REQUIRED: ✗ NONE</span>
      </div>

      {/* Model + backend selector */}
      <div className="flex gap-2 mb-3">
        <select
          value={model}
          onChange={e => { setModel(e.target.value); onModelChange && onModelChange(e.target.value) }}
          style={{
            flex: 1, background: '#0a0a0f', border: '1px solid #2a2a3e',
            borderRadius: 8, padding: '0.4rem 0.75rem', color: '#f1f5f9',
            fontSize: '0.8rem',
          }}>
          <option value="">Select model...</option>
          {models.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
          {model && !models.find(m => m.name === model) && (
            <option value={model}>{model}</option>
          )}
        </select>
        <div style={{ background: '#0a0a0f', border: '1px solid #2a2a3e',
                       borderRadius: 8, padding: '0.4rem 0.75rem',
                       color: '#94a3b8', fontSize: '0.75rem', display: 'flex',
                       alignItems: 'center' }}>
          via {inference?.backend || 'Ollama'}
        </div>
      </div>

      {/* Prompt presets */}
      <div className="flex flex-wrap gap-1.5 mb-2">
        {PRESET_PROMPTS.map(p => (
          <button key={p.label}
                  style={{ background: '#1a1a2e', border: '1px solid #2a2a3e',
                            borderRadius: 6, padding: '0.2rem 0.6rem',
                            color: '#94a3b8', fontSize: '0.68rem', cursor: 'pointer',
                            transition: 'all 0.15s' }}
                  onMouseEnter={e => { e.currentTarget.style.color = '#00d4ff'; e.currentTarget.style.borderColor = '#00d4ff44' }}
                  onMouseLeave={e => { e.currentTarget.style.color = '#94a3b8'; e.currentTarget.style.borderColor = '#2a2a3e' }}
                  onClick={() => setPrompt(p.text)}>
            {p.label}
          </button>
        ))}
      </div>

      {/* Prompt textarea */}
      <textarea
        value={prompt}
        onChange={e => setPrompt(e.target.value)}
        placeholder="Enter a prompt... (try 'Why does AI need expensive GPU hardware?')"
        rows={3}
        style={{
          width: '100%', background: '#0a0a0f', border: '1px solid #2a2a3e',
          borderRadius: 8, padding: '0.6rem 0.75rem', color: '#f1f5f9',
          fontSize: '0.85rem', resize: 'vertical', outline: 'none', marginBottom: '0.75rem',
        }}
      />

      <div className="flex gap-2 mb-4">
        <button className="btn btn-primary" disabled={running || !connected} onClick={startInference}>
          ▶ Run Through SynthGPU
        </button>
        {running && (
          <button className="btn btn-secondary" onClick={stopInference}>■ Stop</button>
        )}
        <button className="btn" style={{ background: '#1a1a2e', border: '1px solid #2a2a3e',
                                          color: '#94a3b8', fontSize: '0.75rem' }}
                onClick={() => { setOutput(''); setMetrics(null); setError(null) }}>
          Clear
        </button>
        {output && (
          <button className="btn" style={{ background: '#1a1a2e', border: '1px solid #2a2a3e',
                                            color: '#94a3b8', fontSize: '0.75rem', marginLeft: 'auto' }}
                  onClick={() => navigator.clipboard.writeText(output)}>
            Copy
          </button>
        )}
        {output && (
          <button className="btn" style={{ background: '#7c3aed22', border: '1px solid #7c3aed44',
                                            color: '#a78bfa', fontSize: '0.72rem' }}
                  onClick={() => setShowRawJson(!showRawJson)}>
            {showRawJson ? 'Hide' : 'View'} Raw JSON
          </button>
        )}
      </div>

      {!connected && (
        <div style={{ color: '#f59e0b', fontSize: '0.8rem', padding: '0.75rem',
                      background: '#f59e0b11', border: '1px solid #f59e0b33',
                      borderRadius: 8, marginBottom: '0.75rem' }}>
          Connect a backend above to run inference
        </div>
      )}

      {/* Preflight warning */}
      {preflightResult && !running && (
        <div style={{
          padding: '0.6rem 0.85rem', borderRadius: 8, marginBottom: '0.75rem',
          fontSize: '0.78rem',
          background: preflightResult.status === 'fast' ? '#052e1644' : preflightResult.status === 'marginal' ? '#451a0344' : '#450a0a44',
          border: `1px solid ${preflightResult.status === 'fast' ? '#10b98155' : preflightResult.status === 'marginal' ? '#f59e0b55' : '#ef444455'}`,
          color: preflightResult.status === 'fast' ? '#10b981' : preflightResult.status === 'marginal' ? '#f59e0b' : '#ef4444',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>{preflightResult.message}</span>
            <span style={{ fontFamily: 'monospace', fontWeight: 700, marginLeft: 12 }}>
              Est: {preflightResult.estimated_tps} tok/sec
            </span>
          </div>
          {preflightResult.status === 'slow' && (
            <div style={{ marginTop: 4, fontSize: '0.72rem', opacity: 0.85 }}>
              {preflightResult.recommendation}
              <button onClick={() => { setModel('tinyllama:latest'); onModelChange?.('tinyllama:latest') }}
                      style={{ marginLeft: 8, color: '#00d4ff', textDecoration: 'underline',
                               background: 'none', border: 'none', cursor: 'pointer', fontSize: '0.72rem' }}>
                Switch to tinyllama
              </button>
            </div>
          )}
          <div style={{ marginTop: 4, fontSize: '0.68rem', opacity: 0.6, fontFamily: 'monospace' }}>
            Free RAM: {preflightResult.free_ram_mb}MB &nbsp;|&nbsp;
            Model: {preflightResult.model_size_mb}MB &nbsp;|&nbsp;
            KV cache: {preflightResult.kv_cache_mb}MB &nbsp;|&nbsp;
            Context: {preflightResult.safe_ctx} tokens
          </div>
        </div>
      )}

      {/* Phase indicator */}
      {inferencePhase && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: '0.75rem' }}>
          {inferencePhase === 'loading_model' && (
            <div style={{ color: '#f59e0b', fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: '1.1rem' }}>⏳</span>
              <div>
                <div style={{ fontWeight: 600 }}>Loading model into virtual VRAM...</div>
                <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>
                  {(elapsedMs / 1000).toFixed(1)}s elapsed — first run takes 5–30 seconds, subsequent runs are instant
                </div>
              </div>
            </div>
          )}
          {inferencePhase === 'generating' && (
            <div style={{ color: '#00d4ff', fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: '1.1rem', animation: 'pulse 1s infinite' }}>⚡</span>
              <div>
                <div style={{ fontWeight: 600 }}>
                  Generating... {metrics?.currentTps?.toFixed(1) || '0.0'} tok/sec
                </div>
                <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>
                  {metrics?.totalTokens || 0} tokens &nbsp;·&nbsp; {(elapsedMs / 1000).toFixed(1)}s elapsed
                </div>
              </div>
            </div>
          )}
          {inferencePhase === 'done' && (
            <div style={{ color: '#10b981', fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: '1.1rem' }}>✓</span>
              <div>
                <div style={{ fontWeight: 600 }}>
                  Complete — {metrics?.totalTokens || 0} tokens in {(elapsedMs / 1000).toFixed(1)}s
                </div>
                <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>
                  Average: {metrics?.avgTps || '0.0'} tok/sec
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {error && (
        <div style={{ color: '#ef4444', fontSize: '0.8rem', padding: '0.75rem',
                      background: '#ef444411', border: '1px solid #ef444433',
                      borderRadius: 8, marginBottom: '0.75rem' }}>
          {error}
        </div>
      )}

      {/* Raw JSON viewer */}
      {showRawJson && output && (
        <div className="mb-3 p-3 rounded-lg" style={{ background: '#0a0a0f',
                border: '1px solid #7c3aed44', fontSize: '0.7rem',
                fontFamily: 'monospace', color: '#a78bfa', overflow: 'auto', maxHeight: 200 }}>
          <div style={{ color: '#94a3b8', fontSize: '0.65rem', marginBottom: 4 }}>
            Response includes injected SynthGPU metadata:
          </div>
          <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
            {JSON.stringify(rawJson, null, 2)}
          </pre>
        </div>
      )}

      {/* Output area */}
      <div style={{ background: '#0a0a0f', border: '1px solid #2a2a3e',
                    borderRadius: 8, padding: '0.75rem', minHeight: 90,
                    marginBottom: '0.75rem' }}>
        <div style={{ color: '#94a3b8', fontSize: '0.65rem', fontWeight: 700,
                      letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 6 }}>
          Output
        </div>
        <div ref={outputRef}
             style={{ color: '#f1f5f9', fontSize: '0.85rem', lineHeight: 1.6,
                      maxHeight: 200, overflowY: 'auto', whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word' }}>
          {output || <span style={{ color: '#94a3b8' }}>Output will appear here...</span>}
          {running && <span className="blink" style={{ color: '#00d4ff' }}>█</span>}
        </div>
      </div>

      {/* SynthGPU Metrics */}
      {metrics && (
        <div style={{ background: '#0a0a0f', border: '1px solid #00d4ff22',
                      borderRadius: 8, padding: '0.75rem' }}>
          <div style={{ color: '#94a3b8', fontSize: '0.65rem', fontWeight: 700,
                        letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 8 }}>
            SynthGPU Metrics
          </div>
          <div className="grid grid-cols-2 gap-2">
            {[
              { label: 'Tokens Generated', value: metrics.totalTokens },
              { label: 'Speed', value: `${metrics.currentTps} tok/sec` },
              { label: 'Avg Speed', value: `${metrics.avgTps} tok/sec` },
              { label: 'Avg Latency', value: `${metrics.latencyMs?.toFixed(0)}ms/token` },
              { label: 'vRAM Used', value: `~${(inference?.memory?.vram_used_mb || 0).toFixed(0)} MB` },
              { label: 'KV Cache', value: `${(inference?.memory?.kv_cache_mb || 0).toFixed(1)} MB ↑` },
              { label: 'vRAM Source', value: 'System RAM pool' },
              { label: 'Physical GPU', value: '✗ NONE — 100% SynthGPU' },
            ].map(({ label, value }) => (
              <div key={label} className="flex justify-between py-1"
                   style={{ borderBottom: '1px solid #1a1a2e' }}>
                <span style={{ color: '#94a3b8', fontSize: '0.7rem' }}>{label}</span>
                <span style={{ color: '#00d4ff', fontSize: '0.72rem', fontWeight: 600,
                               fontFamily: 'monospace' }}>{value}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}


// ── Panel C: Inference Telemetry ─────────────────────────────────────────────
function InferenceTelemetry({ inference }) {
  const [speedHistory, setSpeedHistory] = useState([])
  const prevTokenCount = useRef(0)

  useEffect(() => {
    const recentTokens = inference?.recent_tokens || []
    if (recentTokens.length > 0) {
      const last = recentTokens[recentTokens.length - 1]
      if (last.total_tokens !== prevTokenCount.current) {
        prevTokenCount.current = last.total_tokens
        setSpeedHistory(prev => {
          const next = [...prev, {
            token: last.total_tokens,
            tps: last.tokens_per_sec || 0,
          }]
          return next.slice(-60)
        })
      }
    }
  }, [inference?.recent_tokens])

  const sessions = inference?.session_history || []
  const active = inference?.active_session
  const memInfo = inference?.memory_extension || {}
  const modelMb = inference?.memory?.model_weights_mb || 0
  const kvMb = inference?.memory?.kv_cache_mb || 0
  const totalMb = inference?.memory?.vram_total_mb || 4096
  const usedMb = inference?.memory?.vram_used_mb || 0

  const CustomTooltip = ({ active: a, payload, label }) => {
    if (!a || !payload?.length) return null
    return (
      <div style={{ background: '#1a1a2e', border: '1px solid #2a2a3e',
                    borderRadius: 6, padding: '6px 10px' }}>
        <div style={{ color: '#94a3b8', fontSize: '0.65rem' }}>Token #{label}</div>
        <div style={{ color: '#00d4ff', fontSize: '0.75rem', fontWeight: 700 }}>
          {payload[0]?.value?.toFixed(1)} tok/sec
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <div className="card-title">Inference Telemetry</div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        {/* Token speed chart */}
        <div>
          <div style={{ color: '#94a3b8', fontSize: '0.7rem', fontWeight: 700,
                        textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
            Token Speed
          </div>
          <div style={{ height: 120 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={speedHistory} margin={{ top: 5, right: 5, left: -25, bottom: 5 }}>
                <defs>
                  <linearGradient id="tpsGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#00d4ff" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3e" />
                <XAxis dataKey="token" tick={{ fill: '#94a3b8', fontSize: 9 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 9 }} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="tps" stroke="#00d4ff" strokeWidth={2}
                      fill="url(#tpsGrad)" dot={false} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div style={{ color: '#94a3b8', fontSize: '0.7rem', marginTop: 4 }}>
            Current: <span style={{ color: '#00d4ff', fontWeight: 700 }}>
              {inference?.current_tokens_per_sec?.toFixed(1) || '—'} tok/sec
            </span>
          </div>
        </div>

        {/* vRAM breakdown */}
        <div>
          <div style={{ color: '#94a3b8', fontSize: '0.7rem', fontWeight: 700,
                        textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
            vRAM During Inference
          </div>
          {[
            { label: 'Model weights', value: `~${modelMb.toFixed(0)} MB`, color: '#7c3aed' },
            { label: 'KV cache', value: `${kvMb.toFixed(1)} MB ↑`, color: '#00d4ff' },
            { label: 'Total used', value: `${usedMb.toFixed(0)} MB`, color: '#f1f5f9' },
            { label: 'Free vRAM', value: `${(totalMb - usedMb).toFixed(0)} MB`, color: '#10b981' },
          ].map(({ label, value, color }) => (
            <div key={label} className="flex justify-between py-1.5"
                 style={{ borderBottom: '1px solid #1a1a2e' }}>
              <div className="flex items-center gap-2">
                <div style={{ width: 8, height: 8, borderRadius: 2, background: color }} />
                <span style={{ color: '#94a3b8', fontSize: '0.72rem' }}>{label}</span>
              </div>
              <span style={{ color, fontSize: '0.75rem', fontWeight: 600,
                             fontFamily: 'monospace' }}>{value}</span>
            </div>
          ))}
          <div style={{ marginTop: 8, padding: '6px 8px', background: '#10b98111',
                        border: '1px solid #10b98133', borderRadius: 6 }}>
            <span style={{ color: '#10b981', fontSize: '0.65rem', fontWeight: 700 }}>
              Source: System RAM pool (NOT hard drive)
            </span>
          </div>
        </div>
      </div>

      {/* Session history */}
      {sessions.length > 0 && (
        <div>
          <div className="card-title mb-2">Session History</div>
          <div style={{ border: '1px solid #2a2a3e', borderRadius: 8, overflow: 'hidden' }}>
            {sessions.slice().reverse().map((s, i) => (
              <div key={i} className="flex items-center gap-3 px-3 py-2"
                   style={{ borderBottom: i < sessions.length - 1 ? '1px solid #1a1a2e' : 'none' }}>
                <span style={{ color: '#00d4ff', fontSize: '0.75rem', fontWeight: 600,
                               minWidth: 100 }}>{s.model}</span>
                <span style={{ color: '#94a3b8', fontSize: '0.7rem', flex: 1,
                               overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  "{s.prompt_preview}"
                </span>
                <span style={{ color: '#f1f5f9', fontSize: '0.7rem', fontFamily: 'monospace',
                               whiteSpace: 'nowrap' }}>
                  {s.tokens} tok
                </span>
                <span style={{ color: '#10b981', fontSize: '0.7rem', fontFamily: 'monospace',
                               whiteSpace: 'nowrap' }}>
                  {s.avg_tps} t/s
                </span>
                <span style={{ color: '#10b981', fontSize: '0.65rem' }}>✓</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {active && (
        <div className="mt-3 p-3 rounded-lg flex items-center gap-3"
             style={{ background: '#00d4ff0a', border: '1px solid #00d4ff33' }}>
          <div className="w-2 h-2 rounded-full blink"
               style={{ background: '#00d4ff', boxShadow: '0 0 6px #00d4ff', minWidth: 8 }} />
          <div style={{ fontSize: '0.75rem' }}>
            <span style={{ color: '#00d4ff', fontWeight: 600 }}>{active.model}</span>
            <span style={{ color: '#94a3b8' }}> · Token {active.tokens_so_far} · </span>
            <span style={{ color: '#f1f5f9' }}>"{active.prompt_preview}"</span>
          </div>
        </div>
      )}
    </div>
  )
}


// ── Main LLMInference Tab ─────────────────────────────────────────────────────
export default function LLMInference({ telemetry }) {
  const [selectedModel, setSelectedModel] = useState('')
  const inference = telemetry?.inference || {}
  const memory = telemetry?.memory || {}
  const systemRam = telemetry?.system_ram

  const enrichedInference = {
    ...inference,
    memory: {
      ...memory,
      kv_cache_mb: memory.kv_cache_mb || 0,
      model_weights_mb: memory.model_weights_mb || 0,
      vram_used_mb: memory.vram_used_mb || 0,
      vram_total_mb: memory.vram_total_mb || 4096,
    },
  }

  // Auto-select best model based on free RAM when models load
  const models = enrichedInference?.available_models || []
  const freeRamMB = systemRam?.available_mb || 1500
  useEffect(() => {
    if (selectedModel || models.length === 0) return
    const comfortable = [...models]
      .filter(m => (m.size_mb || 0) < freeRamMB * 0.7)
      .sort((a, b) => b.size_mb - a.size_mb)
    if (comfortable.length > 0) {
      setSelectedModel(comfortable[0].name)
    } else {
      const smallest = [...models].sort((a, b) => a.size_mb - b.size_mb)[0]
      if (smallest) setSelectedModel(smallest.name)
    }
  }, [models.length, freeRamMB])

  return (
    <div className="flex flex-col gap-4">
      <BackendConnector
        inference={enrichedInference}
        onModelSelect={setSelectedModel}
        onConnected={() => {}}
        systemRam={systemRam}
      />
      <InferenceConsole
        inference={enrichedInference}
        selectedModel={selectedModel}
        onModelChange={setSelectedModel}
      />
      <InferenceTelemetry inference={enrichedInference} />
    </div>
  )
}
