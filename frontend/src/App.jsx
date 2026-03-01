import { useState, useEffect, useRef, useCallback } from 'react'
import DevicePanel from './components/DevicePanel.jsx'
import WarpMonitor from './components/WarpMonitor.jsx'
import MemoryGauge from './components/MemoryGauge.jsx'
import PerformanceChart from './components/PerformanceChart.jsx'
import BenchmarkRunner from './components/BenchmarkRunner.jsx'
import TokenGenerator from './components/TokenGenerator.jsx'
import ModelUploader from './components/ModelUploader.jsx'
import EconomicsPanel from './components/EconomicsPanel.jsx'
import LLMInference from './components/LLMInference.jsx'

const TABS = ['Dashboard', 'Benchmarks', 'Demo', 'ONNX Model', 'LLM Inference']

const TAB_ICONS = {
  'Dashboard': '⬛',
  'Benchmarks': '📊',
  'Demo': '🎬',
  'ONNX Model': '🧠',
  'LLM Inference': '⚡',
}

const DEMO_STEPS = [
  { title: 'The Device', text: 'This machine has no GPU. SynthGPU just registered one in software.' },
  { title: 'The Performance', text: 'Real AI workloads. Real performance. No GPU hardware.' },
  { title: 'The Token Generator', text: 'This is what ChatGPT does per token. Running on SynthGPU.' },
  { title: 'The Economics', text: '83% cost reduction. Instant availability. Any machine.' },
  { title: 'The Real Model', text: 'A real language model answering a question about GPU hardware — running through SynthGPU\'s virtual device. No physical GPU. On this machine. Right now.' },
]

export default function App() {
  const [tab, setTab] = useState('Dashboard')
  const [telemetry, setTelemetry] = useState(null)
  const [connected, setConnected] = useState(false)
  const [chartData, setChartData] = useState([])
  const [demoMode, setDemoMode] = useState(false)
  const [demoStep, setDemoStep] = useState(0)
  const [demoText, setDemoText] = useState('')
  const wsRef = useRef(null)
  const reconnectTimer = useRef(null)
  const demoTimer = useRef(null)

  const connectWS = useCallback(() => {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const host = window.location.host
    const ws = new WebSocket(`${proto}://${host}/ws/telemetry`)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      clearTimeout(reconnectTimer.current)
    }

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (data.type === 'connected' || data.type === 'telemetry') {
        setTelemetry(data)
        if (data.scheduler) {
          const t = new Date()
          const label = `${t.getHours().toString().padStart(2,'0')}:${t.getMinutes().toString().padStart(2,'0')}:${t.getSeconds().toString().padStart(2,'0')}`
          setChartData(prev => {
            const next = [...prev, {
              time: label,
              throughput: data.scheduler.warp_throughput_per_sec || 0,
              utilization: data.scheduler.utilization_pct || 0,
            }]
            return next.slice(-60)
          })
        }
      }
    }

    ws.onclose = () => {
      setConnected(false)
      reconnectTimer.current = setTimeout(connectWS, 2000)
    }
    ws.onerror = () => ws.close()
  }, [])

  useEffect(() => {
    connectWS()
    return () => {
      clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connectWS])

  const runDemoStep = (step) => {
    setDemoStep(step)
    setDemoText(DEMO_STEPS[step].text)
    const tabMap = ['Dashboard', 'Benchmarks', 'Demo', 'Dashboard', 'LLM Inference']
    setTab(tabMap[step])

    // Step 5: auto-fill inference prompt and run if connected
    if (step === 4) {
      setTimeout(() => {
        const inferenceConnected = telemetry?.inference?.backend_status === 'connected'
        if (!inferenceConnected) {
          // Stay on LLM Inference tab, user sees "connect" message
        }
      }, 500)
    }

    if (step < DEMO_STEPS.length - 1) {
      demoTimer.current = setTimeout(() => runDemoStep(step + 1), 7000)
    } else {
      demoTimer.current = setTimeout(() => {
        setDemoMode(false)
        setDemoText('')
        setTab('Dashboard')
      }, 10000)
    }
  }

  const startDemoMode = () => {
    setDemoMode(true)
    setDemoStep(0)
    setTab('Dashboard')
    runDemoStep(0)
  }

  const stopDemo = () => {
    setDemoMode(false)
    setDemoText('')
    clearTimeout(demoTimer.current)
  }

  const sched = telemetry?.scheduler || {}
  const mem = telemetry?.memory || {}
  const dev = telemetry?.device || {}
  const inference = telemetry?.inference || {}

  // Header inference badge
  const inferenceActive = inference?.active === true
  const backendConnected = inference?.backend_status === 'connected'
  const backendName = inference?.backend
  const activeModel = inference?.active_model

  let headerBadge = null
  if (inferenceActive && activeModel) {
    headerBadge = { color: '#00d4ff', label: `● Generating: ${activeModel}`, pulse: true }
  } else if (backendConnected) {
    headerBadge = { color: '#10b981', label: `● ${backendName} Connected`, pulse: false }
  } else {
    headerBadge = { color: '#ef4444', label: '✗ No LLM Backend', pulse: false }
  }

  return (
    <div className={`min-h-screen flex flex-col ${demoMode ? 'demo-mode-border' : ''}`}
         style={{ background: '#0a0a0f' }}>

      {/* Header */}
      <header style={{ background: '#12121a', borderBottom: '1px solid #2a2a3e' }}
              className="px-6 py-3 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="text-2xl">⚡</div>
          <div>
            <div className="font-bold text-white text-base leading-tight">SynthGPU</div>
            <div style={{ color: '#94a3b8', fontSize: '0.65rem' }}>Virtual GPU Accelerator v0.2-beta</div>
          </div>
          <span className="badge-no-gpu ml-2">NO PHYSICAL GPU</span>
        </div>

        <div className="flex items-center gap-3">
          {/* LLM Backend badge */}
          <button
            onClick={() => setTab('LLM Inference')}
            style={{
              background: inferenceActive ? '#00d4ff11' : backendConnected ? '#10b98111' : '#ef444411',
              border: `1px solid ${inferenceActive ? '#00d4ff44' : backendConnected ? '#10b98144' : '#ef444433'}`,
              borderRadius: 8, padding: '0.3rem 0.75rem', cursor: 'pointer',
              display: 'flex', alignItems: 'center', gap: 6,
              transition: 'all 0.2s',
            }}>
            <div style={{
              width: 6, height: 6, borderRadius: '50%',
              background: headerBadge.color,
              boxShadow: `0 0 ${headerBadge.pulse ? '8px' : '4px'} ${headerBadge.color}88`,
              animation: headerBadge.pulse ? 'blink 1s step-end infinite' : 'none',
            }} />
            <span style={{ fontSize: '0.7rem', fontWeight: 600, color: headerBadge.color }}>
              {headerBadge.label}
            </span>
          </button>

          {/* Connection status */}
          {connected ? (
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-400"
                   style={{ boxShadow: '0 0 6px #10b981' }} />
              <span style={{ color: '#10b981', fontSize: '0.75rem', fontWeight: 600 }}>LIVE</span>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-yellow-500 blink" />
              <span style={{ color: '#f59e0b', fontSize: '0.75rem' }}>Reconnecting...</span>
            </div>
          )}

          {demoMode ? (
            <button className="btn btn-secondary" onClick={stopDemo}>■ Stop Demo</button>
          ) : (
            <button className="btn btn-primary" onClick={startDemoMode}>▶ Investor Demo</button>
          )}
        </div>
      </header>

      {/* Demo Banner */}
      {demoMode && (
        <div className="px-6 py-2 flex items-center gap-4"
             style={{ background: '#00d4ff11', borderBottom: '1px solid #00d4ff33' }}>
          <span style={{ color: '#00d4ff', fontSize: '0.7rem', fontWeight: 700, letterSpacing: '0.1em' }}>
            DEMO STEP {demoStep + 1}/{DEMO_STEPS.length} — {DEMO_STEPS[demoStep].title.toUpperCase()}
          </span>
          <span style={{ color: '#f1f5f9', fontSize: '0.85rem', fontStyle: 'italic' }}>
            "{demoText}"
          </span>
          {demoStep === 4 && !backendConnected && (
            <button className="btn btn-primary" style={{ fontSize: '0.7rem', padding: '0.25rem 0.65rem',
                                                          marginLeft: 'auto' }}
                    onClick={() => setTab('LLM Inference')}>
              Connect Now
            </button>
          )}
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside style={{ background: '#0d0d16', borderRight: '1px solid #2a2a3e',
                        width: '200px', minWidth: '200px' }}
               className="flex flex-col p-4 gap-1">
          {TABS.map(t => (
            <button key={t}
                    onClick={() => setTab(t)}
                    className="text-left px-3 py-2 rounded-lg text-sm font-medium transition-all"
                    style={{
                      background: tab === t ? '#00d4ff1a' : 'transparent',
                      color: tab === t ? '#00d4ff' : '#94a3b8',
                      border: tab === t ? '1px solid #00d4ff33' : '1px solid transparent',
                    }}>
              {TAB_ICONS[t]} {t}
            </button>
          ))}

          <div className="mt-auto pt-4" style={{ borderTop: '1px solid #2a2a3e' }}>
            <div className="card-title mb-2">Quick Stats</div>
            <div className="flex flex-col gap-2">
              {[
                { val: sched.compute_units || '--', label: 'Compute Units' },
                { val: (sched.warps_executed || 0).toLocaleString(), label: 'Warps Executed' },
                { val: `${mem.utilization_pct || 0}%`, label: 'vRAM Used', color: '#10b981' },
                {
                  val: inference?.current_tokens_per_sec
                    ? `${inference.current_tokens_per_sec.toFixed(1)} t/s`
                    : '—',
                  label: 'Inference Speed',
                  color: inference?.active ? '#00d4ff' : '#94a3b8',
                },
                {
                  val: inference?.active_model || 'None',
                  label: 'Active Model',
                  color: inference?.active_model ? '#00d4ff' : '#94a3b8',
                  small: true,
                },
              ].map(({ val, label, color, small }) => (
                <div key={label}>
                  <div style={{ color: color || '#00d4ff', fontSize: small ? '0.75rem' : '1rem',
                                 fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis',
                                 whiteSpace: 'nowrap' }}>
                    {val}
                  </div>
                  <div style={{ color: '#94a3b8', fontSize: '0.65rem' }}>{label}</div>
                </div>
              ))}
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto p-6">
          {tab === 'Dashboard' && (
            <div className="flex flex-col gap-4">
              <div className="grid grid-cols-1 gap-4" style={{ gridTemplateColumns: '1fr 1fr' }}>
                <DevicePanel telemetry={telemetry} connected={connected}
                             onNavigateToLLM={() => setTab('LLM Inference')} />
                <MemoryGauge memory={mem} inference={inference} />
              </div>
              <WarpMonitor scheduler={sched} inference={inference} />
              <PerformanceChart data={chartData} />
            </div>
          )}
          {tab === 'Benchmarks' && <BenchmarkRunner />}
          {tab === 'Demo' && (
            <div className="flex flex-col gap-4">
              <TokenGenerator />
              <EconomicsPanel />
            </div>
          )}
          {tab === 'ONNX Model' && <ModelUploader />}
          {tab === 'LLM Inference' && <LLMInference telemetry={telemetry} />}
        </main>
      </div>
    </div>
  )
}
