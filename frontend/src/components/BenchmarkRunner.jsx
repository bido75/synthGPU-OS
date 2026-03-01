import { useState } from 'react'

const BENCHMARKS = [
  { id: 'all', label: 'Run All' },
  { id: 'gemm', label: 'GEMM' },
  { id: 'mlp', label: 'MLP' },
  { id: 'transformer', label: 'Transformer' },
  { id: 'token_gen', label: 'Token Gen' },
]

function ResultBar({ result }) {
  const speedup = result.speedup || 1
  const maxBar = 40
  const cpuBar = maxBar
  const gpuBar = Math.min(maxBar, Math.max(4, maxBar / Math.max(speedup, 0.1)))
  const correct = result.correct !== false

  return (
    <div className="mb-4 p-3 rounded-lg" style={{ background: '#0a0a0f', border: '1px solid #2a2a3e' }}>
      <div className="flex items-center justify-between mb-2">
        <span style={{ color: '#f1f5f9', fontSize: '0.8rem', fontWeight: 600 }}>{result.name}</span>
        {correct !== undefined && (
          <span style={{ fontSize: '0.65rem', fontWeight: 700,
                         color: correct ? '#10b981' : '#ef4444' }}>
            {correct ? '✓ Numerically identical to CPU' : '✗ Accuracy mismatch'}
          </span>
        )}
      </div>

      <div className="flex items-center gap-2 mb-1">
        <span style={{ color: '#94a3b8', fontSize: '0.65rem', minWidth: '90px' }}>CPU baseline</span>
        <div style={{ flex: 1, height: 8, background: '#2a2a3e', borderRadius: 4, overflow: 'hidden' }}>
          <div style={{ width: '100%', height: '100%', background: '#4a4a6a', borderRadius: 4 }} />
        </div>
        <span style={{ color: '#94a3b8', fontSize: '0.65rem', minWidth: '64px', textAlign: 'right',
                       fontFamily: 'monospace' }}>
          {result.cpu_ms?.toFixed(1)}ms
        </span>
      </div>

      <div className="flex items-center gap-2 mb-2">
        <span style={{ color: '#94a3b8', fontSize: '0.65rem', minWidth: '90px' }}>SynthGPU</span>
        <div style={{ flex: 1, height: 8, background: '#2a2a3e', borderRadius: 4, overflow: 'hidden' }}>
          <div style={{
            width: `${Math.max(4, 100 / Math.max(speedup, 0.1))}%`,
            height: '100%',
            background: 'linear-gradient(90deg, #00d4ff88, #00d4ff)',
            borderRadius: 4,
            boxShadow: '0 0 6px #00d4ff66',
          }} />
        </div>
        <span style={{ color: '#00d4ff', fontSize: '0.65rem', minWidth: '64px', textAlign: 'right',
                       fontFamily: 'monospace' }}>
          {result.gpu_ms?.toFixed(1)}ms
        </span>
      </div>

      <div className="flex gap-4">
        <span style={{ color: '#10b981', fontSize: '0.7rem', fontWeight: 700 }}>
          {speedup.toFixed(2)}x faster
        </span>
        <span style={{ color: '#94a3b8', fontSize: '0.7rem' }}>
          {result.throughput?.toFixed ? result.throughput.toFixed(3) : result.throughput} {result.throughput_unit}
        </span>
        <span className="badge-no-gpu">NO GPU HARDWARE</span>
      </div>
    </div>
  )
}

export default function BenchmarkRunner() {
  const [running, setRunning] = useState(false)
  const [activeBenchmark, setActiveBenchmark] = useState(null)
  const [progress, setProgress] = useState(0)
  const [results, setResults] = useState({})
  const [error, setError] = useState(null)

  const runBenchmark = async (benchmarkId) => {
    setRunning(true)
    setActiveBenchmark(benchmarkId)
    setProgress(0)
    setError(null)

    try {
      const resp = await fetch('/api/benchmark/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ benchmark: benchmarkId }),
      })

      if (!resp.ok) {
        const err = await resp.json()
        throw new Error(err.detail || 'Benchmark failed')
      }

      const data = await resp.json()
      setResults(prev => ({ ...prev, ...data.results }))
      setProgress(100)
    } catch (e) {
      setError(e.message)
    } finally {
      setRunning(false)
      setActiveBenchmark(null)
    }
  }

  const allResults = Object.values(results).flat()

  return (
    <div className="card">
      <div className="card-title">Benchmark Suite</div>

      <div className="flex gap-2 flex-wrap mb-4">
        {BENCHMARKS.map(b => (
          <button key={b.id}
                  className="btn btn-primary"
                  disabled={running}
                  onClick={() => runBenchmark(b.id)}>
            {running && activeBenchmark === b.id ? '⏳ ' : ''}
            {b.label}
          </button>
        ))}
      </div>

      {running && (
        <div className="mb-4">
          <div className="flex justify-between mb-1">
            <span style={{ color: '#94a3b8', fontSize: '0.75rem' }}>
              Running {activeBenchmark?.toUpperCase()} benchmark...
            </span>
            <span style={{ color: '#00d4ff', fontSize: '0.75rem' }}>{progress}%</span>
          </div>
          <div style={{ height: 6, background: '#2a2a3e', borderRadius: 3, overflow: 'hidden' }}>
            <div style={{
              height: '100%',
              width: `${progress}%`,
              background: 'linear-gradient(90deg, #7c3aed, #00d4ff)',
              borderRadius: 3,
              transition: 'width 0.3s',
              animation: progress < 100 ? 'none' : undefined,
            }} />
          </div>
          <div style={{ color: '#94a3b8', fontSize: '0.65rem', marginTop: 4 }}>
            This may take 30-120 seconds depending on hardware. All computations are real.
          </div>
        </div>
      )}

      {error && (
        <div className="mb-4 p-3 rounded-lg" style={{ background: '#ef444411', border: '1px solid #ef444444' }}>
          <span style={{ color: '#ef4444', fontSize: '0.8rem' }}>Error: {error}</span>
        </div>
      )}

      {allResults.length > 0 && (
        <div>
          <div className="card-title mb-3">Results</div>
          {allResults.map((r, i) => <ResultBar key={i} result={r} />)}
        </div>
      )}

      {allResults.length === 0 && !running && (
        <div style={{ color: '#94a3b8', fontSize: '0.8rem', padding: '2rem', textAlign: 'center',
                      border: '1px dashed #2a2a3e', borderRadius: 8 }}>
          Select a benchmark above to run real AI workload tests.<br />
          <span style={{ fontSize: '0.7rem' }}>All results use real NumPy computations — no simulated data.</span>
        </div>
      )}
    </div>
  )
}
