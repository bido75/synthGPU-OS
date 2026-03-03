import { useState, useEffect } from 'react'

export default function DemoReadyBadge() {
  const [health, setHealth] = useState(null)

  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch('http://localhost:8000/api/health/demo_ready')
        if (r.ok) setHealth(await r.json())
      } catch (_) {}
    }
    check()
    const id = setInterval(check, 30000)
    return () => clearInterval(id)
  }, [])

  if (!health) return null

  const cfg = {
    ready:               { label: '✓ Demo Ready', cls: 'bg-green-900/40 text-green-400 border-green-700' },
    ready_with_warnings: { label: '⚠ Demo Ready', cls: 'bg-amber-900/40 text-amber-400 border-amber-800' },
    not_ready:           { label: '✗ Not Ready',  cls: 'bg-red-900/40  text-red-400   border-red-800'   },
  }[health.status] || { label: '…', cls: 'text-gray-500 border-gray-700' }

  return (
    <div className={`relative group px-3 py-1 rounded border text-xs font-medium cursor-default select-none ${cfg.cls}`}>
      {cfg.label}

      <div className="absolute top-8 right-0 hidden group-hover:block z-50 w-72 bg-gray-950 border border-gray-700 rounded-lg p-3 shadow-xl">
        <div className="text-gray-300 font-semibold mb-2 text-xs">Demo Pre-Flight</div>
        <div className="text-xs space-y-1">
          <div className="flex justify-between">
            <span className="text-gray-500">RAM Free</span>
            <span className={health.free_mb > 1500 ? 'text-green-400' : 'text-amber-400'}>
              {health.free_mb}MB
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">SynthGPU</span>
            <span className="text-cyan-400">{health.synthgpu_mb}MB</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Ollama</span>
            <span className={health.ollama_ok ? 'text-green-400' : 'text-red-400'}>
              {health.ollama_ok ? 'Connected' : 'Not running'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Model</span>
            <span className="text-cyan-400">{health.recommended_model}</span>
          </div>
        </div>

        {health.issues?.length > 0 && (
          <div className="mt-2 pt-2 border-t border-gray-800 space-y-1">
            {health.issues.map((i, idx) => (
              <div key={idx} className="text-red-400 text-xs">✗ {i}</div>
            ))}
          </div>
        )}
        {health.warnings?.length > 0 && (
          <div className="mt-2 pt-2 border-t border-gray-800 space-y-1">
            {health.warnings.map((w, idx) => (
              <div key={idx} className="text-amber-400 text-xs">⚠ {w}</div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
