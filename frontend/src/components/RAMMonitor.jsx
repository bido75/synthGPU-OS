import { useState, useEffect } from 'react'

export default function RAMMonitor() {
  const [ram, setRam] = useState(null)
  const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'

  useEffect(() => {
    const fetchRam = async () => {
      try {
        const r = await fetch(`${apiUrl}/api/system/ram`)
        if (r.ok) setRam(await r.json())
      } catch (_) {}
    }
    fetchRam()
    const id = setInterval(fetchRam, 2000)
    return () => clearInterval(id)
  }, [apiUrl])

  if (!ram) return null

  const colors = {
    healthy:  { bar: '#00B4CC', text: 'text-cyan-400',  bg: 'bg-cyan-900/20',  border: 'border-cyan-800' },
    warning:  { bar: '#F59E0B', text: 'text-amber-400', bg: 'bg-amber-900/20', border: 'border-amber-800' },
    critical: { bar: '#EF4444', text: 'text-red-400',   bg: 'bg-red-900/20',   border: 'border-red-800' },
  }[ram.status] || { bar: '#00B4CC', text: 'text-cyan-400', bg: 'bg-cyan-900/20', border: 'border-cyan-800' }

  return (
    <div className={`rounded-lg border p-4 ${colors.bg} ${colors.border}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="text-xs font-semibold tracking-widest text-gray-400 uppercase">System RAM</div>
        <div className={`text-xs font-medium ${colors.text}`}>{ram.message}</div>
      </div>

      <div className="relative h-2 bg-gray-800 rounded-full overflow-hidden mb-3">
        <div
          className="absolute left-0 top-0 h-full rounded-full transition-all duration-500"
          style={{ width: `${ram.percent_used}%`, backgroundColor: colors.bar }}
        />
      </div>

      <div className="grid grid-cols-3 gap-2 text-center">
        <div>
          <div className="text-xs text-gray-500 uppercase tracking-wide">Total</div>
          <div className="text-sm font-mono text-gray-300">{(ram.total_mb / 1024).toFixed(0)}GB</div>
        </div>
        <div>
          <div className="text-xs text-gray-500 uppercase tracking-wide">Used</div>
          <div className={`text-sm font-mono ${colors.text}`}>{(ram.used_mb / 1024).toFixed(1)}GB</div>
        </div>
        <div>
          <div className="text-xs text-gray-500 uppercase tracking-wide">Free</div>
          <div className="text-sm font-mono text-green-400">{(ram.free_mb / 1024).toFixed(1)}GB</div>
        </div>
      </div>

      <div className="mt-3 pt-3 border-t border-gray-800 flex justify-between text-xs">
        <span className="text-gray-500">SynthGPU process</span>
        <span className="text-cyan-400 font-mono">{ram.synthgpu_mb}MB</span>
      </div>

      {ram.swapping && (
        <div className="mt-2 text-xs text-red-400 flex items-center gap-1">
          <span>&#9888;</span>
          <span>Disk swap active — performance degraded</span>
        </div>
      )}
    </div>
  )
}
