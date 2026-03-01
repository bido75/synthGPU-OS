import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: '#1a1a2e', border: '1px solid #2a2a3e', borderRadius: 8, padding: '8px 12px' }}>
      <div style={{ color: '#94a3b8', fontSize: '0.7rem', marginBottom: 4 }}>{label}</div>
      {payload.map(p => (
        <div key={p.name} style={{ color: p.color, fontSize: '0.75rem' }}>
          {p.name}: <strong>{typeof p.value === 'number' ? p.value.toFixed(2) : p.value}</strong>
          {p.name === 'Throughput' ? ' w/s' : '%'}
        </div>
      ))}
    </div>
  )
}

export default function PerformanceChart({ data }) {
  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <div className="card-title mb-0">Live Performance</div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <div style={{ width: 12, height: 2, background: '#00d4ff', borderRadius: 1 }} />
            <span style={{ color: '#94a3b8', fontSize: '0.65rem' }}>Warp Throughput (w/s)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div style={{ width: 12, height: 2, background: '#7c3aed', borderRadius: 1 }} />
            <span style={{ color: '#94a3b8', fontSize: '0.65rem' }}>Utilization (%)</span>
          </div>
        </div>
      </div>

      <div style={{ height: 200 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3e" />
            <XAxis dataKey="time" tick={{ fill: '#94a3b8', fontSize: 10 }}
                   interval="preserveStartEnd" />
            <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="throughput" name="Throughput"
                  stroke="#00d4ff" strokeWidth={2} dot={false}
                  activeDot={{ r: 4, fill: '#00d4ff' }} isAnimationActive={false} />
            <Line type="monotone" dataKey="utilization" name="Utilization"
                  stroke="#7c3aed" strokeWidth={2} dot={false}
                  activeDot={{ r: 4, fill: '#7c3aed' }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {data.length === 0 && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
                      justifyContent: 'center', color: '#94a3b8', fontSize: '0.8rem' }}>
          Waiting for data...
        </div>
      )}
    </div>
  )
}
