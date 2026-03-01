import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const WHO_BENEFITS = [
  'Startups blocked by GPU cost',
  'Edge devices (IoT, embedded)',
  'Developing world markets',
  'Any CPU-only cloud instance',
  'Air-gapped / secure environments',
]

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: '#1a1a2e', border: '1px solid #2a2a3e', borderRadius: 8, padding: '8px 12px' }}>
      <div style={{ color: '#f1f5f9', fontSize: '0.75rem', fontWeight: 700 }}>{label}</div>
      <div style={{ color: payload[0]?.color, fontSize: '0.75rem' }}>
        ${payload[0]?.value?.toLocaleString()}/mo
      </div>
    </div>
  )
}

export default function EconomicsPanel() {
  const [data, setData] = useState(null)
  const [animate, setAnimate] = useState(false)

  useEffect(() => {
    fetch('/api/economics')
      .then(r => r.json())
      .then(d => { setData(d); setTimeout(() => setAnimate(true), 200) })
      .catch(() => {})
  }, [])

  const comparisons = data?.comparisons || [
    { name: 'NVIDIA H100', cost_per_hour: 32.77, monthly_cost: 23594, hardware_required: true, color: '#7c3aed' },
    { name: 'NVIDIA A100', cost_per_hour: 12.24, monthly_cost: 8813, hardware_required: true, color: '#6d28d9' },
    { name: 'SynthGPU', cost_per_hour: 5.44, monthly_cost: 3917, hardware_required: false, color: '#00d4ff' },
  ]

  const chartData = comparisons.map(c => ({
    name: c.name.replace('NVIDIA ', '').replace('SynthGPU', 'SynthGPU\n(CPU-only)'),
    cost: c.monthly_cost,
    color: c.color,
  }))

  return (
    <div className="card">
      <div className="card-title">The Economics of SynthGPU</div>

      {/* Comparison table */}
      <div className="mb-4 overflow-x-auto">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #2a2a3e' }}>
              {['', 'Cost/hr', 'Monthly (1 unit)', 'Wait Time', 'Hardware', 'Savings'].map(h => (
                <th key={h} style={{ color: '#94a3b8', fontWeight: 600, padding: '8px 12px',
                                     textAlign: h === '' ? 'left' : 'center', fontSize: '0.65rem',
                                     textTransform: 'uppercase', letterSpacing: '0.08em' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {comparisons.map((c, i) => {
              const savings = comparisons[0].monthly_cost - c.monthly_cost
              const savingsPct = Math.round((savings / comparisons[0].monthly_cost) * 100)
              return (
                <tr key={i} style={{ borderBottom: '1px solid #1a1a2e',
                                     background: c.hardware_required === false ? '#00d4ff0a' : 'transparent' }}>
                  <td style={{ padding: '10px 12px' }}>
                    <span style={{ color: c.color, fontWeight: 700 }}>{c.name}</span>
                    {!c.hardware_required && (
                      <span className="badge-no-gpu ml-2" style={{ fontSize: '0.6rem' }}>YOU ARE HERE</span>
                    )}
                  </td>
                  <td style={{ textAlign: 'center', color: '#f1f5f9', fontFamily: 'monospace',
                               padding: '10px 12px' }}>${c.cost_per_hour}</td>
                  <td style={{ textAlign: 'center', color: '#f1f5f9', fontFamily: 'monospace',
                               padding: '10px 12px' }}>${c.monthly_cost?.toLocaleString()}</td>
                  <td style={{ textAlign: 'center', color: '#94a3b8', padding: '10px 12px',
                               fontSize: '0.75rem' }}>{c.wait_time}</td>
                  <td style={{ textAlign: 'center', padding: '10px 12px' }}>
                    {c.hardware_required
                      ? <span style={{ color: '#ef4444' }}>Required</span>
                      : <span style={{ color: '#10b981', fontWeight: 700 }}>✗ None needed</span>}
                  </td>
                  <td style={{ textAlign: 'center', padding: '10px 12px' }}>
                    {i === 0 ? <span style={{ color: '#94a3b8' }}>—</span> :
                     <span style={{ color: '#10b981', fontWeight: 700 }}>
                       {savingsPct > 0 ? `${savingsPct}% less` : '—'}
                     </span>}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Savings highlight */}
      <div className="p-4 rounded-xl mb-4"
           style={{ background: '#10b98111', border: '1px solid #10b98133' }}>
        <div style={{ color: '#10b981', fontSize: '1.4rem', fontWeight: 800 }}>
          ${data?.monthly_savings?.toLocaleString() || '19,677'}/month saved
        </div>
        <div style={{ color: '#94a3b8', fontSize: '0.8rem', marginTop: 2 }}>
          vs NVIDIA H100 · {data?.savings_pct || 83}% cost reduction · Instant availability
        </div>
      </div>

      {/* Bar chart */}
      <div style={{ height: 200 }} className="mb-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3e" />
            <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <YAxis tickFormatter={v => `$${(v / 1000).toFixed(0)}k`}
                   tick={{ fill: '#94a3b8', fontSize: 10 }} />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="cost" radius={[4, 4, 0, 0]} isAnimationActive={animate}>
              {chartData.map((entry, i) => (
                <Cell key={i} fill={entry.color}
                      style={{ filter: entry.color === '#00d4ff' ? 'drop-shadow(0 0 8px #00d4ff88)' : 'none' }} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Who benefits */}
      <div>
        <div style={{ color: '#f1f5f9', fontWeight: 700, marginBottom: '0.75rem', fontSize: '0.85rem' }}>
          Who Benefits:
        </div>
        <div className="grid grid-cols-2 gap-2">
          {WHO_BENEFITS.map(w => (
            <div key={w} className="flex items-center gap-2">
              <div style={{ width: 6, height: 6, background: '#10b981', borderRadius: '50%',
                            boxShadow: '0 0 4px #10b981' }} />
              <span style={{ color: '#94a3b8', fontSize: '0.75rem' }}>{w}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
