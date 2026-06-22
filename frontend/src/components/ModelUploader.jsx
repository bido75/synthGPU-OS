import { useEffect, useState, useRef } from 'react'

export default function ModelUploader() {
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [running, setRunning] = useState(false)
  const [model, setModel] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [provider, setProvider] = useState('cpu')
  const [providerStatus, setProviderStatus] = useState({ cpu: true, openvino: false })
  const inputRef = useRef(null)

  useEffect(() => {
    fetch('/api/model/providers')
      .then(resp => resp.ok ? resp.json() : Promise.reject(new Error('Provider status unavailable')))
      .then(setProviderStatus)
      .catch(() => setProviderStatus({ cpu: true, openvino: false }))
  }, [])

  const uploadFile = async (file) => {
    if (!file.name.endsWith('.onnx')) {
      setError('Only .onnx files are supported')
      return
    }
    setUploading(true)
    setError(null)
    setResult(null)

    const form = new FormData()
    form.append('file', file)

    try {
      const resp = await fetch('/api/model/upload', { method: 'POST', body: form })
      if (!resp.ok) {
        const err = await resp.json()
        throw new Error(err.detail || 'Upload failed')
      }
      const data = await resp.json()
      setModel(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setUploading(false)
    }
  }

  const runInference = async () => {
    if (!model) return
    setRunning(true)
    setError(null)

    try {
      const resp = await fetch(`/api/model/${model.model_id}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_shape: [1], dtype: 'float32', provider }),
      })
      let data
      const ct = resp.headers.get('content-type') || ''
      if (ct.includes('application/json')) {
        data = await resp.json()
      } else {
        const text = await resp.text()
        throw new Error(`Server error (${resp.status}): ${text.slice(0, 120)}`)
      }
      if (!resp.ok) throw new Error(data.detail || 'Inference failed')
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setRunning(false)
    }
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) uploadFile(file)
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <div className="card-title mb-0">ONNX Model Runner</div>
        <span className="badge-no-gpu">ONNX Runtime Instrumentation</span>
      </div>

      {!model ? (
        <div
          className="flex flex-col items-center justify-center p-8 rounded-xl cursor-pointer"
          style={{
            border: `2px dashed ${dragging ? '#00d4ff' : '#2a2a3e'}`,
            background: dragging ? '#00d4ff0a' : '#0a0a0f',
            transition: 'all 0.2s',
            minHeight: 160,
          }}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}>
          <input ref={inputRef} type="file" accept=".onnx" className="hidden"
                 onChange={e => e.target.files[0] && uploadFile(e.target.files[0])} />
          <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🧠</div>
          <div style={{ color: '#f1f5f9', fontWeight: 600, marginBottom: 4 }}>
            {uploading ? 'Uploading...' : 'Drop an ONNX model here'}
          </div>
          <div style={{ color: '#94a3b8', fontSize: '0.75rem' }}>
            or click to browse · Supported: .onnx files up to 500MB
          </div>
        </div>
      ) : (
        <div>
          <div className="p-3 rounded-lg mb-4"
               style={{ background: '#0a0a0f', border: '1px solid #00d4ff33' }}>
            <div className="flex items-center justify-between mb-2">
              <span style={{ color: '#f1f5f9', fontWeight: 600 }}>{model.filename}</span>
              <span style={{ color: '#10b981', fontSize: '0.75rem', fontWeight: 700 }}>✓ Loaded</span>
            </div>
            <div style={{ color: '#94a3b8', fontSize: '0.75rem' }}>
              Size: {model.size_mb} MB
            </div>
            {model.inputs?.map(inp => (
              <div key={inp.name} style={{ color: '#94a3b8', fontSize: '0.75rem' }}>
                Input: <span style={{ fontFamily: 'monospace', color: '#f1f5f9' }}>
                  {inp.dtype}[{inp.shape?.join(', ')}]
                </span>
              </div>
            ))}
          </div>

          <div className="flex gap-2 mb-4 items-center">
            <label style={{ color: '#94a3b8', fontSize: '0.75rem' }} htmlFor="onnx-provider">
              Provider
            </label>
            <select
              id="onnx-provider"
              value={provider}
              disabled={running}
              onChange={event => { setProvider(event.target.value); setResult(null) }}
              style={{ background: '#0a0a0f', color: '#f1f5f9', border: '1px solid #2a2a3e', borderRadius: 6, padding: '0.45rem' }}>
              <option value="cpu">CPU</option>
              <option value="openvino" disabled={!providerStatus.openvino}>
                OpenVINO{providerStatus.openvino ? '' : ' (unavailable)'}
              </option>
            </select>
            <button className="btn btn-primary" disabled={running} onClick={runInference}>
              {running ? '⏳ Running...' : '▶ Run Inference'}
            </button>
            <button className="btn btn-secondary" onClick={() => { setModel(null); setResult(null) }}>
              ✕ Remove
            </button>
          </div>

          {result && (
            <div className="p-3 rounded-lg"
                 style={{ background: '#0a0a0f', border: '1px solid #2a2a3e' }}>
              <div className="card-title mb-2">Result</div>
              {[
                { label: 'Output shapes', value: result.output_shapes?.map(s => `[${s}]`).join(', ') || '--' },
                { label: 'Inference time', value: `${result.elapsed_ms}ms` },
                { label: 'Throughput', value: `${result.throughput_per_sec} inferences/sec` },
                { label: 'Device', value: result.device || 'SynthGPU Virtual Accelerator' },
                { label: 'Provider', value: result.provider || '--' },
                { label: 'Profiled node time', value: `${result.profiled_node_total_ms ?? 0}ms` },
                { label: 'Correctness gate', value: result.correctness_verified ? 'Verified' : 'Not verified' },
              ].map(({ label, value }) => (
                <div key={label} className="flex justify-between py-1"
                     style={{ borderBottom: '1px solid #2a2a3e' }}>
                  <span style={{ color: '#94a3b8', fontSize: '0.75rem' }}>{label}</span>
                  <span style={{ color: '#00d4ff', fontSize: '0.75rem', fontFamily: 'monospace' }}>{value}</span>
                </div>
              ))}
              {result.unsupported_ops?.length > 0 && (
                <div className="mt-3" style={{ color: '#f59e0b', fontSize: '0.75rem' }}>
                  Unvalidated ops: {result.unsupported_ops.join(', ')}
                </div>
              )}
              {result.per_node_timing_ms?.length > 0 && (
                <div className="mt-3">
                  <div style={{ color: '#f1f5f9', fontSize: '0.75rem', fontWeight: 600, marginBottom: 4 }}>
                    Real per-node timing
                  </div>
                  {result.per_node_timing_ms.map((node, index) => (
                    <div key={`${node.node_name}-${index}`} className="flex justify-between py-1"
                         style={{ borderBottom: '1px solid #2a2a3e', gap: 12 }}>
                      <span style={{ color: '#94a3b8', fontSize: '0.7rem', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {node.node_name} ({node.op_name})
                      </span>
                      <span style={{ color: '#00d4ff', fontSize: '0.7rem', fontFamily: 'monospace', whiteSpace: 'nowrap' }}>
                        {node.duration_ms}ms
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="mt-3 p-3 rounded-lg"
             style={{ background: '#ef444411', border: '1px solid #ef444444' }}>
          <span style={{ color: '#ef4444', fontSize: '0.8rem' }}>⚠ {error}</span>
        </div>
      )}
    </div>
  )
}
