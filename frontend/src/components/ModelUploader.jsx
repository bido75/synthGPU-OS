import { useState, useRef } from 'react'

export default function ModelUploader() {
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [running, setRunning] = useState(false)
  const [model, setModel] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

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

    const inputShape = model.inputs?.[0]?.shape?.map(d =>
      typeof d === 'number' ? d : (d === null || d === 'batch_size') ? 1 : parseInt(d) || 1
    ) || [1, 3, 224, 224]

    try {
      const resp = await fetch(`/api/model/${model.model_id}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_shape: inputShape, dtype: 'float32' }),
      })
      if (!resp.ok) {
        const err = await resp.json()
        throw new Error(err.detail || 'Inference failed')
      }
      const data = await resp.json()
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
        <span className="badge-no-gpu">SynthGPU Execution Provider</span>
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

          <div className="flex gap-2 mb-4">
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
              ].map(({ label, value }) => (
                <div key={label} className="flex justify-between py-1"
                     style={{ borderBottom: '1px solid #2a2a3e' }}>
                  <span style={{ color: '#94a3b8', fontSize: '0.75rem' }}>{label}</span>
                  <span style={{ color: '#00d4ff', fontSize: '0.75rem', fontFamily: 'monospace' }}>{value}</span>
                </div>
              ))}
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
