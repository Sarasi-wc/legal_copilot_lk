import { useState } from 'react'
import axios from 'axios'

const API_URL = '/api'

/**
 * Preview-only corpus construction panel. Calls POST /preview-corpus, which
 * mirrors CorpusBuilder.process_document() (OCR -> segmentation -> metadata)
 * but writes nothing to data/raw, data/processed, or the live search index —
 * the Ask a Question tab is completely unaffected by anything done here.
 */
function UploadPreview() {
  const [file, setFile] = useState(null)
  const [actName, setActName] = useState('')
  const [year, setYear] = useState(2024)
  const [forceOcr, setForceOcr] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [preview, setPreview] = useState(null)
  const [selectedPassageId, setSelectedPassageId] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) {
      setError('Please choose a PDF first.')
      return
    }

    setLoading(true)
    setError(null)
    setPreview(null)

    const formData = new FormData()
    formData.append('file', file)
    formData.append('act_name', actName || file.name)
    formData.append('year', String(year))
    formData.append('force_ocr', String(forceOcr))

    try {
      const response = await axios.post(`${API_URL}/preview-corpus`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setPreview(response.data)
      setSelectedPassageId(response.data.passages[0]?.passage_id || '')
    } catch (err) {
      setError(err.response?.data?.detail || 'Preview failed')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const selectedPassage = preview?.passages.find((p) => p.passage_id === selectedPassageId)

  return (
    <>
      <div className="panel-intro">
        <h3>📤 Upload &amp; Preview</h3>
        <p>
          Preview how a PDF would be processed by the corpus construction pipeline — text
          extraction, OCR quality validation, hierarchical segmentation, and metadata enrichment
          (the same components <code>CorpusBuilder</code> uses, Chapter 5 §5.4.1).
        </p>
        <div className="info-callout">
          🔒 <strong>Preview only.</strong> Nothing here is written to the corpus, the search
          index, or disk — the live <strong>Ask a Question</strong> tab is completely unaffected.
          To add a document to the real corpus, add it to <code>data/manifest_demo.json</code>{' '}
          and run <code>build-corpus</code> / <code>build-indices</code> (Chapter 5 §5.5.1).
        </div>
      </div>

      <form onSubmit={handleSubmit} className="question-form upload-form">
        <div className="upload-dropzone">
          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => setFile(e.target.files[0] || null)}
          />
          {file && (
            <p className="upload-filename">
              📄 {file.name} ({(file.size / 1024).toFixed(1)} KB)
            </p>
          )}
        </div>

        <div className="upload-fields">
          <div className="setting-group">
            <label>Act / document name</label>
            <input
              type="text"
              value={actName}
              onChange={(e) => setActName(e.target.value)}
              placeholder={file ? file.name : 'e.g. Penal Code of Sri Lanka'}
            />
          </div>
          <div className="setting-group">
            <label>Year</label>
            <input
              type="number"
              value={year}
              min={1800}
              max={2100}
              onChange={(e) => setYear(parseInt(e.target.value, 10) || 2024)}
            />
          </div>
          <div className="setting-group checkbox-inline">
            <label>
              <input type="checkbox" checked={forceOcr} onChange={(e) => setForceOcr(e.target.checked)} />
              Force OCR (scanned document)
            </label>
          </div>
        </div>

        <div className="button-group">
          <button type="submit" disabled={loading} className="submit-button">
            {loading ? '🔍 Processing...' : '🔍 Run Preview Pipeline'}
          </button>
        </div>
      </form>

      {error && (
        <div className="error-box">
          <h3>❌ Error</h3>
          <p>{error}</p>
        </div>
      )}

      {preview && (
        <div className="results-section">
          <div className="success-box">
            Processed <strong>{preview.filename}</strong> — preview only, not added to the
            corpus or index.
          </div>

          <div className="citations-section">
            <div className="metrics">
              <div className="metric">
                <strong>Extraction Method</strong>
                <span className="metric-text">{preview.extraction_method}</span>
              </div>
              <div className="metric">
                <strong>OCR Quality Score</strong>
                <span>{(preview.quality_score * 100).toFixed(1)}%</span>
              </div>
              <div className="metric">
                <strong>Sections Found</strong>
                <span>{preview.num_sections}</span>
              </div>
              <div className="metric">
                <strong>Passages Created</strong>
                <span>{preview.num_passages}</span>
              </div>
            </div>
          </div>

          {preview.is_valid ? (
            <div className="success-box">
              ✅ Quality meets the {(preview.threshold * 100).toFixed(0)}% accuracy threshold for
              a document from {preview.threshold === 0.95 ? 'post' : 'pre'}-2000.
            </div>
          ) : (
            <div className="warning-box">
              ⚠️ Quality score ({(preview.quality_score * 100).toFixed(1)}%) is below the{' '}
              {(preview.threshold * 100).toFixed(0)}% threshold — this document would require
              manual correction or re-OCR before inclusion in the real corpus (Appendix E, OCR
              Protocol).
            </div>
          )}

          <div className="citations-section">
            <h3>📄 Extracted Text (preview)</h3>
            <p className="muted">First 3,000 of {preview.raw_text_len.toLocaleString()} characters extracted</p>
            <textarea className="text-preview" value={preview.raw_text_snippet} readOnly rows={8} />
          </div>

          <div className="citations-section">
            <h3>🧩 Segmented Passages</h3>
            {preview.passages.length === 0 ? (
              <p className="muted">
                No structured Parts/Chapters/Sections detected — the segmenter fell back to plain
                text chunking, or the document contains no extractable text.
              </p>
            ) : (
              <>
                <div className="table-wrap">
                  <table className="passages-table">
                    <thead>
                      <tr>
                        <th>Passage ID</th>
                        <th>Level</th>
                        <th>Title</th>
                        <th>Length</th>
                        <th>Over 1,500 cap</th>
                      </tr>
                    </thead>
                    <tbody>
                      {preview.passages.slice(0, 150).map((p) => (
                        <tr key={p.passage_id}>
                          <td>{p.passage_id}</td>
                          <td>{p.level}</td>
                          <td>{(p.title || '').slice(0, 60)}</td>
                          <td>{p.length.toLocaleString()}</td>
                          <td>{p.length > 1500 ? '⚠️' : ''}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {preview.passages.length > 150 && (
                  <p className="muted">Showing first 150 of {preview.passages.length} passages.</p>
                )}

                <details className="details-section">
                  <summary>🔍 Inspect a single passage</summary>
                  <div className="details-content">
                    <select
                      value={selectedPassageId}
                      onChange={(e) => setSelectedPassageId(e.target.value)}
                    >
                      {preview.passages.map((p) => (
                        <option key={p.passage_id} value={p.passage_id}>
                          {p.passage_id}
                        </option>
                      ))}
                    </select>
                    {selectedPassage && (
                      <pre className="json-view">{JSON.stringify(selectedPassage, null, 2)}</pre>
                    )}
                  </div>
                </details>
              </>
            )}
          </div>

          <div className="citations-section">
            <h3>🏷️ Act-Level Metadata</h3>
            <p className="muted">
              Amendments, repeal status, and cross-references detected by <code>MetadataExtractor</code>.
            </p>
            <pre className="json-view">{JSON.stringify(preview.act_metadata, null, 2)}</pre>
          </div>
        </div>
      )}
    </>
  )
}

export default UploadPreview
