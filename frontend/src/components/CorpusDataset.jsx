import { useEffect, useState } from 'react'
import axios from 'axios'
import { VerticalBarChart } from './Charts'

const API_URL = '/api'

const QUERY_TYPES = [
  { label: 'Factual', count: 223, color: '#1f77b4' },
  { label: 'Interpretive', count: 58, color: '#ff7f0e' },
  { label: 'Procedural', count: 47, color: '#2ca02c' },
  { label: 'Cross-reference', count: 6, color: '#d62728' },
]
const QAC_TOTAL = 334

const PANEL_ROWS = [
  { dimension: 'Correctness', qac: 4.6, abstention: 4.61 },
  { dimension: 'Completeness', qac: 4.48, abstention: 4.56 },
  { dimension: 'Clarity', qac: 4.65, abstention: 4.68 },
]

/**
 * Static corpus/dataset overview (Chapter 5 §5.3), mirroring the Streamlit
 * Corpus & Dataset tab. Only the raw PDF listing is fetched live (from
 * GET /corpus-info) — everything else mirrors Streamlit's own hardcoded
 * figures so the two UIs stay in sync with each other and with Chapter 5.
 */
function CorpusDataset() {
  const [pdfs, setPdfs] = useState(null)
  const [pdfError, setPdfError] = useState(null)

  useEffect(() => {
    axios
      .get(`${API_URL}/corpus-info`)
      .then((res) => setPdfs(res.data.raw_pdfs))
      .catch((err) => setPdfError(err.response?.data?.detail || 'Could not load corpus info'))
  }, [])

  return (
    <>
      <div className="panel-intro">
        <h3>📂 Corpus Overview</h3>
        <div className="metrics">
          <div className="metric">
            <strong>Total Passages</strong>
            <span>793</span>
          </div>
          <div className="metric">
            <strong>Source Documents</strong>
            <span>1</span>
          </div>
          <div className="metric">
            <strong>Embedding Dimensions</strong>
            <span>768</span>
          </div>
          <div className="metric">
            <strong>Chunk Size (max chars)</strong>
            <span>1,500</span>
          </div>
        </div>
        <p style={{ marginTop: '1rem' }}>
          The corpus is built from the <strong>Constitution of Sri Lanka (1978)</strong>,
          OCR-extracted and segmented at passage level. Passage IDs follow the format{' '}
          <code>ACT__1978_SEC_&lt;article&gt;</code>. Sections longer than 1,500 characters are
          split into <code>_CHUNK_N</code> sub-passages at sentence boundaries. Dense embeddings
          use <code>sentence-transformers/all-mpnet-base-v2</code> (768-dim bi-encoder).
        </p>
      </div>

      <div className="citations-section">
        <h3>📄 Raw PDF Sources</h3>
        {pdfError && <p className="muted">{pdfError}</p>}
        {!pdfError && !pdfs && <p className="muted">Loading…</p>}
        {pdfs && pdfs.length === 0 && <p className="muted">No PDFs found in data/raw/acts/.</p>}
        {pdfs && pdfs.length > 0 && (
          <>
            <div className="data-table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>File</th>
                    <th className="numeric">Size</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {pdfs.map((p) => (
                    <tr key={p.file}>
                      <td>{p.file}</td>
                      <td className="numeric">{p.size_mb.toFixed(1)} MB</td>
                      <td>{p.active ? '✅ Active (demo corpus)' : '⬜ Available (not indexed)'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="muted">
              Only PDFs listed in <code>data/manifest_demo.json</code> are indexed. Adding an Act
              requires updating the manifest and rebuilding indices.
            </p>
          </>
        )}
      </div>

      <div className="citations-section">
        <h3>📋 Q-A-C Evaluation Dataset</h3>
        <div className="metrics">
          <div className="metric">
            <strong>Total Items</strong>
            <span>334</span>
          </div>
          <div className="metric">
            <strong>Human-Authored</strong>
            <span>75</span>
          </div>
          <div className="metric">
            <strong>AI-Generated (expert-reviewed)</strong>
            <span>259</span>
          </div>
          <div className="metric">
            <strong>Panel Reviewers</strong>
            <span>5</span>
          </div>
        </div>
        <p style={{ margin: '1rem 0' }}>
          The <strong>Question–Answer–Citation (Q-A-C) dataset</strong> is the first benchmark for
          Sri Lankan legal AI. Each item contains a natural-language question, a gold reference
          answer, and one or more gold passage IDs.
        </p>

        <div className="diagnostics-grid">
          <div className="data-table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Query Type</th>
                  <th className="numeric">Count</th>
                  <th className="numeric">Proportion</th>
                </tr>
              </thead>
              <tbody>
                {QUERY_TYPES.map((qt) => (
                  <tr key={qt.label}>
                    <td>{qt.label}</td>
                    <td className="numeric">{qt.count}</td>
                    <td className="numeric">{((qt.count / QAC_TOTAL) * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            <VerticalBarChart
              labels={QUERY_TYPES.map((q) => q.label)}
              values={QUERY_TYPES.map((q) => q.count)}
              colors={QUERY_TYPES.map((q) => q.color)}
            />
          </div>
        </div>
      </div>

      <div className="citations-section">
        <h3>👩‍⚖️ Panel Review Summary</h3>
        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Dimension</th>
                <th className="numeric">QAC (N=334)</th>
                <th className="numeric">Abstention (N=20)</th>
              </tr>
            </thead>
            <tbody>
              {PANEL_ROWS.map((row) => (
                <tr key={row.dimension}>
                  <td>{row.dimension}</td>
                  <td className="numeric">{row.qac.toFixed(2)}</td>
                  <td className="numeric">{row.abstention.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted">
          5-rater LLB panel reviewed all 354 items. Pairwise MAD = 0.151; 99.9% of scores within
          0.5 pts. Ceiling effect (82% scores = 5) renders κ undefined — MAD is the primary
          agreement metric.
        </p>
      </div>
    </>
  )
}

export default CorpusDataset
