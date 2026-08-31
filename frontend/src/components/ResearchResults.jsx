import { useEffect, useState } from 'react'
import axios from 'axios'
import { GroupedRetrievalChart } from './Charts'

const API_URL = '/api'

const CONDITIONS = [
  { key: 'bm25', label: 'BM25 only' },
  { key: 'dense', label: 'Dense only' },
  { key: 'hybrid', label: 'Hybrid (no rerank)' },
  { key: 'hybrid_rerank', label: 'Hybrid + Rerank (full system) ★' },
]

const SIG_ROWS = [
  { comparison: 'BM25 vs Dense', pValue: '1.0000', significant: 'No' },
  { comparison: 'BM25 vs Hybrid', pValue: '<0.0001', significant: 'Yes ✅' },
  { comparison: 'BM25 vs Hybrid + Rerank', pValue: '<0.0001', significant: 'Yes ✅' },
  { comparison: 'Dense vs Hybrid', pValue: '<0.0001', significant: 'Yes ✅' },
  { comparison: 'Dense vs Hybrid + Rerank', pValue: '<0.0001', significant: 'Yes ✅' },
  { comparison: 'Hybrid vs Hybrid + Rerank', pValue: '0.0014', significant: 'Yes ✅' },
]

const RQ3_ROWS = [
  { metric: 'Answers with inline citations', rag: '10 / 10', noRag: '0 / 10' },
  { metric: 'Mean faithfulness score', rag: '0.900', noRag: 'n/a (unverifiable)' },
  { metric: 'Citation format compliance', rag: '100%', noRag: '0%' },
  { metric: 'Abstentions triggered', rag: '0 / 10', noRag: '0 / 10' },
  { metric: 'Observed factual errors', rag: 'None detected (NLI)', noRag: 'Multiple hallucinations' },
]

const NLI_ROWS = [
  { cls: 'Entailed (faithful claims)', range: '0.951 – 0.994', mean: '0.980', n: '11' },
  { cls: 'Not-entailed (hallucinated)', range: '0.000 – 0.287', mean: '0.031', n: '11' },
  { cls: 'Operational threshold', range: '—', mean: '0.30', n: '—' },
]

/**
 * Chapter 6 results, mirroring the Streamlit Research Results tab. Only the
 * RQ2 retrieval table/chart is fetched live (GET /research-results, backed by
 * results/eval_post_qac_fixes_v2/results_summary.json — the authoritative
 * N=334 evaluation); the remaining tables are static, matching Streamlit's
 * own hardcoded figures for the same reason as the Corpus & Dataset tab.
 */
function ResearchResults() {
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    axios
      .get(`${API_URL}/research-results`)
      .then((res) => setResults(res.data))
      .catch((err) =>
        setError(
          err.response?.status === 404
            ? 'Results file not found at results/eval_post_qac_fixes_v2/results_summary.json.'
            : err.response?.data?.detail || 'Could not load research results'
        )
      )
  }, [])

  const rq1 = results?.rq1
  const metrics = ['mrr_mean', 'recall@10_mean', 'ndcg@10_mean']
  const metricLabels = { 'mrr_mean': 'MRR', 'recall@10_mean': 'Recall@10', 'ndcg@10_mean': 'NDCG@10' }
  const maxByMetric = {}
  if (rq1) {
    metrics.forEach((m) => {
      maxByMetric[m] = Math.max(...CONDITIONS.map((c) => rq1[c.key][m]))
    })
  }

  return (
    <>
      <div className="panel-intro">
        <h3>📊 RQ2 — Retrieval Effectiveness (N=334)</h3>

        {error && <p className="muted">{error}</p>}
        {!error && !rq1 && <p className="muted">Loading…</p>}

        {rq1 && (
          <>
            <div className="data-table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Condition</th>
                    {metrics.map((m) => (
                      <th className="numeric" key={m}>
                        {metricLabels[m]}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {CONDITIONS.map((c) => (
                    <tr key={c.key}>
                      <td>{c.label}</td>
                      {metrics.map((m) => (
                        <td
                          key={m}
                          className={`numeric ${rq1[c.key][m] === maxByMetric[m] ? 'highlight-cell' : ''}`}
                        >
                          {rq1[c.key][m].toFixed(3)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="muted">
              ★ Full system. Article-level relevance: retrieved passage matches gold section
              number. Authoritative results from{' '}
              <code>results/eval_post_qac_fixes_v2/</code>.
            </p>

            <GroupedRetrievalChart
              conditions={['BM25 only', 'Dense only', 'Hybrid\n(no rerank)', 'Hybrid + Rerank\n(full system)']}
              series={[
                { name: 'MRR', color: '#1f77b4', values: CONDITIONS.map((c) => rq1[c.key]['mrr_mean']) },
                {
                  name: 'Recall@10',
                  color: '#2ca02c',
                  values: CONDITIONS.map((c) => rq1[c.key]['recall@10_mean']),
                },
                {
                  name: 'NDCG@10',
                  color: '#ff7f0e',
                  values: CONDITIONS.map((c) => rq1[c.key]['ndcg@10_mean']),
                },
              ]}
            />
            <p className="muted">
              Bonferroni-corrected pairwise Wilcoxon signed-rank tests: Hybrid+Rerank vs BM25
              p&lt;0.0001; vs Dense p&lt;0.0001; vs Hybrid p=0.0014 (Bonferroni-adjusted). BM25 vs
              Dense p=1.000 (not significant).
            </p>
          </>
        )}
      </div>

      <div className="citations-section">
        <h3>📐 Statistical Significance (MRR, Bonferroni-corrected Wilcoxon)</h3>
        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Comparison</th>
                <th className="numeric">p-value</th>
                <th>Significant?</th>
              </tr>
            </thead>
            <tbody>
              {SIG_ROWS.map((row) => (
                <tr key={row.comparison}>
                  <td>{row.comparison}</td>
                  <td className="numeric">{row.pValue}</td>
                  <td>{row.significant}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="citations-section">
        <h3>📝 RQ3 — Generation Fidelity vs No-RAG Baseline (N=10)</h3>
        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>RAG (Full Pipeline)</th>
                <th>No-RAG Baseline (GPT-3.5-turbo)</th>
              </tr>
            </thead>
            <tbody>
              {RQ3_ROWS.map((row) => (
                <tr key={row.metric}>
                  <td>{row.metric}</td>
                  <td>{row.rag}</td>
                  <td>{row.noRag}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted">
          Faithfulness verified via RoBERTa-large-MNLI (threshold 0.30, aggregate gate 0.80).
          10-item qualitative comparison spanning 8 constitutional domains.
        </p>
      </div>

      <div className="citations-section">
        <h3>🚫 Abstention Evaluation (N=20)</h3>
        <div className="metrics">
          <div className="metric">
            <strong>Precision</strong>
            <span>1.000</span>
          </div>
          <div className="metric">
            <strong>Recall</strong>
            <span>1.000</span>
          </div>
          <div className="metric">
            <strong>F1</strong>
            <span>1.000</span>
          </div>
        </div>
        <p className="muted" style={{ marginTop: '1rem' }}>
          5/5 in-corpus queries correctly answered; 15/15 out-of-corpus queries correctly
          abstained. Fix applied: Layer 0 threshold −1.0 → −0.5 in <code>evidence_planner.py</code>{' '}
          + pre-generation query-scope classifier in <code>openai_generator.py</code>.
        </p>
      </div>

      <div className="citations-section">
        <h3>🔬 NLI Threshold Calibration (N=22 pairs)</h3>
        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Class</th>
                <th>Score Range</th>
                <th className="numeric">Mean</th>
                <th className="numeric">N</th>
              </tr>
            </thead>
            <tbody>
              {NLI_ROWS.map((row) => (
                <tr key={row.cls}>
                  <td>{row.cls}</td>
                  <td>{row.range}</td>
                  <td className="numeric">{row.mean}</td>
                  <td className="numeric">{row.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted">
          Gap of 0.664 between highest not-entailed (0.287) and lowest entailed (0.951) score.
          Perfect separation (F1=1.0) at threshold 0.29; rounded to 0.30 as safety margin.
        </p>
      </div>
    </>
  )
}

export default ResearchResults
