import Gauge from './Gauge'
import { RetrievalBarChart, CitationValidityBar } from './Charts'

/**
 * Live diagnostic view of the most recent answer from the Ask a Question tab:
 * retrieval scores, citation validity, and NLI faithfulness verification.
 */
function AnswerDiagnostics({ lastResult, lastQuestion }) {
  if (!lastResult) {
    return (
      <div className="panel-intro">
        <h3>📈 Answer Diagnostics</h3>
        <p className="muted">
          Ask a question in the <strong>⚖️ Ask a Question</strong> tab to see its diagnostics
          here.
        </p>
      </div>
    )
  }

  if (lastResult.abstained) {
    return (
      <>
        <div className="panel-intro">
          <h3>📈 Answer Diagnostics</h3>
          <p>
            <strong>Question:</strong> {lastQuestion}
          </p>
        </div>
        <div className="warning-box">
          ⚠️ <strong>System abstained</strong>
          <br />
          Reason: {lastResult.abstention_reason || 'unknown'}
        </div>
        <p className="muted">
          No citations, faithfulness score, or retrieval-score chart apply to an abstained
          response — abstention short-circuits generation and verification.
        </p>
      </>
    )
  }

  const citations = lastResult.citations || []
  const verification = lastResult.verification || {}
  const faith = verification.faithfulness || {}
  const faithScore = faith.faithfulness_score
  const citVal = verification.citation_validation || {}
  const valid = citVal.valid_citations ?? citations.filter((c) => c.has_matching_evidence).length
  const total = citVal.total_citations ?? citations.length
  const passed = verification.verification_passed

  const retrievalScores = lastResult.retrieval_scores || []
  const citedIds = new Set(citations.map((c) => c.passage_id).filter(Boolean))

  return (
    <>
      <div className="panel-intro">
        <h3>📈 Answer Diagnostics</h3>
        <p className="muted">
          Live diagnostic view of the most recent answer from the{' '}
          <strong>⚖️ Ask a Question</strong> tab: retrieval scores, citation validity, and NLI
          faithfulness verification.
        </p>
        <p>
          <strong>Question:</strong> {lastQuestion}
        </p>
      </div>

      <div className="citations-section">
        <div className="metrics">
          <div className="metric">
            <strong>Retrieval Method</strong>
            <span className="metric-text">{lastResult.retrieval_method || 'n/a'}</span>
          </div>
          <div className="metric">
            <strong>Passages Retrieved</strong>
            <span>{lastResult.num_retrieved ?? 0}</span>
          </div>
          <div className="metric">
            <strong>Evidence Used</strong>
            <span>{lastResult.evidence_used ?? 0}</span>
          </div>
          <div className="metric">
            <strong>Verification</strong>
            <span className="metric-text">
              {passed ? '✅ Passed' : verification.faithfulness ? '⚠️ Issues' : 'n/a (disabled)'}
            </span>
          </div>
        </div>
      </div>

      <div className="diagnostics-grid">
        <div className="citations-section">
          <h3>Faithfulness (NLI verification)</h3>
          {faithScore !== undefined && faithScore !== null ? (
            <>
              <Gauge value={faithScore} label="Faithfulness score" />
              <p className="muted">
                Bands mirror the verifier's own thresholds (Ch5, Table 5.5): &lt;50%
                hard-abstention zone · 50–80% low-confidence · ≥80% passes the aggregate
                faithfulness gate.
              </p>
            </>
          ) : (
            <p className="muted">Verification was disabled for this query.</p>
          )}
        </div>

        <div className="citations-section">
          <h3>Citation validity</h3>
          {total > 0 ? (
            <CitationValidityBar valid={valid} total={total} />
          ) : (
            <p className="muted">No citations were generated for this answer.</p>
          )}
        </div>
      </div>

      <div className="citations-section">
        <h3>Retrieved passage scores</h3>
        {retrievalScores.length > 0 ? (
          <>
            <RetrievalBarChart items={retrievalScores} citedIds={citedIds} />
            <p className="muted">
              Green bars are passages the generator actually cited in the answer; blue bars were
              retrieved but not cited.
            </p>
          </>
        ) : (
          <p className="muted">No retrieval score data available for this answer.</p>
        )}
      </div>
    </>
  )
}

export default AnswerDiagnostics
