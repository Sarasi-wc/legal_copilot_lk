import { useState } from 'react'
import axios from 'axios'

const API_URL = '/api'

const exampleQuestions = [
  'What are the fundamental rights guaranteed under Article 14 of the Constitution?',
  'How is the Constitution of Sri Lanka amended?',
  'What language is designated as the official language of Sri Lanka?',
  'What are the qualifications required to become President of Sri Lanka?',
]

function AskQuestion({ settings, onAnswered }) {
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!question.trim()) {
      setError('Please enter a question')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await axios.post(`${API_URL}/answer`, {
        question,
        retrieval_method: settings.retrievalMethod,
        top_k: settings.topK,
        include_verification: settings.includeVerification,
      })

      setResult(response.data)
      onAnswered(response.data, question)
    } catch (err) {
      setError(err.response?.data?.detail || 'Error processing question')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const loadExample = (example) => {
    setQuestion(example)
  }

  return (
    <>
      <div className="examples-section">
        <h3>📝 Example Questions</h3>
        <div className="example-buttons">
          {exampleQuestions.map((example, idx) => (
            <button key={idx} onClick={() => loadExample(example)} className="example-button">
              {example}
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="question-form">
        <h3>❓ Ask Your Legal Question</h3>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g., What are the fundamental rights guaranteed under Article 14 of the Constitution?"
          rows={4}
          className="question-input"
        />

        <div className="button-group">
          <button type="submit" disabled={loading} className="submit-button">
            {loading ? '🔍 Searching...' : '🔍 Ask Question'}
          </button>

          <button
            type="button"
            onClick={() => {
              setQuestion('')
              setResult(null)
              setError(null)
            }}
            className="clear-button"
          >
            🗑️ Clear
          </button>
        </div>
      </form>

      {error && (
        <div className="error-box">
          <h3>❌ Error</h3>
          <p>{error}</p>
        </div>
      )}

      {result && (
        <div className="results-section">
          {result.abstained ? (
            <div className="warning-box">
              <h3>⚠️ Unable to Answer</h3>
              <p>
                <strong>Reason:</strong> {result.abstention_reason}
              </p>
              <p>{result.answer}</p>
            </div>
          ) : (
            <>
              <div className="answer-box">
                <h3>📋 Answer</h3>
                <p>{result.answer}</p>
              </div>

              {result.citations && result.citations.length > 0 && (
                <div className="citations-section">
                  <h3>📚 Citations</h3>
                  {result.citations.map((citation, idx) => (
                    <div key={idx} className="citation-box">
                      {citation.has_matching_evidence ? '✅' : '❌'}
                      <strong> Citation {idx + 1}:</strong> [{citation.act_name},{' '}
                      {citation.section_label || 'Article'} {citation.section}]
                    </div>
                  ))}
                </div>
              )}

              {result.verification && (
                <div className="verification-section">
                  <h3>✓ Verification</h3>
                  {result.verification.verification_passed ? (
                    <div className="success-box">
                      ✅ <strong>Verification Passed</strong>
                      <br />
                      Faithfulness Score:{' '}
                      {(result.verification.faithfulness.faithfulness_score * 100).toFixed(1)}%
                    </div>
                  ) : (
                    <div className="warning-box">
                      ⚠️ <strong>Verification Issues Detected</strong>
                      <br />
                      Faithfulness Score:{' '}
                      {(result.verification.faithfulness.faithfulness_score * 100).toFixed(1)}%
                    </div>
                  )}

                  <div className="metrics">
                    <div className="metric">
                      <strong>Valid Citations</strong>
                      <span>
                        {result.verification.citation_validation?.valid_citations ?? 0}/
                        {result.verification.citation_validation?.total_citations ?? 0}
                      </span>
                    </div>
                    {result.verification.citation_coverage && (
                      <div className="metric">
                        <strong>Coverage</strong>
                        <span>
                          {((result.verification.citation_coverage.citation_density ?? 0) * 100).toFixed(
                            1
                          )}
                          %
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              <details className="details-section">
                <summary>🔍 Query Details</summary>
                <div className="details-content">
                  <p>
                    <strong>Query Type:</strong> {result.query_metadata?.query_type || 'N/A'}
                  </p>
                  <p>
                    <strong>Retrieval Method:</strong> {result.retrieval_method}
                  </p>
                  <p>
                    <strong>Passages Retrieved:</strong>{' '}
                    {result.num_retrieved || result.evidence_used || 'N/A'}
                  </p>
                  <p>
                    <strong>Evidence Used:</strong> {result.evidence_used}
                  </p>
                  {result.query_metadata?.section_refs && result.query_metadata.section_refs.length > 0 && (
                    <p>
                      <strong>Section References:</strong>{' '}
                      {result.query_metadata.section_refs.join(', ')}
                    </p>
                  )}
                </div>
              </details>

              <p className="diagnostics-hint">
                📈 Open the <strong>Answer Diagnostics</strong> tab to see this answer's retrieval
                scores, citation validity, and faithfulness gauge.
              </p>
            </>
          )}
        </div>
      )}
    </>
  )
}

export default AskQuestion
