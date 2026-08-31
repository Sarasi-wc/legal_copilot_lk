/**
 * Horizontal bar chart of retrieved passage scores, cited passages highlighted.
 * Labels are disambiguated (not just deduplicated) because a long section split
 * into several chunks can produce multiple passages sharing the same Act +
 * Article label (e.g. two _CHUNK_N passages both under Article 13) — without
 * disambiguation those would be indistinguishable in the legend.
 */
export function RetrievalBarChart({ items, citedIds }) {
  if (!items || items.length === 0) return null

  const maxScore = Math.max(...items.map((i) => i.score || 0), 0.1)
  const seen = {}
  const rows = [...items]
    .sort((a, b) => (b.score || 0) - (a.score || 0))
    .map((item, idx) => {
      const base = item.section_number
        ? `${(item.act_name || '?').slice(0, 3)}. Art ${item.section_number}`
        : item.passage_id || '?'
      seen[base] = (seen[base] || 0) + 1
      const label = seen[base] === 1 ? base : `${base} (${seen[base]})`
      return { ...item, label, key: `${item.passage_id || 'p'}-${idx}` }
    })

  return (
    <div className="retrieval-chart">
      {rows.map((row) => (
        <div className="retrieval-row" key={row.key}>
          <span className="retrieval-label">{row.label}</span>
          <div className="retrieval-track">
            <div
              className="retrieval-fill"
              style={{
                width: `${((row.score || 0) / maxScore) * 100}%`,
                background: citedIds.has(row.passage_id) ? '#28a745' : '#8ca9c9',
              }}
            />
          </div>
          <span className="retrieval-score">{(row.score || 0).toFixed(3)}</span>
        </div>
      ))}
    </div>
  )
}

/** Simple vertical bar chart (e.g. query-type distribution counts). */
export function VerticalBarChart({ labels, values, colors, valueFormatter }) {
  const max = Math.max(...values, 1)
  const fmt = valueFormatter || ((v) => v)

  return (
    <div className="vbar-chart">
      {labels.map((label, i) => (
        <div className="vbar-col" key={label}>
          <span className="vbar-value">{fmt(values[i])}</span>
          <div className="vbar-track">
            <div
              className="vbar-fill"
              style={{ height: `${(values[i] / max) * 100}%`, background: colors[i % colors.length] }}
            />
          </div>
          <span className="vbar-label">{label}</span>
        </div>
      ))}
    </div>
  )
}

/**
 * Grouped vertical bar chart for RQ2 retrieval effectiveness (3 metrics x 4
 * conditions). Y-axis is zoomed to [yMin, yMax] rather than [0, 1] — matching
 * the Streamlit matplotlib chart's ylim(0.55, 1.02) — because all scores fall
 * in a narrow high band and a full 0-1 axis would flatten the visible gaps
 * between conditions to the point of being unreadable.
 */
export function GroupedRetrievalChart({ conditions, series, yMin = 0.55, yMax = 1.02 }) {
  const scale = (v) => Math.max(0, Math.min(100, ((v - yMin) / (yMax - yMin)) * 100))

  return (
    <div className="grouped-chart-wrap">
      <div className="grouped-chart">
        {conditions.map((cond, ci) => (
          <div className="grouped-col" key={cond}>
            <div className="grouped-bars">
              {series.map((s) => (
                <div className="grouped-bar-wrap" key={s.name}>
                  <span className="grouped-bar-value">{s.values[ci].toFixed(3)}</span>
                  <div className="grouped-bar-track">
                    <div
                      className="grouped-bar-fill"
                      style={{ height: `${scale(s.values[ci])}%`, background: s.color }}
                    />
                  </div>
                </div>
              ))}
            </div>
            <span className="grouped-col-label">{cond}</span>
          </div>
        ))}
      </div>
      <div className="grouped-legend">
        {series.map((s) => (
          <span key={s.name}>
            <span className="legend-swatch" style={{ background: s.color }} /> {s.name}
          </span>
        ))}
      </div>
    </div>
  )
}

/** Single stacked bar: valid (green) vs invalid/fabricated (red) citations. */
export function CitationValidityBar({ valid, total }) {
  const invalid = Math.max(total - valid, 0)
  const validPct = total > 0 ? (valid / total) * 100 : 0
  const invalidPct = total > 0 ? (invalid / total) * 100 : 0

  return (
    <div className="citation-bar-wrap">
      <div className="citation-bar">
        <div
          className="citation-bar-segment"
          style={{ width: `${validPct}%`, background: '#28a745' }}
        />
        <div
          className="citation-bar-segment"
          style={{ width: `${invalidPct}%`, background: '#dc3545' }}
        />
      </div>
      <div className="citation-bar-legend">
        <span>
          <span className="legend-swatch" style={{ background: '#28a745' }} /> Valid ({valid})
        </span>
        <span>
          <span className="legend-swatch" style={{ background: '#dc3545' }} /> Invalid ({invalid})
        </span>
      </div>
    </div>
  )
}
