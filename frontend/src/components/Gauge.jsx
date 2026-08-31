const RED = '#dc3545'
const AMBER = '#ffc107'
const GREEN = '#28a745'

function polarToCartesian(cx, cy, r, angleDeg) {
  const rad = (angleDeg * Math.PI) / 180
  return { x: cx + r * Math.cos(rad), y: cy - r * Math.sin(rad) }
}

// Filled annulus wedge (donut segment) between two angles — the SVG
// equivalent of matplotlib's Wedge, used for each coloured band.
function describeAnnulusSegment(cx, cy, rOuter, rInner, startAngle, endAngle) {
  const outerStart = polarToCartesian(cx, cy, rOuter, startAngle)
  const outerEnd = polarToCartesian(cx, cy, rOuter, endAngle)
  const innerEnd = polarToCartesian(cx, cy, rInner, endAngle)
  const innerStart = polarToCartesian(cx, cy, rInner, startAngle)
  const largeArc = startAngle - endAngle > 180 ? 1 : 0
  return [
    `M ${outerStart.x} ${outerStart.y}`,
    `A ${rOuter} ${rOuter} 0 ${largeArc} 1 ${outerEnd.x} ${outerEnd.y}`,
    `L ${innerEnd.x} ${innerEnd.y}`,
    `A ${rInner} ${rInner} 0 ${largeArc} 0 ${innerStart.x} ${innerStart.y}`,
    'Z',
  ].join(' ')
}

/**
 * Semicircular 0-1 gauge with red/amber/green bands and a needle.
 * Band edges default to the verifier's own thresholds (hard-abstention 0.50,
 * aggregate faithfulness gate 0.80 — Chapter 5, Table 5.5) so the chart reads
 * directly against the documented decision boundaries, not arbitrary cut-offs.
 */
function Gauge({ value, label, low = 0.5, high = 0.8 }) {
  const v = Math.max(0, Math.min(1, value))
  const cx = 110
  const cy = 100
  const rOuter = 90
  const rInner = 58
  const bands = [
    [0, low, RED],
    [low, high, AMBER],
    [high, 1, GREEN],
  ]
  const needleAngle = 180 - v * 180
  const needleTip = polarToCartesian(cx, cy, rOuter * 0.82, needleAngle)

  return (
    <div className="gauge">
      <svg viewBox="0 0 220 115" className="gauge-svg">
        {bands.map(([start, end, color]) => (
          <path
            key={color}
            d={describeAnnulusSegment(cx, cy, rOuter, rInner, 180 - start * 180, 180 - end * 180)}
            fill={color}
            opacity="0.9"
          />
        ))}
        <line
          x1={cx}
          y1={cy}
          x2={needleTip.x}
          y2={needleTip.y}
          stroke="#212529"
          strokeWidth="3"
          strokeLinecap="round"
        />
        <circle cx={cx} cy={cy} r="5" fill="#212529" />
      </svg>
      <div className="gauge-value">{(v * 100).toFixed(1)}%</div>
      <div className="gauge-label">{label}</div>
    </div>
  )
}

export default Gauge
