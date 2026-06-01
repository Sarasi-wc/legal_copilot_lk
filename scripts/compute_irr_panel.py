"""
compute_irr_panel.py
--------------------
Compute inter-rater reliability (quadratic-weighted Cohen's κ) from the
five-reviewer panel CSV (qac_panel_review_all.csv).

For each of the C(5,2)=10 reviewer pairs, κ is computed per dimension
(Correctness, Completeness, Clarity) and as an overall mean.  The mean
across all 10 pairs is the aggregate IRR estimate.

Continuous panel scores (e.g. 4.7) are rounded to the nearest integer
before κ computation; the resulting ceiling distribution is documented.

Usage:
    venv/bin/python scripts/compute_irr_panel.py [--items QAC_001-QAC_075] [--output results/irr]

Output:
    Console summary + results/irr/irr_panel_results.json
"""

import argparse
import csv
import itertools
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PANEL_CSV    = PROJECT_ROOT / "data/evaluation/qac_panel_review_all.csv"

REVIEWER_NAMES = [
    "Chathuri Madushika (LLB OUSL)",
    "Rohana Dasanayaka (LLB OUSL)",
    "Dulani Kulasooriya (LLB OUSL)",
    "Dulshan Liyanaarachchi (LLB OUSL)",
    "Inoshika Kodithuwakku (LLB OUSL)",
]
REVIEWER_COL_START = 11
COLS_PER_REVIEWER  = 5          # Correctness, Completeness, Clarity, Comments, Date
DIM_LABELS = ["Correctness", "Completeness", "Clarity"]
DIM_OFFSETS = [0, 1, 2]         # col offsets within a reviewer block

LANDIS_KOCH = [
    (float("-inf"), 0.00, "Poor"),
    (0.00, 0.20, "Slight"),
    (0.21, 0.40, "Fair"),
    (0.41, 0.60, "Moderate"),
    (0.61, 0.75, "Substantial"),
    (0.75, 0.80, "Substantial (meets target)"),
    (0.81, 1.01, "Almost perfect"),
]


def interpret(kappa: float) -> str:
    for lo, hi, label in LANDIS_KOCH:
        if lo <= kappa < hi:
            return label
    return "Almost perfect"


def std_round(x: float) -> int:
    """Round half-up (standard rounding), then clamp to 1-5."""
    return max(1, min(5, int(x + 0.5)))


def parse_item_range(spec: str):
    if not spec:
        return None
    if "-" in spec and "," not in spec:
        parts = spec.split("-", 1)
        try:
            lo = int(parts[0].replace("QAC_", ""))
            hi = int(parts[1].replace("QAC_", ""))
            return {f"QAC_{n:03d}" for n in range(lo, hi + 1)}
        except ValueError:
            pass
    return {s.strip() for s in spec.split(",")}


def load_panel(csv_path: Path, item_filter) -> dict:
    """
    Returns {item_id: {reviewer_idx: {dim_label: int_rating}}}.
    Continuous scores are rounded to nearest integer (1-5).
    """
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    data = {}
    seen = set()
    for row in rows[3:]:           # 3 header rows
        if not row or not row[0].strip():
            continue
        item_id = row[0].strip()
        if item_id in seen:
            continue
        seen.add(item_id)
        if item_filter and item_id not in item_filter:
            continue

        reviewer_data = {}
        for r_idx in range(len(REVIEWER_NAMES)):
            base = REVIEWER_COL_START + r_idx * COLS_PER_REVIEWER
            dim_ratings = {}
            for d_idx, dim in enumerate(DIM_LABELS):
                col = base + DIM_OFFSETS[d_idx]
                raw = row[col].strip().lstrip(".") if col < len(row) else ""
                try:
                    dim_ratings[dim] = std_round(float(raw))
                except ValueError:
                    dim_ratings[dim] = None
            reviewer_data[r_idx] = dim_ratings
        data[item_id] = reviewer_data

    return data


def compute_all_pairwise(panel_data: dict, output_dir: Path):
    items = sorted(panel_data.keys())
    n_items = len(items)

    if n_items == 0:
        print("ERROR: No items found after applying filter.")
        return

    # ── Distribution of rounded scores ───────────────────────────────────────
    all_scores = []
    for item in items:
        for r_idx in panel_data[item]:
            for dim in DIM_LABELS:
                v = panel_data[item][r_idx].get(dim)
                if v is not None:
                    all_scores.append(v)
    score_dist = Counter(all_scores)

    print()
    print("=" * 72)
    print("PANEL INTER-RATER RELIABILITY  (quadratic-weighted Cohen's κ)")
    print("=" * 72)
    print(f"Items      : {n_items}  ({items[0]} – {items[-1]})")
    print(f"Reviewers  : {len(REVIEWER_NAMES)}")
    print(f"Score dist : { {k: score_dist[k] for k in sorted(score_dist)} }  (after rounding)")
    print()

    # ── Pairwise κ ────────────────────────────────────────────────────────────
    pair_results = []
    labels = list(range(1, 6))

    for r1_idx, r2_idx in itertools.combinations(range(len(REVIEWER_NAMES)), 2):
        pair_label = f"R{r1_idx+1}×R{r2_idx+1}"
        dim_kappas = []
        dim_detail = {}

        for dim in DIM_LABELS:
            y1, y2 = [], []
            for item in items:
                v1 = panel_data[item][r1_idx].get(dim)
                v2 = panel_data[item][r2_idx].get(dim)
                if v1 is not None and v2 is not None:
                    y1.append(v1)
                    y2.append(v2)

            if len(y1) < 2:
                dim_detail[dim] = None
                continue

            k_w = cohen_kappa_score(y1, y2, labels=labels, weights="quadratic")
            dim_detail[dim] = round(k_w, 4)
            dim_kappas.append(k_w)

        mean_k = round(sum(dim_kappas) / len(dim_kappas), 4) if dim_kappas else None
        pair_results.append({
            "pair":        pair_label,
            "r1":          REVIEWER_NAMES[r1_idx],
            "r2":          REVIEWER_NAMES[r2_idx],
            "dimensions":  dim_detail,
            "mean_kappa":  mean_k,
            "interpretation": interpret(mean_k) if mean_k is not None else "N/A",
        })

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"{'Pair':<8}  {'Correctness':>12}  {'Completeness':>12}  {'Clarity':>9}  {'Mean κ':>8}  Interpretation")
    print("-" * 72)
    for pr in pair_results:
        d = pr["dimensions"]
        c  = f"{d['Correctness']:.4f}"  if d['Correctness']  is not None else "—"
        p  = f"{d['Completeness']:.4f}" if d['Completeness'] is not None else "—"
        cl = f"{d['Clarity']:.4f}"      if d['Clarity']      is not None else "—"
        mk = f"{pr['mean_kappa']:.4f}"  if pr['mean_kappa']  is not None else "—"
        print(f"{pr['pair']:<8}  {c:>12}  {p:>12}  {cl:>9}  {mk:>8}  {pr['interpretation']}")

    # ── Aggregate across all pairs ────────────────────────────────────────────
    valid_means = [pr["mean_kappa"] for pr in pair_results if pr["mean_kappa"] is not None]
    agg_kappa = round(sum(valid_means) / len(valid_means), 4) if valid_means else None

    per_dim_agg = {}
    for dim in DIM_LABELS:
        vals = [pr["dimensions"][dim] for pr in pair_results if pr["dimensions"].get(dim) is not None]
        per_dim_agg[dim] = round(sum(vals) / len(vals), 4) if vals else None

    print("-" * 72)
    print(f"{'MEAN':8}  {per_dim_agg['Correctness']:>12.4f}  {per_dim_agg['Completeness']:>12.4f}  "
          f"{per_dim_agg['Clarity']:>9.4f}  {agg_kappa:>8.4f}  {interpret(agg_kappa)}")
    print()
    meets_target = agg_kappa is not None and agg_kappa >= 0.75
    print(f"Target κ ≥ 0.75 : {'MET ✓' if meets_target else 'NOT MET ✗'}  "
          f"(mean pairwise weighted κ = {agg_kappa:.4f})")

    print()
    print("NOTE: Near-zero or low κ with high mean scores (4.3–4.9 range) indicates")
    print("      a ceiling effect — insufficient score variance for κ to be informative.")
    print("      ICC(2,1) is the recommended metric for continuous panel ratings;")
    print("      pairwise MAD and % within 0.5 pts are reported in Section 6.3.4.")
    print("=" * 72)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = output_dir / "irr_panel_results.json"
    output = {
        "run_timestamp":       datetime.utcnow().isoformat() + "Z",
        "source_csv":          str(PANEL_CSV),
        "n_items":             n_items,
        "item_range":          f"{items[0]}–{items[-1]}",
        "n_reviewers":         len(REVIEWER_NAMES),
        "reviewer_names":      REVIEWER_NAMES,
        "score_distribution":  {str(k): score_dist[k] for k in sorted(score_dist)},
        "pair_results":        pair_results,
        "aggregate": {
            "mean_kappa_correctness":  per_dim_agg["Correctness"],
            "mean_kappa_completeness": per_dim_agg["Completeness"],
            "mean_kappa_clarity":      per_dim_agg["Clarity"],
            "mean_kappa_overall":      agg_kappa,
            "interpretation":          interpret(agg_kappa) if agg_kappa else "N/A",
            "meets_target_0_75":       bool(meets_target),
        },
        "ceiling_effect_note": (
            "Scores clustered 4–5 after rounding; κ underestimates agreement. "
            "ICC(2,1) and MAD are primary agreement metrics (Section 6.3.4)."
        ),
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items",  default="QAC_001-QAC_075",
                        help="Item range (default: QAC_001-QAC_075)")
    parser.add_argument("--output", default="results/irr",
                        help="Output directory (default: results/irr)")
    args = parser.parse_args()

    item_filter = parse_item_range(args.items) if args.items else None
    out_dir = Path(args.output) if os.path.isabs(args.output) else PROJECT_ROOT / args.output

    panel_data = load_panel(PANEL_CSV, item_filter)
    compute_all_pairwise(panel_data, out_dir)


if __name__ == "__main__":
    main()
