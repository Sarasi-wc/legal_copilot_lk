#!/usr/bin/env python3
"""
Compute inter-rater reliability (Cohen's κ) between two expert annotators
for the Q-A-C dataset rubric review.

Expects two CSV files in the qac_expert_review.csv format, each with
Correctness / Completeness / Clarity columns filled in for overlapping items.

Usage:
    python scripts/compute_irr.py \
        --e1 data/evaluation/qac_expert_review.csv \
        --e2 data/evaluation/qac_expert_review_e2.csv \
        [--items QAC_001-QAC_075] \
        [--output results/irr/]

The --items flag accepts a range (QAC_001-QAC_075), a comma-separated list
(QAC_001,QAC_003), or is omitted to use all items that both files share.

Output:
    Console summary table + results/irr/irr_results.json
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

from sklearn.metrics import cohen_kappa_score

# ── Landis & Koch (1977) interpretation ──────────────────────────────────────
LANDIS_KOCH = [
    (float("-inf"), 0.00, "Poor"),
    (0.00, 0.20, "Slight"),
    (0.21, 0.40, "Fair"),
    (0.41, 0.60, "Moderate"),
    (0.61, 0.75, "Substantial"),
    (0.75, 0.80, "Substantial (meets target)"),
    (0.81, 1.01, "Almost perfect"),
]

DIMENSIONS = [
    "Correctness (1=Poor  5=Excellent)",
    "Completeness (1=Poor  5=Excellent)",
    "Clarity (1=Poor  5=Excellent)",
]
DIM_LABELS = ["Correctness", "Completeness", "Clarity"]


def interpret(kappa: float) -> str:
    for lo, hi, label in LANDIS_KOCH:
        if lo <= kappa < hi:
            return label
    return "Almost perfect"


def parse_rating(value: str):
    """Return int 1–5 or None if blank/invalid."""
    v = value.strip()
    if not v:
        return None
    try:
        r = int(v)
        if 1 <= r <= 5:
            return r
    except ValueError:
        pass
    return None


def load_ratings(path: str) -> dict:
    """Return {item_id: {dim_label: rating_int}} for all rated rows."""
    ratings = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row.get("Item ID", "").strip()
            if not item_id:
                continue
            item_ratings = {}
            for dim, label in zip(DIMENSIONS, DIM_LABELS):
                item_ratings[label] = parse_rating(row.get(dim, ""))
            ratings[item_id] = item_ratings
    return ratings


def parse_item_range(spec: str) -> set:
    """Parse 'QAC_001-QAC_075' or 'QAC_001,QAC_003,...' into a set of IDs."""
    if "-" in spec and "," not in spec:
        parts = spec.split("-", 1)
        prefix = "QAC_"
        try:
            lo = int(parts[0].replace(prefix, ""))
            hi = int(parts[1].replace(prefix, ""))
            return {f"{prefix}{n:03d}" for n in range(lo, hi + 1)}
        except ValueError:
            pass
    return {s.strip() for s in spec.split(",")}


def compute_irr(e1_path: str, e2_path: str, item_filter, output_dir: str):
    e1 = load_ratings(e1_path)
    e2 = load_ratings(e2_path)

    # Intersect by item ID, then apply optional filter
    common = set(e1.keys()) & set(e2.keys())
    if item_filter:
        common &= item_filter
    common = sorted(common)

    if not common:
        print("ERROR: No overlapping rated items found between the two files.")
        sys.exit(1)

    # Collect paired ratings per dimension; skip items where either is missing
    paired: dict[str, list[tuple[int, int]]] = {d: [] for d in DIM_LABELS}
    skipped_per_dim: dict[str, list[str]] = {d: [] for d in DIM_LABELS}

    for item_id in common:
        for dim in DIM_LABELS:
            r1 = e1[item_id].get(dim)
            r2 = e2[item_id].get(dim)
            if r1 is not None and r2 is not None:
                paired[dim].append((r1, r2))
            else:
                skipped_per_dim[dim].append(item_id)

    # ── Per-dimension κ ───────────────────────────────────────────────────────
    results = {}
    print()
    print("=" * 65)
    print("INTER-RATER RELIABILITY RESULTS")
    print("=" * 65)
    print(f"E1 file : {e1_path}")
    print(f"E2 file : {e2_path}")
    print(f"Items   : {len(common)} overlapping  |  filter: {item_filter or 'all'}")
    print()
    print(f"{'Dimension':<14}  {'N pairs':>7}  {'Unweighted κ':>13}  {'Weighted κ (quad)':>18}  Interpretation")
    print("-" * 65)

    kappas_weighted = []
    for dim in DIM_LABELS:
        pairs = paired[dim]
        n = len(pairs)
        if n < 2:
            print(f"{dim:<14}  {n:>7}  {'—':>13}  {'—':>18}  Insufficient data")
            results[dim] = {"n": n, "kappa_unweighted": None, "kappa_weighted": None}
            continue

        y1 = [p[0] for p in pairs]
        y2 = [p[1] for p in pairs]

        # sklearn requires both arrays to have the same set of labels
        labels = list(range(1, 6))
        k_uw = cohen_kappa_score(y1, y2, labels=labels, weights=None)
        k_w  = cohen_kappa_score(y1, y2, labels=labels, weights="quadratic")
        kappas_weighted.append(k_w)

        interp = interpret(k_w)
        print(f"{dim:<14}  {n:>7}  {k_uw:>13.4f}  {k_w:>18.4f}  {interp}")
        results[dim] = {
            "n": n,
            "kappa_unweighted": round(k_uw, 4),
            "kappa_weighted_quadratic": round(k_w, 4),
            "interpretation": interp,
            "skipped_items": skipped_per_dim[dim],
        }

    # ── Overall (mean weighted κ across dimensions) ───────────────────────────
    if kappas_weighted:
        overall = sum(kappas_weighted) / len(kappas_weighted)
        overall_interp = interpret(overall)
        print("-" * 65)
        print(f"{'Overall (mean)':14}  {'':>7}  {'':>13}  {overall:>18.4f}  {overall_interp}")
        meets_target = overall >= 0.75
        print()
        print(f"Target κ ≥ 0.75 : {'MET ✓' if meets_target else 'NOT MET ✗'}  (overall weighted κ = {overall:.4f})")
    else:
        overall = None
        overall_interp = "N/A"
        meets_target = False

    # ── Disagreement table ────────────────────────────────────────────────────
    print()
    print("DISAGREEMENT TABLE  (|E1 − E2| ≥ 2 on any dimension)")
    print("-" * 65)
    disagreements = []
    for item_id in common:
        row_issues = []
        for dim in DIM_LABELS:
            r1 = e1[item_id].get(dim)
            r2 = e2[item_id].get(dim)
            if r1 is not None and r2 is not None and abs(r1 - r2) >= 2:
                row_issues.append(f"{dim}: E1={r1} E2={r2} Δ={abs(r1-r2)}")
        if row_issues:
            disagreements.append({"item_id": item_id, "issues": row_issues})
            print(f"  {item_id}: {'; '.join(row_issues)}")

    if not disagreements:
        print("  None — all pairs within 1 point on every dimension.")

    print()
    print(f"Total disagreement items: {len(disagreements)} / {len(common)}")
    print("=" * 65)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "irr_results.json")
    output = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "e1_file": e1_path,
        "e2_file": e2_path,
        "n_items": len(common),
        "item_ids": common,
        "dimensions": results,
        "overall_kappa_weighted": round(overall, 4) if overall is not None else None,
        "overall_interpretation": overall_interp,
        "meets_target_0_75": bool(meets_target),
        "disagreements": disagreements,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute inter-rater reliability (Cohen's κ)")
    parser.add_argument(
        "--e1",
        default="data/evaluation/qac_expert_review.csv",
        help="Path to Expert 1 annotation CSV (default: qac_expert_review.csv)",
    )
    parser.add_argument(
        "--e2",
        default="data/evaluation/qac_expert_review_e2.csv",
        help="Path to Expert 2 annotation CSV (default: qac_expert_review_e2.csv)",
    )
    parser.add_argument(
        "--items",
        default="QAC_001-QAC_075",
        help="Item range or list to evaluate (default: QAC_001-QAC_075)",
    )
    parser.add_argument(
        "--output",
        default="results/irr",
        help="Output directory for JSON results (default: results/irr)",
    )
    args = parser.parse_args()

    item_filter = parse_item_range(args.items) if args.items else None

    # Resolve paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    e1_path = args.e1 if os.path.isabs(args.e1) else os.path.join(project_root, args.e1)
    e2_path = args.e2 if os.path.isabs(args.e2) else os.path.join(project_root, args.e2)
    out_dir  = args.output if os.path.isabs(args.output) else os.path.join(project_root, args.output)

    for path, label in [(e1_path, "E1"), (e2_path, "E2")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} file not found: {path}")
            sys.exit(1)

    compute_irr(e1_path, e2_path, item_filter, out_dir)


if __name__ == "__main__":
    main()
