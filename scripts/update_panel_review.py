"""
update_panel_review.py
----------------------
Three operations in one pass:

1. Parse the 54-item expert-review CSV (QAC_301–334 + ABS_001–020) and
   add a `panel_review` field to each of the 34 matching QAC items in
   qac_dataset.json.

2. Set gold_answers[0].annotator_id = "LEGAL_EXPERT_1" for QAC_076–334
   (259 items whose annotator_id was incorrectly left blank).

3. Create data/evaluation/qac_panel_review_all.csv — a comprehensive
   5-reviewer sheet covering all 354 items (334 QAC + 20 ABS).
   QAC_301–334 and ABS_001–020 are pre-populated from the source CSV;
   QAC_001–300 rows have blank rating columns ready for reviewers.

Usage:
    venv/bin/python scripts/update_panel_review.py [--dry-run]

Inputs:
    data/evaluation/qac_dataset.json
    data/evaluation/abstention_test_set.json
    "12. Final Report/Dataset/Q-A-C dataset for expert review - 301-354.csv"
        (path relative to project root's parent — see SOURCE_CSV below)

Outputs (unless --dry-run):
    data/evaluation/qac_dataset.json          (in-place, backup at .json.bak_panel)
    data/evaluation/qac_panel_review_all.csv  (new file)
"""

import argparse
import csv
import json
import os
import re
import shutil
from copy import deepcopy
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
QAC_JSON      = PROJECT_ROOT / "data/evaluation/qac_dataset.json"
ABS_JSON      = PROJECT_ROOT / "data/evaluation/abstention_test_set.json"
SOURCE_CSV    = (PROJECT_ROOT.parent.parent /
                 "12. Final Report/Dataset/Q-A-C dataset for expert review - 301-354.csv")
OUTPUT_CSV    = PROJECT_ROOT / "data/evaluation/qac_panel_review_all.csv"

# ---------------------------------------------------------------------------
# CSV column layout
# ---------------------------------------------------------------------------
REVIEWER_NAMES = [
    "Chathuri Madushika (LLB OUSL)",
    "Rohana Dasanayaka (LLB OUSL)",
    "Dulani Kulasooriya (LLB OUSL)",
    "Dulshan Liyanaarachchi (LLB OUSL)",
    "Inoshika Kodithuwakku (LLB OUSL)",
]
# Each reviewer block: Correctness, Completeness, Clarity, Comments, Date
REVIEWER_COL_START = 11   # first reviewer starts at col index 11
COLS_PER_REVIEWER  = 5


def reviewer_cols(reviewer_idx: int):
    base = REVIEWER_COL_START + reviewer_idx * COLS_PER_REVIEWER
    return {
        "correctness":  base,
        "completeness": base + 1,
        "clarity":      base + 2,
        "comments":     base + 3,
        "date":         base + 4,
    }


def safe_float(val: str):
    """Parse a rating string like '4.7' or '.4.7' (typo in source CSV)."""
    val = val.strip().lstrip(".")
    try:
        return float(val)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 1. Parse source CSV
# ---------------------------------------------------------------------------
def parse_source_csv(path: Path) -> dict:
    """
    Returns a dict keyed by item_id (e.g. 'QAC_301', 'ABS_001') with:
        {
          'row': list of raw CSV cells,
          'panel_review': {
              'n_reviewers': int,
              'mean_correctness': float,
              'mean_completeness': float,
              'mean_clarity': float,
              'mean_overall': float,
              'reviewer_scores': [{'reviewer':str, 'correctness':float,
                                   'completeness':float, 'clarity':float,
                                   'comments':str, 'date':str}, ...]
          }
        }
    Rows with no item_id or whose item_id doesn't start with QAC_/ABS_ are
    skipped (handles trailing empty rows and the stray QAC_334 duplicate).
    """
    results = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        rows = list(reader)

    # rows[0] = header (reviewer names), rows[1] = sub-header, rows[2..] = data
    seen_item_ids: set[str] = set()

    for row in rows[2:]:
        if not row:
            continue
        item_id = row[0].strip()
        if not re.match(r"^(QAC|ABS)_\d+$", item_id):
            continue
        if item_id in seen_item_ids:
            continue   # skip stray duplicate at bottom of CSV
        seen_item_ids.add(item_id)

        reviewer_scores = []
        for r_idx, r_name in enumerate(REVIEWER_NAMES):
            cols = reviewer_cols(r_idx)
            # Pad row if shorter than expected
            def get_col(c):
                return row[c].strip() if c < len(row) else ""

            correctness  = safe_float(get_col(cols["correctness"]))
            completeness = safe_float(get_col(cols["completeness"]))
            clarity      = safe_float(get_col(cols["clarity"]))

            reviewer_scores.append({
                "reviewer":    r_name,
                "correctness":  correctness,
                "completeness": completeness,
                "clarity":      clarity,
                "comments":     get_col(cols["comments"]),
                "date":         get_col(cols["date"]),
            })

        valid_scores = [s for s in reviewer_scores
                        if None not in (s["correctness"], s["completeness"], s["clarity"])]
        n = len(valid_scores)

        if n == 0:
            mean_c = mean_p = mean_cl = mean_ov = None
        else:
            mean_c  = round(sum(s["correctness"]  for s in valid_scores) / n, 3)
            mean_p  = round(sum(s["completeness"] for s in valid_scores) / n, 3)
            mean_cl = round(sum(s["clarity"]      for s in valid_scores) / n, 3)
            mean_ov = round((mean_c + mean_p + mean_cl) / 3, 3)

        results[item_id] = {
            "row": row,
            "panel_review": {
                "n_reviewers":       n,
                "mean_correctness":  mean_c,
                "mean_completeness": mean_p,
                "mean_clarity":      mean_cl,
                "mean_overall":      mean_ov,
                "reviewer_scores":   reviewer_scores,
            }
        }

    return results


# ---------------------------------------------------------------------------
# 2. Update qac_dataset.json
# ---------------------------------------------------------------------------
def update_qac_json(qac_path: Path, csv_data: dict, dry_run: bool):
    with open(qac_path) as fh:
        data = json.load(fh)

    updated_panel = 0
    updated_annotator = 0

    for item in data["items"]:
        item_id = item["item_id"]

        # Operation A: add panel_review for QAC_301–334
        if item_id in csv_data and item_id.startswith("QAC_"):
            item["panel_review"] = csv_data[item_id]["panel_review"]
            updated_panel += 1

        # Operation B: update annotator_id for QAC_076–334
        m = re.match(r"QAC_(\d+)$", item_id)
        if m:
            num = int(m.group(1))
            if num >= 76:
                for ans in item["gold_answers"]:
                    if ans.get("annotator_id", "") == "":
                        ans["annotator_id"] = "LEGAL_EXPERT_1"
                        updated_annotator += 1

    print(f"panel_review added to {updated_panel} QAC items")
    print(f"annotator_id updated to LEGAL_EXPERT_1 for {updated_annotator} gold_answer entries")

    if not dry_run:
        backup = qac_path.with_suffix(".json.bak_panel")
        shutil.copy2(qac_path, backup)
        print(f"Backup written: {backup}")
        with open(qac_path, "w") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        print(f"Updated: {qac_path}")
    else:
        print("[dry-run] qac_dataset.json NOT written")


# ---------------------------------------------------------------------------
# 3. Build comprehensive panel-review CSV
# ---------------------------------------------------------------------------
CSV_HEADER_ROW1 = (
    ["Item ID", "Query Type", "Difficulty", "Legal Domain",
     "Question", "Reference Answer", "Constitutional Articles",
     "Supporting Passage(s)", "Annotated By", "Annotation Date",
     "--- EXPERT REVIEW ---"]
    + [name for name in REVIEWER_NAMES
       for _ in range(COLS_PER_REVIEWER)]
)

CSV_HEADER_ROW2 = (
    ["", "", "", "", "", "", "", "", "", "", ""]
    + ["Correctness (1=Poor 5=Excellent)",
       "Completeness (1=Poor 5=Excellent)",
       "Clarity (1=Poor 5=Excellent)",
       "Expert Comments", "Review Date"] * len(REVIEWER_NAMES)
)


def _citations_text(item: dict) -> tuple[str, str]:
    """Return (constitutional_articles_str, supporting_passages_str) for a QAC item."""
    citations = item["gold_answers"][0].get("citations", [])
    articles = ", ".join(
        f"Article {c['section_number']}" for c in citations
        if c.get("section_number")
    )
    passages = " || ".join(
        c.get("passage_text", "")[:150] for c in citations
        if c.get("passage_text")
    )
    return articles, passages


def build_comprehensive_csv(
    qac_json_path: Path,
    abs_json_path: Path,
    csv_data: dict,       # pre-parsed rows from source CSV keyed by item_id
    output_path: Path,
    dry_run: bool,
):
    with open(qac_json_path) as fh:
        qac_data = json.load(fh)
    with open(abs_json_path) as fh:
        abs_data = json.load(fh)

    rows = [CSV_HEADER_ROW1, CSV_HEADER_ROW2]

    # ----- QAC items -----
    for item in qac_data["items"]:
        item_id = item["item_id"]
        arts, passages = _citations_text(item)
        annotated_by = item["gold_answers"][0].get("annotator_id", "") or "LEGAL_EXPERT_1"

        base = [
            item_id,
            item.get("query_type", ""),
            item.get("difficulty", ""),
            item.get("legal_domain", ""),
            item.get("question", ""),
            item["gold_answers"][0].get("answer_text", ""),
            arts,
            passages,
            annotated_by,
            "",   # annotation date not stored in JSON
        ]

        if item_id in csv_data:
            # Pre-fill from source CSV — copy original rating cells verbatim
            src_row = csv_data[item_id]["row"]
            # col 10 is the separator; cols 11..end are rating cells
            rating_cells = src_row[10:] if len(src_row) > 10 else []
            # Pad to expected length (1 separator + 5 reviewers × 5 cols)
            expected_len = 1 + len(REVIEWER_NAMES) * COLS_PER_REVIEWER
            while len(rating_cells) < expected_len:
                rating_cells.append("")
            rows.append(base + rating_cells)
        else:
            # Blank rating columns
            blank_ratings = [""] * (1 + len(REVIEWER_NAMES) * COLS_PER_REVIEWER)
            rows.append(base + blank_ratings)

    # ----- ABS items -----
    for item in abs_data["items"]:
        item_id = item["item_id"]
        # For ABS items, get reference answer from source CSV if available
        ref_answer = ""
        if item_id in csv_data:
            src_row = csv_data[item_id]["row"]
            ref_answer = src_row[5] if len(src_row) > 5 else ""

        base = [
            item_id,
            item.get("query_type", ""),
            "N/A (OOC)",
            item.get("domain", "").replace("_", " ").title(),
            item.get("question", ""),
            ref_answer,
            "N/A",
            "N/A",
            "ABSTENTION_EVAL",
            "",
        ]

        if item_id in csv_data:
            src_row = csv_data[item_id]["row"]
            rating_cells = src_row[10:] if len(src_row) > 10 else []
            expected_len = 1 + len(REVIEWER_NAMES) * COLS_PER_REVIEWER
            while len(rating_cells) < expected_len:
                rating_cells.append("")
            rows.append(base + rating_cells)
        else:
            blank_ratings = [""] * (1 + len(REVIEWER_NAMES) * COLS_PER_REVIEWER)
            rows.append(base + blank_ratings)

    pre_populated = sum(1 for r in rows[2:] if r[0] in csv_data)
    print(f"Comprehensive CSV: {len(rows)-2} items, {pre_populated} pre-populated, "
          f"{len(rows)-2-pre_populated} blank")

    if not dry_run:
        with open(output_path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerows(rows)
        print(f"Written: {output_path}")
    else:
        print(f"[dry-run] {output_path} NOT written")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and report without writing any files")
    args = parser.parse_args()

    print(f"Source CSV : {SOURCE_CSV}")
    print(f"QAC JSON   : {QAC_JSON}")
    print(f"ABS JSON   : {ABS_JSON}")
    print(f"Output CSV : {OUTPUT_CSV}")
    print()

    csv_data = parse_source_csv(SOURCE_CSV)
    print(f"Parsed {len(csv_data)} items from source CSV: "
          f"{sorted(csv_data.keys())[:3]}...{sorted(csv_data.keys())[-3:]}")
    print()

    update_qac_json(QAC_JSON, csv_data, dry_run=args.dry_run)
    print()

    build_comprehensive_csv(QAC_JSON, ABS_JSON, csv_data, OUTPUT_CSV, dry_run=args.dry_run)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
