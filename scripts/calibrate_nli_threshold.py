"""
Empirical calibration of the NLI entailment threshold for legal faithfulness checking.

Creates known correct and incorrect claim-evidence pairs drawn from the Constitution of
Sri Lanka and computes NLI scores via roberta-large-mnli. Then finds the threshold that
maximises F1 for distinguishing entailed (faithful) from non-entailed (hallucinated) claims.

Usage:
    python scripts/calibrate_nli_threshold.py [--output results/nli_calibration.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Ground-truth calibration pairs
# Each pair is (evidence, claim, label)
#   label=1 → entailed   (faithful claim, should score high)
#   label=0 → not entailed (hallucination/wrong, should score low)
#
# Evidence texts are verbatim or near-verbatim from the Constitution of Sri Lanka.
# ---------------------------------------------------------------------------

CALIBRATION_PAIRS = [
    # --- Article 9: Buddhism ---
    (
        "The Republic of Sri Lanka shall give to Buddhism the foremost place and accordingly "
        "it shall be the duty of the State to protect and foster the Buddha Sasana, while "
        "assuring to all religions the rights granted by Articles 10 and 14(1)(e).",
        "Buddhism has the foremost place in Sri Lanka.",
        1  # correct
    ),
    (
        "The Republic of Sri Lanka shall give to Buddhism the foremost place and accordingly "
        "it shall be the duty of the State to protect and foster the Buddha Sasana, while "
        "assuring to all religions the rights granted by Articles 10 and 14(1)(e).",
        "The State is obliged to protect and foster Buddhism.",
        1  # correct paraphrase
    ),
    (
        "The Republic of Sri Lanka shall give to Buddhism the foremost place and accordingly "
        "it shall be the duty of the State to protect and foster the Buddha Sasana, while "
        "assuring to all religions the rights granted by Articles 10 and 14(1)(e).",
        "Christianity is the official religion of Sri Lanka.",
        0  # hallucination
    ),
    (
        "The Republic of Sri Lanka shall give to Buddhism the foremost place and accordingly "
        "it shall be the duty of the State to protect and foster the Buddha Sasana, while "
        "assuring to all religions the rights granted by Articles 10 and 14(1)(e).",
        "All religions have equal status under the Constitution.",
        0  # incorrect — Buddhism has foremost place, not equality
    ),

    # --- Article 10: Freedom of thought ---
    (
        "Every person is entitled to freedom of thought, conscience and religion, "
        "including the freedom to have or to adopt a religion or belief of his choice.",
        "Every person has freedom of thought, conscience and religion.",
        1  # correct
    ),
    (
        "Every person is entitled to freedom of thought, conscience and religion, "
        "including the freedom to have or to adopt a religion or belief of his choice.",
        "People are free to adopt any religion of their choice.",
        1  # correct paraphrase
    ),
    (
        "Every person is entitled to freedom of thought, conscience and religion, "
        "including the freedom to have or to adopt a religion or belief of his choice.",
        "Freedom of thought can be restricted for national security.",
        0  # hallucination — Art 10 has no restriction clause
    ),
    (
        "Every person is entitled to freedom of thought, conscience and religion, "
        "including the freedom to have or to adopt a religion or belief of his choice.",
        "Only citizens are entitled to freedom of religion.",
        0  # incorrect — applies to every 'person', not just citizens
    ),

    # --- Article 18/19: Official languages ---
    (
        "The Official Language of Sri Lanka shall be Sinhala. Tamil shall also be an "
        "official language.",
        "Sinhala and Tamil are both official languages of Sri Lanka.",
        1  # correct
    ),
    (
        "The Official Language of Sri Lanka shall be Sinhala. Tamil shall also be an "
        "official language.",
        "Sinhala is an official language of Sri Lanka.",
        1  # correct (subset of truth)
    ),
    (
        "The Official Language of Sri Lanka shall be Sinhala. Tamil shall also be an "
        "official language.",
        "English is the official language of Sri Lanka.",
        0  # hallucination
    ),
    (
        "The Official Language of Sri Lanka shall be Sinhala. Tamil shall also be an "
        "official language.",
        "Only Sinhala is recognised as an official language.",
        0  # incorrect — Tamil is also official
    ),

    # --- Article 30: Term of President ---
    (
        "There shall be a President of the Republic of Sri Lanka, who is the Head of "
        "the State, the Head of the Executive and of the Government, and the Commander-"
        "in-Chief of the Armed Forces. The President shall be elected by the People and "
        "shall hold office for a term of six years.",
        "The President holds office for a term of six years.",
        1  # correct
    ),
    (
        "There shall be a President of the Republic of Sri Lanka, who is the Head of "
        "the State, the Head of the Executive and of the Government, and the Commander-"
        "in-Chief of the Armed Forces. The President shall be elected by the People and "
        "shall hold office for a term of six years.",
        "The President is elected directly by the people.",
        1  # correct
    ),
    (
        "There shall be a President of the Republic of Sri Lanka, who is the Head of "
        "the State, the Head of the Executive and of the Government, and the Commander-"
        "in-Chief of the Armed Forces. The President shall be elected by the People and "
        "shall hold office for a term of six years.",
        "The President holds office for a term of five years.",
        0  # incorrect — it is six years
    ),
    (
        "There shall be a President of the Republic of Sri Lanka, who is the Head of "
        "the State, the Head of the Executive and of the Government, and the Commander-"
        "in-Chief of the Armed Forces. The President shall be elected by the People and "
        "shall hold office for a term of six years.",
        "Parliament elects the President.",
        0  # incorrect — the people elect the President
    ),

    # --- Article 82: Constitutional amendment ---
    (
        "A Bill for the amendment of any provision of the Constitution shall, on the "
        "certificate of the President, be passed by the special majority required by "
        "Parliament, that is to say, by not less than two-thirds of the whole number "
        "of Members of Parliament (including those not present) voting in its favour.",
        "Amending the Constitution requires a two-thirds majority of all Members of Parliament.",
        1  # correct
    ),
    (
        "A Bill for the amendment of any provision of the Constitution shall, on the "
        "certificate of the President, be passed by the special majority required by "
        "Parliament, that is to say, by not less than two-thirds of the whole number "
        "of Members of Parliament (including those not present) voting in its favour.",
        "A constitutional amendment requires a simple majority in Parliament.",
        0  # incorrect — requires two-thirds
    ),

    # --- Article 70: Dissolution of Parliament ---
    (
        "The President may, by Proclamation, dissolve Parliament after the expiration "
        "of one year from the date of its first meeting after a General Election.",
        "The President can dissolve Parliament by Proclamation after it has met for at least one year.",
        1  # correct
    ),
    (
        "The President may, by Proclamation, dissolve Parliament after the expiration "
        "of one year from the date of its first meeting after a General Election.",
        "Parliament can be dissolved by the Prime Minister.",
        0  # incorrect — the President dissolves Parliament
    ),

    # --- Article 148: Consolidated Fund ---
    (
        "All revenues, loans raised by the Government of Sri Lanka and all other moneys "
        "received by the Government of Sri Lanka shall, subject to the provisions of this "
        "Constitution, form one Consolidated Fund.",
        "All government revenues and loans form the Consolidated Fund.",
        1  # correct
    ),
    (
        "All revenues, loans raised by the Government of Sri Lanka and all other moneys "
        "received by the Government of Sri Lanka shall, subject to the provisions of this "
        "Constitution, form one Consolidated Fund.",
        "The Consolidated Fund is managed by the Central Bank.",
        0  # hallucination — not stated in this passage
    ),
]


def compute_nli_scores(pairs, model_name="roberta-large-mnli", device=None):
    """Compute NLI entailment scores for all pairs."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading NLI model: {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    scores = []
    labels = []

    for i, (evidence, claim, label) in enumerate(pairs, 1):
        inputs = tokenizer(
            evidence, claim,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            # roberta-large-mnli: 0=CONTRADICTION, 1=NEUTRAL, 2=ENTAILMENT
            entailment_prob = probs[0][2].item()

        scores.append(entailment_prob)
        labels.append(label)

        verdict = "ENTAIL" if label == 1 else "NOT"
        logger.info(
            f"[{i:02d}/{len(pairs)}] {verdict}  score={entailment_prob:.4f}  "
            f"claim='{claim[:60]}...'"
        )

    return scores, labels


def find_optimal_threshold(scores, labels, grid_size=100):
    """Find threshold that maximises F1 on the calibration set."""
    thresholds = np.linspace(0, 1, grid_size + 1)
    best_f1 = 0.0
    best_threshold = 0.5
    best_stats = {}

    for t in thresholds:
        preds = [1 if s >= t else 0 for s in scores]
        tp = sum(1 for p, g in zip(preds, labels) if p == 1 and g == 1)
        fp = sum(1 for p, g in zip(preds, labels) if p == 1 and g == 0)
        fn = sum(1 for p, g in zip(preds, labels) if p == 0 and g == 1)
        tn = sum(1 for p, g in zip(preds, labels) if p == 0 and g == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        acc = (tp + tn) / len(labels)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)
            best_stats = {
                "threshold": float(t),
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": acc,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn
            }

    return best_threshold, best_stats


def print_results(scores, labels, best_threshold, best_stats):
    """Print calibration results."""
    correct_scores = [s for s, l in zip(scores, labels) if l == 1]
    incorrect_scores = [s for s, l in zip(scores, labels) if l == 0]

    print("\n=== NLI THRESHOLD CALIBRATION RESULTS ===")
    print(f"\nCalibration set: {len(labels)} pairs "
          f"({sum(labels)} entailed, {len(labels)-sum(labels)} not entailed)")
    print(f"\nEntailed claims NLI scores:")
    print(f"  mean={np.mean(correct_scores):.4f}  "
          f"min={np.min(correct_scores):.4f}  "
          f"max={np.max(correct_scores):.4f}  "
          f"median={np.median(correct_scores):.4f}")
    print(f"\nNon-entailed claims NLI scores:")
    print(f"  mean={np.mean(incorrect_scores):.4f}  "
          f"min={np.min(incorrect_scores):.4f}  "
          f"max={np.max(incorrect_scores):.4f}  "
          f"median={np.median(incorrect_scores):.4f}")

    print(f"\n--- Optimal threshold: {best_threshold:.4f} ---")
    print(f"  F1={best_stats['f1']:.4f}  "
          f"Precision={best_stats['precision']:.4f}  "
          f"Recall={best_stats['recall']:.4f}  "
          f"Accuracy={best_stats['accuracy']:.4f}")
    print(f"  TP={best_stats['tp']}  FP={best_stats['fp']}  "
          f"FN={best_stats['fn']}  TN={best_stats['tn']}")

    print(f"\nCurrent production threshold (0.15):")
    preds_015 = [1 if s >= 0.15 else 0 for s in scores]
    tp = sum(1 for p, g in zip(preds_015, labels) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(preds_015, labels) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(preds_015, labels) if p == 0 and g == 1)
    tn = sum(1 for p, g in zip(preds_015, labels) if p == 0 and g == 0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"  F1={f1:.4f}  Precision={prec:.4f}  Recall={rec:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    if best_threshold > 0.15:
        print(f"\nRECOMMENDATION: Raise threshold from 0.15 to {best_threshold:.2f}")
        print(f"  This reduces false positives (hallucinations accepted) "
              f"at cost of more false negatives (faithful claims rejected).")
    else:
        print(f"\nRECOMMENDATION: Current threshold 0.15 is at or above optimal; keep it.")


def main():
    parser = argparse.ArgumentParser(description="Calibrate NLI faithfulness threshold")
    parser.add_argument('--output', type=Path, default=Path('results/nli_calibration.json'))
    parser.add_argument('--model', default='roberta-large-mnli')
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    scores, labels = compute_nli_scores(CALIBRATION_PAIRS, model_name=args.model)
    best_threshold, best_stats = find_optimal_threshold(scores, labels)
    print_results(scores, labels, best_threshold, best_stats)

    result = {
        "model": args.model,
        "n_pairs": len(CALIBRATION_PAIRS),
        "n_entailed": sum(labels),
        "n_not_entailed": len(labels) - sum(labels),
        "scores": scores,
        "labels": labels,
        "optimal_threshold": best_threshold,
        "optimal_stats": best_stats,
        "current_threshold": 0.15,
        "pairs": [
            {"evidence": e[:100], "claim": c, "label": l, "score": s}
            for (e, c, l), s in zip(CALIBRATION_PAIRS, scores)
        ]
    }

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Calibration results saved to {args.output}")

    return best_threshold


if __name__ == '__main__':
    main()
