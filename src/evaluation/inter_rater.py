"""
Inter-rater reliability metrics for Q-A-C dataset annotation (RO4).

Implements Cohen's kappa (unweighted and quadratic-weighted) as required by
the methodology (Section 3.4.4): targeting κ ≥ 0.75 across all annotation
dimensions (correctness, completeness, clarity, citation accuracy).

Usage:
    from src.evaluation.inter_rater import cohen_kappa, weighted_cohen_kappa,
                                           compute_all_kappas

    # Correctness ratings from two annotators (0–5 scale)
    rater1 = [4, 3, 5, 2, 4, 3, 5, 4]
    rater2 = [4, 4, 5, 2, 3, 3, 4, 4]

    kappa = cohen_kappa(rater1, rater2)
    w_kappa = weighted_cohen_kappa(rater1, rater2)
"""

from typing import List, Dict, Optional
import numpy as np
from src.utils import get_logger

logger = get_logger(__name__)

# Minimum acceptable κ per methodology (Section 3.4.4)
KAPPA_TARGET = 0.75


def cohen_kappa(
    rater1: List[int],
    rater2: List[int],
    labels: Optional[List[int]] = None
) -> float:
    """
    Compute unweighted Cohen's kappa between two raters.

    κ = (Po - Pe) / (1 - Pe)
    where Po = observed agreement, Pe = expected agreement by chance.

    Args:
        rater1: Ratings from first annotator
        rater2: Ratings from second annotator
        labels: Ordered list of all possible rating values.
                Inferred from data if None.

    Returns:
        Cohen's kappa coefficient (-1 to 1).
        κ < 0   : worse than chance
        0 ≤ κ < 0.2  : slight agreement
        0.2 ≤ κ < 0.4: fair agreement
        0.4 ≤ κ < 0.6: moderate agreement
        0.6 ≤ κ < 0.8: substantial agreement
        0.8 ≤ κ ≤ 1.0: almost perfect agreement
    """
    if len(rater1) != len(rater2):
        raise ValueError("Rater lists must have equal length")
    if len(rater1) == 0:
        raise ValueError("Rater lists must not be empty")

    if labels is None:
        labels = sorted(set(rater1) | set(rater2))

    n = len(rater1)
    label_index = {label: i for i, label in enumerate(labels)}
    k = len(labels)

    # Build confusion matrix
    cm = np.zeros((k, k), dtype=float)
    for r1, r2 in zip(rater1, rater2):
        cm[label_index[r1], label_index[r2]] += 1

    # Observed agreement (Po)
    po = np.trace(cm) / n

    # Expected agreement (Pe)
    row_marginals = cm.sum(axis=1) / n
    col_marginals = cm.sum(axis=0) / n
    pe = np.dot(row_marginals, col_marginals)

    if pe == 1.0:
        return 1.0  # Perfect marginal distribution — kappa undefined, return 1

    kappa = (po - pe) / (1 - pe)
    return float(kappa)


def weighted_cohen_kappa(
    rater1: List[int],
    rater2: List[int],
    labels: Optional[List[int]] = None,
    weight_scheme: str = 'quadratic'
) -> float:
    """
    Compute weighted Cohen's kappa (quadratic or linear weights).

    Weighted kappa penalises disagreements proportionally to their distance,
    which is appropriate for ordinal annotation scales (0–5 rubric).

    Args:
        rater1: Ratings from first annotator
        rater2: Ratings from second annotator
        labels: Ordered list of all possible rating values.
        weight_scheme: 'quadratic' (default, standard for ordinal) or 'linear'

    Returns:
        Weighted Cohen's kappa coefficient.
    """
    if len(rater1) != len(rater2):
        raise ValueError("Rater lists must have equal length")
    if len(rater1) == 0:
        raise ValueError("Rater lists must not be empty")

    if labels is None:
        labels = sorted(set(rater1) | set(rater2))

    n = len(rater1)
    label_index = {label: i for i, label in enumerate(labels)}
    k = len(labels)

    # Weight matrix
    weights = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            if weight_scheme == 'quadratic':
                weights[i, j] = (i - j) ** 2 / (k - 1) ** 2
            else:  # linear
                weights[i, j] = abs(i - j) / (k - 1)

    # Confusion matrix
    cm = np.zeros((k, k), dtype=float)
    for r1, r2 in zip(rater1, rater2):
        cm[label_index[r1], label_index[r2]] += 1

    # Expected matrix (outer product of marginals)
    row_marginals = cm.sum(axis=1)
    col_marginals = cm.sum(axis=0)
    expected = np.outer(row_marginals, col_marginals) / n

    # Standard weighted kappa: κ_w = 1 - (Σ w_ij * O_ij) / (Σ w_ij * E_ij)
    numerator = np.sum(weights * cm)
    denominator = np.sum(weights * expected)
    if denominator == 0:
        return 1.0

    kappa = 1 - (numerator / denominator)
    return float(kappa)


def interpret_kappa(kappa: float) -> str:
    """Return Landis & Koch (1977) interpretation of kappa value."""
    if kappa < 0:
        return "worse than chance"
    elif kappa < 0.20:
        return "slight"
    elif kappa < 0.40:
        return "fair"
    elif kappa < 0.60:
        return "moderate"
    elif kappa < 0.80:
        return "substantial"
    else:
        return "almost perfect"


def compute_all_kappas(
    annotation_data: List[Dict],
    dimensions: List[str] = None
) -> Dict:
    """
    Compute Cohen's kappa for all annotation dimensions across the dataset.

    Args:
        annotation_data: List of dicts, each with keys like
            'rater1_correctness', 'rater2_correctness',
            'rater1_completeness', 'rater2_completeness', etc.
        dimensions: Dimensions to compute kappa for.
            Defaults to ['correctness', 'completeness', 'clarity'].

    Returns:
        Dictionary with kappa values per dimension and an overall summary.
    """
    if dimensions is None:
        dimensions = ['correctness', 'completeness', 'clarity']

    results = {}

    for dim in dimensions:
        r1_key = f'rater1_{dim}'
        r2_key = f'rater2_{dim}'

        # Filter items where both raters provided ratings
        pairs = [
            (item[r1_key], item[r2_key])
            for item in annotation_data
            if r1_key in item and r2_key in item
            and item[r1_key] is not None and item[r2_key] is not None
        ]

        if not pairs:
            logger.warning(f"No annotation pairs found for dimension: {dim}")
            results[dim] = {
                'n_pairs': 0,
                'kappa': None,
                'weighted_kappa': None,
                'interpretation': 'insufficient data',
                'meets_target': False
            }
            continue

        r1_ratings = [p[0] for p in pairs]
        r2_ratings = [p[1] for p in pairs]

        try:
            kappa = cohen_kappa(r1_ratings, r2_ratings)
            w_kappa = weighted_cohen_kappa(r1_ratings, r2_ratings)
            interpretation = interpret_kappa(w_kappa)
            meets_target = w_kappa >= KAPPA_TARGET

            results[dim] = {
                'n_pairs': len(pairs),
                'kappa': round(kappa, 4),
                'weighted_kappa': round(w_kappa, 4),
                'interpretation': interpretation,
                'meets_target': meets_target
            }

            logger.info(
                f"[{dim}] κ={kappa:.4f}  weighted-κ={w_kappa:.4f}  "
                f"({interpretation})  {'✓' if meets_target else '✗'} "
                f"target={KAPPA_TARGET}"
            )

        except Exception as e:
            logger.error(f"Error computing kappa for {dim}: {e}")
            results[dim] = {
                'n_pairs': len(pairs),
                'kappa': None,
                'weighted_kappa': None,
                'interpretation': f'error: {e}',
                'meets_target': False
            }

    # Overall summary
    valid = [r for r in results.values() if r['weighted_kappa'] is not None]
    results['summary'] = {
        'dimensions_assessed': len(dimensions),
        'dimensions_meeting_target': sum(1 for r in valid if r['meets_target']),
        'overall_weighted_kappa_mean': (
            round(np.mean([r['weighted_kappa'] for r in valid]), 4)
            if valid else None
        ),
        'target_kappa': KAPPA_TARGET,
        'ready_for_main_annotation': all(r['meets_target'] for r in valid)
    }

    return results
