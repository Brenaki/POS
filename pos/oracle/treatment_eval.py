"""Evaluate one fitted pool under the ADR 0020 v2 metric set.

Both treatments are Decision Tree pools, so soft combination is a genuine
`predict_proba` average (`mean_probability`) — the ADR 0016 normalised-margin
fallback, which only existed for the Perceptron, is not used here.

Two readings of Oracle_N are recorded side by side (ADR 0020 §5):

- `oracle_curve` is the **project's** Oracle_N — the fraction of samples on
  which at least N of the M classifiers are correct (README, subproject
  document, schema v1). Order-free and decreasing; its first point is the
  classic pool oracle, the only point comparable to Table 7.
- `oracle_greedy_curve` is the greedy nested construction: at each step, the
  classifier that adds the most newly covered samples. Increasing, and its
  last point is that same classic oracle. It is a POS contribution that
  depends on the greedy ordering, so no published number matches it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pos.diversity import double_fault_matrix
from pos.oracle.correctness_matrix import build_correctness_matrix
from pos.oracle.des_comparison import effective_k, evaluate_des
from pos.oracle.fusion_predict import majority_vote_predict, mean_probs_predict
from pos.oracle.metrics import oracle_balanced_recall, prediction_metrics
from pos.oracle.oracle_curve_greedy import atleast_curve, greedy_oracle_curve


def _mean_pairwise_double_fault(y_test, preds) -> float:
    dfm = double_fault_matrix(y_test, preds)
    m = dfm.shape[0]
    if m < 2:
        return 0.0
    off = dfm[~np.eye(m, dtype=bool)]
    return float(off.mean())


def evaluate_treatment(
    pool, X_test, y_test, X_dsel, y_dsel, des_methods, random_state
) -> dict[str, Any]:
    """Greedy + at-least Oracle curves, MVR, mean probability, DCS/DES."""
    matrix = build_correctness_matrix(pool, X_test, y_test)
    preds = np.array([clf.predict(X_test) for clf in pool])
    mv_metrics = prediction_metrics(y_test, majority_vote_predict(pool, X_test))
    try:
        mp_metrics = prediction_metrics(y_test, mean_probs_predict(pool, X_test))
    except (AttributeError, ValueError):
        mp_metrics = None

    metrics: dict[str, Any] = {
        "n_classifiers": len(pool),
        "oracle_curve": atleast_curve(matrix),
        "oracle_greedy_curve": greedy_oracle_curve(matrix),
        "majority_vote": mv_metrics["acc"],
        "majority_vote_metrics": mv_metrics,
        "mean_probs": mp_metrics["acc"] if mp_metrics else None,
        "mean_probs_metrics": mp_metrics,
        "oracle_1_balanced": oracle_balanced_recall(matrix, y_test, 1),
        "individual_accuracies": [float(matrix[:, j].mean())
                                  for j in range(len(pool))],
        "double_fault_mean": _mean_pairwise_double_fault(y_test, preds),
        "correctness_matrix": matrix,
    }
    if des_methods:
        des, des_preds, des_notes = evaluate_des(
            pool, X_dsel, y_dsel, X_test, y_test,
            random_state=random_state, methods=tuple(des_methods),
        )
        metrics["des"] = des
        metrics["des_preds"] = des_preds
        metrics["des_notes"] = des_notes
        metrics["k_eff"] = effective_k(len(y_dsel))
        metrics["n_dsel"] = int(len(y_dsel))
    return metrics
