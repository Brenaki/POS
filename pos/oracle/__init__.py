"""Oracle_N package: generalized Oracle upper bound for MCS.

Public API:
    build_correctness_matrix(pool, X, y) → (n, M) matrix of 0/1
    oracle_n_accuracy(matrix, n) → float
    oracle_curve(matrix) → {N: acc}
    oracle_curve_array(matrix) → [acc_1, ..., acc_M]
    majority_vote_accuracy(pool, X, y) → float
    mean_probs_accuracy(pool, X, y) → float
    load_arff_dataset(path) → (X, y)
    list_datasets() → [Path]
"""

from pos.oracle.arff_loader import list_datasets, load_arff_dataset
from pos.oracle.comparison import majority_vote_accuracy, mean_probs_accuracy
from pos.oracle.correctness_matrix import build_correctness_matrix
from pos.oracle.experiment import run_experiment
from pos.oracle.oracle_curve import oracle_curve, oracle_curve_array
from pos.oracle.oracle_n import oracle_n_accuracy, oracle_n_vector

__all__ = [
    "build_correctness_matrix",
    "oracle_n_accuracy",
    "oracle_n_vector",
    "oracle_curve",
    "oracle_curve_array",
    "majority_vote_accuracy",
    "mean_probs_accuracy",
    "load_arff_dataset",
    "list_datasets",
    "run_experiment",
]
