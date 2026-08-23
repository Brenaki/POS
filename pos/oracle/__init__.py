"""Oracle_N package: generalized Oracle upper bound for MCS.

Public API:
    build_correctness_matrix(pool, X, y) → (n, M) matrix of 0/1
    oracle_n_accuracy(matrix, n) → float
    oracle_curve(matrix) → {N: acc}
    oracle_curve_array(matrix) → [acc_1, ..., acc_M]
    majority_vote_accuracy(pool, X, y) → float
    mean_probs_accuracy(pool, X, y) → float
    evaluate_des(pool, X_dsel, y_dsel, X, y) → (accuracies, notes)
    load_arff_dataset(path) → (X, y)
    list_datasets() → [Path]
    run_experiment(dataset_path, ...) → results dict
    evaluate_pool(pool, X, y) → dict (individual + ensemble metrics)
    build_rf_pool(X_train, y_train, M, random_state) → list[DecisionTree]
    build_catalog(dataset_dir) → pd.DataFrame
"""

from pos.oracle.arff_loader import list_datasets, load_arff_dataset
from pos.oracle.comparison import (
    majority_vote_accuracy,
    mean_decision_accuracy,
    mean_probs_accuracy,
    soft_fusion_accuracy,
)
from pos.oracle.correctness_matrix import build_correctness_matrix
from pos.oracle.dataset_catalog import build_catalog, load_catalog, save_catalog
from pos.oracle.des_comparison import best_des, des_columns, evaluate_des
from pos.oracle.des_methods import DES_METHODS, PRIMARY_METHODS
from pos.oracle.experiment import run_experiment
from pos.oracle.metrics import prediction_metrics
from pos.oracle.oracle_curve import oracle_curve, oracle_curve_array
from pos.oracle.oracle_n import oracle_n_accuracy, oracle_n_vector
from pos.oracle.pool_evaluation import evaluate_pool
from pos.oracle.random_forest_pool import build_rf_pool
from pos.oracle.run_recorder import record_run

__all__ = [
    "build_correctness_matrix",
    "oracle_n_accuracy",
    "oracle_n_vector",
    "oracle_curve",
    "oracle_curve_array",
    "majority_vote_accuracy",
    "mean_probs_accuracy",
    "mean_decision_accuracy",
    "soft_fusion_accuracy",
    "evaluate_des",
    "best_des",
    "DES_METHODS",
    "PRIMARY_METHODS",
    "des_columns",
    "load_arff_dataset",
    "list_datasets",
    "run_experiment",
    "evaluate_pool",
    "prediction_metrics",
    "build_rf_pool",
    "build_catalog",
    "save_catalog",
    "load_catalog",
    "record_run",
]
