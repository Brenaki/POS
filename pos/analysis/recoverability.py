"""What predicts recoverability — the share of the Oracle gap DS converts.

ADR 0019, from the external review. `DF/e^2` predicts how *large* the gap is
(rho = -0.86) but not how much of it dynamic selection reaches. That second
question is the one a practitioner actually has, and the qualitative reading so
far — non-linear boundary + low dimension + globally weak base learner => more
to gain — is testable.

Features are dataset-level complexity measures plus pool-level statistics;
the target is the mean `recovered` per (dataset, mode). Evaluation is
leave-one-dataset-out, because with 29 datasets any in-sample R^2 is a
description of the sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pos.complexity.fast_adapter import complexity_data3
from pos.oracle.arff_loader import list_datasets, load_arff_dataset
from pos.pool.complexity_voter import measure_names

GROUPS = ["overlapping", "neighborhood"]
POOL_FEATURES = ["mean_individual_acc", "df_ratio_exact", "oracle_1", "majority_vote"]
META_FEATURES = ["n_classes", "n_features", "n_samples", "imbalance_ratio"]


def dataset_complexity_features(dataset_dir: Path | str | None = None) -> pd.DataFrame:
    """The eleven complexity measures of each dataset, computed once."""
    names = [n.split(".")[1] for n in measure_names(GROUPS)]
    rows = []
    for path in list_datasets(dataset_dir):
        X, y = load_arff_dataset(path)
        values = complexity_data3(X, y, GROUPS)
        rows.append({"dataset": path.stem,
                     **dict(zip(names, values, strict=True))})
    return pd.DataFrame(rows)


def build_feature_frame(df: pd.DataFrame, complexity: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, mode): features + the `recovered` target."""
    cols = [c for c in POOL_FEATURES + META_FEATURES if c in df.columns]
    agg = (df.dropna(subset=["recovered"])
             .groupby(["dataset", "mode"], as_index=False)[[*cols, "recovered"]]
             .mean())
    return agg.merge(complexity, on="dataset", how="left").dropna()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    drop = {"dataset", "mode", "recovered"}
    return [c for c in frame.columns if c not in drop]


def lodo_regression(frame: pd.DataFrame, model=None) -> dict:
    """Leave-one-dataset-out predictions of `recovered`. Returns scores."""
    from scipy.stats import spearmanr
    from sklearn.base import clone

    feats = feature_columns(frame)
    # `model or default` would call len() on an unfitted ensemble and raise;
    # only None means "use the default".
    if model is None:
        model = ridge_model()
    y_true, y_pred = [], []
    for held in sorted(frame["dataset"].unique()):
        train = frame[frame["dataset"] != held]
        test = frame[frame["dataset"] == held]
        est = clone(model)
        est.fit(train[feats], train["recovered"])
        y_true.extend(test["recovered"].tolist())
        y_pred.extend(np.asarray(est.predict(test[feats])).tolist())
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    rho, p = spearmanr(y_true, y_pred)
    return {
        "n": len(y_true),
        "r2_lodo": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "spearman": float(rho), "p_value": float(p),
        "mae": float(np.abs(y_true - y_pred).mean()),
        "baseline_mae": float(np.abs(y_true - y_true.mean()).mean()),
    }


def ridge_model():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))


def forest_model(random_state: int = 42):
    return RandomForestRegressor(n_estimators=300, random_state=random_state,
                                 min_samples_leaf=2)
