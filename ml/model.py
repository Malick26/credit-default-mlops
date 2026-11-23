"""The model and its evaluation. Higher ROC-AUC is better."""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.data import FEATURES, TARGET

DEFAULT_PARAMS = {
    "max_iter": 200,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "random_state": 42,
}


def build_model(params: dict | None = None) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**{**DEFAULT_PARAMS, **(params or {})})


def train(train_df: pd.DataFrame, params: dict | None = None) -> HistGradientBoostingClassifier:
    model = build_model(params)
    model.fit(train_df[FEATURES], train_df[TARGET])
    return model


def evaluate(model, evaluation: pd.DataFrame) -> dict:
    """Score any model with predict_proba (sklearn or MLflow-loaded) on held-out rows."""
    scores = model.predict_proba(evaluation[FEATURES])[:, 1]
    return {
        "roc_auc": float(roc_auc_score(evaluation[TARGET], scores)),
        "average_precision": float(average_precision_score(evaluation[TARGET], scores)),
        "positive_rate": float(evaluation[TARGET].mean()),
    }
