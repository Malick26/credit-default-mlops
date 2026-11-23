"""Quality gates for continuous training. The metric is ROC-AUC: higher is better.

The thresholds are fixed before any evaluation and recorded in ml/README_experiments.md.
"""
MIN_ROC_AUC = 0.70          # quality floor: no model below this goes to production
MIN_IMPROVEMENT = 0.015     # a candidate must beat the champion by at least this much
MAX_DROP = 0.02             # retrain when the champion loses this much from its approval score


def retrain_reason(champion_auc: float, approved_auc: float, champion_version_data: str,
                   dataset_version: str) -> str | None:
    """Why the champion should be retrained on this delivery, or None to keep it as is."""
    if champion_auc < MIN_ROC_AUC:
        return f"champion ROC-AUC {champion_auc:.4f} is below the floor {MIN_ROC_AUC}"
    if approved_auc - champion_auc >= MAX_DROP:
        return f"champion ROC-AUC dropped by {approved_auc - champion_auc:.4f} (limit {MAX_DROP})"
    if dataset_version != champion_version_data:
        return f"new validated dataset version {dataset_version} (champion trained on {champion_version_data})"
    return None


def promotion_reason(candidate_auc: float, champion_auc: float | None) -> tuple[bool, str]:
    """(promote?, reason). With no champion, only the quality floor applies (bootstrap)."""
    if candidate_auc < MIN_ROC_AUC:
        return False, f"candidate ROC-AUC {candidate_auc:.4f} is below the floor {MIN_ROC_AUC}"
    if champion_auc is None:
        return True, f"first model, ROC-AUC {candidate_auc:.4f} reaches the floor {MIN_ROC_AUC}"
    gain = candidate_auc - champion_auc
    if gain < MIN_IMPROVEMENT:
        return False, f"improvement {gain:+.4f} is below the minimum {MIN_IMPROVEMENT}"
    return True, f"improvement {gain:+.4f} reaches the minimum {MIN_IMPROVEMENT}"
