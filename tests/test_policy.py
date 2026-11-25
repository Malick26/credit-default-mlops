"""The promotion and retraining rules, with candidates that pass and fail."""
import pytest

from ml.policy import MAX_DROP, MIN_IMPROVEMENT, MIN_ROC_AUC, promotion_reason, retrain_reason


@pytest.mark.parametrize("candidate, champion, expected", [
    (0.78, 0.75, True),                            # clear improvement
    (0.75 + MIN_IMPROVEMENT, 0.75, True),          # exactly the minimum improvement
    (0.76, 0.75, False),                           # better, but not enough
    (0.74, 0.75, False),                           # worse than the champion
    (0.69, 0.60, False),                           # big gain but below the quality floor
    (MIN_ROC_AUC, None, True),                     # bootstrap at the floor
    (0.65, None, False),                           # bootstrap below the floor
])
def test_promotion_rule(candidate, champion, expected):
    promote, reason = promotion_reason(candidate, champion)
    assert promote is expected
    assert reason


@pytest.mark.parametrize("current, approved, trained_on, delivered, retrain", [
    (0.78, 0.78, "v2", "v2", False),               # same data, same score: keep
    (0.78, 0.78, "v2", "v3", True),                # new validated version
    (0.69, 0.75, "v2", "v2", True),                # below the floor
    (0.78 - MAX_DROP, 0.78, "v2", "v2", True),     # dropped by the limit
    (0.77, 0.78, "v2", "v2", False),               # small drop is tolerated
])
def test_retraining_rule(current, approved, trained_on, delivered, retrain):
    assert (retrain_reason(current, approved, trained_on, delivered) is not None) is retrain
