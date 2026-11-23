"""The data contract stops a bad delivery before any production change."""
import pytest

from ml.data import TARGET, validate
from tests.helpers import synthetic_data


def _split():
    df = synthetic_data(rows=200)
    df["source_row"] = range(len(df))
    return df.iloc[:150].copy(), df.iloc[150:].copy()


def test_valid_delivery_passes():
    validate(*_split())


@pytest.mark.parametrize("break_it, message", [
    (lambda tr, ev: tr.drop(columns=["age"]), "missing columns"),
    (lambda tr, ev: tr.assign(age=None), "missing values"),
    (lambda tr, ev: tr.assign(**{TARGET: 2}), "must be 0 or 1"),
    (lambda tr, ev: tr.assign(**{TARGET: 0}), "single class"),
    (lambda tr, ev: tr.assign(age=12), "out-of-range"),
    (lambda tr, ev: tr.iloc[0:0], "no rows"),
])
def test_bad_training_file_is_rejected(break_it, message):
    train, evaluation = _split()
    with pytest.raises(ValueError, match=message):
        validate(break_it(train, evaluation), evaluation)


def test_overlap_between_train_and_evaluation_is_rejected():
    train, evaluation = _split()
    evaluation.iloc[0, evaluation.columns.get_loc("source_row")] = train["source_row"].iloc[0]
    with pytest.raises(ValueError, match="both train and evaluation"):
        validate(train, evaluation)
