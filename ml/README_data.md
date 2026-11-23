# Data

## Source and licence

- Dataset: **Default of Credit Card Clients** (I-Cheng Yeh, 2009)
- Provider: UCI Machine Learning Repository, dataset id 350
- URL: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Licence: CC BY 4.0
- Snapshot: `default+of+credit+card+clients.zip` from the UCI static URL
  (https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip)

30,000 credit card clients in Taiwan (April to September 2005), 23 features
(credit limit, demographics, repayment status, bill and payment amounts over 6 months)
and a binary label `default` (default payment the next month).

## Versions

The source has no date per client, so new deliveries are simulated with fixed,
reproducible batches (`ml/prepare_data.py`, seed 42).

| Version | Training rows | What changed |
|---|---|---|
| v1 | 3,000 | First usable delivery, used for the baseline model |
| v2 | 21,000 | +18,000 newly labelled clients (large delivery, should trigger retraining) |
| v3 | 24,000 | +3,000 newly labelled clients (small delivery) |

## Rebuild the versions

```
curl -L -o data/raw/credit.zip "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
unzip -o data/raw/credit.zip -d data/raw
python ml/prepare_data.py --raw "data/raw/default of credit card clients.xls"
```

`python ml/prepare_data.py` without `--raw` downloads the file itself, but it can fail
with a certificate error on some Python installs; `curl` then works.

To get the published versions instead: `dvc pull data/v1.dvc` (or v2, v3). The DVC remote
is the `credit-default-mlops/` folder of the course Azure Blob container; the SAS token is
set locally with `dvc remote modify --local azure sas_token ...` and in CI from the
`AZURE_SAS_TOKEN` secret.

## Split design

- 20% of the clients (6,000) are held out once, stratified on the label, and saved as
  `evaluation.csv` in every version.
- These rows are never used for fitting or tuning, so the champion and the candidate are
  always compared on the same observations.
- `prepare_data.py` checks that no `source_row` appears in both `train.csv` and
  `evaluation.csv`.
- The source has no time order, so the split is random (no chronology to preserve).

## Identifiers

| Version | Git commit / tag | DVC hash (`data/<version>.dvc`) |
|---|---|---|
| v1 | TODO (tag after merge) | `0587b91afa18a8efca52f6a0df90062e.dir` |
| v2 | TODO (tag after merge) | `0d69b04d4b5d20a6d55b092137cf0213.dir` |
| v3 | TODO (tag after merge) | `e45020d121acb59cf797087f63c41563.dir` |
