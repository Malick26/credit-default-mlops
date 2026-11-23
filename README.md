# credit-default-mlops

Predicts whether a credit card client will default on next month's payment, and runs the
model in production with CI/CD and continuous training.

## Branch flow

- `main`: production. Every merge deploys to the online service.
- `staging`: release candidate, validated in a staging container.
- `dev`: integration branch.
- `feature/*`: short-lived branches created from `dev`.

Changes go `feature/*` -> `dev` -> `staging` -> `main`, always through a reviewed pull
request. The three long-lived branches are protected: no direct push, 1 approval and the
CI checks are required to merge.

## Workflow triggers

TODO (filled in with the CI/CD and CT workflows)

## Data

See [ml/README_data.md](ml/README_data.md).
