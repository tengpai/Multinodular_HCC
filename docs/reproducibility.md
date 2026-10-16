# Reproducing the current analysis

Use Python 3.12, repository-root working directory, and approved private data.
Run `python -m MODULE --help` for all arguments. Paths below are placeholders;
write results outside the repository. Data are not bundled or downloaded.

## Habitat selection

```sh
python -m current_analysis.habitat_train_only --raw /private/habitat_raw.csv --labels /private/labels.csv --output /private/habitat_run --stage prepare
python -m current_analysis.habitat_train_only --raw /private/habitat_raw.csv --labels /private/labels.csv --output /private/habitat_run --stage select
```

Both stages must use the same portable code and inputs. Its run signature differs
from the old workstation script; do not mix old preparation checkpoints with new
selection runs. The original input hashes are checked by default. Keep the
complete produced lock manifest and artifacts together.

Five-fold training-only out-of-fold comparison uses seed 20260718, correlation
threshold 0.90, univariable Cox score-test P < .05, at most 25 screened features,
Elastic Net l1_ratio 0.5, and final Cox L2 penalty 0.1. Candidate selection and
nested candidate-selection assessment are distinct outputs. The locked choice is
max-H2 (eight retained features; pooled OOF 0.6244224609541; nested OOF
0.6113551971428). This release did not rerun the expensive raw-feature selection;
it ports its scientific functions and checks the existing locked artifacts.

## Frozen inference or train-only reconstruction

Prefer `locked_inference` to score new inputs without fitting. For exact study
reconstruction and validation of the archived training procedure:

```sh
python -m current_analysis.fusion_locked_core --habitat-dir /private/habitat_run --fusion-input /private/fusion_input.csv --output /private/fusion_run
python -m current_analysis.fusion_statistics --scores /private/fusion_run/fusion_scores_locked.csv --output /private/statistics
python -m current_analysis.complementarity --scores /private/fusion_run/fusion_scores_locked.csv --manifest /private/fusion_run/fusion_core_manifest.json --output /private/complementarity
python -m current_analysis.clinical_benchmarks --input /private/evaluation_input.csv --output /private/benchmarks
python -m current_analysis.probability_diagnostics --scores /private/fusion_run/fusion_scores_locked.csv --output /private/probability
python -m current_analysis.fusion_diagnostics --scores /private/fusion_run/fusion_scores_locked.csv --manifest /private/fusion_run/fusion_core_manifest.json --output /private/diagnostics --bootstrap 1000
```

Fusion construction estimates its scales and Cox coefficients in training only
(L2 0.01); validation patients do not alter these estimates. Single-score
training Cox models supply probability estimates for calibration and IPCW IBS;
these probability models differ from the multivariable fusion construction.
Numerical PH tests report actual rank/KM transform P values without an all-pass
label. Bootstrap internal validation refits only fusion coefficients, retaining
the five existing components and original training scaling. It is not full
pipeline internal validation.

Linear variance-contribution decompositions are descriptive; their percentages
are not an estimate of independent biological information. Correlation/VIF and
paired clinical benchmark comparisons have separate interpretations.

## Reproducibility limits

The original OneKey environment is required for archived feature-extraction
notebooks. Its internal penalty-selection helper cannot be fully audited from
the notebook alone; do not infer a pure LASSO or lambda-min rule from its name.
Earlier H1, DCA, NRI and IDI code in `legacy/` does not establish current H2
results. Historical clinical imputation requires separate execution provenance;
these scripts consume the supplied clinical table and do not prove that its
original imputation excluded outcome information. Restricted data access remains
necessary for an independent end-to-end replication.
