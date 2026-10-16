# Multinodular HCC CT fusion analysis

Code supporting **CT-based Tumor Heterogeneity Fusion Model for Prognosis in
Multinodular Hepatocellular Carcinoma: A Multicenter Study**.

## Current manuscript version

The current habitat component is **max-H2**, selected from 12 prespecified
aggregation/subregion candidates using training-set cross-validation. The old
max-H1 notebook and earlier fusion analyses are preserved under `legacy/` and
must not be used to reproduce the current manuscript.

`current_analysis/` provides portable command-line entry points for the current
habitat selection, locked fusion model, clinical benchmarks, correlation/VIF,
calibration/prediction error, numerical proportional-hazards diagnostics, and
fusion-stage bootstrap internal validation. Input/output paths are arguments.
Scientific functions ported unchanged from the locked analysis are listed in
`config/current_analysis_provenance.json`; reporting corrections are identified
explicitly there. The PH correction replaces an incorrect interpretation of a
library return value with numerical Schoenfeld tests. It does not alter scores.

## Quick start

Use Python 3.12 in an isolated environment:

```sh
python -m venv .venv
# Activate .venv using your operating system's usual command.
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Apply the released aggregate parameters to an appropriately constructed private
component-score table, without fitting on the input patients:

```sh
python -m current_analysis.locked_inference --input /private/component_scores.csv --output /private/predictions.csv
```

Add `--compute-clinical` only when the table contains the nine correctly coded
clinical inputs. AFP and CA199 in this interface are binary indicators, not raw
laboratory concentrations. See [input schema](docs/input_schema.md).

All commands are run from the repository root. Generated tables contain private
data and must be stored outside the repository. Exact study reproduction requires
approved access to the original input tables and locked artifacts; they are not
part of this public release.

## Repository layout

- `current_analysis/`: current max-H2 statistical analysis and frozen inference.
- `config/locked_fusion_model.json`: aggregate training means/scales, clinical
  weights, fusion coefficients and training-median cutoffs; no patient records.
- `config/current_analysis_provenance.json`: source hashes and change scope.
- `docs/`: schemas, commands, verification and limits.
- `01_score_construction/`: original OneKey notebooks for radiomics and fixed
  ImageNet feature extraction, with outputs cleared and local paths anonymized.
- `legacy/`: superseded H1 and earlier exploratory code, clearly separated.
- `tests/`: synthetic-data tests; patient data are never required by CI.

The OneKey notebooks require the original proprietary `onekey_algo` environment.
Installing the statistical requirements does not install OneKey or reproduce CT
preprocessing, segmentation or all neural-network feature extraction. The
reference PyRadiomics settings are in `config/pyradiomics_extraction.yaml`.
WES processing and patient sequencing data are outside this release's scope.

## Reproduction and validation

See [reproduction commands](docs/reproducibility.md) and the aggregate
[release validation record](docs/release_validation.json).
The portable fusion entry point reproduced the archived three-cohort five-score
C-indices (approximately 0.703, 0.604 and 0.614). Same-cohort clinical benchmarking
reproduced all 24 point/interval rows and 48 paired-difference rows without a
numerical difference. This validates the port against the locked inputs; it does
not resolve upstream historical clinical-imputation provenance.

Test set 2 evaluates PFS transfer of an RFS-derived score and must be interpreted
separately. Unadjusted paired intervals do not establish superiority over every
clinical comparator. Internal bootstrap validation fixes component scores and
does not repeat their entire feature/candidate selection.

## Data availability

Raw CT and WES data are not public. De-identified feature tables, component scores
and necessary clinical variables may be requested from the corresponding authors
subject to institutional approval and a data-use agreement. No patient-level
table is distributed here. Public aggregate parameters do not constitute access
to those restricted data.

## Acknowledgments and license

Feature-extraction notebooks use the OneKey radiomics platform and third-party
PyRadiomics/PyTorch tools. Statistical code uses the open-source libraries listed
in `requirements.txt`; their licenses apply separately. The repository's existing
research-use license is unchanged; see `LICENSE.txt`.
