# onekey_HCC_multinodular

Analysis code for: **"Multicenter Integration of CT Radiomics, Deep Learning, and Habitat Imaging for Predicting Prognosis of Multinodular Hepatocellular Carcinoma with Whole-Exome Sequencing Validation"**

## Overview

This repository contains the **imaging-analysis code** used to construct and validate a five-dimensional CT fusion score predicting recurrence-free survival in CNLC stage Ib–IIb multinodular hepatocellular carcinoma (HCC).

The radiomic feature extraction, deep-learning feature extraction, habitat clustering, and feature-selection notebooks were run on the **OneKey radiomics platform** (OnekeyAI Lab, www.medai.icu) and are provided here as the actual study notebooks with **all cell outputs cleared and local file paths anonymized**. The fusion-model scripts are the actual custom Python scripts used in the study.

> **Provenance note.** Notebook step files (`*.ipynb`) are the OneKey-platform notebooks used in the study; outputs were stripped and absolute machine paths were replaced with placeholders (e.g., `<PROJECT_ROOT>`, `<RADIO_DATA_ROOT>`). The `.py` files under `02_fusion_model/` are the custom analysis scripts (paths anonymized; docstrings cleaned). No algorithm or parameter was altered. Superseded/exploratory iterations were removed for clarity.

> **Code scope.** This release covers the core **imaging pipeline** — five-score construction (`01_score_construction/`) and fusion modeling, risk stratification, subgroup analysis, and NRI/IDI/time-dependent AUC (`02_fusion_model/`). Other analyses described in the manuscript were performed with standard, publicly available open-source libraries and are not separately scripted here because they are fully specified in the Methods and reproducible from the released component-score table: SHAP and permutation-importance interpretability (`shap`, `scikit-learn`), bootstrap C-index confidence intervals and pairwise comparisons (`numpy`/`scipy` + `lifelines`), generalization-degradation summaries (arithmetic on reported C-indices), and inter-score correlation/variance-inflation-factor analysis (`pandas`/`scipy`, `statsmodels`). The whole-exome sequencing (WES) construct-validity analyses were post-hoc and exploratory and were performed with standard genomics tools (per-nodule somatic-mutation calling, MATH/Shannon clonal-diversity, Jaccard clonal-origin, and mutational-signature analysis); the genomic analysis scripts and sequencing data are **not** included in this repository (consistent with patient-privacy and third-party sequencing-vendor constraints). See Data Availability.


## Repository Structure

```
onekey_HCC_multinodular/
├── README.md
├── LICENSE.txt
├── requirements.txt
├── .gitignore
├── config/
│   └── pyradiomics_extraction.yaml          # Reference summary of PyRadiomics v3.0 parameters
│
├── 01_score_construction/                   # OneKey notebooks (outputs cleared, paths anonymized)
│   ├── rad_diff_traditional/                #   Rad_Diff: 3-phase inter-tumor difference radiomics + LASSO-Cox
│   │   ├── config.yaml
│   │   ├── Step0_data_split.ipynb
│   │   ├── Step1_radiomics_features.ipynb
│   │   ├── Step2_clinical_model.ipynb       #   (habitat-cohort-aligned variant used in the integrated analysis)
│   │   ├── Step3_combined_model.ipynb       #   (habitat-cohort-aligned variant used in the integrated analysis)
│   │   ├── Step4_1_cindex.ipynb
│   │   ├── Step4_2_time_dependent_auc.ipynb
│   │   └── Step4_3_model_comparison.ipynb
│   ├── dl_maxpooling_resnet50/              #   DL_MaxPooling: ResNet50 + max-pooling + Elastic Net-Cox
│   │   ├── config.yaml
│   │   ├── Step1_dl_features.ipynb
│   │   ├── Step2_clinical_model.ipynb
│   │   ├── Step3_combined_model.ipynb
│   │   ├── Step4_1_cindex.ipynb
│   │   └── Step4_2_time_dependent_auc.ipynb
│   ├── dl_heterogeneity_densenet121/        #   DL_Heterogeneity: DenseNet121 + SD/Range + Elastic Net-Cox
│   │   ├── config.yaml
│   │   ├── Step1_dl_features.ipynb
│   │   ├── Step2_clinical_model.ipynb
│   │   ├── Step3_combined_model.ipynb
│   │   ├── Step4_1_cindex.ipynb
│   │   └── Step4_2_time_dependent_auc.ipynb
│   └── habitat/                             #   Habitat_MaxH1: portal-phase K-means(K=3) sub-regions + LASSO-Cox survival
│       ├── config.yaml
│       └── Step1_habitat_survival.ipynb     #   LASSO-Cox survival on max-aggregated H1 features
│
└── 02_fusion_model/                         # Custom Python scripts (standalone)
    ├── cox_fusion_complete_analysis.py      #   Five-score Z-score + multivariable Cox + KM/calibration/DCA
    ├── subgroup_all_cohorts.py              #   11-subgroup risk stratification
    └── time_dependent_idi_nri.py            #   NRI / IDI / time-dependent AUC
```


## Dependencies

See `requirements.txt`. Core: PyRadiomics v3.0, torchvision (ImageNet-pretrained ResNet50/DenseNet121), lifelines, scikit-survival, scikit-learn, numpy, pandas, scipy, matplotlib. The reference environment used to produce the reported statistical results is documented at the top of `requirements.txt`.

## How to Run

The notebooks and scripts expect de-identified feature tables and a label file. Before running, set the data paths (placeholders `<PROJECT_ROOT>`, `<RADIO_DATA_ROOT>`, `<IMAGE_DATA_ROOT>`, `<HABITAT_RESULT_ROOT>`) in each `config.yaml` / script header to your local data location. Because the raw imaging and clinical data are restricted (see below), exact numerical reproduction requires the original dataset.

**Runtime note.** The `01_score_construction/` notebooks (`*.ipynb`) were run on the proprietary **OneKey radiomics platform** (`import onekey_algo ...`) and require that environment to execute end-to-end; they are provided as a transparent, human-readable record of the exact feature-extraction and feature-selection pipeline. The fusion-model scripts under `02_fusion_model/` are **standalone Python** and run on the open-source stack in `requirements.txt` (lifelines, scikit-survival, scikit-learn, numpy, pandas, scipy, matplotlib) given the de-identified component-score table.

## Data Availability

Owing to patient privacy, raw CT (DICOM) data and whole-exome sequencing data are not publicly available. De-identified radiomic features, the five component scores with the training-set Z-score parameters and fusion coefficients, and the clinical variables required to reproduce the main imaging results are available from the corresponding author upon reasonable request under an institutional data-use agreement.

## Acknowledgments

- OneKey radiomics platform (OnekeyAI Lab, www.medai.icu) — base analysis framework for feature extraction and selection notebooks
- PyRadiomics (van Griethuysen et al., Cancer Research, 2017)
- torchvision — ImageNet-pretrained deep learning models
- lifelines, scikit-survival, scikit-learn — survival modeling and statistics

## License

Released for research purposes only. Copyright (c) 2026, The First Affiliated
Hospital of Guangxi Medical University. See `LICENSE.txt`. Third-party packages
(PyRadiomics, PyTorch/torchvision, lifelines, scikit-survival, etc.) remain under
their own licenses.

