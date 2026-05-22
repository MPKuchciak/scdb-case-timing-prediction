# Analysis of U.S. Supreme Court Case Duration: A Study on Prediction and Data Leakage

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=flat&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) ![SHAP](https://img.shields.io/badge/SHAP-FF6B6B?style=flat&logoColor=white) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project conducts a comprehensive analysis of U.S. Supreme Court case durations using the Supreme Court Database (SCDB). The core of the project involves developing and comparing two distinct modeling approaches: one that intentionally includes features prone to data leakage to establish a performance ceiling, and a second, more realistic model that carefully controls for such information. The goal is to build a practical predictive model while demonstrating the impact of data leakage on model performance and interpretation using Explainable AI (XAI).

---

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Workflow and Usage](#workflow-and-usage)
- [Data](#data)
- [Key Results](#key-results)
- [License](#license)

---

## Project Structure

```
scdb-case-timing-prediction/
├── .gitattributes
├── .gitignore
├── data/
│   ├── model_results/
│   │   └── model_comparison_comprehensive.csv  # All model run metrics
│   ├── raw/                                     # Original, immutable data (not committed)
│   │   ├── SCDB_2024_01_caseCentered_Docket.csv
│   │   └── SCDB_2024_01_caseCentered_Vote.csv
│   └── processed/                               # Cleaned, transformed data (not committed)
│       ├── scdb_processed_part2.csv
│       └── scdb_eda.csv
├── notebooks/
│   ├── 000_PackageInstallation.ipynb
│   ├── 001_DataCleaning.ipynb
│   ├── 002_EDA.ipynb
│   └── 003_XGB_XAI.ipynb
├── src/
│   ├── model_utils.py                           # Model loading and data split utilities
│   └── toc_generator.py                         # Notebook TOC generator utility
├── models/                                      # Saved trained models (.pkl, .joblib)
├── presentation/                                # Slides and presentation materials
├── docs/
│   ├── SCDB_2024_01_codebook.pdf
│   └── variable_description.pdf
├── archive/
├── README.md
└── requirements.txt
```

---

## Installation

1. **Clone the repository:**
```bash
   git clone https://github.com/MPKuchciak/scdb-case-timing-prediction.git
   cd scdb-case-timing-prediction
```

2. **Create a virtual environment (recommended):**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
   pip install -r requirements.txt
```

---

## Workflow and Usage

Run the notebooks in order for full reproducibility.

**1. [notebooks/000_PackageInstallation.ipynb](./notebooks/000_PackageInstallation.ipynb)**
Utility notebook for verifying the environment and managing `requirements.txt`. Includes guidance for both `venv` and `conda` setup.

**2. [notebooks/001_DataCleaning.ipynb](./notebooks/001_DataCleaning.ipynb)**
Loads raw SCDB data, performs cleaning, handles missing values, engineers features (including case duration), and saves the processed dataset to `data/processed/`.

**3. [notebooks/002_EDA.ipynb](./notebooks/002_EDA.ipynb)**
Exploratory Data Analysis on the processed data — distributions, trends, and correlations between variables.

**4. [notebooks/003_XGB_XAI.ipynb](./notebooks/003_XGB_XAI.ipynb)**
Core modeling notebook. Develops and compares multiple XGBoost models across leaky and non-leaky scenarios using Optuna for hyperparameter tuning. Uses SHAP and DALEX for model interpretation.

---

## Data

The data comes from the **Supreme Court Database (SCDB)**, covering case outcomes from 1946 to the present.

- `data/raw/` — original, unaltered CSV files from the SCDB website (not committed to the repo)
- `data/processed/` — cleaned and transformed data used for modeling (not committed)
- `data/model_results/` — model comparison metrics across all training runs
- `docs/` — official SCDB codebook and variable descriptions

---

## Key Results

Ten models were trained across three scenarios (basic, leakage-controlled, full leakage) with baseline and Optuna-tuned variants. All models used log-transformed targets where noted.

**Model with Data Leakage (log target, Optuna-tuned):** Test R² = **0.967**, Test RMSE = 0.124 — serves as a theoretical upper benchmark. Features like `decisionType` and `partyWinning` dominate, artificially inflating performance.

**Model without Data Leakage (log target, Optuna-tuned):** Test R² = **0.480**, Test RMSE = 41.0 days — the practical model using only features available at prediction time.

**Feature Importance (Practical Model):** SHAP and DALEX analysis agree on the top predictors:
1. **Days: Term Start to Argument** — by far the strongest predictor; timing within the term drives duration more than case content
2. **Natural Court Period** — different court compositions show distinct efficiency patterns
3. **Consolidated Dockets** — direct complexity indicator
4. **Law Type (Statutory vs Constitutional)** — case type affects processing time

**Key insight:** The leakage comparison starkly illustrates the danger of post-decision features. The practical model confirms that *when* a case is argued in the term is more predictive than *what* the case is about.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
