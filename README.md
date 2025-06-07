# Analysis of U.S. Supreme Court Case Duration: A Study on Prediction and Data Leakage

<!-- [![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
The repository is organized to ensure clarity and reproducibility.

```
scdb-case-timing-prediction/
├── .git/                             # Git's internal directory
├── .gitattributes                    # Defines attributes for paths
├── .gitignore                        # Specifies intentionally untracked files
├── data/
│   ├── raw/                          # For original, immutable data
│   │   ├── SCDB_2024_01_caseCentered_Docket.csv
│   │   └── SCDB_2024_01_caseCentered_Vote.csv # If used
│   ├── processed/                      # For cleaned, transformed data 
│   │   ├── scdb_processed_part1.csv    # Output of 001_DataCleaning.ipynb
│   │   └── scdb_eda.csv                # Data for EDA notebook
│   └── external/                       # For any other external data
├── notebooks/                        # For all Jupyter notebooks
│   ├── 001_DataCleaning.ipynb
│   ├── 002_EDA.ipynb                 # Renamed from eda.ipynb for sequence
│   └── 003_ModelXGboost.ipynb        # Renamed from ModelXGboost.ipynb for sequence
├── src/                              # For Python source code (.py files, utility scripts)
├── models/                           # For saved trained models (e.g., .pkl, .joblib files)
├── presentation/                     # For slides, presentation materials
├── reports/                          # For generated reports, figures
│   └── figures/                      # For plots saved from notebooks/scripts
├── docs/                             # For documentation
│   ├── SCDB_2024_01_codebook.pdf
│   └── variable_description.pdf
├── archive/                          # For old/unused files
├── README.md                         # Project overview, setup, how to run
└── requirements.txt                  # Python package dependencies
```

---

## Installation

To get this project up and running on your local machine, follow these steps.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/scdb-case-timing-prediction.git](https://github.com/your-username/scdb-case-timing-prediction.git)
    cd scdb-case-timing-prediction
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies from `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **PyTorch Installation (if needed):**
    Codes require PyTorch, install it using the command from the [official website](https://pytorch.org/get-started/locally/).

    * **For CPU-only:**
        ```bash
        pip install torch torchvision torchaudio
        ```
    * **For GPU (NVIDIA with CUDA):** Check your CUDA version and use the corresponding command from the PyTorch website. For example, for CUDA 12.1:
        ```bash
        pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
        ```

---

## Workflow and Usage

The project workflow is organized into a series of Jupyter notebooks. Please run them in the specified order for full reproducibility.

* **1. [notebooks/000_PackageInstallation.ipynb](./notebooks/000_PackageInstallation.ipynb)**
    * A utility notebook to guide you through installing all necessary packages and dependencies.

* **2. [notebooks/001_DataCleaning.ipynb](./notebooks/001_DataCleaning.ipynb)**
    * This notebook loads the raw data from `data/raw/`, performs extensive cleaning, handles missing values, engineers new features (like case duration), and saves the final cleaned dataset to `data/processed/`.

* **3. [notebooks/002_EDA.ipynb](./notebooks/002_EDA.ipynb)**
    * Conducts a thorough Exploratory Data Analysis (EDA) on the processed data to uncover trends, distributions, and correlations between variables.

* **4. [notebooks/003_XGB_XAI.ipynb](./notebooks/003_XGB_XAI.ipynb)**
    * This is the core modeling notebook. It develops and compares multiple XGBoost models, including scenarios with and without data leakage. It evaluates their performance and uses Explainable AI (XAI) techniques like SHAP to interpret the differences in feature importance between the models.

---

## Data

The data for this project comes from the **Supreme Court Database (SCDB)**, a comprehensive dataset covering case outcomes from 1946 to the present.

* **`data/raw/`**: Contains the original, unaltered CSV files downloaded from the SCDB website.
* **`data/processed/`**: Contains the cleaned and transformed data used for analysis and modeling.
* **`docs/`**: This directory contains the official **`SCDB_2024_01_codebook.pdf`** and a **`variable_description.pdf`**, which provide detailed information on every variable in the dataset.

---

## Key Results

The comparative modeling approach yielded critical insights into both prediction and the practical challenges of data leakage.

* **Modeling Scenarios**: Two primary modeling scenarios were evaluated to understand the impact of feature availability:
    * **Model with Data Leakage**: This model was trained with all available features, including those known after a case is decided. It achieved a very high **R-squared of 0.92**, serving as a theoretical upper benchmark.
    * **Model without Data Leakage**: This model was carefully trained using only features that would be available at the time of prediction. It achieved a more realistic and practical **R-squared of 0.78**.

* **Impact of Data Leakage**: The comparison starkly illustrates the effect of data leakage. In the "leaky" model, features like `decisionType` and `partyWinning` were overwhelmingly dominant, artificially inflating performance and providing no real predictive value for future cases.

* **Feature Importance (Practical Model)**: In the practical, non-leaky model, XAI analysis with SHAP revealed that the most influential predictors of case duration were:
    1.  **Certification Reason (`certReason`)**: The legal grounds on which the case was granted review.
    2.  **Issue Area (`issueArea`)**: The substantive legal topic of the case (e.g., Civil Rights, Criminal Procedure, Economics).
    3.  **Case Origin (`caseOrigin`)**: The lower court or state from which the case was appealed.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.