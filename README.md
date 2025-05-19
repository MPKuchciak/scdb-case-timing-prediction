```
scdb-case-timing-prediction/
├── .git/                             # Git's internal directory
├── .gitattributes                    # Defines attributes for paths
├── .gitignore                        # Specifies intentionally untracked files
├── data/
│   ├── raw/                          # For original, immutable data
│   │   ├── SCDB_2024_01_caseCentered_Docket.csv
│   │   └── SCDB_2024_01_caseCentered_Vote.csv # If used
│   ├── processed/                      # For cleaned, transformed data (intermediate/final)
│   │   ├── scdb_processed_part1.csv    # Output of 001_DataCleaning.ipynb
│   │   └── scdb_eda.csv                # Data for EDA notebook (maybe same as processed)
│   └── external/                       # (Optional) For any other external data
├── notebooks/                        # For all Jupyter notebooks
│   ├── 001_DataCleaning.ipynb
│   ├── 002_EDA.ipynb                 # Renamed from eda.ipynb for sequence
│   ├── 003_ModelXGboost.ipynb        # Renamed from ModelXGboost.ipynb for sequence
│   └── (Untitled-1 - Copy.ipynb)     # Consider renaming or moving to archive/ if a scratchpad
├── src/                              # For Python source code (.py files, utility scripts)
├── models/                           # For saved trained models (e.g., .pkl, .joblib files)
├── presentation/                     # For slides, presentation materials
├── reports/                          # (Consider adding) For generated reports, figures
│   └── figures/                      # For plots saved from notebooks/scripts
├── docs/                             # (Consider adding) For documentation
│   ├── SCDB_2024_01_codebook.pdf
│   └── variable_description.pdf
├── archive/                          # For old/unused files
├── README.md                         # Project overview, setup, how to run
└── requirements.txt                  # Python package dependencies
```