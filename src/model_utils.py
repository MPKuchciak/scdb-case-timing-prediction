# src/model_utils.py

import os
import glob
import json
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple, Any

def list_trained_models(models_base_dir: str = "../models") -> List[Dict[str, Any]]:
    """
    Scans the base directory for trained models and returns their info.
    Sorts models by timestamp, from newest to oldest.
    """
    model_infos = []
    if not os.path.isdir(models_base_dir):
        print(f"Warning: Models base directory '{models_base_dir}' not found.")
        return model_infos

    search_pattern = os.path.join(models_base_dir, "*_v*")
    potential_dirs = glob.glob(search_pattern)

    for item_path in potential_dirs:
        if os.path.isdir(item_path):
            model_name_versioned = os.path.basename(item_path)
            metadata_path = os.path.join(item_path, f"{model_name_versioned}_metadata.json")
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    model_infos.append({
                        'name': model_name_versioned,
                        'path': item_path,
                        'timestamp': metadata.get('timestamp'),
                        'test_rmse': metadata.get('performance_metrics', {}).get('test', {}).get('rmse')
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not process metadata for {model_name_versioned}: {e}")

    return sorted(model_infos, key=lambda x: x.get('timestamp', ''), reverse=True)

def load_model_artifacts(model_dir_path: str) -> Optional[Dict[str, Any]]:
    """
    Loads a saved model pipeline, metadata, and Optuna study (if available).
    """
    if not os.path.isdir(model_dir_path):
        print(f"Error: Invalid directory path '{model_dir_path}'.")
        return None

    model_name = os.path.basename(model_dir_path)
    model_path = os.path.join(model_dir_path, f"{model_name}.joblib")
    metadata_path = model_path.replace(".joblib", "_metadata.json")

    optuna_study_path = os.path.join(model_dir_path, f"{model_name}_optuna_study.joblib")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    artifacts = {}
    try:
        pipeline = joblib.load(model_path)
        artifacts['pipeline'] = pipeline
        artifacts['preprocessor'] = pipeline.named_steps.get('preprocessor')
        artifacts['xgb_model'] = pipeline.named_steps.get('regressor')
        print(f"Successfully loaded model pipeline from: {model_path}")
    except Exception as e:
        print(f"Error loading model pipeline from {model_path}: {e}")
        return None

    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            artifacts['metadata'] = json.load(f)
    else:
        artifacts['metadata'] = {}
        print(f"Warning: Metadata file not found at {metadata_path}")
        
    # --- ADDED: Load Optuna study if it exists ---
    if os.path.exists(optuna_study_path):
        try:
            artifacts['optuna_study'] = joblib.load(optuna_study_path)
            print(f"Successfully loaded Optuna study from: {optuna_study_path}")
        except Exception as e:
            print(f"Warning: Could not load Optuna study file: {e}")
            
    return artifacts

def get_data_splits_from_metadata(X_full: pd.DataFrame, y_full: pd.Series, metadata: Dict[str, Any]) -> Optional[Tuple]:
    """
    Recreates data splits (train, val, test) using parameters stored in model metadata.
    """
    split_params = metadata.get('split_params')
    stratify_used = metadata.get('stratification_used_in_split', False)
    
    if not split_params or None in [split_params.get('test_size'), split_params.get('eval_size'), split_params.get('random_state_split')]:
        print("Error: Metadata is missing required split parameters. Please retrain a model to generate new metadata.")
        return None

    print(f"Recreating splits with params: {split_params}, Stratify: {stratify_used}")

    stratify_col = None
    if stratify_used:
        try:
            y_bins = pd.qcut(y_full, q=min(5, y_full.nunique()), labels=False, duplicates='drop')
            if y_bins.nunique() > 1:
                stratify_col = y_bins
        except Exception as e:
            print(f"Warning: Could not create stratification bins during recreation: {e}")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_full, y_full, test_size=split_params['test_size'], random_state=split_params['random_state_split'],
        shuffle=True, stratify=stratify_col
    )
    
    val_size = split_params['eval_size'] / (1 - split_params['test_size'])
    stratify_col_train_val = stratify_col.loc[X_train_val.index] if stratify_used and stratify_col is not None and not X_train_val.empty else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=split_params['random_state_split'],
        shuffle=True, stratify=stratify_col_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test