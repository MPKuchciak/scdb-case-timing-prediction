# File: src/train_model.py

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import json
import glob
import re
from datetime import datetime
from tqdm import tqdm
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any
import pynvml

# Optimization libraries
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import (
    train_test_split, cross_val_score, 
    RandomizedSearchCV, learning_curve, validation_curve
)

# Preprocessing and metrics
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import uniform, randint

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class XGBoostModelTrainer:
    """
    XGBoost training pipeline with parameter optimization, CUDA support,
    automatic versioning, and XAI preparation for Supreme Court Case Duration prediction / analysis.
    """
    
    def __init__(self, output_dir: str = "../models", enable_cuda: bool = True, 
                 random_state: int = 420, #plot_dir: str = "../plots", 
                 organize_by_model: bool = True):
        """
        Initialize the trainer.
        
        Args:
            output_dir (str): Directory to save models and metadata
            enable_cuda (bool): Whether to attempt CUDA acceleration
            plot_dir (str): Directory to save plots -> deleted
            random_state (int): Global random seed
            organize_by_model (bool): Create separate folders for each model
        """
        self.output_dir = output_dir
        #self.plot_dir = plot_dir
        self.random_state = random_state
        self.enable_cuda = enable_cuda
        self.organize_by_model = organize_by_model
        self.device_info = self._setup_device()
        
        # Create directories
        for directory in [output_dir]: #, plot_dir
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
    
    def _setup_device(self) -> Dict[str, Any]:
        """Setup and detect CUDA availability for XGBoost using pynvml."""
        device_info = {
            'cuda_available': False,
            'cuda_device_count': 0,
            'device_name': 'cpu',
            'tree_method': 'hist'
        }

        if self.enable_cuda:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    device_info['cuda_available'] = True
                    device_info['device_name'] = 'cuda'
                    device_info['tree_method'] = 'gpu_hist'
                    device_info['cuda_device_count'] = device_count
                    print(f"CUDA detected: {device_count} GPU(s) available.")
                else:
                    print("CUDA enabled, but no NVIDIA GPUs found. Using CPU.")
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as e:
                print(f"CUDA check failed (could not communicate with NVIDIA driver): {e}. Using CPU.")
            except Exception as e:
                print(f"An unexpected error occurred during CUDA check: {e}. Using CPU.")
        else:
            print("CUDA disabled by user. Using CPU.")

        return device_info

    # def _setup_device(self) -> Dict[str, Any]: -> old part for deletion
    #     """Setup and detect CUDA availability for XGBoost."""
    #     device_info = {
    #         'cuda_available': False,
    #         'cuda_device_count': 0,
    #         'device_name': 'cpu',
    #         'tree_method': 'hist'
    #     }
        
    #     if self.enable_cuda: 
    #         try:
    #             # Check if CUDA is available for XGBoost
    #             import subprocess
    #             result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    #             if result.returncode == 0:
    #                 device_info['cuda_available'] = True
    #                 device_info['device_name'] = 'cuda'
    #                 device_info['tree_method'] = 'gpu_hist'
    #                 # Try to get device count
    #                 try:
    #                     gpu_count = result.stdout.count('GPU') #3?
    #                     device_info['cuda_device_count'] = gpu_count
    #                 except:
    #                     device_info['cuda_device_count'] = 1
    #                 print(f"CUDA detected: {device_info['cuda_device_count']} GPU(s) available")
    #             else:
    #                 print("CUDA not available, using CPU")
    #         except Exception as e:
    #             print(f"CUDA check failed: {e}, using CPU")
    #     else:
    #         print("CUDA disabled, using CPU")
            
    #     return device_info


    
    def _get_next_version(self, base_name: str) -> str:
        """
        Get the next available version number for a model name.
        
        Args:
            base_name (str): Base model name
            
        Returns:
            str: Model name with version (e.g., model_v1, model_v2)
        """
        # Look for existing models with this base name
        if self.organize_by_model:
            # Check in organized directories
            pattern = os.path.join(self.output_dir, f"{base_name}_v*")
            existing_dirs = glob.glob(pattern)
            
            if not existing_dirs:
                # Also check for direct files in output_dir
                pattern = os.path.join(self.output_dir, f"{base_name}_v*.joblib")
                existing_files = glob.glob(pattern)
                if not existing_files:
                    direct_pattern = os.path.join(self.output_dir, f"{base_name}.joblib")
                    if os.path.exists(direct_pattern):
                        return f"{base_name}_v2"
                    else:
                        return f"{base_name}_v1"
                else:
                    # Extract version from files
                    versions = []
                    for file in existing_files:
                        match = re.search(rf"{base_name}_v(\d+)\.joblib", file)
                        if match:
                            versions.append(int(match.group(1)))
                    next_version = max(versions) + 1 if versions else 1
                    return f"{base_name}_v{next_version}"
            else:
                # Extract version from directory names
                versions = []
                for directory in existing_dirs:
                    match = re.search(rf"{base_name}_v(\d+)$", directory)
                    if match:
                        versions.append(int(match.group(1)))
                next_version = max(versions) + 1 if versions else 1
                return f"{base_name}_v{next_version}"
        else:
            # Original file-based versioning
            pattern = os.path.join(self.output_dir, f"{base_name}_v*.joblib")
            existing_files = glob.glob(pattern)
            
            if not existing_files:
                direct_pattern = os.path.join(self.output_dir, f"{base_name}.joblib")
                if os.path.exists(direct_pattern):
                    return f"{base_name}_v2"
                else:
                    return f"{base_name}_v1"
            
            versions = []
            for file in existing_files:
                match = re.search(rf"{base_name}_v(\d+)\.joblib", file)
                if match:
                    versions.append(int(match.group(1)))
            
            next_version = max(versions) + 1 if versions else 1
            return f"{base_name}_v{next_version}"
    
    def _create_model_directory(self, model_name: str) -> Dict[str, str]:
        """
        Create organized directory structure for a model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Dict with paths for model files
        """
        if self.organize_by_model:
            model_dir = os.path.join(self.output_dir, model_name)
            #plot_dir = os.path.join(self.plot_dir, model_name) # old
            plot_dir = os.path.join(model_dir, 'plots')
            
            for directory in [model_dir, plot_dir]:
                if not os.path.exists(directory):
                    os.makedirs(directory)
            
            return {
                'model_dir': model_dir,
                'plot_dir': plot_dir,
                'model_path': os.path.join(model_dir, f"{model_name}.joblib"),
                'metadata_path': os.path.join(model_dir, f"{model_name}_metadata.json"),
                'plot_path': os.path.join(plot_dir, f"{model_name}_diagnostics.pdf")
            }
        else:
            return {
                'model_dir': self.output_dir,
                'plot_dir': self.plot_dir,
                'model_path': os.path.join(self.output_dir, f"{model_name}.joblib"),
                'metadata_path': os.path.join(self.output_dir, f"{model_name}_metadata.json"),
                'plot_path': os.path.join(self.plot_dir, f"{model_name}_diagnostics.pdf")
            }
    
    def prepare_data_splits(self, X: pd.DataFrame, y: pd.Series, 
                        test_size: float = 0.2, eval_size: float = 0.15,
                        stratify_by_target: bool = True) -> Tuple:
        """
        Create train/validation/test splits with optional stratification.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion for test set (default: 0.2)
            eval_size: Proportion for validation set (default: 0.15)
            stratify_by_target: Whether to stratify by binned target values
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
            Tuple of splits AND a boolean indicating if stratification was used.
        """
        print("Creating data splits...")
        
        # For stratification on regression target, bin the values
        stratify_col = None
        stratification_was_used = False # Initialize as False
        if stratify_by_target:
            try:
                # Create bins for stratification
                y_bins = pd.qcut(y, q=min(5, y.nunique()), labels=False, duplicates='drop')
                if y_bins.nunique() > 1:
                    stratify_col = y_bins
                    stratification_was_used = True # Set to True if successful
                    print("Using stratified sampling based on duration bins.")
                else:
                    print("Warning: Could not create enough bins for stratification. Proceeding without.")
            except Exception as e:
                print(f"Warning: Could not create stratification bins: {e}. Proceeding without stratification.")
        
        # First split: separate test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            shuffle=True, stratify=stratify_col
        )
        
        # Calculate validation size from remaining data
        # We want 15% of total data as validation
        # We have 80% as train_val, so validation should be 15/80 = 0.1875 of train_val
        val_size_from_train_val = eval_size / (1 - test_size)
        
        # Second split: separate train and validation
        stratify_col_train_val = None
        if stratification_was_used and stratify_col is not None:
            stratify_col_train_val = stratify_col.loc[X_train_val.index]
            if stratify_col_train_val.nunique() < 2:
                # If the smaller dataset doesn't have enough bin diversity, can't stratify
                stratify_col_train_val = None

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_from_train_val, 
            random_state=self.random_state, shuffle=True, 
            stratify=stratify_col_train_val
        )
        
        # Verify splits
        train_pct = len(X_train) / len(X) * 100
        val_pct = len(X_val) / len(X) * 100
        test_pct = len(X_test) / len(X) * 100
        
        print(f"Data splits created:")
        print(f"   Train: {X_train.shape[0]} samples ({train_pct:.1f}%)")
        print(f"   Validation: {X_val.shape[0]} samples ({val_pct:.1f}%)")
        print(f"   Test: {X_test.shape[0]} samples ({test_pct:.1f}%)")
        print(f"   Total: {len(X)} samples (sum: {train_pct + val_pct + test_pct:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, stratification_was_used
    
    def create_preprocessor(self, numerical_features: List[str], 
                          categorical_features: List[str]) -> ColumnTransformer:
        """Create optimized preprocessing pipeline."""
        print("Creating preprocessing pipeline...")
        
        # Numerical preprocessing with robust scaling
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing optimized for XGBoost 3.0
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(
                handle_unknown='ignore', 
                sparse_output=False, 
                drop='first',
                max_categories=50  # Limit categories for memory efficiency
            ))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        print(f"Preprocessor created for {len(numerical_features)} numerical and {len(categorical_features)} categorical features")
        return preprocessor
    
    def get_base_xgboost_params(self) -> Dict[str, Any]:
        """Get base XGBoost parameters optimized for XGBoost 3.0."""
        base_params = {
            'objective': 'reg:squarederror',
            'random_state': self.random_state,
            'n_jobs': -1,
            'tree_method': self.device_info['tree_method'],
        }
        
        # Add GPU-specific parameters if CUDA is available
        if self.device_info['cuda_available']:
            base_params.update({
                'gpu_id': 0,
                'predictor': 'gpu_predictor'
            })
            
        return base_params
    
    def get_optimization_search_spaces(self) -> Dict[str, Dict]:
        """Define search spaces for different optimization methods."""
        return {
            'random_search': {
                'n_estimators': [400, 2500],
                'learning_rate': [0.01, 0.3],
                'max_depth': [4, 10],
                'min_child_weight': [1, 15],
                'subsample': [0.6, 1.0],
                'colsample_bytree': [0.6, 1.0],
                'reg_alpha': [1e-8, 10.0],
                'reg_lambda': [1e-8, 10.0],
                'gamma': [1e-8, 1.0]
            },
            'optuna': {
                'n_estimators': (400, 2500),
                'learning_rate': (0.01, 0.3),
                'max_depth': (4, 10),
                'min_child_weight': (1, 15),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'reg_alpha': (1e-8, 10.0),
                'reg_lambda': (1e-8, 10.0),
                'gamma': (1e-8, 1.0)
            }
        }
    
    def optimize_parameters_optuna(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray,
                                  n_trials: int = 100, timeout: int = 3600,
                                  early_stopping_rounds: int = 50,
                                  search_space: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Optimize XGBoost parameters using Optuna.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            n_trials: Number of optimization trials
            timeout: Maximum optimization time in seconds
            early_stopping_rounds: Early stopping rounds for XGBoost
            
        Returns:
            Dict containing best parameters and study results
        """
        print(f"Starting Optuna optimization with {n_trials} trials (max {timeout}s)...")
        
        base_params = self.get_base_xgboost_params()
        
        if search_space is None:
            print("Using default Optuna search space.")
            # Define a default search space here if you want
            default_search_space = {
                'n_estimators': {'type': 'int', 'low': 400, 'high': 2500},
                'learning_rate': {'type': 'float', 'low': 1e-2, 'high': 0.3, 'log': True},
                'max_depth': {'type': 'int', 'low': 4, 'high': 10},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0, 'step': 0.05},
                'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0, 'step': 0.05},
                'gamma': {'type': 'float', 'low': 1e-8, 'high': 1.0, 'log': True},
                'min_child_weight': {'type': 'int', 'low': 1, 'high': 15},
                'reg_lambda': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
                'reg_alpha': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
            }
            active_search_space = default_search_space
        else:
            print("Using custom search space provided from the notebook.")
            active_search_space = search_space
    
        def objective(trial):
            param = {}
            # Dynamically create trials based on the search space dictionary
            for name, config in active_search_space.items():
                param_type = config.get('type', 'float') # Default to float if not specified

                if param_type == 'categorical':
                    param[name] = trial.suggest_categorical(name, config['choices'])
                elif param_type == 'int':
                    param[name] = trial.suggest_int(name, config['low'], config['high'], step=config.get('step', 1))
                elif param_type == 'float':
                    param[name] = trial.suggest_float(name, config['low'], config['high'], step=config.get('step'), log=config.get('log', False))

            params = base_params.copy()
            params.update(param)
            params['early_stopping_rounds'] = early_stopping_rounds

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            y_val_pred = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, y_val_pred))

        # Create study
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        print(f"Optuna optimization completed! Best RMSE: {study.best_value:.4f}")
        
        best_params = base_params.copy()
        best_params.update(study.best_params)
        best_params['early_stopping_rounds'] = early_stopping_rounds
        
        return {
            'best_params': best_params,
            'best_score': study.best_value,
            'study': study,
            'n_trials': len(study.trials)
        }
    
    def optimize_parameters_sklearn(self, X_train: np.ndarray, y_train: np.ndarray,
                                    cv: int = 3, n_iter: int = 50,
                                    early_stopping_rounds: int = 50) -> Dict[str, Any]:
        """
        Optimize parameters using sklearn's RandomizedSearchCV.
        Early stopping is disabled during the CV process itself to avoid fit errors,
        but the early_stopping_rounds parameter is retained for the final model params.
        """
        
        # Prepare constructor parameters for XGBRegressor used within RandomizedSearchCV.
        # We explicitly remove 'early_stopping_rounds' for the CV part.
        xgb_constructor_params_for_cv = self.get_base_xgboost_params()
        if 'early_stopping_rounds' in xgb_constructor_params_for_cv:
            del xgb_constructor_params_for_cv['early_stopping_rounds']
        # Also ensure no lingering eval_set if it was somehow in base_params
        if 'eval_set' in xgb_constructor_params_for_cv:
             del xgb_constructor_params_for_cv['eval_set']

        search_spaces_config = self.get_optimization_search_spaces()
        random_space_config = search_spaces_config['random_search']
        
        param_dist = {}
        for param, bounds_config in random_space_config.items():
            if not (isinstance(bounds_config, list) and len(bounds_config) == 2):
                print(f"Warning: Configuration for {param} in random_search space is not a list of [low, high]. Skipping.")
                continue

            low_bound, high_bound = bounds_config[0], bounds_config[1]

            if param in ['n_estimators', 'max_depth', 'min_child_weight']:
                if high_bound < low_bound:
                    print(f"Warning: For integer param {param}, high_bound {high_bound} < low_bound {low_bound}. Using low_bound only.")
                    param_dist[param] = randint(low_bound, low_bound + 1) 
                else:
                    param_dist[param] = randint(low_bound, high_bound + 1)
            else: # Assuming float parameters
                if high_bound < low_bound:
                    print(f"Warning: For float param {param}, high_bound {high_bound} < low_bound {low_bound}. Using low_bound value.")
                    param_dist[param] = uniform(loc=low_bound, scale=0)
                else:
                    param_dist[param] = uniform(loc=low_bound, scale=high_bound - low_bound)

        print(f"Starting Randomized Search with {n_iter} iterations and {cv}-fold CV (early stopping disabled for CV process)...")
        
        search = RandomizedSearchCV(
            estimator=xgb.XGBRegressor(**xgb_constructor_params_for_cv), # No early stopping params here
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1,
            error_score='raise' # Helps in debugging if other issues arise
        )
        
        # Fit without any early stopping related fit_params
        search.fit(X_train, y_train) 
        
        print(f"Random search completed!")
        print(f"Best RMSE: {-search.best_score_:.4f}")
        
        # Start with a fresh set of base parameters for the final model.
        final_best_params = self.get_base_xgboost_params()
        # Update with the hyperparameters found by RandomizedSearchCV.
        final_best_params.update(search.best_params_)
        
        # Now, add the original 'early_stopping_rounds' to these best_params.
        # This ensures that when the final model is trained (outside this function, in train_model),
        # it will use the desired early stopping rounds.
        final_best_params['early_stopping_rounds'] = early_stopping_rounds
        
        return {
            'best_params': final_best_params,
            'best_score': -search.best_score_,
            'search_results': search,
            'cv_results': search.cv_results_
        }
    
    def perform_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                                params: Dict[str, Any], cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation with given parameters.
        
        Args:
            X: Feature matrix
            y: Target variable
            params: XGBoost parameters
            cv: Number of cross-validation folds
            
        Returns:
            Dict with CV results
        """
        print(f"Performing {cv}-fold cross-validation...")
        
        # Create a copy of params without early_stopping_rounds for CV
        # because sklearn's cross_val_score doesn't provide validation sets
        cv_params = params.copy()
        if 'early_stopping_rounds' in cv_params:
            cv_params.pop('early_stopping_rounds')
            print("Note: Removed early_stopping_rounds for cross-validation")
        
        model = xgb.XGBRegressor(**cv_params)
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            model, X, y, cv=cv, 
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        cv_scores = -cv_scores  # Convert back to positive RMSE
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_folds': cv
        }
    
    def perform_advanced_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                                        params: Dict[str, Any], cv: int = 5) -> Dict[str, Any]:
        """
        Perform advanced cross-validation with early stopping support.
        Uses the validation fold directly for early stopping instead of creating another split.
        
        Args:
            X: Feature matrix
            y: Target variable
            params: XGBoost parameters (can include early_stopping_rounds)
            cv: Number of cross-validation folds
            
        Returns:
            Dict with CV results including early stopping info
        """
        from sklearn.model_selection import KFold
        
        print(f"Performing advanced {cv}-fold cross-validation with early stopping support...")
        
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        cv_scores = []
        early_stopping_info = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"  Fold {fold + 1}/{cv}", end='')
            
            # Split data for this fold
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train model
            model = xgb.XGBRegressor(**params)
            
            if 'early_stopping_rounds' in params and params['early_stopping_rounds'] > 0:
                # Use the fold's validation set for early stopping
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    verbose=False
                )
                
                early_stopping_info.append({
                    'fold': fold + 1,
                    'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else params.get('n_estimators', 100),
                    'best_score': model.best_score if hasattr(model, 'best_score') else None
                })
            else:
                # Train without early stopping
                model.fit(X_train_fold, y_train_fold)
                
                early_stopping_info.append({
                    'fold': fold + 1,
                    'best_iteration': params.get('n_estimators', 100),
                    'best_score': None
                })
            
            # Predict on validation fold
            y_pred = model.predict(X_val_fold)
            fold_rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            cv_scores.append(fold_rmse)
            
            print(f" - RMSE: {fold_rmse:.4f}")
        
        cv_scores = np.array(cv_scores)
        
        # Calculate average best iteration for early stopping
        avg_best_iter = np.mean([info['best_iteration'] for info in early_stopping_info if info['best_iteration'] is not None])
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_folds': cv,
            'early_stopping_info': early_stopping_info,
            'avg_best_iteration': avg_best_iter
        }

    def plot_cv_results(self, cv_results: Dict[str, Any], save_path: Optional[str] = None):
        """Create enhanced visualization for cross-validation results"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        cv_scores = cv_results['cv_scores']
        
        # Plot 1: CV Scores with confidence interval
        folds = range(1, len(cv_scores) + 1)
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        axes[0,0].plot(folds, cv_scores, 'o-', color='blue', linewidth=2, markersize=8, label='Fold Scores')
        axes[0,0].axhline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.3f}')
        axes[0,0].fill_between(range(len(cv_scores)+2), 
                            mean_score - std_score, 
                            mean_score + std_score, 
                            alpha=0.2, color='red', label=f'±1 STD: {std_score:.3f}')
        axes[0,0].set_xlabel('Fold Number')
        axes[0,0].set_ylabel('RMSE')
        axes[0,0].set_title('Cross-Validation Scores Across Folds')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_xlim(0.5, len(cv_scores) + 0.5)
        
        # Plot 2: Cumulative mean and variance
        cumulative_mean = np.array([cv_scores[:i+1].mean() for i in range(len(cv_scores))])
        cumulative_std = np.array([cv_scores[:i+1].std() for i in range(len(cv_scores))])
        
        axes[0,1].plot(folds, cumulative_mean, 'g-', linewidth=2, label='Cumulative Mean')
        axes[0,1].fill_between(folds, 
                            cumulative_mean - cumulative_std, 
                            cumulative_mean + cumulative_std, 
                            alpha=0.3, color='green')
        axes[0,1].set_xlabel('Number of Folds')
        axes[0,1].set_ylabel('Cumulative Mean RMSE')
        axes[0,1].set_title('CV Score Stability')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Early Stopping Iterations (if available)
        if 'early_stopping_info' in cv_results and cv_results['early_stopping_info']:
            early_stop_info = cv_results['early_stopping_info']
            iterations = [info['best_iteration'] for info in early_stop_info if info['best_iteration'] is not None]
            
            if iterations:
                axes[1,0].plot(range(1, len(iterations)+1), iterations, 'o-', color='orange', 
                            linewidth=2, markersize=8, label='Best Iteration')
                axes[1,0].axhline(np.mean(iterations), color='red', linestyle='--', 
                                linewidth=2, label=f'Mean: {np.mean(iterations):.0f}')
                axes[1,0].fill_between(range(len(iterations)+2), 
                                    np.mean(iterations) - np.std(iterations), 
                                    np.mean(iterations) + np.std(iterations), 
                                    alpha=0.2, color='red')
                axes[1,0].set_xlabel('Fold Number')
                axes[1,0].set_ylabel('Best Iteration (Early Stopping)')
                axes[1,0].set_title('Early Stopping Analysis')
                axes[1,0].legend()
                axes[1,0].grid(True, alpha=0.3)
                axes[1,0].set_xlim(0.5, len(iterations) + 0.5)
        else:
            # If no early stopping, show score distribution
            axes[1,0].violinplot([cv_scores], positions=[1], showmeans=True, showmedians=True)
            axes[1,0].set_ylabel('RMSE')
            axes[1,0].set_title('Score Distribution')
            axes[1,0].set_xticks([1])
            axes[1,0].set_xticklabels(['CV Scores'])
            axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Performance comparison - sorted scores
        sorted_scores = np.sort(cv_scores)
        axes[1,1].bar(range(1, len(sorted_scores)+1), sorted_scores, 
                    color=plt.cm.RdYlGn_r(sorted_scores / sorted_scores.max()), 
                    edgecolor='black', linewidth=1)
        axes[1,1].axhline(mean_score, color='red', linestyle='--', linewidth=2, 
                        label=f'Mean: {mean_score:.3f}')
        axes[1,1].set_xlabel('Fold (sorted by performance)')
        axes[1,1].set_ylabel('RMSE')
        axes[1,1].set_title('Sorted Fold Performance')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Add overall statistics as text
        stats_text = f'Mean: {mean_score:.3f}\nStd: {std_score:.3f}\nMin: {cv_scores.min():.3f}\nMax: {cv_scores.max():.3f}'
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show() #to comment out after giving out for mark !!!!!!
        
        return fig

    def plot_training_curves(self, training_history: Dict[str, Any], save_path: Optional[str] = None):
        """Plot training curves showing model learning progress"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Assuming training_history contains eval results from XGBoost
        if 'validation_0' in training_history:
            train_scores = training_history['validation_0']['rmse']
            epochs = range(1, len(train_scores) + 1)
            
            # Plot 1: Training curve
            axes[0,0].plot(epochs, train_scores, 'b-', linewidth=2, label='Training RMSE')
            if 'validation_1' in training_history:
                val_scores = training_history['validation_1']['rmse']
                axes[0,0].plot(epochs, val_scores, 'r-', linewidth=2, label='Validation RMSE')
                
                # Find best epoch
                best_epoch = np.argmin(val_scores) + 1
                best_score = val_scores[best_epoch - 1]
                axes[0,0].plot(best_epoch, best_score, 'go', markersize=10, 
                            label=f'Best: {best_score:.3f} @ epoch {best_epoch}')
            
            axes[0,0].set_xlabel('Epoch')
            axes[0,0].set_ylabel('RMSE')
            axes[0,0].set_title('Training Progress')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # Plot 2: Log scale view
            axes[0,1].semilogy(epochs, train_scores, 'b-', linewidth=2, label='Training RMSE')
            if 'validation_1' in training_history:
                axes[0,1].semilogy(epochs, val_scores, 'r-', linewidth=2, label='Validation RMSE')
            axes[0,1].set_xlabel('Epoch')
            axes[0,1].set_ylabel('RMSE (log scale)')
            axes[0,1].set_title('Training Progress (Log Scale)')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            
            # Plot 3: Improvement rate
            if len(train_scores) > 1:
                improvement = -np.diff(train_scores)
                axes[1,0].plot(epochs[1:], improvement, 'g-', linewidth=2)
                axes[1,0].axhline(0, color='red', linestyle='--', alpha=0.5)
                axes[1,0].set_xlabel('Epoch')
                axes[1,0].set_ylabel('RMSE Improvement')
                axes[1,0].set_title('Training Improvement per Epoch')
                axes[1,0].grid(True, alpha=0.3)
            
            # Plot 4: Overfitting detection
            if 'validation_1' in training_history:
                gap = np.array(val_scores) - np.array(train_scores)
                axes[1,1].plot(epochs, gap, 'orange', linewidth=2)
                axes[1,1].axhline(0, color='red', linestyle='--', alpha=0.5)
                axes[1,1].fill_between(epochs, 0, gap, where=(gap > 0), 
                                    alpha=0.3, color='red', label='Overfitting')
                axes[1,1].set_xlabel('Epoch')
                axes[1,1].set_ylabel('Validation - Training RMSE')
                axes[1,1].set_title('Overfitting Analysis')
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show() #to comment out after giving out for mark !!!!!!
        
        return fig


    def create_diagnostic_plots(self, results: Dict[str, Any], plot_path: str) -> str:
        """
        Create comprehensive diagnostic plots for model evaluation.
        
        Args:
            results: Training results dictionary
            plot_path: Full path where to save the PDF
            
        Returns:
            str: Path to saved PDF with all plots
        """
        print("Creating diagnostic plots...")
        
        with PdfPages(plot_path) as pdf:
            # Plot 1: Prediction vs Actual
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Train set
            axes[0,0].scatter(results['y_train'], results['y_train_pred'], alpha=0.6)
            axes[0,0].plot([results['y_train'].min(), results['y_train'].max()], 
                          [results['y_train'].min(), results['y_train'].max()], 'r--', lw=2)
            axes[0,0].set_xlabel('Actual Duration (days)')
            axes[0,0].set_ylabel('Predicted Duration (days)')
            axes[0,0].set_title(f'Train Set: R² = {results["metrics"]["train"]["r2"]:.4f}')
            axes[0,0].grid(True, alpha=0.3)
            
            # Validation set
            axes[0,1].scatter(results['y_val'], results['y_val_pred'], alpha=0.6, color='orange')
            axes[0,1].plot([results['y_val'].min(), results['y_val'].max()], 
                          [results['y_val'].min(), results['y_val'].max()], 'r--', lw=2)
            axes[0,1].set_xlabel('Actual Duration (days)')
            axes[0,1].set_ylabel('Predicted Duration (days)')
            axes[0,1].set_title(f'Validation Set: R² = {results["metrics"]["validation"]["r2"]:.4f}')
            axes[0,1].grid(True, alpha=0.3)
            
            # Test set
            axes[1,0].scatter(results['y_test'], results['y_test_pred'], alpha=0.6, color='green')
            axes[1,0].plot([results['y_test'].min(), results['y_test'].max()], 
                          [results['y_test'].min(), results['y_test'].max()], 'r--', lw=2)
            axes[1,0].set_xlabel('Actual Duration (days)')
            axes[1,0].set_ylabel('Predicted Duration (days)')
            axes[1,0].set_title(f'Test Set: R² = {results["metrics"]["test"]["r2"]:.4f}')
            axes[1,0].grid(True, alpha=0.3)
            
            # Residuals
            residuals = results['y_test'] - results['y_test_pred']
            axes[1,1].scatter(results['y_test_pred'], residuals, alpha=0.6, color='purple')
            axes[1,1].axhline(y=0, color='r', linestyle='--')
            axes[1,1].set_xlabel('Predicted Duration (days)')
            axes[1,1].set_ylabel('Residuals')
            axes[1,1].set_title('Residual Plot (Test Set)')
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Plot 2: Feature Importance
            fig, ax = plt.subplots(figsize=(12, 8))
            top_features = results['feature_importance'].head(20)
            
            sns.barplot(data=top_features, y='feature', x='importance', ax=ax)
            ax.set_title('Top 20 Most Important Features')
            ax.set_xlabel('Feature Importance')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Plot 3: Training History (if available)
            if hasattr(results['xgb_model'], 'evals_result_'):
                fig, ax = plt.subplots(figsize=(10, 6))
                
                eval_results = results['xgb_model'].evals_result_
                if 'validation_0' in eval_results:
                    epochs = range(len(eval_results['validation_0']['rmse']))
                    ax.plot(epochs, eval_results['validation_0']['rmse'], 
                           label='Validation RMSE', linewidth=2)
                    ax.set_xlabel('Boosting Round')
                    ax.set_ylabel('RMSE')
                    ax.set_title('Training History')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # Plot 4: Metrics Comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metrics_df = pd.DataFrame(results['metrics']).T
            metrics_df[['rmse', 'mae']].plot(kind='bar', ax=ax)
            ax.set_title('Model Performance Across Splits')
            ax.set_ylabel('Error (days)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Plot 5: Cross-validation results (if available)
            if 'cv_results' in results and results['cv_results'] is not None:
                fig, ax = plt.subplots(figsize=(10, 6))
                cv_res = results['cv_results']
                ax.boxplot([cv_res['cv_scores']])
                ax.set_title(f'Cross-Validation Results ({cv_res["cv_folds"]} folds)')
                ax.set_ylabel('RMSE (days)')
                ax.set_xlabel('Cross-Validation')
                ax.grid(True, alpha=0.3)
                
                # Add mean and std text
                ax.text(0.7, max(cv_res['cv_scores']) * 0.9, 
                       f'Mean: {cv_res["cv_mean"]:.2f}\nStd: {cv_res["cv_std"]:.2f}',
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        print(f"Diagnostic plots saved: {plot_path}")
        return plot_path
    
####################################################################### TESTING ##########################################
    def create_time_based_splits(self, X: pd.DataFrame, y: pd.Series, 
                            time_col: str = 'year_of_argument',
                            test_size: float = 0.2) -> Tuple:
        """
        Create time-based train/test split for temporal validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            time_col: Column name for time-based sorting
            test_size: Proportion of data to use as test (from most recent)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if time_col not in X.columns:
            raise ValueError(f"Time column '{time_col}' not found in data")
        
        # Combine X and y with index preservation
        data = X.copy()
        data['_target'] = y
        
        # Sort by time
        data_sorted = data.sort_values(time_col)
        
        # Split point
        split_idx = int(len(data_sorted) * (1 - test_size))
        
        # Create splits
        train_data = data_sorted.iloc[:split_idx]
        test_data = data_sorted.iloc[split_idx:]
        
        # Separate features and target
        X_train = train_data.drop('_target', axis=1)
        y_train = train_data['_target']
        X_test = test_data.drop('_target', axis=1)
        y_test = test_data['_target']
        
        print(f"Time-based split created:")
        print(f"  Train: {len(X_train)} samples (up to {train_data[time_col].max()})")
        print(f"  Test: {len(X_test)} samples (from {test_data[time_col].min()})")
        
        return X_train, X_test, y_train, y_test

    def select_top_features_from_importance(self, 
                                        feature_importance_df: pd.DataFrame,
                                        X: pd.DataFrame,
                                        threshold: Union[int, float, str] = 30) -> List[str]:
        """
        Select features based on importance scores.
        
        Args:
            feature_importance_df: DataFrame with 'feature' and 'importance' columns
            X: Original feature matrix (to validate feature names)
            threshold: Either:
                - int: Select top N features
                - float (0-1): Select features capturing X% of total importance
                - str: 'median' or 'mean' to use statistical threshold
                
        Returns:
            List of selected feature names
        """
        # Sort by importance
        sorted_importance = feature_importance_df.sort_values('importance', ascending=False)
        
        if isinstance(threshold, int):
            # Select top N
            selected_features = sorted_importance.head(threshold)['feature'].tolist()
            print(f"Selected top {threshold} features")
            
        elif isinstance(threshold, float) and 0 < threshold < 1:
            # Select features that capture X% of importance
            sorted_importance['cumsum'] = sorted_importance['importance'].cumsum()
            total_importance = sorted_importance['importance'].sum()
            threshold_value = total_importance * threshold
            selected_features = sorted_importance[sorted_importance['cumsum'] <= threshold_value]['feature'].tolist()
            print(f"Selected features capturing {threshold*100:.1f}% of total importance")
            
        elif threshold == 'median':
            # Select features above median importance
            median_importance = sorted_importance['importance'].median()
            selected_features = sorted_importance[sorted_importance['importance'] > median_importance]['feature'].tolist()
            print(f"Selected features above median importance ({median_importance:.4f})")
            
        elif threshold == 'mean':
            # Select features above mean importance
            mean_importance = sorted_importance['importance'].mean()
            selected_features = sorted_importance[sorted_importance['importance'] > mean_importance]['feature'].tolist()
            print(f"Selected features above mean importance ({mean_importance:.4f})")
            
        else:
            raise ValueError(f"Invalid threshold: {threshold}")
        
        # Validate features exist in X
        valid_features = [f for f in selected_features if f in X.columns]
        if len(valid_features) < len(selected_features):
            print(f"Warning: {len(selected_features) - len(valid_features)} features not found in data")
        
        print(f"Final selection: {len(valid_features)} features")
        return valid_features

    def create_ensemble_predictions(self, model_paths: List[str], X_test: pd.DataFrame,
                                weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Create ensemble predictions from multiple saved models.
        
        Args:
            model_paths: List of paths to saved model files
            X_test: Test data for predictions
            weights: Optional weights for each model (must sum to 1)
            
        Returns:
            Array of ensemble predictions
        """
        if weights is not None and abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        predictions = []
        
        for i, model_path in enumerate(model_paths):
            if not os.path.exists(model_path):
                print(f"Warning: Model not found at {model_path}, skipping...")
                continue
                
            # Load model
            pipeline = joblib.load(model_path)
            
            # Make predictions
            pred = pipeline.predict(X_test)
            predictions.append(pred)
            print(f"Loaded model {i+1}/{len(model_paths)}: {os.path.basename(model_path)}")
        
        if not predictions:
            raise ValueError("No valid models found")
        
        # Create ensemble
        predictions_array = np.array(predictions)
        
        if weights is None:
            # Simple average
            ensemble_pred = np.mean(predictions_array, axis=0)
            print("Created simple average ensemble")
        else:
            # Weighted average
            ensemble_pred = np.average(predictions_array, axis=0, weights=weights[:len(predictions)])
            print("Created weighted average ensemble")
        
        return ensemble_pred

    def evaluate_ensemble(self, y_true: np.ndarray, ensemble_pred: np.ndarray,
                        individual_preds: Optional[List[np.ndarray]] = None) -> Dict[str, float]:
        """
        Evaluate ensemble performance and compare to individual models.
        
        Args:
            y_true: True target values
            ensemble_pred: Ensemble predictions
            individual_preds: Optional list of individual model predictions
            
        Returns:
            Dictionary of metrics
        """
        ensemble_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, ensemble_pred)),
            'mae': mean_absolute_error(y_true, ensemble_pred),
            'r2': r2_score(y_true, ensemble_pred)
        }
        
        print(f"\nEnsemble Performance:")
        print(f"  RMSE: {ensemble_metrics['rmse']:.2f}")
        print(f"  MAE: {ensemble_metrics['mae']:.2f}")
        print(f"  R²: {ensemble_metrics['r2']:.4f}")
        
        if individual_preds:
            print(f"\nIndividual Model Performance:")
            for i, pred in enumerate(individual_preds):
                rmse = np.sqrt(mean_squared_error(y_true, pred))
                print(f"  Model {i+1} RMSE: {rmse:.2f}")
        
        return ensemble_metrics
####################################################################### TESTING ##########################################





    def train_model(self, 
                   X: pd.DataFrame, 
                   y: pd.Series,
                   numerical_features: List[str], 
                   categorical_features: List[str],
                   model_name: str = 'scdb_duration_model',
                   optimization_method: Optional[str] = None,
                   optimization_params: Optional[Dict] = None,
                   custom_params: Optional[Dict] = None,
                   generate_custom_params: bool = False,
                   custom_params_seed: Optional[int] = None,
                   early_stopping_rounds: int = 50,
                   test_size: float = 0.2,
                   eval_size: float = 0.15,
                   create_plots: bool = True,
                   perform_cv: bool = False,
                   cv_folds: int = 5,
                   advanced_cv: bool = False,
                   plot_cv_results: bool = False,
                   plot_training_curves: bool = False) -> Dict[str, Any]:
        """
        Main training function with parameter optimization options.
        
        Args:
            X: Feature matrix
            y: Target variable (duration_days)
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            model_name: Base model name (versioning will be automatic)
            optimization_method: None, 'optuna', or 'random'
            optimization_params: Parameters for optimization method
            custom_params: Custom XGBoost parameters
            generate_custom_params: Generate custom params using Optuna ranges
            custom_params_seed: Seed for custom parameter generation
            early_stopping_rounds: Early stopping rounds for XGBoost
            test_size: Test set proportion
            eval_size: Validation set proportion
            create_plots: Whether to create diagnostic plots
            perform_cv: Whether to perform cross-validation
            cv_folds: Number of CV folds
            advanced_cv: Use advanced CV that supports early stopping
            
        Returns:
            dict: Comprehensive results for XAI and analysis
        """
        print("Starting Enhanced XGBoost Training Pipeline")
        print("=" * 70)
        print(f"Device: {self.device_info['device_name'].upper()}")
        
        start_time = datetime.now()
        
        # Auto-increment version
        versioned_model_name = self._get_next_version(model_name)
        print(f"Model name: {versioned_model_name}")
        
        # Create organized directory structure
        paths = self._create_model_directory(versioned_model_name)
        print(f"Model directory: {paths['model_dir']}")
        
        # Data preparation
        X_train, X_val, X_test, y_train, y_val, y_test, stratification_used = self.prepare_data_splits(
            X, y, test_size, eval_size
        )
        
        # Preprocessing
        preprocessor = self.create_preprocessor(numerical_features, categorical_features)
        print("Fitting preprocessor...")
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)
        
        # Get feature names
        try:
            feature_names_out = preprocessor.get_feature_names_out().tolist()
        except:
            feature_names_out = [f"feature_{i}" for i in range(X_train_processed.shape[1])]
        
        print(f"Preprocessing complete: {len(feature_names_out)} features")
        
        # Parameter optimization or generation
        optimization_results = None
        cv_results = None
        
        if optimization_method:
            opt_params = optimization_params or {}
            opt_params['early_stopping_rounds'] = early_stopping_rounds
            
            if optimization_method == 'optuna':
                optimization_results = self.optimize_parameters_optuna(
                    X_train_processed, y_train, X_val_processed, y_val, **opt_params
                )

                if optimization_results:
                    # Save the Optuna study object
                    study_path = os.path.join(paths['model_dir'], f"{versioned_model_name}_optuna_study.joblib")
                    joblib.dump(optimization_results['study'], study_path)
                    print(f"Optuna study saved: {study_path}")
                    
                    # Also save as database for Optuna dashboard compatibility - for later checkout 
                    try:
                        import sqlite3
                        db_path = os.path.join(paths['model_dir'], f"{versioned_model_name}_optuna.db")
                        optimization_results['study'].trials_dataframe().to_sql(
                            'trials', sqlite3.connect(db_path), if_exists='replace', index=False
                        )
                        print(f"Optuna database saved: {db_path}")
                    except Exception as e:
                        print(f"Could not save Optuna database: {e}")

                best_params = optimization_results['best_params']
                
            elif optimization_method == 'random':
                optimization_results = self.optimize_parameters_sklearn(
                    X_train_processed, y_train, **opt_params
                )
                best_params = optimization_results['best_params']
                
            else:
                raise ValueError(f"Unknown optimization method: {optimization_method}. Use 'optuna' or 'random'")
                
        else:
            # Use custom, generated, or default parameters
            best_params = self.get_base_xgboost_params()
            
            if generate_custom_params:
                print("Generating custom parameters using Optuna ranges...")
                generated_params = self.generate_custom_parameters(custom_params_seed)
                best_params.update(generated_params)
                print("Generated parameters:")
                for key, value in generated_params.items():
                    print(f"  {key}: {value}")
                    
            elif custom_params:
                best_params.update(custom_params)
                
            else:
                # Add some defaults
                best_params.update({
                    'n_estimators': 1000,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'min_child_weight': 3,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0
                })
            
            best_params['early_stopping_rounds'] = early_stopping_rounds
        
        # Cross-validation (if requested)
        if perform_cv:
            print(f"\nPerforming {cv_folds}-fold cross-validation...")
            X_full_processed = preprocessor.transform(X)
            
            if advanced_cv:
                cv_results = self.perform_advanced_cross_validation(
                    X_full_processed, y, best_params, cv_folds
                )
                print(f"Advanced CV RMSE: {cv_results['cv_mean']:.2f} ± {cv_results['cv_std']:.2f}")
                print(f"Average best iteration: {cv_results['avg_best_iteration']:.0f}")
            else:
                cv_results = self.perform_cross_validation(
                    X_full_processed, y, best_params, cv_folds
                )
                print(f"CV RMSE: {cv_results['cv_mean']:.2f} ± {cv_results['cv_std']:.2f}")
                
        # Train final model
        print(f"\nTraining final model with parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        xgb_model = xgb.XGBRegressor(**best_params)
        
        # Train with progress tracking
        print("\nTraining in progress...")
        xgb_model.fit(
            X_train_processed, y_train,
            eval_set=[(X_train_processed, y_train), (X_val_processed, y_val)],
            verbose=False
        )
        
        training_history = xgb_model.evals_result()

        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', xgb_model)
        ])
        
        # Evaluate model
        print("\nEvaluating model performance...")
        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)
        y_test_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train': {
                'mse': mean_squared_error(y_train, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'mae': mean_absolute_error(y_train, y_train_pred),
                'r2': r2_score(y_train, y_train_pred)
            },
            'validation': {
                'mse': mean_squared_error(y_val, y_val_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'mae': mean_absolute_error(y_val, y_val_pred),
                'r2': r2_score(y_val, y_val_pred)
            },
            'test': {
                'mse': mean_squared_error(y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'mae': mean_absolute_error(y_test, y_test_pred),
                'r2': r2_score(y_test, y_test_pred)
            }
        }
        
        # Print metrics
        print("\nPERFORMANCE METRICS:")
        print("-" * 50)
        for split, split_metrics in metrics.items():
            print(f"{split.upper():>12}: RMSE={split_metrics['rmse']:6.2f} | "
                  f"MAE={split_metrics['mae']:6.2f} | R²={split_metrics['r2']:6.4f}")
        
        if cv_results:
            cv_type = "Advanced CV" if advanced_cv else "CV"
            print(f"{cv_type + ' RESULTS':>12}: RMSE={cv_results['cv_mean']:6.2f} ± {cv_results['cv_std']:5.2f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names_out,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTOP 10 MOST IMPORTANT FEATURES:")
        print(feature_importance.head(10)[['feature', 'importance']].to_string(index=False))
        
        # Prepare results dictionary
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        # Prepare XAI-ready data
        X_train_processed_df = pd.DataFrame(
            X_train_processed, columns=feature_names_out, index=X_train.index
        )
        X_val_processed_df = pd.DataFrame(
            X_val_processed, columns=feature_names_out, index=X_val.index
        )
        X_test_processed_df = pd.DataFrame(
            X_test_processed, columns=feature_names_out, index=X_test.index
        )
        
        # Create comprehensive metadata
        model_metadata = {
            'model_name': versioned_model_name,
            'timestamp': start_time.isoformat(),
            'training_duration_seconds': training_duration,
            'optimization_method': optimization_method,
            'generated_custom_params': generate_custom_params,
            'early_stopping_rounds': early_stopping_rounds,
            'advanced_cv_used': advanced_cv if perform_cv else False,
            'device_info': self.device_info,
            'data_info': {
                'total_samples': len(X),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'n_features_original': X.shape[1],
                'n_features_processed': len(feature_names_out),
                'numerical_features': numerical_features,
                'categorical_features': categorical_features
            },
        'stratification_used_in_split': stratification_used,
        'split_params': {
            'test_size': test_size,
            'eval_size': eval_size,
            'random_state_split': self.random_state
        },
            'model_params': best_params,
            'performance_metrics': metrics,
            'cross_validation': cv_results,
            'optimization_results': optimization_results,
            'best_iteration': getattr(xgb_model, 'best_iteration', None),
            'training_history': training_history 
        }
        
        # Save model and metadata
        joblib.dump(pipeline, paths['model_path'])
        with open(paths['metadata_path'], 'w') as f:
            json.dump(model_metadata, f, indent=2, default=str)
        
        print(f"\nModel saved: {paths['model_path']}")
        print(f"Metadata saved: {paths['metadata_path']}")
        
        # Prepare comprehensive results
        results = {
            # Core model objects for XAI
            'pipeline': pipeline,
            'xgb_model': xgb_model,
            'preprocessor': preprocessor,
            
            # History
            'model_metadata': model_metadata,

            # Original data splits
            'X_train_original': X_train,
            'X_val_original': X_val,
            'X_test_original': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            
            # Processed data (ready for XAI tools)
            'X_train_processed': X_train_processed_df,
            'X_val_processed': X_val_processed_df,
            'X_test_processed': X_test_processed_df,
            
            # Predictions
            'y_train_pred': y_train_pred,
            'y_val_pred': y_val_pred,
            'y_test_pred': y_test_pred,
            
            # Analysis and metadata
            'feature_names': feature_names_out,
            'feature_importance': feature_importance,
            'metrics': metrics,
            'cv_results': cv_results,
            'model_metadata': model_metadata,
            'model_name': versioned_model_name,
            'model_path': paths['model_path'],
            'optimization_results': optimization_results,
            'paths': paths
        }
        
        # Create diagnostic plots
        if create_plots:
            plot_path = self.create_diagnostic_plots(results, paths['plot_path'])
            results['plot_path'] = plot_path
        
        if create_plots and plot_cv_results and cv_results:
            print("Creating CV visualization...")
            # Use paths['plot_dir'] directly
            cv_plot_path = os.path.join(paths['plot_dir'], f"{versioned_model_name}_cv_results.png")
            self.plot_cv_results(cv_results, cv_plot_path)
            results['cv_plot_path'] = cv_plot_path
        
        # Create training curve plots if enabled
        if create_plots and plot_training_curves and training_history:
            print("Creating training curve plots...")
            # Use paths['plot_dir'] directly
            training_curve_path = os.path.join(paths['plot_dir'], f"{versioned_model_name}_training_curves.png")
            self.plot_training_curves(training_history, training_curve_path)
            results['training_curve_path'] = training_curve_path

        print(f"\nTraining completed successfully!")
        print(f"Total time: {training_duration:.1f}s")
        print(f"Final test RMSE: {metrics['test']['rmse']:.2f} days")
        if cv_results:
            cv_type = "Advanced CV" if advanced_cv else "Cross-validation"
            print(f"{cv_type} RMSE: {cv_results['cv_mean']:.2f} ± {cv_results['cv_std']:.2f} days")
        print("=" * 70)
        
        return results


# Feature importance analysis by category
def analyze_feature_importance_by_category(self, feature_importance_df, feature_categories):
    """Group feature importance by conceptual categories"""
    category_importance = {}
    for category, features in feature_categories.items():
        cat_features = [f for f in features if f in feature_importance_df['feature'].values]
        cat_importance = feature_importance_df[
            feature_importance_df['feature'].isin(cat_features)
        ]['importance'].sum()
        category_importance[category] = cat_importance
    return category_importance

# Add feature selection option
def select_top_features(self, X, y, feature_importance_df, top_n=30):
    """Select top N most important features"""
    top_features = feature_importance_df.head(top_n)['feature'].tolist()
    return X[top_features]

def load_model_for_xai(model_path: str) -> Dict[str, Any]:
    """Load a saved model with all components for XAI analysis."""
    print(f"Loading model from: {model_path}")
    
    pipeline = joblib.load(model_path)
    
    # Load metadata if available
    metadata_path = model_path.replace('.joblib', '_metadata.json')
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return {
        'pipeline': pipeline,
        'preprocessor': pipeline.named_steps['preprocessor'],
        'xgb_model': pipeline.named_steps['regressor'],
        'metadata': metadata
    }


# Usage examples
if __name__ == '__main__':
    print("Enhanced XGBoost Model Trainer")
    print("=" * 50)
    print("Features:")
    print("- Automatic CUDA detection and GPU acceleration")
    print("- Parameter optimization (Optuna, Random Search)")
    print("- Automatic model versioning")
    print("- Organized model directories")
    print("- Comprehensive diagnostic plots") 
    print("- XAI-ready outputs")
    print("- Cross-validation support")
    print("- Custom parameter generation")
    print("- Detailed performance tracking")
    print("\nUsage examples:")
    print("1. Basic training:")
    print("   trainer = XGBoostModelTrainer()")
    print("   results = trainer.train_model(X, y, numerical_features, categorical_features)")
    print("\n2. With Optuna optimization:")
    print("   results = trainer.train_model(X, y, numerical_features, categorical_features,")
    print("                                optimization_method='optuna',")
    print("                                optimization_params={'n_trials': 100})")
    print("\n3. With custom parameters:")
    print("   custom_params = {'n_estimators': 500, 'learning_rate': 0.1}")
    print("   results = trainer.train_model(X, y, numerical_features, categorical_features,")
    print("                                custom_params=custom_params)")
    print("\n4. With cross-validation:")
    print("   results = trainer.train_model(X, y, numerical_features, categorical_features,")
    print("                                perform_cv=True, cv_folds=5)")