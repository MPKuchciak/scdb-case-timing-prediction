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
    Enhanced XGBoost training pipeline with parameter optimization, CUDA support,
    automatic versioning, and comprehensive XAI preparation for Supreme Court case duration prediction.
    """
    
    def __init__(self, output_dir: str = "../models", enable_cuda: bool = True, 
                 plot_dir: str = "../plots", random_state: int = 42, 
                 organize_by_model: bool = True):
        """
        Initialize the enhanced trainer.
        
        Args:
            output_dir (str): Directory to save models and metadata
            enable_cuda (bool): Whether to attempt CUDA acceleration
            plot_dir (str): Directory to save plots
            random_state (int): Global random seed
            organize_by_model (bool): Create separate folders for each model
        """
        self.output_dir = output_dir
        self.plot_dir = plot_dir
        self.random_state = random_state
        self.enable_cuda = enable_cuda
        self.organize_by_model = organize_by_model
        self.device_info = self._setup_device()
        
        # Create directories
        for directory in [output_dir, plot_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
    
    def _setup_device(self) -> Dict[str, Any]:
        """Setup and detect CUDA availability for XGBoost."""
        device_info = {
            'cuda_available': False,
            'cuda_device_count': 0,
            'device_name': 'cpu',
            'tree_method': 'hist'
        }
        
        if self.enable_cuda:
            try:
                # Check if CUDA is available for XGBoost
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    device_info['cuda_available'] = True
                    device_info['device_name'] = 'cuda'
                    device_info['tree_method'] = 'gpu_hist'
                    # Try to get device count
                    try:
                        gpu_count = result.stdout.count('GPU ')
                        device_info['cuda_device_count'] = gpu_count
                    except:
                        device_info['cuda_device_count'] = 1
                    print(f"CUDA detected: {device_info['cuda_device_count']} GPU(s) available")
                else:
                    print("CUDA not available, using CPU")
            except Exception as e:
                print(f"CUDA check failed: {e}, using CPU")
        else:
            print("CUDA disabled, using CPU")
            
        return device_info
    
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
            plot_dir = os.path.join(self.plot_dir, model_name)
            
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
    
    def generate_custom_parameters(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate custom parameters using the same ranges as Optuna optimization.
        
        Args:
            seed (int): Random seed for reproducible parameter generation
            
        Returns:
            Dict of sampled parameters
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Sample parameters using the same ranges as Optuna
        custom_params = {
            'n_estimators': np.random.randint(400, 2501),
            'learning_rate': np.random.lognormal(np.log(0.05), 0.5),  # Log-normal around 0.05
            'max_depth': np.random.randint(4, 11),
            'subsample': np.round(np.random.uniform(0.6, 1.0), 2),
            'colsample_bytree': np.round(np.random.uniform(0.6, 1.0), 2),
            'gamma': np.random.lognormal(np.log(1e-6), 2),  # Log-normal
            'min_child_weight': np.random.randint(1, 16),
            'reg_lambda': np.random.lognormal(np.log(1), 1),  # Log-normal around 1
            'reg_alpha': np.random.lognormal(np.log(0.1), 1),  # Log-normal around 0.1
        }
        
        # Clip values to ensure they're within bounds
        custom_params['learning_rate'] = np.clip(custom_params['learning_rate'], 0.01, 0.3)
        custom_params['gamma'] = np.clip(custom_params['gamma'], 1e-8, 1.0)
        custom_params['reg_lambda'] = np.clip(custom_params['reg_lambda'], 1e-8, 10.0)
        custom_params['reg_alpha'] = np.clip(custom_params['reg_alpha'], 1e-8, 10.0)
        
        return custom_params
    
    def prepare_data_splits(self, X: pd.DataFrame, y: pd.Series, 
                           test_size: float = 0.2, eval_size: float = 0.15) -> Tuple:
        """Create stratified train/validation/test splits."""
        print("Creating data splits...")
        
        # Initial train-test split
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, shuffle=True
        )
        
        # Further split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=eval_size, 
            random_state=self.random_state, shuffle=True
        )
        
        print(f"Data splits created:")
        print(f"   Train: {X_train.shape[0]} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Validation: {X_val.shape[0]} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   Test: {X_test.shape[0]} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
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
        
        Args:
            X_train, y_train: Training data
            cv: Number of cross-validation folds
            n_iter: Number of iterations for random search
            early_stopping_rounds: Early stopping rounds for XGBoost
            
        Returns:
            Dict containing best parameters and search results
        """
        base_params = self.get_base_xgboost_params()
        base_params['early_stopping_rounds'] = early_stopping_rounds
        
        search_spaces = self.get_optimization_search_spaces()
        
        param_dist = {}
        random_space = search_spaces['random_search']
        for param, bounds in random_space.items():
            if isinstance(bounds, list) and len(bounds) == 2:
                if param in ['n_estimators', 'max_depth', 'min_child_weight']:
                    param_dist[param] = range(bounds[0], bounds[1] + 1)
                else:
                    param_dist[param] = np.linspace(bounds[0], bounds[1], 20)
                    
        print(f"Starting Randomized Search with {n_iter} iterations and {cv}-fold CV...")
        search = RandomizedSearchCV(
            xgb.XGBRegressor(**base_params),
            param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )
        
        # Fit
        search.fit(X_train, y_train)
        
        print(f"Random search completed!")
        print(f"  Best RMSE: {-search.best_score_:.4f}")
        
        # Combine best parameters with base parameters
        best_params = base_params.copy()
        best_params.update(search.best_params_)
        
        return {
            'best_params': best_params,
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
        Perform advanced cross-validation that supports early stopping.
        
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
            print(f"  Fold {fold + 1}/{cv}")
            
            # Split data for this fold
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Further split training data for early stopping if needed
            if 'early_stopping_rounds' in params and params['early_stopping_rounds'] > 0:
                # Split train into train/early_stop_val (80/20)
                split_idx = int(0.8 * len(X_train_fold))
                X_train_es = X_train_fold[:split_idx]
                X_val_es = X_train_fold[split_idx:]
                y_train_es = y_train_fold[:split_idx]
                y_val_es = y_train_fold[split_idx:]
                
                # Train with early stopping
                model = xgb.XGBRegressor(**params)
                model.fit(
                    X_train_es, y_train_es,
                    eval_set=[(X_val_es, y_val_es)],
                    verbose=False
                )
                
                early_stopping_info.append({
                    'fold': fold + 1,
                    'best_iteration': getattr(model, 'best_iteration', params.get('n_estimators', 100)),
                    'best_score': getattr(model, 'best_score', None)
                })
            else:
                # Train without early stopping
                model = xgb.XGBRegressor(**params)
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
            
            print(f"    Fold {fold + 1} RMSE: {fold_rmse:.4f}")
        
        cv_scores = np.array(cv_scores)
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_folds': cv,
            'early_stopping_info': early_stopping_info,
            'avg_best_iteration': np.mean([info['best_iteration'] for info in early_stopping_info])
        }
    
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
                   advanced_cv: bool = False) -> Dict[str, Any]:
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
        if self.device_info['cuda_available']:
            print(f"CUDA Devices: {self.device_info['cuda_device_count']}")
        print("=" * 70)
        
        start_time = datetime.now()
        
        # Auto-increment version
        versioned_model_name = self._get_next_version(model_name)
        print(f"Model name: {versioned_model_name}")
        
        # Create organized directory structure
        paths = self._create_model_directory(versioned_model_name)
        print(f"Model directory: {paths['model_dir']}")
        
        # Data preparation
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data_splits(
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
                # Add some sensible defaults
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
            eval_set=[(X_val_processed, y_val)],
            verbose=False
        )
        
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
            'model_params': best_params,
            'performance_metrics': metrics,
            'cross_validation': cv_results,
            'optimization_results': optimization_results,
            'best_iteration': getattr(xgb_model, 'best_iteration', None)
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
        
        print(f"\nTraining completed successfully!")
        print(f"Total time: {training_duration:.1f}s")
        print(f"Final test RMSE: {metrics['test']['rmse']:.2f} days")
        if cv_results:
            cv_type = "Advanced CV" if advanced_cv else "Cross-validation"
            print(f"{cv_type} RMSE: {cv_results['cv_mean']:.2f} ± {cv_results['cv_std']:.2f} days")
        print("=" * 70)
        
        return results



# Add these features to your trainer:

# 1. Feature importance analysis by category
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

# 2. Add stratified sampling option for train/test split
def prepare_data_splits(self, X, y, test_size=0.2, eval_size=0.15, stratify_on=None):
    """Enhanced splitting with optional stratification"""
    if stratify_on is not None:
        # For regression, bin the target for stratification
        y_bins = pd.qcut(y, q=5, labels=['Very Fast', 'Fast', 'Normal', 'Slow', 'Very Slow'])
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y_bins
        )
    # ... rest of the method

# 3. Add feature selection option
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