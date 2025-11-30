"""
Model Trainer Module for Parkinson's Disease Predictor

Implements train-on-launch ML pipeline with:
- 5-fold stratified cross-validation
- Optuna hyperparameter tuning
- Model comparison (XGBoost, Random Forest, SVM)
- SHAP explainability with matplotlib plots
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Optional imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from .data_loader import DataLoader


class ModelTrainer:
    """
    Trains and manages ML models for Parkinson's disease prediction.
    
    Features:
    - Auto-trains on first launch if no cached model exists
    - Hyperparameter optimization with Optuna
    - Compares XGBoost, Random Forest, and SVM
    - Generates SHAP explanations
    - Saves best model with metrics
    """
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        """
        Initialize ModelTrainer.
        
        Args:
            data_dir: Directory for cached datasets
            models_dir: Directory for saved models
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.data_loader = DataLoader(data_dir=data_dir)
        
        # Set instance paths based on models_dir
        self.MODEL_PATH = os.path.join(models_dir, "best_model.pkl")
        self.SCALER_PATH = os.path.join(models_dir, "scaler.pkl")
        self.METRICS_PATH = os.path.join(models_dir, "metrics.json")
        self.SHAP_PLOT_PATH = os.path.join(models_dir, "shap_summary.png")
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.metrics = {}
        self.shap_explainer = None
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Check library availability
        print(f"\n🤖 ModelTrainer initialized")
        print(f"   - XGBoost available: {XGBOOST_AVAILABLE}")
        print(f"   - Optuna available: {OPTUNA_AVAILABLE}")
        print(f"   - SHAP available: {SHAP_AVAILABLE}")
    
    def model_exists(self) -> bool:
        """Check if a trained model exists."""
        return (os.path.exists(self.MODEL_PATH) and 
                os.path.exists(self.SCALER_PATH))
    
    def load_model(self) -> bool:
        """
        Load pre-trained model and scaler.
        
        Returns:
            True if model loaded successfully
        """
        try:
            if self.model_exists():
                self.model = joblib.load(self.MODEL_PATH)
                self.scaler = joblib.load(self.SCALER_PATH)
                
                if os.path.exists(self.METRICS_PATH):
                    with open(self.METRICS_PATH, 'r') as f:
                        self.metrics = json.load(f)
                
                self.feature_names = self.data_loader.get_feature_names()
                
                print(f"✅ Loaded pre-trained model from {self.MODEL_PATH}")
                if self.metrics:
                    print(f"   - Model type: {self.metrics.get('model_type', 'Unknown')}")
                    print(f"   - Test accuracy: {self.metrics.get('test_accuracy', 'N/A'):.4f}")
                    print(f"   - F1 score: {self.metrics.get('test_f1', 'N/A'):.4f}")
                
                return True
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def save_model(self) -> None:
        """Save trained model, scaler, and metrics."""
        try:
            joblib.dump(self.model, self.MODEL_PATH)
            joblib.dump(self.scaler, self.SCALER_PATH)
            
            with open(self.METRICS_PATH, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
            
            print(f"✅ Model saved to {self.MODEL_PATH}")
            print(f"✅ Metrics saved to {self.METRICS_PATH}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def _create_objective(self, model_type: str, X: np.ndarray, y: np.ndarray):
        """
        Create Optuna objective function for hyperparameter tuning.
        
        Args:
            model_type: 'xgboost', 'random_forest', or 'svm'
            X: Training features
            y: Training labels
            
        Returns:
            Objective function for Optuna
        """
        def objective(trial):
            if model_type == 'xgboost' and XGBOOST_AVAILABLE:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'random_state': 42,
                    'use_label_encoder': False,
                    'eval_metric': 'logloss'
                }
                model = xgb.XGBClassifier(**params)
                
            elif model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': 42
                }
                model = RandomForestClassifier(**params)
                
            elif model_type == 'svm':
                params = {
                    'C': trial.suggest_float('C', 0.1, 100, log=True),
                    'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                    'probability': True,
                    'random_state': 42
                }
                if params['kernel'] == 'poly':
                    params['degree'] = trial.suggest_int('degree', 2, 5)
                model = SVC(**params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # 5-fold cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
            
            return scores.mean()
        
        return objective
    
    def _tune_model(self, model_type: str, X: np.ndarray, y: np.ndarray, 
                    n_trials: int = 50) -> Tuple[Any, Dict]:
        """
        Tune hyperparameters for a model type.
        
        Args:
            model_type: Type of model to tune
            X: Training features
            y: Training labels
            n_trials: Number of Optuna trials
            
        Returns:
            Tuple of (best_model, best_params)
        """
        if not OPTUNA_AVAILABLE:
            print(f"⚠️ Optuna not available, using default parameters for {model_type}")
            return self._create_default_model(model_type), {}
        
        print(f"\n🔧 Tuning {model_type}...")
        
        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        objective = self._create_objective(model_type, X, y)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"   Best F1 score: {best_score:.4f}")
        print(f"   Best params: {best_params}")
        
        # Create model with best params
        if model_type == 'xgboost' and XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(
                **best_params,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif model_type == 'random_forest':
            model = RandomForestClassifier(**best_params, random_state=42)
        elif model_type == 'svm':
            model = SVC(**best_params, probability=True, random_state=42)
        
        return model, best_params
    
    def _create_default_model(self, model_type: str) -> Any:
        """Create model with default parameters."""
        if model_type == 'xgboost' and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'svm':
            return SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42
            )
    
    def _evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
    
    def _generate_shap_explanations(self, model: Any, X: np.ndarray, 
                                     feature_names: List[str]) -> None:
        """
        Generate SHAP explanations and save plot.
        
        Args:
            model: Trained model
            X: Feature data for explanation
            feature_names: List of feature names
        """
        if not SHAP_AVAILABLE:
            print("⚠️ SHAP not available, skipping explanations")
            return
        
        print("\n📊 Generating SHAP explanations...")
        
        try:
            # Create explainer based on model type
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                self.shap_explainer = shap.TreeExplainer(model)
            else:
                # Kernel SHAP for other models (slower but general)
                background = shap.sample(X, min(100, len(X)))
                self.shap_explainer = shap.KernelExplainer(model.predict_proba, background)
            
            # Calculate SHAP values (use subset for speed)
            X_sample = X[:min(500, len(X))]
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
            
            # Create summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, 
                X_sample, 
                feature_names=feature_names,
                show=False,
                plot_size=(10, 8)
            )
            plt.tight_layout()
            plt.savefig(self.SHAP_PLOT_PATH, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ SHAP summary plot saved to {self.SHAP_PLOT_PATH}")
            
        except Exception as e:
            print(f"⚠️ Error generating SHAP explanations: {e}")
    
    def train(self, n_trials: int = 50, force: bool = False) -> Dict[str, Any]:
        """
        Train models with hyperparameter optimization.
        
        Args:
            n_trials: Number of Optuna trials per model
            force: If True, retrain even if model exists
            
        Returns:
            Dictionary with training results
        """
        # Check for existing model
        if self.model_exists() and not force:
            print("\n✅ Pre-trained model found. Loading...")
            self.load_model()
            return self.metrics
        
        print("\n" + "="*60)
        print("🚀 Starting Model Training Pipeline")
        print("="*60)
        
        # Load data
        print("\n📁 Loading datasets...")
        X_train, X_test, y_train, y_test = self.data_loader.get_train_test_data()
        self.feature_names = list(X_train.columns)
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Features: {len(self.feature_names)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Models to compare
        model_types = ['random_forest']
        if XGBOOST_AVAILABLE:
            model_types.insert(0, 'xgboost')  # XGBoost first if available
        model_types.append('svm')
        
        # Train and compare models
        results = {}
        best_model = None
        best_score = 0
        best_model_type = None
        best_params = {}
        
        for model_type in model_types:
            print(f"\n{'='*40}")
            print(f"Training: {model_type.upper()}")
            print('='*40)
            
            # Tune hyperparameters
            model, params = self._tune_model(
                model_type, X_train_scaled, y_train, 
                n_trials=n_trials if OPTUNA_AVAILABLE else 1
            )
            
            # Train on full training set
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            metrics = self._evaluate_model(model, X_test_scaled, y_test)
            results[model_type] = {
                'model': model,
                'params': params,
                'metrics': metrics
            }
            
            print(f"\n📈 {model_type} Results:")
            print(f"   Accuracy:  {metrics['accuracy']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall:    {metrics['recall']:.4f}")
            print(f"   F1 Score:  {metrics['f1']:.4f}")
            print(f"   ROC AUC:   {metrics['roc_auc']:.4f}")
            
            # Track best model by F1 score
            if metrics['f1'] > best_score:
                best_score = metrics['f1']
                best_model = model
                best_model_type = model_type
                best_params = params
        
        # Select best model
        print(f"\n{'='*60}")
        print(f"🏆 Best Model: {best_model_type.upper()}")
        print(f"   F1 Score: {best_score:.4f}")
        print('='*60)
        
        self.model = best_model
        
        # Store metrics
        self.metrics = {
            'model_type': best_model_type,
            'best_params': best_params,
            'test_accuracy': results[best_model_type]['metrics']['accuracy'],
            'test_precision': results[best_model_type]['metrics']['precision'],
            'test_recall': results[best_model_type]['metrics']['recall'],
            'test_f1': results[best_model_type]['metrics']['f1'],
            'test_roc_auc': results[best_model_type]['metrics']['roc_auc'],
            'feature_names': self.feature_names,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'trained_at': datetime.now().isoformat(),
            'all_model_results': {
                model_type: res['metrics'] 
                for model_type, res in results.items()
            }
        }
        
        # Generate SHAP explanations
        self._generate_shap_explanations(self.model, X_train_scaled, self.feature_names)
        
        # Save model and metrics
        self.save_model()
        
        # Print classification report
        y_pred = self.model.predict(X_test_scaled)
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                     target_names=['Low Risk', 'High Risk']))
        
        return self.metrics
    
    def predict(self, features: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Make prediction on input features.
        
        Args:
            features: Input features (1D or 2D array)
            
        Returns:
            Tuple of (prediction, probability, confidence_interval)
        """
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("No trained model available. Call train() first.")
        
        # Ensure 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        # Calculate confidence interval using model's inherent uncertainty
        # For ensemble models, use the variance of tree predictions
        if hasattr(self.model, 'estimators_'):
            tree_preds = np.array([
                tree.predict_proba(features_scaled)[0, 1] 
                for tree in self.model.estimators_
            ])
            std = np.std(tree_preds)
            confidence_interval = np.array([
                max(0, probability[1] - 1.96 * std),
                min(1, probability[1] + 1.96 * std)
            ])
        else:
            # For non-ensemble models, use a simpler approach
            confidence_interval = np.array([
                max(0, probability[1] - 0.1),
                min(1, probability[1] + 0.1)
            ])
        
        return prediction, probability[1], confidence_interval
    
    def predict_with_explanation(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make prediction with SHAP explanation.
        
        Args:
            features: Input features
            
        Returns:
            Dictionary with prediction and explanation
        """
        prediction, probability, confidence = self.predict(features)
        
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'confidence_interval': confidence.tolist(),
            'risk_level': 'High' if probability > 0.5 else 'Low',
            'risk_percentage': probability * 100
        }
        
        # Generate SHAP explanation if available
        if SHAP_AVAILABLE and self.shap_explainer is not None:
            try:
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                shap_values = self.shap_explainer.shap_values(features_scaled)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                # Get feature contributions
                contributions = {}
                for i, (name, value, shap_val) in enumerate(
                    zip(self.feature_names, features.flatten(), shap_values.flatten())
                ):
                    contributions[name] = {
                        'value': float(value),
                        'shap_value': float(shap_val),
                        'impact': 'Increases risk' if shap_val > 0 else 'Decreases risk'
                    }
                
                result['feature_contributions'] = contributions
                
            except Exception as e:
                result['explanation_error'] = str(e)
        
        return result
    
    def get_shap_plot(self, features: np.ndarray) -> Optional[str]:
        """
        Generate SHAP waterfall plot for a single prediction.
        
        Args:
            features: Input features
            
        Returns:
            Path to saved plot or None
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return None
        
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            shap_values = self.shap_explainer.shap_values(features_scaled)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Create waterfall plot
            plt.figure(figsize=(10, 6))
            
            # Sort features by absolute SHAP value
            indices = np.argsort(np.abs(shap_values.flatten()))[::-1]
            
            values = shap_values.flatten()[indices]
            names = [self.feature_names[i] for i in indices]
            
            colors = ['#ff6b6b' if v > 0 else '#4ecdc4' for v in values]
            
            plt.barh(range(len(values)), values, color=colors)
            plt.yticks(range(len(values)), names)
            plt.xlabel('SHAP Value (Impact on Prediction)')
            plt.title('Feature Contributions to Prediction')
            plt.tight_layout()
            
            plot_path = os.path.join(self.models_dir, 'current_prediction_shap.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            print(f"Error generating SHAP plot: {e}")
            return None
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            self.load_model()
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_).flatten()
        else:
            return {}
        
        return dict(zip(self.feature_names, importances))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if not self.metrics:
            self.load_model()
        
        return {
            'model_type': self.metrics.get('model_type', 'Unknown'),
            'accuracy': self.metrics.get('test_accuracy', 0),
            'f1_score': self.metrics.get('test_f1', 0),
            'roc_auc': self.metrics.get('test_roc_auc', 0),
            'training_samples': self.metrics.get('training_samples', 0),
            'trained_at': self.metrics.get('trained_at', 'Unknown'),
            'feature_names': self.feature_names
        }


def ensure_model_trained(data_dir: str = "data", n_trials: int = 50) -> ModelTrainer:
    """
    Convenience function to ensure a model is trained and ready.
    
    Args:
        data_dir: Directory for datasets
        n_trials: Number of Optuna trials
        
    Returns:
        ModelTrainer with trained model
    """
    trainer = ModelTrainer(data_dir=data_dir)
    trainer.train(n_trials=n_trials)
    return trainer


if __name__ == "__main__":
    # Test the model trainer
    print("Testing ModelTrainer...")
    trainer = ModelTrainer()
    
    # Train with fewer trials for testing
    metrics = trainer.train(n_trials=10)
    
    print("\n📊 Training complete!")
    print(f"Model: {metrics['model_type']}")
    print(f"F1 Score: {metrics['test_f1']:.4f}")
    
    # Test prediction
    print("\n🔮 Testing prediction...")
    dummy_features = np.random.rand(16) * 0.1  # Random features
    result = trainer.predict_with_explanation(dummy_features)
    print(f"Prediction: {result['risk_level']} ({result['risk_percentage']:.1f}%)")
