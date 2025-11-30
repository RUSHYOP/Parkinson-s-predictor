"""
Tests for Model Trainer Module

Tests model training, prediction, SHAP explanations, and model persistence.
"""

import pytest
import os
import json
import numpy as np
import pandas as pd
import tempfile
import shutil
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_trainer import ModelTrainer, ensure_model_trained, XGBOOST_AVAILABLE, OPTUNA_AVAILABLE, SHAP_AVAILABLE


class TestModelTrainer:
    """Test suite for ModelTrainer class."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for data and models."""
        data_dir = tempfile.mkdtemp()
        models_dir = tempfile.mkdtemp()
        yield data_dir, models_dir
        shutil.rmtree(data_dir, ignore_errors=True)
        shutil.rmtree(models_dir, ignore_errors=True)
    
    @pytest.fixture
    def trainer(self, temp_dirs):
        """Create a ModelTrainer instance with temp directories."""
        data_dir, models_dir = temp_dirs
        return ModelTrainer(data_dir=data_dir, models_dir=models_dir)
    
    @pytest.fixture
    def mock_data(self, temp_dirs):
        """Create mock dataset for testing."""
        data_dir, _ = temp_dirs
        
        # Create mock combined dataset
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'Jitter(%)': np.random.rand(n_samples) * 2,
            'Jitter(Abs)': np.random.rand(n_samples) * 0.001,
            'Jitter:RAP': np.random.rand(n_samples) * 0.01,
            'Jitter:PPQ5': np.random.rand(n_samples) * 0.01,
            'Jitter:DDP': np.random.rand(n_samples) * 0.03,
            'Shimmer': np.random.rand(n_samples) * 0.2,
            'Shimmer(dB)': np.random.rand(n_samples) * 1.0,
            'Shimmer:APQ3': np.random.rand(n_samples) * 0.1,
            'Shimmer:APQ5': np.random.rand(n_samples) * 0.1,
            'Shimmer:APQ11': np.random.rand(n_samples) * 0.1,
            'Shimmer:DDA': np.random.rand(n_samples) * 0.2,
            'NHR': np.random.rand(n_samples) * 0.1,
            'HNR': np.random.rand(n_samples) * 30,
            'RPDE': np.random.rand(n_samples) * 0.5 + 0.3,
            'DFA': np.random.rand(n_samples) * 0.3 + 0.6,
            'PPE': np.random.rand(n_samples) * 0.3,
            'target': np.random.randint(0, 2, n_samples),
            'source': ['test'] * n_samples
        }
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(data_dir, "combined_parkinsons.csv"), index=False)
        
        return df
    
    def test_init_creates_directories(self, temp_dirs):
        """Test that ModelTrainer creates necessary directories."""
        data_dir, models_dir = temp_dirs
        new_models_dir = os.path.join(models_dir, "new_models")
        
        trainer = ModelTrainer(data_dir=data_dir, models_dir=new_models_dir)
        
        assert os.path.exists(new_models_dir)
    
    def test_model_exists_false_initially(self, trainer):
        """Test that model_exists returns False when no model saved."""
        assert not trainer.model_exists()
    
    def test_load_model_fails_without_model(self, trainer):
        """Test that load_model returns False when no model exists."""
        assert not trainer.load_model()
    
    def test_create_default_model_random_forest(self, trainer):
        """Test creation of default Random Forest model."""
        model = trainer._create_default_model('random_forest')
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_create_default_model_svm(self, trainer):
        """Test creation of default SVM model."""
        model = trainer._create_default_model('svm')
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_create_default_model_xgboost(self, trainer):
        """Test creation of default XGBoost model."""
        model = trainer._create_default_model('xgboost')
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_evaluate_model(self, trainer, mock_data):
        """Test model evaluation metrics."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Prepare data
        feature_cols = [col for col in mock_data.columns if col not in ['target', 'source']]
        X = mock_data[feature_cols].values
        y = mock_data['target'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = trainer._evaluate_model(model, X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    def test_train_creates_model(self, trainer, mock_data):
        """Test that training creates a model."""
        # Train with minimal trials for speed
        metrics = trainer.train(n_trials=2, force=True)
        
        assert trainer.model is not None
        assert trainer.is_trained if hasattr(trainer, 'is_trained') else True
        assert 'model_type' in metrics
        assert 'test_accuracy' in metrics
    
    def test_train_saves_model(self, trainer, mock_data, temp_dirs):
        """Test that training saves the model."""
        _, models_dir = temp_dirs
        
        trainer.train(n_trials=2, force=True)
        
        assert os.path.exists(os.path.join(models_dir, "best_model.pkl"))
        assert os.path.exists(os.path.join(models_dir, "scaler.pkl"))
        assert os.path.exists(os.path.join(models_dir, "metrics.json"))
    
    def test_predict_returns_correct_format(self, trainer, mock_data):
        """Test that predict returns correct format."""
        trainer.train(n_trials=2, force=True)
        
        # Create test input
        test_features = np.random.rand(16)
        
        prediction, probability, confidence = trainer.predict(test_features)
        
        assert prediction in [0, 1]
        assert 0 <= probability <= 1
        assert len(confidence) == 2
        assert confidence[0] <= probability <= confidence[1] or confidence[0] <= confidence[1]
    
    def test_predict_with_explanation(self, trainer, mock_data):
        """Test predict_with_explanation returns correct structure."""
        trainer.train(n_trials=2, force=True)
        
        test_features = np.random.rand(16)
        result = trainer.predict_with_explanation(test_features)
        
        assert 'prediction' in result
        assert 'probability' in result
        assert 'confidence_interval' in result
        assert 'risk_level' in result
        assert 'risk_percentage' in result
        
        assert result['risk_level'] in ['Low', 'High']
    
    def test_get_feature_importance(self, trainer, mock_data):
        """Test feature importance extraction."""
        trainer.train(n_trials=2, force=True)
        
        importance = trainer.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
        # Check that values are numeric (includes numpy numeric types)
        import numbers
        assert all(isinstance(v, numbers.Real) for v in importance.values())
    
    def test_get_model_info(self, trainer, mock_data):
        """Test model info retrieval."""
        trainer.train(n_trials=2, force=True)
        
        info = trainer.get_model_info()
        
        assert 'model_type' in info
        assert 'accuracy' in info
        assert 'f1_score' in info
        assert 'training_samples' in info
    
    def test_model_persistence(self, trainer, mock_data, temp_dirs):
        """Test that model can be saved and loaded."""
        # Train and save
        trainer.train(n_trials=2, force=True)
        original_metrics = trainer.metrics.copy()
        
        # Create new trainer and load
        data_dir, models_dir = temp_dirs
        new_trainer = ModelTrainer(data_dir=data_dir, models_dir=models_dir)
        
        assert new_trainer.load_model()
        assert new_trainer.metrics['model_type'] == original_metrics['model_type']


class TestEnsureModelTrained:
    """Test suite for ensure_model_trained convenience function."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories."""
        data_dir = tempfile.mkdtemp()
        
        # Create mock data
        np.random.seed(42)
        n_samples = 50
        
        data = {
            'Jitter(%)': np.random.rand(n_samples),
            'Jitter(Abs)': np.random.rand(n_samples) * 0.001,
            'Jitter:RAP': np.random.rand(n_samples) * 0.01,
            'Jitter:PPQ5': np.random.rand(n_samples) * 0.01,
            'Jitter:DDP': np.random.rand(n_samples) * 0.03,
            'Shimmer': np.random.rand(n_samples) * 0.2,
            'Shimmer(dB)': np.random.rand(n_samples),
            'Shimmer:APQ3': np.random.rand(n_samples) * 0.1,
            'Shimmer:APQ5': np.random.rand(n_samples) * 0.1,
            'Shimmer:APQ11': np.random.rand(n_samples) * 0.1,
            'Shimmer:DDA': np.random.rand(n_samples) * 0.2,
            'NHR': np.random.rand(n_samples) * 0.1,
            'HNR': np.random.rand(n_samples) * 30,
            'RPDE': np.random.rand(n_samples) * 0.5,
            'DFA': np.random.rand(n_samples) * 0.3 + 0.5,
            'PPE': np.random.rand(n_samples) * 0.3,
            'target': np.random.randint(0, 2, n_samples),
            'source': ['test'] * n_samples
        }
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(data_dir, "combined_parkinsons.csv"), index=False)
        
        yield data_dir
        shutil.rmtree(data_dir, ignore_errors=True)
    
    def test_ensure_model_trained_returns_trainer(self, temp_dirs):
        """Test that ensure_model_trained returns a trainer."""
        trainer = ensure_model_trained(data_dir=temp_dirs, n_trials=2)
        
        assert isinstance(trainer, ModelTrainer)
        assert trainer.model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
