"""
Parkinson's Disease Predictor - Source Package

This package contains modules for:
- data_loader: Dataset downloading and preprocessing
- voice_analyzer: Audio biomarker extraction
- model_trainer: ML pipeline with hyperparameter tuning
- database: Session history management
"""

from .data_loader import DataLoader
from .voice_analyzer import VoiceAnalyzer
from .model_trainer import ModelTrainer
from .database import DatabaseManager

__version__ = "2.0.0"
__all__ = ["DataLoader", "VoiceAnalyzer", "ModelTrainer", "DatabaseManager"]
