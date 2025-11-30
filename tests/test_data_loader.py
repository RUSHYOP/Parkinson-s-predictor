"""
Tests for Data Loader Module

Tests dataset downloading, caching, preprocessing, and feature extraction.
"""

import pytest
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader, load_parkinsons_data


class TestDataLoader:
    """Test suite for DataLoader class."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def loader(self, temp_data_dir):
        """Create a DataLoader instance with temp directory."""
        return DataLoader(data_dir=temp_data_dir)
    
    def test_init_creates_directory(self, temp_data_dir):
        """Test that DataLoader creates data directory."""
        new_dir = os.path.join(temp_data_dir, "new_data")
        loader = DataLoader(data_dir=new_dir)
        assert os.path.exists(new_dir)
    
    def test_voice_features_defined(self, loader):
        """Test that voice features are properly defined."""
        features = loader.get_feature_names()
        assert len(features) == 16
        assert 'Jitter(%)' in features
        assert 'Shimmer' in features
        assert 'HNR' in features
        assert 'NHR' in features
        assert 'RPDE' in features
        assert 'DFA' in features
        assert 'PPE' in features
    
    def test_get_feature_names_returns_copy(self, loader):
        """Test that get_feature_names returns a copy, not the original."""
        features1 = loader.get_feature_names()
        features2 = loader.get_feature_names()
        features1.append("test")
        assert "test" not in features2
    
    @patch('requests.get')
    def test_download_file_success(self, mock_get, loader):
        """Test successful file download."""
        mock_response = MagicMock()
        mock_response.headers = {'content-length': '100'}
        mock_response.iter_content.return_value = [b'test data']
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        content = loader._download_file("http://test.com/file", "Test File")
        assert content == "test data"
    
    @patch('requests.get')
    def test_download_file_failure(self, mock_get, loader):
        """Test download failure handling."""
        mock_get.side_effect = Exception("Network error")
        
        with pytest.raises(Exception):
            loader._download_file("http://test.com/file", "Test File")
    
    def test_extract_voice_features_schema(self, loader):
        """Test that extracted features follow correct schema."""
        # Create mock dataframe with expected columns
        mock_df = pd.DataFrame({
            'Jitter(%)': [0.01, 0.02],
            'Shimmer': [0.03, 0.04],
            'HNR': [20.0, 21.0],
            'NHR': [0.01, 0.02],
            'target': [0, 1]
        })
        
        result = loader._extract_voice_features(mock_df, 'test')
        
        assert 'target' in result.columns
        assert 'source' in result.columns
        assert all(result['source'] == 'test')
    
    def test_cached_dataset_loading(self, loader, temp_data_dir):
        """Test that cached datasets are loaded correctly."""
        # Create a mock cached file
        mock_data = pd.DataFrame({
            'Jitter(%)': [0.01],
            'Shimmer': [0.03],
            'HNR': [20.0],
            'target': [0],
            'source': ['test']
        })
        
        cached_path = os.path.join(temp_data_dir, "combined_parkinsons.csv")
        mock_data.to_csv(cached_path, index=False)
        
        # Should load from cache
        result = loader.get_combined_dataset()
        
        assert len(result) == 1
        assert result['source'].iloc[0] == 'test'
    
    def test_dataset_info_structure(self, loader, temp_data_dir):
        """Test that dataset info has correct structure."""
        # Create mock cached data
        mock_data = pd.DataFrame({
            'Jitter(%)': [0.01, 0.02, 0.03],
            'Shimmer': [0.03, 0.04, 0.05],
            'HNR': [20.0, 21.0, 22.0],
            'target': [0, 1, 1],
            'source': ['uci', 'uci', 'oxford']
        })
        
        cached_path = os.path.join(temp_data_dir, "combined_parkinsons.csv")
        mock_data.to_csv(cached_path, index=False)
        
        info = loader.get_dataset_info()
        
        assert 'total_samples' in info
        assert 'uci_samples' in info
        assert 'oxford_samples' in info
        assert 'num_features' in info
        assert 'feature_names' in info
        assert 'target_distribution' in info
        assert 'class_balance' in info
        
        assert info['total_samples'] == 3
        assert info['uci_samples'] == 2
        assert info['oxford_samples'] == 1


class TestLoadParkinsonsData:
    """Test suite for convenience function."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_convenience_function_returns_dataframe(self, temp_data_dir):
        """Test that load_parkinsons_data returns a DataFrame."""
        # Create mock cached data
        mock_data = pd.DataFrame({
            'Jitter(%)': [0.01],
            'Shimmer': [0.03],
            'HNR': [20.0],
            'target': [0],
            'source': ['test']
        })
        
        cached_path = os.path.join(temp_data_dir, "combined_parkinsons.csv")
        mock_data.to_csv(cached_path, index=False)
        
        result = load_parkinsons_data(data_dir=temp_data_dir)
        
        assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
