"""
Tests for Voice Analyzer Module

Tests audio loading, feature extraction, chunking, and fallback behavior.
"""

import pytest
import os
import numpy as np
import tempfile
import shutil
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.voice_analyzer import VoiceAnalyzer, VoiceFeatures, PARSELMOUTH_AVAILABLE, LIBROSA_AVAILABLE


class TestVoiceFeatures:
    """Test suite for VoiceFeatures dataclass."""
    
    def test_default_values(self):
        """Test that VoiceFeatures has correct defaults."""
        features = VoiceFeatures()
        
        assert features.jitter_percent == 0.0
        assert features.shimmer == 0.0
        assert features.hnr == 0.0
        assert features.nhr == 0.0
        assert features.extraction_method == "unknown"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        features = VoiceFeatures(
            jitter_percent=1.5,
            shimmer=0.05,
            hnr=22.0,
            nhr=0.02
        )
        
        result = features.to_dict()
        
        assert isinstance(result, dict)
        assert result['Jitter(%)'] == 1.5
        assert result['Shimmer'] == 0.05
        assert result['HNR'] == 22.0
        assert result['NHR'] == 0.02
        assert len(result) == 16  # All 16 features
    
    def test_to_array(self):
        """Test conversion to numpy array."""
        features = VoiceFeatures(
            jitter_percent=1.5,
            shimmer=0.05
        )
        
        result = features.to_array()
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 16
        assert result[0] == 1.5  # Jitter(%)
        assert result[5] == 0.05  # Shimmer


class TestVoiceAnalyzer:
    """Test suite for VoiceAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a VoiceAnalyzer instance."""
        return VoiceAnalyzer()
    
    def test_is_available(self, analyzer):
        """Test availability check."""
        # Should be available if either library is installed
        expected = PARSELMOUTH_AVAILABLE or LIBROSA_AVAILABLE
        assert analyzer.is_available() == expected
    
    def test_extraction_method_detection(self, analyzer):
        """Test that extraction method is correctly detected."""
        if PARSELMOUTH_AVAILABLE:
            assert analyzer.extraction_method == "parselmouth"
        elif LIBROSA_AVAILABLE:
            assert analyzer.extraction_method == "librosa_fallback"
        else:
            assert analyzer.extraction_method == "none"
    
    def test_supported_formats(self, analyzer):
        """Test that supported formats are defined."""
        assert '.wav' in analyzer.SUPPORTED_FORMATS
        assert '.mp3' in analyzer.SUPPORTED_FORMATS
        assert '.ogg' in analyzer.SUPPORTED_FORMATS
        assert '.flac' in analyzer.SUPPORTED_FORMATS
    
    def test_chunk_audio_short(self, analyzer):
        """Test chunking of short audio."""
        # Create short audio (less than chunk duration)
        sr = 22050
        duration = 5.0  # 5 seconds
        audio = np.random.randn(int(sr * duration))
        
        chunks = analyzer._chunk_audio(audio, sr)
        
        assert len(chunks) == 1
        assert len(chunks[0]) == len(audio)
    
    def test_chunk_audio_long(self, analyzer):
        """Test chunking of long audio."""
        # Create long audio (more than chunk duration)
        sr = 22050
        duration = 25.0  # 25 seconds = 2.5 chunks
        audio = np.random.randn(int(sr * duration))
        
        chunks = analyzer._chunk_audio(audio, sr)
        
        assert len(chunks) == 3  # Should have 3 chunks
        assert len(chunks[0]) == int(analyzer.CHUNK_DURATION * sr)
    
    def test_aggregate_chunk_features_single(self, analyzer):
        """Test aggregation of single chunk."""
        features = [{'jitter_percent': 1.0, 'shimmer': 0.05}]
        
        result = analyzer._aggregate_chunk_features(features)
        
        assert result['jitter_percent'] == 1.0
        assert result['shimmer'] == 0.05
    
    def test_aggregate_chunk_features_multiple(self, analyzer):
        """Test aggregation of multiple chunks."""
        features = [
            {'jitter_percent': 1.0, 'shimmer': 0.04},
            {'jitter_percent': 2.0, 'shimmer': 0.06}
        ]
        
        result = analyzer._aggregate_chunk_features(features)
        
        assert result['jitter_percent'] == 1.5  # Mean of 1.0 and 2.0
        assert result['shimmer'] == 0.05  # Mean of 0.04 and 0.06
    
    def test_get_feature_descriptions(self, analyzer):
        """Test feature descriptions."""
        descriptions = analyzer.get_feature_descriptions()
        
        assert isinstance(descriptions, dict)
        assert 'Jitter(%)' in descriptions
        assert 'Shimmer' in descriptions
        assert 'HNR' in descriptions
        assert len(descriptions) == 16
    
    def test_file_not_found(self, analyzer):
        """Test handling of non-existent file."""
        with pytest.raises(FileNotFoundError):
            analyzer.extract_features("nonexistent_file.wav")
    
    def test_unsupported_format(self, analyzer, tmp_path):
        """Test handling of unsupported format."""
        # Create a dummy file with unsupported extension
        dummy_file = tmp_path / "test.xyz"
        dummy_file.write_text("dummy content")
        
        with pytest.raises(ValueError, match="Unsupported audio format"):
            analyzer.extract_features(str(dummy_file))
    
    @pytest.mark.skipif(not LIBROSA_AVAILABLE, reason="Librosa not available")
    def test_estimate_rpde(self, analyzer):
        """Test RPDE estimation."""
        audio = np.random.randn(22050)  # 1 second of audio
        sr = 22050
        
        rpde = analyzer._estimate_rpde(audio, sr)
        
        assert 0.0 <= rpde <= 1.0
    
    @pytest.mark.skipif(not LIBROSA_AVAILABLE, reason="Librosa not available")
    def test_estimate_dfa(self, analyzer):
        """Test DFA estimation."""
        audio = np.random.randn(22050)
        sr = 22050
        
        dfa = analyzer._estimate_dfa(audio, sr)
        
        assert 0.0 <= dfa <= 2.0
    
    @pytest.mark.skipif(not LIBROSA_AVAILABLE, reason="Librosa not available")
    def test_estimate_ppe(self, analyzer):
        """Test PPE estimation."""
        audio = np.random.randn(22050)
        sr = 22050
        
        ppe = analyzer._estimate_ppe(audio, sr)
        
        assert 0.0 <= ppe <= 1.0


class TestVoiceAnalyzerIntegration:
    """Integration tests requiring actual audio libraries."""
    
    @pytest.fixture
    def analyzer(self):
        return VoiceAnalyzer()
    
    @pytest.fixture
    def sample_audio(self, tmp_path):
        """Create a sample audio file for testing."""
        if not LIBROSA_AVAILABLE:
            pytest.skip("Librosa not available")
        
        import soundfile as sf
        
        # Generate a simple sine wave
        sr = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))
        frequency = 200  # 200 Hz tone
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Add some noise
        audio += 0.1 * np.random.randn(len(audio))
        
        filepath = tmp_path / "test_audio.wav"
        sf.write(str(filepath), audio, sr)
        
        return str(filepath)
    
    @pytest.mark.skipif(not LIBROSA_AVAILABLE, reason="Librosa not available")
    def test_extract_features_from_file(self, analyzer, sample_audio):
        """Test feature extraction from actual audio file."""
        if not analyzer.is_available():
            pytest.skip("No audio processing library available")
        
        features = analyzer.extract_features(sample_audio)
        
        assert isinstance(features, VoiceFeatures)
        assert features.duration_seconds > 0
        assert features.sample_rate > 0
        assert features.extraction_method != "unknown"
    
    @pytest.mark.skipif(not LIBROSA_AVAILABLE, reason="Librosa not available")
    def test_extract_features_from_array(self, analyzer):
        """Test feature extraction from numpy array."""
        if not analyzer.is_available():
            pytest.skip("No audio processing library available")
        
        sr = 22050
        duration = 3.0
        audio = np.random.randn(int(sr * duration)) * 0.5
        
        features = analyzer.extract_features_from_array(audio, sr)
        
        assert isinstance(features, VoiceFeatures)
        assert features.duration_seconds == pytest.approx(duration, rel=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
