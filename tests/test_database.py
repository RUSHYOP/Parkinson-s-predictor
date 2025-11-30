"""
Tests for Database Module

Tests SQLite database operations, CRUD functionality, and session management.
"""

import pytest
import os
import json
import tempfile
import shutil
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import DatabaseManager, PredictionSession, get_db


class TestDatabaseManager:
    """Test suite for DatabaseManager class."""
    
    @pytest.fixture
    def temp_db_dir(self):
        """Create a temporary directory for database."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def db(self, temp_db_dir):
        """Create a DatabaseManager instance with temp database."""
        db_path = os.path.join(temp_db_dir, "test_history.db")
        return DatabaseManager(db_path=db_path)
    
    def test_init_creates_database(self, temp_db_dir):
        """Test that DatabaseManager creates database file."""
        db_path = os.path.join(temp_db_dir, "new_history.db")
        db = DatabaseManager(db_path=db_path)
        
        assert os.path.exists(db_path)
    
    def test_init_creates_tables(self, db):
        """Test that tables are created on init."""
        with db.get_session() as session:
            # Try to query - should not raise error
            result = session.query(PredictionSession).all()
            assert result == []
    
    def test_save_prediction_returns_id(self, db):
        """Test that save_prediction returns a valid ID."""
        session_id = db.save_prediction(
            input_type='symptom',
            features={'test': 1.0},
            risk_score=0.75,
            risk_level='High'
        )
        
        assert session_id is not None
        assert isinstance(session_id, int)
        assert session_id > 0
    
    def test_save_prediction_stores_data(self, db):
        """Test that save_prediction correctly stores data."""
        features = {'Jitter(%)': 1.5, 'Shimmer': 0.05}
        
        session_id = db.save_prediction(
            input_type='voice',
            features=features,
            risk_score=0.65,
            risk_level='Moderate',
            confidence_interval=[0.55, 0.75],
            audio_filename='test.wav',
            audio_duration=5.0,
            notes='Test prediction'
        )
        
        # Retrieve and verify
        prediction = db.get_prediction(session_id)
        
        assert prediction is not None
        assert prediction['input_type'] == 'voice'
        assert prediction['risk_level'] == 'Moderate'
        assert prediction['risk_percentage'] == 65.0
        assert prediction['audio_filename'] == 'test.wav'
        assert prediction['notes'] == 'Test prediction'
        assert prediction['features'] == features
    
    def test_get_prediction_returns_none_for_invalid_id(self, db):
        """Test that get_prediction returns None for invalid ID."""
        result = db.get_prediction(99999)
        assert result is None
    
    def test_get_history_returns_list(self, db):
        """Test that get_history returns a list."""
        history = db.get_history()
        assert isinstance(history, list)
    
    def test_get_history_ordered_by_timestamp(self, db):
        """Test that history is ordered by timestamp descending."""
        # Create multiple predictions
        for i in range(3):
            db.save_prediction(
                input_type='symptom',
                features={'val': i},
                risk_score=0.5,
                risk_level='Moderate'
            )
        
        history = db.get_history()
        
        assert len(history) == 3
        # Most recent should be first
        if len(history) >= 2:
            assert history[0]['id'] > history[1]['id']
    
    def test_get_history_respects_limit(self, db):
        """Test that get_history respects limit parameter."""
        # Create 5 predictions
        for i in range(5):
            db.save_prediction(
                input_type='symptom',
                features={},
                risk_score=0.5,
                risk_level='Moderate'
            )
        
        history = db.get_history(limit=3)
        assert len(history) == 3
    
    def test_get_history_filters_by_input_type(self, db):
        """Test filtering history by input type."""
        db.save_prediction(input_type='symptom', features={}, risk_score=0.5, risk_level='Low')
        db.save_prediction(input_type='voice', features={}, risk_score=0.7, risk_level='High')
        db.save_prediction(input_type='symptom', features={}, risk_score=0.6, risk_level='Moderate')
        
        symptom_history = db.get_history(input_type='symptom')
        voice_history = db.get_history(input_type='voice')
        
        assert len(symptom_history) == 2
        assert len(voice_history) == 1
        assert all(h['input_type'] == 'symptom' for h in symptom_history)
    
    def test_get_history_filters_by_risk_level(self, db):
        """Test filtering history by risk level."""
        db.save_prediction(input_type='symptom', features={}, risk_score=0.2, risk_level='Low')
        db.save_prediction(input_type='symptom', features={}, risk_score=0.7, risk_level='High')
        db.save_prediction(input_type='symptom', features={}, risk_score=0.8, risk_level='High')
        
        high_risk = db.get_history(risk_level='High')
        
        assert len(high_risk) == 2
        assert all(h['risk_level'] == 'High' for h in high_risk)
    
    def test_get_history_count(self, db):
        """Test get_history_count returns correct count."""
        for _ in range(5):
            db.save_prediction(input_type='symptom', features={}, risk_score=0.5, risk_level='Moderate')
        
        count = db.get_history_count()
        assert count == 5
    
    def test_update_notes(self, db):
        """Test updating notes on a prediction."""
        session_id = db.save_prediction(
            input_type='symptom',
            features={},
            risk_score=0.5,
            risk_level='Moderate'
        )
        
        success = db.update_notes(session_id, "Updated notes")
        assert success
        
        prediction = db.get_prediction(session_id)
        assert prediction['notes'] == "Updated notes"
    
    def test_update_notes_invalid_id(self, db):
        """Test updating notes with invalid ID returns False."""
        success = db.update_notes(99999, "Notes")
        assert not success
    
    def test_delete_session_soft_delete(self, db):
        """Test that delete_session performs soft delete."""
        session_id = db.save_prediction(
            input_type='symptom',
            features={},
            risk_score=0.5,
            risk_level='Moderate'
        )
        
        success = db.delete_session(session_id)
        assert success
        
        # Should not appear in history
        history = db.get_history()
        assert len(history) == 0
        
        # But should still exist in DB (soft delete)
        with db.get_session() as session:
            prediction = session.query(PredictionSession).filter(
                PredictionSession.id == session_id
            ).first()
            assert prediction is not None
            assert prediction.is_deleted == True
    
    def test_delete_session_invalid_id(self, db):
        """Test deleting with invalid ID returns False."""
        success = db.delete_session(99999)
        assert not success
    
    def test_clear_all(self, db):
        """Test clearing all history."""
        for _ in range(5):
            db.save_prediction(input_type='symptom', features={}, risk_score=0.5, risk_level='Moderate')
        
        count = db.clear_all()
        
        assert count == 5
        assert db.get_history_count() == 0
    
    def test_get_statistics(self, db):
        """Test get_statistics returns correct structure."""
        db.save_prediction(input_type='symptom', features={}, risk_score=0.2, risk_level='Low')
        db.save_prediction(input_type='voice', features={}, risk_score=0.5, risk_level='Moderate')
        db.save_prediction(input_type='symptom', features={}, risk_score=0.8, risk_level='High')
        
        stats = db.get_statistics()
        
        assert stats['total_predictions'] == 3
        assert stats['symptom_assessments'] == 2
        assert stats['voice_analyses'] == 1
        assert stats['risk_distribution']['low'] == 1
        assert stats['risk_distribution']['moderate'] == 1
        assert stats['risk_distribution']['high'] == 1
        assert 0 <= stats['average_risk_score'] <= 1
    
    def test_export_history_json(self, db):
        """Test exporting history as JSON."""
        db.save_prediction(input_type='symptom', features={'test': 1}, risk_score=0.5, risk_level='Moderate')
        
        export = db.export_history(format='json')
        
        data = json.loads(export)
        assert isinstance(data, list)
        assert len(data) == 1
    
    def test_export_history_csv(self, db):
        """Test exporting history as CSV."""
        db.save_prediction(input_type='symptom', features={}, risk_score=0.5, risk_level='Moderate')
        
        export = db.export_history(format='csv')
        
        assert 'ID' in export or 'id' in export.lower()
        assert 'symptom' in export
    
    def test_export_history_invalid_format(self, db):
        """Test exporting with invalid format raises error."""
        with pytest.raises(ValueError, match="Unsupported format"):
            db.export_history(format='xml')
    
    def test_get_recent_predictions(self, db):
        """Test getting recent predictions."""
        for i in range(10):
            db.save_prediction(input_type='symptom', features={'i': i}, risk_score=0.5, risk_level='Moderate')
        
        recent = db.get_recent_predictions(limit=5)
        
        assert len(recent) == 5


class TestPredictionSession:
    """Test suite for PredictionSession model."""
    
    def test_to_dict(self):
        """Test PredictionSession.to_dict() method."""
        session = PredictionSession(
            id=1,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            input_type='voice',
            features_json='{"test": 1}',
            risk_score=0.75,
            risk_level='High',
            risk_percentage=75.0,
            confidence_lower=0.65,
            confidence_upper=0.85,
            audio_filename='test.wav',
            notes='Test notes'
        )
        
        result = session.to_dict()
        
        assert result['id'] == 1
        assert result['input_type'] == 'voice'
        assert result['features'] == {'test': 1}
        assert result['risk_score'] == 0.75
        assert result['risk_level'] == 'High'
        assert result['confidence_interval'] == [0.65, 0.85]
        assert result['audio_filename'] == 'test.wav'
        assert result['notes'] == 'Test notes'
    
    def test_repr(self):
        """Test PredictionSession string representation."""
        session = PredictionSession(
            id=1,
            input_type='symptom',
            risk_level='High'
        )
        
        repr_str = repr(session)
        
        assert 'id=1' in repr_str
        assert 'symptom' in repr_str
        assert 'High' in repr_str


class TestGetDb:
    """Test suite for get_db convenience function."""
    
    def test_get_db_returns_manager(self, tmp_path):
        """Test that get_db returns a DatabaseManager."""
        db_path = str(tmp_path / "test.db")
        db = get_db(db_path=db_path)
        
        assert isinstance(db, DatabaseManager)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
