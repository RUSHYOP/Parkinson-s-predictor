"""
Database Module for Parkinson's Disease Predictor

SQLAlchemy-based session history management with SQLite backend.
Stores prediction sessions for user history tracking.
"""

import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

Base = declarative_base()


class PredictionSession(Base):
    """
    SQLAlchemy model for storing prediction sessions.
    
    Attributes:
        id: Unique session identifier
        timestamp: When the prediction was made
        input_type: 'symptom' or 'voice'
        features_json: JSON string of input features
        risk_score: Predicted risk probability (0-1)
        risk_level: 'Low', 'Moderate', or 'High'
        risk_percentage: Risk score as percentage
        confidence_lower: Lower bound of confidence interval
        confidence_upper: Upper bound of confidence interval
        shap_values_json: JSON string of SHAP values (if available)
        audio_filename: Original audio filename (for voice input)
        audio_duration: Audio duration in seconds (for voice input)
        notes: User notes about the prediction
    """
    __tablename__ = 'prediction_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    input_type = Column(String(20), nullable=False)  # 'symptom' or 'voice'
    features_json = Column(Text, nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    risk_percentage = Column(Float, nullable=False)
    confidence_lower = Column(Float, nullable=True)
    confidence_upper = Column(Float, nullable=True)
    shap_values_json = Column(Text, nullable=True)
    audio_filename = Column(String(255), nullable=True)
    audio_duration = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    is_deleted = Column(Boolean, default=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'input_type': self.input_type,
            'features': json.loads(self.features_json) if self.features_json else {},
            'risk_score': self.risk_score,
            'risk_level': self.risk_level,
            'risk_percentage': self.risk_percentage,
            'confidence_interval': [self.confidence_lower, self.confidence_upper] 
                                   if self.confidence_lower is not None else None,
            'shap_values': json.loads(self.shap_values_json) if self.shap_values_json else None,
            'audio_filename': self.audio_filename,
            'audio_duration': self.audio_duration,
            'notes': self.notes
        }
    
    def __repr__(self):
        return f"<PredictionSession(id={self.id}, type={self.input_type}, risk={self.risk_level})>"


class DatabaseManager:
    """
    Manages SQLite database for prediction history.
    
    Features:
    - Auto-creates database and tables on first use
    - CRUD operations for prediction sessions
    - Session filtering and search
    - Export functionality
    """
    
    def __init__(self, db_path: str = "data/history.db"):
        """
        Initialize DatabaseManager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Create data directory if needed
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Create engine and session factory
        self.engine = create_engine(
            f'sqlite:///{db_path}',
            echo=False,  # Set to True for SQL debugging
            connect_args={'check_same_thread': False}  # Allow multi-threaded access
        )
        
        self.SessionFactory = sessionmaker(bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        print(f"📁 Database initialized at {db_path}")
    
    @contextmanager
    def get_session(self) -> Session:
        """
        Get a database session with automatic cleanup.
        
        Usage:
            with db.get_session() as session:
                session.query(...)
        """
        session = self.SessionFactory()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def save_prediction(
        self,
        input_type: str,
        features: Dict[str, float],
        risk_score: float,
        risk_level: str,
        confidence_interval: Optional[List[float]] = None,
        shap_values: Optional[Dict[str, float]] = None,
        audio_filename: Optional[str] = None,
        audio_duration: Optional[float] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        Save a prediction session to the database.
        
        Args:
            input_type: 'symptom' or 'voice'
            features: Dictionary of input features
            risk_score: Predicted risk probability (0-1)
            risk_level: 'Low', 'Moderate', or 'High'
            confidence_interval: [lower, upper] bounds
            shap_values: Feature contribution values
            audio_filename: Original audio filename
            audio_duration: Audio duration in seconds
            notes: User notes
            
        Returns:
            ID of the created session
        """
        with self.get_session() as session:
            prediction = PredictionSession(
                timestamp=datetime.utcnow(),
                input_type=input_type,
                features_json=json.dumps(features),
                risk_score=risk_score,
                risk_level=risk_level,
                risk_percentage=risk_score * 100,
                confidence_lower=confidence_interval[0] if confidence_interval else None,
                confidence_upper=confidence_interval[1] if confidence_interval else None,
                shap_values_json=json.dumps(shap_values) if shap_values else None,
                audio_filename=audio_filename,
                audio_duration=audio_duration,
                notes=notes
            )
            
            session.add(prediction)
            session.flush()  # Get the ID before commit
            
            print(f"✅ Saved prediction session #{prediction.id}")
            return prediction.id
    
    def get_prediction(self, session_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a single prediction by ID.
        
        Args:
            session_id: ID of the prediction session
            
        Returns:
            Prediction dictionary or None
        """
        with self.get_session() as session:
            prediction = session.query(PredictionSession).filter(
                PredictionSession.id == session_id,
                PredictionSession.is_deleted == False
            ).first()
            
            return prediction.to_dict() if prediction else None
    
    def get_history(
        self,
        limit: int = 50,
        offset: int = 0,
        input_type: Optional[str] = None,
        risk_level: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get prediction history with optional filtering.
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            input_type: Filter by 'symptom' or 'voice'
            risk_level: Filter by risk level
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            List of prediction dictionaries
        """
        with self.get_session() as session:
            query = session.query(PredictionSession).filter(
                PredictionSession.is_deleted == False
            )
            
            if input_type:
                query = query.filter(PredictionSession.input_type == input_type)
            
            if risk_level:
                query = query.filter(PredictionSession.risk_level == risk_level)
            
            if start_date:
                query = query.filter(PredictionSession.timestamp >= start_date)
            
            if end_date:
                query = query.filter(PredictionSession.timestamp <= end_date)
            
            # Order by most recent first
            query = query.order_by(PredictionSession.timestamp.desc())
            
            # Apply pagination
            predictions = query.offset(offset).limit(limit).all()
            
            return [p.to_dict() for p in predictions]
    
    def get_history_count(
        self,
        input_type: Optional[str] = None,
        risk_level: Optional[str] = None
    ) -> int:
        """
        Get total count of prediction history.
        
        Args:
            input_type: Filter by 'symptom' or 'voice'
            risk_level: Filter by risk level
            
        Returns:
            Total count of predictions
        """
        with self.get_session() as session:
            query = session.query(PredictionSession).filter(
                PredictionSession.is_deleted == False
            )
            
            if input_type:
                query = query.filter(PredictionSession.input_type == input_type)
            
            if risk_level:
                query = query.filter(PredictionSession.risk_level == risk_level)
            
            return query.count()
    
    def update_notes(self, session_id: int, notes: str) -> bool:
        """
        Update notes for a prediction session.
        
        Args:
            session_id: ID of the prediction session
            notes: New notes text
            
        Returns:
            True if updated successfully
        """
        with self.get_session() as session:
            prediction = session.query(PredictionSession).filter(
                PredictionSession.id == session_id
            ).first()
            
            if prediction:
                prediction.notes = notes
                return True
            return False
    
    def delete_session(self, session_id: int) -> bool:
        """
        Soft delete a prediction session.
        
        Args:
            session_id: ID of the prediction session
            
        Returns:
            True if deleted successfully
        """
        with self.get_session() as session:
            prediction = session.query(PredictionSession).filter(
                PredictionSession.id == session_id
            ).first()
            
            if prediction:
                prediction.is_deleted = True
                print(f"🗑️ Deleted prediction session #{session_id}")
                return True
            return False
    
    def clear_all(self, permanent: bool = False) -> int:
        """
        Clear all prediction history.
        
        Args:
            permanent: If True, permanently delete. Otherwise soft delete.
            
        Returns:
            Number of deleted records
        """
        with self.get_session() as session:
            if permanent:
                count = session.query(PredictionSession).delete()
            else:
                count = session.query(PredictionSession).filter(
                    PredictionSession.is_deleted == False
                ).update({'is_deleted': True})
            
            print(f"🗑️ Cleared {count} prediction sessions")
            return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about prediction history.
        
        Returns:
            Dictionary with statistics
        """
        with self.get_session() as session:
            total = session.query(PredictionSession).filter(
                PredictionSession.is_deleted == False
            ).count()
            
            symptom_count = session.query(PredictionSession).filter(
                PredictionSession.is_deleted == False,
                PredictionSession.input_type == 'symptom'
            ).count()
            
            voice_count = session.query(PredictionSession).filter(
                PredictionSession.is_deleted == False,
                PredictionSession.input_type == 'voice'
            ).count()
            
            high_risk = session.query(PredictionSession).filter(
                PredictionSession.is_deleted == False,
                PredictionSession.risk_level == 'High'
            ).count()
            
            moderate_risk = session.query(PredictionSession).filter(
                PredictionSession.is_deleted == False,
                PredictionSession.risk_level == 'Moderate'
            ).count()
            
            low_risk = session.query(PredictionSession).filter(
                PredictionSession.is_deleted == False,
                PredictionSession.risk_level == 'Low'
            ).count()
            
            # Average risk score
            from sqlalchemy import func
            avg_risk = session.query(func.avg(PredictionSession.risk_score)).filter(
                PredictionSession.is_deleted == False
            ).scalar() or 0
            
            return {
                'total_predictions': total,
                'symptom_assessments': symptom_count,
                'voice_analyses': voice_count,
                'risk_distribution': {
                    'high': high_risk,
                    'moderate': moderate_risk,
                    'low': low_risk
                },
                'average_risk_score': float(avg_risk)
            }
    
    def export_history(self, format: str = 'json') -> str:
        """
        Export all prediction history.
        
        Args:
            format: Export format ('json' or 'csv')
            
        Returns:
            Exported data as string
        """
        history = self.get_history(limit=10000)
        
        if format == 'json':
            return json.dumps(history, indent=2, default=str)
        
        elif format == 'csv':
            import csv
            from io import StringIO
            
            output = StringIO()
            if history:
                writer = csv.DictWriter(output, fieldnames=history[0].keys())
                writer.writeheader()
                for row in history:
                    # Flatten nested structures
                    flat_row = {
                        k: json.dumps(v) if isinstance(v, (dict, list)) else v
                        for k, v in row.items()
                    }
                    writer.writerow(flat_row)
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_recent_predictions(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent predictions for quick access.
        
        Args:
            limit: Number of predictions to return
            
        Returns:
            List of recent predictions
        """
        return self.get_history(limit=limit)


# Convenience function for quick database access
def get_db(db_path: str = "data/history.db") -> DatabaseManager:
    """
    Get a DatabaseManager instance.
    
    Args:
        db_path: Path to database file
        
    Returns:
        DatabaseManager instance
    """
    return DatabaseManager(db_path=db_path)


if __name__ == "__main__":
    # Test the database
    print("Testing DatabaseManager...")
    
    db = DatabaseManager()
    
    # Test save
    session_id = db.save_prediction(
        input_type='symptom',
        features={'age': 65, 'tremor': 1, 'bradykinesia': 1},
        risk_score=0.75,
        risk_level='High',
        confidence_interval=[0.65, 0.85],
        notes='Test prediction'
    )
    
    print(f"\nCreated session: {session_id}")
    
    # Test get
    prediction = db.get_prediction(session_id)
    print(f"Retrieved: {prediction}")
    
    # Test history
    history = db.get_history()
    print(f"\nHistory count: {len(history)}")
    
    # Test statistics
    stats = db.get_statistics()
    print(f"\nStatistics: {stats}")
