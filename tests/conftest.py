"""
Pytest configuration and fixtures for Parkinson's Predictor tests.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def sample_features():
    """Sample voice features for testing."""
    return {
        'Jitter(%)': 0.5,
        'Jitter(Abs)': 0.00003,
        'Jitter:RAP': 0.002,
        'Jitter:PPQ5': 0.003,
        'Jitter:DDP': 0.006,
        'Shimmer': 0.03,
        'Shimmer(dB)': 0.3,
        'Shimmer:APQ3': 0.01,
        'Shimmer:APQ5': 0.012,
        'Shimmer:APQ11': 0.015,
        'Shimmer:DDA': 0.03,
        'NHR': 0.02,
        'HNR': 22.0,
        'RPDE': 0.5,
        'DFA': 0.7,
        'PPE': 0.2
    }


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
