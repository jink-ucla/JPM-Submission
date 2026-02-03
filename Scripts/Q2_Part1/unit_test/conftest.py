"""
Pytest configuration and shared fixtures for Q2 tests.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    yield


@pytest.fixture
def tolerance():
    """Standard numerical tolerance for assertions."""
    return 1e-6
