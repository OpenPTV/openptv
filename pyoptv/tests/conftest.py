import pytest
import sys
import os
from pathlib import Path

def find_project_root():
    """Find the project root by looking for pyproject.toml."""
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path(__file__).parent.parent

# Add project src directory to path for development testing
project_root = find_project_root()
src_path = project_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

@pytest.fixture
def test_data_dir():
    """Return the path to test data directory."""
    # Try multiple locations for testing_fodder
    candidates = [
        Path(__file__).parent / "testing_fodder",
        project_root / "testing_fodder", 
        project_root / "py_bind" / "testing_fodder",
        project_root.parent / "py_bind" / "testing_fodder"
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    # If none found, return the expected location
    return Path(__file__).parent / "testing_fodder"

@pytest.fixture  
def calibration_data_dir(test_data_dir):
    """Return the calibration test data directory."""
    return test_data_dir / "calibration"
