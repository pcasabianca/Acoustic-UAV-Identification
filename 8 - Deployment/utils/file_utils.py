"""
Utility functions for file operations.
"""

from pathlib import Path
from typing import List


def ensure_directories(directories: List[Path]) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def get_files_by_pattern(directory: Path, pattern: str) -> List[Path]:
    """
    Get all files matching a pattern in a directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern (e.g., '*.wav')
        
    Returns:
        List of matching file paths
    """
    return sorted(Path(directory).glob(pattern))
