"""
Utility functions for logging setup and management.
"""

import logging
import time
from pathlib import Path
from datetime import datetime


def cleanup_old_logs(log_dir: Path, days: int = 7) -> int:
    """
    Remove log files older than specified days.
    
    Args:
        log_dir: Directory containing log files
        days: Keep logs from last N days
        
    Returns:
        Number of files removed
    """
    try:
        current_time = time.time()
        cutoff_time = current_time - (days * 86400)  # days in seconds
        
        removed_count = 0
        for log_file in log_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                removed_count += 1
        
        if removed_count > 0:
            print(f"[CLEANUP] Removed {removed_count} old log file(s)")
        
        return removed_count
    except Exception as e:
        print(f"[WARN] Log cleanup failed: {e}")
        return 0


def setup_logger(name: str, log_dir: Path, log_level: str = 'INFO', 
                console_output: bool = True) -> logging.Logger:
    """
    Configure and return a logger with file and optional console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        console_output: Enable console output
        
    Returns:
        Configured logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Cleanup old logs (keep only last 7 days)
    cleanup_old_logs(log_dir, days=7)
    
    level = getattr(logging, log_level.upper())
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    log_file = log_dir / f"detector_{datetime.now().strftime('%Y%m%d')}.log"
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(fh)
    if console_output:
        logger.addHandler(ch)
    
    return logger
