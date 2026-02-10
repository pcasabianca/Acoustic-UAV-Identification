"""
Configuration manager for drone detection system.
"""

import json
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: str = "deployment_config.json"):
        """
        Initialize config manager.
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config_path = Path(config_path)
        self.config = self.load()
    
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def reload(self) -> bool:
        """
        Reload configuration from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.config = self.load()
            return True
        except Exception as e:
            print(f"Failed to reload config: {e}")
            return False
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get nested configuration value.
        
        Args:
            *keys: Nested keys (e.g., 'audio', 'sample_rate')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value
    
    def validate(self) -> bool:
        """
        Validate configuration structure.
        
        Returns:
            True if valid
        """
        required_sections = ['audio', 'detection', 'models', 'feature_extraction', 'output']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing config section: {section}")
        
        return True
