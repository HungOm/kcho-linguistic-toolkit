"""
Configuration system for K'Cho toolkit.

Provides default configurations and user-overridable settings via YAML/JSON config files.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class DataSourcesConfig:
    """Configuration for data source paths."""
    gold_standard_collocations: str = 'data/gold_standard_collocations.txt'
    gold_standard_corpus: str = 'data/gold_standard_kcho_english.json'
    sample_corpus: str = 'data/sample_corpus.txt'
    custom_corpus: str = 'data/custom_corpus.txt'
    aligned_corpus: str = 'data/parallel_corpora/aligned_cho_english.csv'
    linguistic_data: str = 'kcho/data/linguistic_data.json'


@dataclass
class CollocationConfig:
    """Configuration for collocation extraction."""
    window_size: int = 5
    min_freq: int = 5
    measures: list = None
    linguistic_patterns: list = None
    
    def __post_init__(self):
        if self.measures is None:
            self.measures = ['pmi', 'tscore', 'dice', 'log_likelihood', 'npmi']
        if self.linguistic_patterns is None:
            self.linguistic_patterns = ['VP', 'PP', 'APP', 'AGR', 'AUX', 'COMP', 'MWE']


@dataclass
class OutputConfig:
    """Configuration for output formats."""
    formats: list = None
    include_metadata: bool = True
    output_dir: str = 'output'
    
    def __post_init__(self):
        if self.formats is None:
            self.formats = ['csv', 'json', 'txt']


@dataclass
class KchoConfig:
    """Main configuration class."""
    data_sources: DataSourcesConfig = None
    collocation: CollocationConfig = None
    output: OutputConfig = None
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = DataSourcesConfig()
        if self.collocation is None:
            self.collocation = CollocationConfig()
        if self.output is None:
            self.output = OutputConfig()


def load_config(config_path: Optional[str] = None) -> KchoConfig:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
        
    Returns:
        KchoConfig instance
    """
    # Start with defaults
    config = KchoConfig()
    
    if config_path and Path(config_path).exists():
        config_path = Path(config_path)
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # Merge user config with defaults
        config = _merge_config(config, user_config)
    
    # Apply environment variable overrides
    config = _apply_env_overrides(config)
    
    return config


def _merge_config(default_config: KchoConfig, user_config: Dict[str, Any]) -> KchoConfig:
    """Merge user configuration with defaults."""
    config_dict = asdict(default_config)
    
    def deep_merge(base_dict: dict, update_dict: dict) -> dict:
        """Deep merge two dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict
    
    merged_dict = deep_merge(config_dict, user_config)
    
    # Reconstruct config objects
    return KchoConfig(
        data_sources=DataSourcesConfig(**merged_dict['data_sources']),
        collocation=CollocationConfig(**merged_dict['collocation']),
        output=OutputConfig(**merged_dict['output'])
    )


def _apply_env_overrides(config: KchoConfig) -> KchoConfig:
    """Apply environment variable overrides."""
    # Data source overrides
    if os.getenv('KCHO_CUSTOM_CORPUS'):
        config.data_sources.custom_corpus = os.getenv('KCHO_CUSTOM_CORPUS')
    if os.getenv('KCHO_GOLD_STANDARD'):
        config.data_sources.gold_standard_collocations = os.getenv('KCHO_GOLD_STANDARD')
    
    # Collocation overrides
    if os.getenv('KCHO_WINDOW_SIZE'):
        config.collocation.window_size = int(os.getenv('KCHO_WINDOW_SIZE'))
    if os.getenv('KCHO_MIN_FREQ'):
        config.collocation.min_freq = int(os.getenv('KCHO_MIN_FREQ'))
    
    # Output overrides
    if os.getenv('KCHO_OUTPUT_DIR'):
        config.output.output_dir = os.getenv('KCHO_OUTPUT_DIR')
    
    return config


def save_config_template(output_path: str = 'config.yaml'):
    """Save a configuration template file."""
    config = KchoConfig()
    config_dict = asdict(config)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    print(f"Configuration template saved to {output_path}")


def validate_config(config: KchoConfig) -> bool:
    """Validate configuration settings."""
    errors = []
    
    # Validate data source paths
    for attr_name in dir(config.data_sources):
        if not attr_name.startswith('_'):
            path = getattr(config.data_sources, attr_name)
            if not Path(path).exists():
                errors.append(f"Data source path does not exist: {path}")
    
    # Validate collocation settings
    if config.collocation.window_size < 1:
        errors.append("Window size must be >= 1")
    if config.collocation.min_freq < 1:
        errors.append("Minimum frequency must be >= 1")
    
    # Validate output settings
    valid_formats = ['csv', 'json', 'txt']
    for fmt in config.output.formats:
        if fmt not in valid_formats:
            errors.append(f"Invalid output format: {fmt}")
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


# Default configuration instance
DEFAULT_CONFIG = KchoConfig()
