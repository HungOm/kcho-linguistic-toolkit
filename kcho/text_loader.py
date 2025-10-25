"""
Universal text loader for K'Cho toolkit.

Handles loading K'Cho text from various file formats (txt, json) with automatic structure detection.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union
import re

logger = logging.getLogger(__name__)


class TextLoader:
    """Universal text loader for K'Cho files."""
    
    @staticmethod
    def load_from_file(file_path: str) -> List[str]:
        """
        Load K'Cho sentences from any text or JSON file.
        
        Args:
            file_path: Path to input file
            
        Returns:
            List of K'Cho sentences
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() == '.txt':
            return TextLoader.load_from_txt(str(file_path))
        elif file_path.suffix.lower() == '.json':
            return TextLoader.load_from_json(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    @staticmethod
    def load_from_txt(file_path: str) -> List[str]:
        """
        Load from plain text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of sentences
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by lines and filter empty lines
        sentences = [line.strip() for line in content.split('\n') if line.strip()]
        
        logger.info(f"Loaded {len(sentences)} sentences from {file_path}")
        return sentences
    
    @staticmethod
    def load_from_json(file_path: str, structure_type: str = 'auto') -> List[str]:
        """
        Load from JSON file with automatic structure detection.
        
        Args:
            file_path: Path to JSON file
            structure_type: 'auto', 'bible', or 'simple'
            
        Returns:
            List of sentences
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if structure_type == 'auto':
            structure_type = TextLoader._detect_json_structure(data)
        
        if structure_type == 'nested':
            return TextLoader.load_from_json_nested(data)
        elif structure_type == 'simple':
            return TextLoader.load_from_json_simple(data)
        else:
            raise ValueError(f"Unknown JSON structure type: {structure_type}")
    
    @staticmethod
    def load_from_json_nested(data: Union[Dict, str]) -> List[str]:
        """
        Extract text from nested JSON structure (like hierarchical data).
        
        Args:
            data: JSON data or file path
            
        Returns:
            List of sentences
        """
        if isinstance(data, str):
            with open(data, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        sentences = []
        
        def extract_text_recursive(obj, path=""):
            """Recursively extract text from nested structure."""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    extract_text_recursive(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    current_path = f"{path}[{i}]" if path else f"[{i}]"
                    extract_text_recursive(item, current_path)
            elif isinstance(obj, str) and obj.strip():
                # Check if this looks like K'Cho text
                if TextLoader._is_kcho_text(obj):
                    sentences.append(obj.strip())
        
        extract_text_recursive(data)
        
        logger.info(f"Extracted {len(sentences)} K'Cho sentences from nested JSON structure")
        return sentences
    
    @staticmethod
    def load_from_json_simple(data: Union[Dict, List, str]) -> List[str]:
        """
        Load from simple JSON structure (list of strings or dict with text fields).
        
        Args:
            data: JSON data or file path
            
        Returns:
            List of sentences
        """
        if isinstance(data, str):
            with open(data, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        sentences = []
        
        if isinstance(data, list):
            # List of strings
            for item in data:
                if isinstance(item, str) and TextLoader._is_kcho_text(item):
                    sentences.append(item.strip())
        elif isinstance(data, dict):
            # Dictionary with text fields
            for key, value in data.items():
                if isinstance(value, str) and TextLoader._is_kcho_text(value):
                    sentences.append(value.strip())
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and TextLoader._is_kcho_text(item):
                            sentences.append(item.strip())
        
        logger.info(f"Extracted {len(sentences)} K'Cho sentences from simple JSON structure")
        return sentences
    
    @staticmethod
    def _detect_json_structure(data: Any) -> str:
        """
        Detect JSON structure type automatically.
        
        Args:
            data: JSON data
            
        Returns:
            Structure type ('nested' or 'simple')
        """
        if isinstance(data, dict):
            # Check for nested structure with numeric keys or verse-like keys
            for key, value in data.items():
                if isinstance(value, dict):
                    sub_keys = list(value.keys())
                    # Check for numeric keys or verse-like patterns (s1, 1, 2, etc.)
                    if any(re.match(r'^[s]?\d+$', k) for k in sub_keys):
                        return 'nested'
                    # Also check if any sub-value is a dict (deeper nesting)
                    if any(isinstance(v, dict) for v in value.values()):
                        return 'nested'
        
        return 'simple'
    
    @staticmethod
    def _is_kcho_text(text: str) -> bool:
        """
        Check if text appears to be K'Cho text.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be K'Cho
        """
        if not text or len(text.strip()) < 3:
            return False
        
        # Check for K'Cho-specific characters and patterns
        kcho_indicators = [
            'ah', 'ci', 'noh', 'am', 'ung', 'kh', 'ng', 'hm', 'k\'', 'ng\'',
            'Khanpughi', 'khomdek', 'tüisho', 'akdei', 'akhmüp'
        ]
        
        text_lower = text.lower()
        indicator_count = sum(1 for indicator in kcho_indicators if indicator in text_lower)
        
        # If we find multiple K'Cho indicators, likely K'Cho text
        return indicator_count >= 2 or any(indicator in text_lower for indicator in ['ah', 'ci', 'noh'])
    
    @staticmethod
    def load_from_csv(file_path: str, text_column: str = 'Text_Cho') -> List[str]:
        """
        Load K'Cho text from CSV file.
        
        Args:
            file_path: Path to CSV file
            text_column: Name of column containing K'Cho text
            
        Returns:
            List of sentences
        """
        import csv
        
        sentences = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if text_column in row and TextLoader._is_kcho_text(row[text_column]):
                    sentences.append(row[text_column].strip())
        
        logger.info(f"Loaded {len(sentences)} K'Cho sentences from CSV {file_path}")
        return sentences
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """
        Get information about a text file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        info = {
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'file_extension': file_path.suffix.lower(),
            'exists': True
        }
        
        # Try to load and analyze content
        try:
            if file_path.suffix.lower() == '.txt':
                sentences = TextLoader.load_from_txt(str(file_path))
                info['sentence_count'] = len(sentences)
                info['avg_sentence_length'] = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            elif file_path.suffix.lower() == '.json':
                sentences = TextLoader.load_from_json(str(file_path))
                info['sentence_count'] = len(sentences)
                info['avg_sentence_length'] = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            else:
                info['sentence_count'] = 'unknown'
                info['avg_sentence_length'] = 'unknown'
        except Exception as e:
            info['error'] = str(e)
            info['sentence_count'] = 'error'
            info['avg_sentence_length'] = 'error'
        
        return info
