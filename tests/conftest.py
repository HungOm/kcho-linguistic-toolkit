"""
Test configuration and utilities for the K'Cho Linguistic Toolkit test suite.
"""

import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def sample_corpus():
    """Provide a sample corpus for testing."""
    return [
        "Om noh Yong am paapai pe ci",
        "Ak'hmó lùum ci",
        "Om noh Yong am paapai pe ci",
        "Ak'hmó lùum ci",
        "Om noh Yong am paapai pe ci"
    ]


@pytest.fixture
def sample_corpus_file():
    """Provide a temporary corpus file for testing."""
    corpus_content = [
        "Om noh Yong am paapai pe ci",
        "Ak'hmó lùum ci",
        "Om noh Yong am paapai pe ci",
        "Ak'hmó lùum ci",
        "Om noh Yong am paapai pe ci"
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for line in corpus_content:
            f.write(line + '\n')
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)


@pytest.fixture
def empty_corpus_file():
    """Provide an empty corpus file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("")
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)


@pytest.fixture
def single_word_corpus():
    """Provide a single word corpus for testing."""
    return ["Om"]


@pytest.fixture
def empty_corpus():
    """Provide an empty corpus for testing."""
    return []


@pytest.fixture
def large_corpus():
    """Provide a larger corpus for testing."""
    base_sentences = [
        "Om noh Yong am paapai pe ci",
        "Ak'hmó lùum ci",
        "Om noh Yong am paapai pe ci"
    ]
    
    large_corpus = []
    for _ in range(50):
        large_corpus.extend(base_sentences)
    
    return large_corpus


@pytest.fixture
def output_file():
    """Provide a temporary output file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)


class TestDataProvider:
    """Provide test data for various scenarios."""
    
    @staticmethod
    def get_kcho_words():
        """Get a list of K'Cho words for testing."""
        return [
            "Om", "noh", "Yong", "am", "paapai", "pe", "ci",
            "Ak'hmó", "lùum", "hmó", "lùum"
        ]
    
    @staticmethod
    def get_kcho_sentences():
        """Get a list of K'Cho sentences for testing."""
        return [
            "Om noh Yong am paapai pe ci",
            "Ak'hmó lùum ci",
            "Om noh Yong am paapai pe ci",
            "Ak'hmó lùum ci",
            "Om noh Yong am paapai pe ci"
        ]
    
    @staticmethod
    def get_mixed_content():
        """Get mixed content for testing edge cases."""
        return [
            "Om noh Yong am paapai pe ci",  # Normal sentence
            "Ak'hmó lùum ci",               # Another normal sentence
            "",                             # Empty sentence
            "Om",                          # Single word
            "Om noh Yong am paapai pe ci",  # Repeat
            "Ak'hmó lùum ci"               # Repeat
        ]
    
    @staticmethod
    def get_special_characters():
        """Get text with special characters for testing."""
        return [
            "Ak'hmó lùum ci",      # Accented characters
            "Om noh Yong",        # Regular characters
            "Ak'hmó lùum ci",     # Mixed
        ]


@pytest.fixture
def test_data_provider():
    """Provide test data provider."""
    return TestDataProvider()


# Markers for different test types
pytestmark = [
    pytest.mark.unit,
]
