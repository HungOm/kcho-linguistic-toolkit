"""
Unit tests for text normalization functionality.

Tests the KChoNormalizer class and normalization functions.
"""

import pytest
from kcho.normalize import KChoNormalizer, normalize_text, tokenize


class TestKChoNormalizer:
    """Test the KChoNormalizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = KChoNormalizer()
    
    def test_initialization(self):
        """Test KChoNormalizer initialization."""
        assert isinstance(self.normalizer, KChoNormalizer)
        assert hasattr(self.normalizer, 'normalize_text')
        assert hasattr(self.normalizer, 'tokenize')
    
    def test_normalize_basic(self):
        """Test basic text normalization."""
        text = "Om noh Yong am paapai pe ci"
        normalized = self.normalizer.normalize_text(text)
        
        # Should return a string
        assert isinstance(normalized, str)
        # Should not be empty
        assert len(normalized) > 0
    
    def test_normalize_empty_string(self):
        """Test normalization of empty string."""
        normalized = self.normalizer.normalize_text("")
        assert normalized == ""
    
    def test_normalize_whitespace(self):
        """Test normalization of whitespace."""
        text = "  Om   noh   Yong  "
        normalized = self.normalizer.normalize_text(text)
        
        # Should handle whitespace properly
        assert isinstance(normalized, str)
        # Should not have excessive whitespace
        assert "  " not in normalized.strip()
    
    def test_tokenize_basic(self):
        """Test basic tokenization."""
        text = "Om noh Yong am paapai pe ci"
        tokens = self.normalizer.tokenize(text)
        
        # Should return a list
        assert isinstance(tokens, list)
        # Should have tokens
        assert len(tokens) > 0
        # All tokens should be strings
        for token in tokens:
            assert isinstance(token, str)
    
    def test_tokenize_empty_string(self):
        """Test tokenization of empty string."""
        tokens = self.normalizer.tokenize("")
        assert tokens == []
    
    def test_tokenize_single_word(self):
        """Test tokenization of single word."""
        tokens = self.normalizer.tokenize("Om")
        assert tokens == ["Om"]
    
    def test_tokenize_with_punctuation(self):
        """Test tokenization with punctuation."""
        text = "Om noh Yong, am paapai pe ci!"
        tokens = self.normalizer.tokenize(text)
        
        # Should handle punctuation
        assert isinstance(tokens, list)
        assert len(tokens) > 0
    
    def test_normalize_special_characters(self):
        """Test normalization of special characters."""
        text = "Ak'hmó lùum ci"
        normalized = self.normalizer.normalize_text(text)
        
        # Should handle special characters
        assert isinstance(normalized, str)
        assert len(normalized) > 0
    
    def test_tokenize_special_characters(self):
        """Test tokenization of special characters."""
        text = "Ak'hmó lùum ci"
        tokens = self.normalizer.tokenize(text)
        
        # Should handle special characters
        assert isinstance(tokens, list)
        assert len(tokens) > 0
    
    def test_consistency(self):
        """Test consistency of normalization and tokenization."""
        text = "Om noh Yong am paapai pe ci"
        
        # Multiple calls should give consistent results
        norm1 = self.normalizer.normalize_text(text)
        norm2 = self.normalizer.normalize_text(text)
        assert norm1 == norm2
        
        tokens1 = self.normalizer.tokenize(text)
        tokens2 = self.normalizer.tokenize(text)
        assert tokens1 == tokens2


class TestNormalizeText:
    """Test the normalize_text function."""
    
    def test_normalize_text_basic(self):
        """Test basic normalize_text function."""
        text = "Om noh Yong am paapai pe ci"
        normalized = normalize_text(text)
        
        # Should return a string
        assert isinstance(normalized, str)
        # Should not be empty
        assert len(normalized) > 0
    
    def test_normalize_text_empty(self):
        """Test normalize_text with empty string."""
        normalized = normalize_text("")
        assert normalized == ""
    
    def test_normalize_text_none(self):
        """Test normalize_text with None."""
        normalized = normalize_text(None)
        assert normalized == ""
    
    def test_normalize_text_whitespace(self):
        """Test normalize_text with whitespace."""
        text = "  Om   noh   Yong  "
        normalized = normalize_text(text)
        
        # Should handle whitespace
        assert isinstance(normalized, str)
        assert len(normalized) > 0


class TestTokenize:
    """Test the tokenize function."""
    
    def test_tokenize_basic(self):
        """Test basic tokenize function."""
        text = "Om noh Yong am paapai pe ci"
        tokens = tokenize(text)
        
        # Should return a list
        assert isinstance(tokens, list)
        # Should have tokens
        assert len(tokens) > 0
        # All tokens should be strings
        for token in tokens:
            assert isinstance(token, str)
    
    def test_tokenize_empty(self):
        """Test tokenize with empty string."""
        tokens = tokenize("")
        assert tokens == []
    
    def test_tokenize_none(self):
        """Test tokenize with None."""
        tokens = tokenize(None)
        assert tokens == []
    
    def test_tokenize_single_word(self):
        """Test tokenize with single word."""
        tokens = tokenize("Om")
        assert tokens == ["Om"]
    
    def test_tokenize_special_characters(self):
        """Test tokenize with special characters."""
        text = "Ak'hmó lùum ci"
        tokens = tokenize(text)
        
        # Should handle special characters
        assert isinstance(tokens, list)
        assert len(tokens) > 0


@pytest.mark.integration
class TestNormalizationIntegration:
    """Integration tests for normalization."""
    
    def test_full_normalization_pipeline(self):
        """Test complete normalization pipeline."""
        texts = [
            "Om noh Yong am paapai pe ci",
            "Ak'hmó lùum ci",
            "Om noh Yong am paapai pe ci",
            "Ak'hmó lùum ci",
            "Om noh Yong am paapai pe ci"
        ]
        
        normalizer = KChoNormalizer()
        
        for text in texts:
            # Test normalization
            normalized = normalizer.normalize_text(text)
            assert isinstance(normalized, str)
            
            # Test tokenization
            tokens = normalizer.tokenize(text)
            assert isinstance(tokens, list)
            
            # Test standalone functions
            norm_func = normalize_text(text)
            tokens_func = tokenize(text)
            
            assert isinstance(norm_func, str)
            assert isinstance(tokens_func, list)
    
    def test_edge_cases(self):
        """Test various edge cases."""
        normalizer = KChoNormalizer()
        
        edge_cases = [
            "",                    # Empty string
            " ",                   # Single space
            "   ",                 # Multiple spaces
            "Om",                  # Single word
            "Om noh",              # Two words
            "Om noh Yong am paapai pe ci",  # Normal sentence
            "Ak'hmó lùum ci",      # Special characters
            "Om, noh Yong!",       # Punctuation
            "Om\nnoh\nYong",      # Newlines
            "Om\tnoh\tYong",       # Tabs
        ]
        
        for text in edge_cases:
            # Should not raise exceptions
            normalized = normalizer.normalize_text(text)
            tokens = normalizer.tokenize(text)
            
            assert isinstance(normalized, str)
            assert isinstance(tokens, list)
    
    def test_unicode_handling(self):
        """Test Unicode character handling."""
        normalizer = KChoNormalizer()
        
        unicode_texts = [
            "Ak'hmó lùum ci",      # Accented characters
            "Om noh Yong",        # Regular characters
            "Ak'hmó lùum ci",     # Mixed
        ]
        
        for text in unicode_texts:
            normalized = normalizer.normalize_text(text)
            tokens = normalizer.tokenize(text)
            
            assert isinstance(normalized, str)
            assert isinstance(tokens, list)
            
            # Should preserve Unicode characters
            if "hmó" in text:
                assert "hmó" in normalized or "hmó" in " ".join(tokens)
    
    def test_performance(self):
        """Test performance with larger texts."""
        normalizer = KChoNormalizer()
        
        # Create a larger text
        base_text = "Om noh Yong am paapai pe ci "
        large_text = base_text * 100  # 100 repetitions
        
        # Should handle large text efficiently
        normalized = normalizer.normalize_text(large_text)
        tokens = normalizer.tokenize(large_text)
        
        assert isinstance(normalized, str)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
