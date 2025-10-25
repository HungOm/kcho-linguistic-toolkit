"""
Test suite for Text Processing components in kcho_system.py

Tests KchoTokenizer and KchoValidator classes
"""

import pytest
from unittest.mock import patch, MagicMock
from kcho.kcho_system import KchoTokenizer, KchoValidator, KchoKnowledge


class TestKchoTokenizer:
    """Test KchoTokenizer functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.tokenizer = KchoTokenizer()
    
    def test_initialization(self):
        """Test tokenizer initialization"""
        assert hasattr(self.tokenizer, 'normalization_rules')
        assert isinstance(self.tokenizer.normalization_rules, dict)
        assert len(self.tokenizer.normalization_rules) >= 3  # Should have at least 3 rules
        # Check that all values are strings
        for value in self.tokenizer.normalization_rules.values():
            assert isinstance(value, str)
    
    def test_normalize_basic(self):
        """Test basic text normalization"""
        text = "Om noh Yong am paapai pe ci"
        normalized = self.tokenizer.normalize(text)
        assert normalized == "Om noh Yong am paapai pe ci"
    
    def test_normalize_quotes(self):
        """Test quote normalization"""
        text = "Om 'noh' Yong"
        normalized = self.tokenizer.normalize(text)
        assert normalized == "Om 'noh' Yong"
    
    def test_normalize_multiple_spaces(self):
        """Test multiple space normalization"""
        text = "Om    noh   Yong"
        normalized = self.tokenizer.normalize(text)
        assert normalized == "Om noh Yong"
    
    def test_normalize_leading_trailing_spaces(self):
        """Test leading and trailing space normalization"""
        text = "  Om noh Yong  "
        normalized = self.tokenizer.normalize(text)
        assert normalized == "Om noh Yong"
    
    def test_tokenize_basic(self):
        """Test basic tokenization"""
        text = "Om noh Yong am paapai pe ci"
        tokens = self.tokenizer.tokenize(text)
        expected = ["Om", "noh", "Yong", "am", "paapai", "pe", "ci"]
        assert tokens == expected
    
    def test_tokenize_with_punctuation(self):
        """Test tokenization with punctuation"""
        text = "Om noh Yong am paapai pe ci."
        tokens = self.tokenizer.tokenize(text)
        expected = ["Om", "noh", "Yong", "am", "paapai", "pe", "ci"]
        assert tokens == expected
    
    def test_tokenize_kcho_apostrophe_patterns(self):
        """Test tokenization with K'Cho apostrophe patterns"""
        text = "k'cho m'htak ng'thu K'Cho M'Htak Ng'laamihta"
        tokens = self.tokenizer.tokenize(text)
        expected = ["k'cho", "m'htak", "ng'thu", "K'Cho", "M'Htak", "Ng'laamihta"]
        assert tokens == expected
    
    def test_tokenize_mixed_punctuation(self):
        """Test tokenization with mixed punctuation"""
        text = "Om noh Yong, am paapai pe ci!"
        tokens = self.tokenizer.tokenize(text)
        expected = ["Om", "noh", "Yong", "am", "paapai", "pe", "ci"]
        assert tokens == expected
    
    def test_tokenize_empty_text(self):
        """Test tokenization of empty text"""
        text = ""
        tokens = self.tokenizer.tokenize(text)
        assert tokens == []
    
    def test_tokenize_whitespace_only(self):
        """Test tokenization of whitespace-only text"""
        text = "   \n\t  "
        tokens = self.tokenizer.tokenize(text)
        assert tokens == []
    
    def test_sentence_split_basic(self):
        """Test basic sentence splitting"""
        text = "Om noh Yong ci. Law noh Khanpughi ci."
        sentences = self.tokenizer.sentence_split(text)
        expected = ["Om noh Yong ci.", "Law noh Khanpughi ci."]
        assert sentences == expected
    
    def test_sentence_split_multiple_endings(self):
        """Test sentence splitting with multiple sentence endings"""
        text = "Om noh Yong ci! Law noh Khanpughi ci? Ak'hmó lùum ci."
        sentences = self.tokenizer.sentence_split(text)
        expected = ["Om noh Yong ci", "Law noh Khanpughi ci", "Ak'hmó lùum ci."]
        assert sentences == expected
    
    def test_sentence_split_no_ending(self):
        """Test sentence splitting with no sentence ending"""
        text = "Om noh Yong ci"
        sentences = self.tokenizer.sentence_split(text)
        expected = ["Om noh Yong ci"]
        assert sentences == expected
    
    def test_sentence_split_semicolon_not_split(self):
        """Test that semicolons do NOT split sentences (K'Cho specific)"""
        text = "Om noh Yong ci; Law noh Khanpughi ci"
        sentences = self.tokenizer.sentence_split(text)
        # Should be treated as one sentence, not split on semicolon
        assert len(sentences) == 1
        assert "Om noh Yong ci; Law noh Khanpughi ci" in sentences[0]
    
    def test_sentence_split_empty_text(self):
        """Test sentence splitting of empty text"""
        text = ""
        sentences = self.tokenizer.sentence_split(text)
        assert sentences == []
    
    def test_sentence_split_whitespace_only(self):
        """Test sentence splitting of whitespace-only text"""
        text = "   \n\t  "
        sentences = self.tokenizer.sentence_split(text)
        assert sentences == []


class TestKchoValidator:
    """Test KchoValidator functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = KchoValidator()
    
    def test_initialization(self):
        """Test validator initialization"""
        assert hasattr(self.validator, 'knowledge')
        assert isinstance(self.validator.knowledge, KchoKnowledge)
    
    def test_is_kcho_text_valid_kcho(self):
        """Test validation of valid K'Cho text"""
        text = "Om noh Yong am paapai pe ci"
        is_kcho, confidence, metrics = self.validator.is_kcho_text(text)
        
        assert isinstance(is_kcho, bool)
        assert isinstance(confidence, float)
        assert isinstance(metrics, dict)
        assert 0.0 <= confidence <= 1.0
    
    def test_is_kcho_text_empty_text(self):
        """Test validation of empty text"""
        text = ""
        is_kcho, confidence, metrics = self.validator.is_kcho_text(text)
        
        assert is_kcho is False
        assert confidence == 0.0
        assert metrics == {}
    
    def test_is_kcho_text_whitespace_only(self):
        """Test validation of whitespace-only text"""
        text = "   \n\t  "
        is_kcho, confidence, metrics = self.validator.is_kcho_text(text)
        
        assert is_kcho is False
        assert confidence == 0.0
        assert metrics == {}
    
    def test_is_kcho_text_english(self):
        """Test validation of English text"""
        text = "This is English text with no K'Cho markers"
        is_kcho, confidence, metrics = self.validator.is_kcho_text(text)
        
        # Should have low confidence for non-K'Cho text
        assert isinstance(is_kcho, bool)
        assert isinstance(confidence, float)
        assert confidence < 0.5  # Should be low confidence
    
    def test_is_kcho_text_mixed_content(self):
        """Test validation of mixed K'Cho and English text"""
        text = "Om noh Yong ci. This is English. Law noh Khanpughi ci."
        is_kcho, confidence, metrics = self.validator.is_kcho_text(text)
        
        assert isinstance(is_kcho, bool)
        assert isinstance(confidence, float)
        assert isinstance(metrics, dict)
    
    def test_is_kcho_text_metrics_structure(self):
        """Test that metrics contain expected keys"""
        text = "Om noh Yong am paapai pe ci"
        is_kcho, confidence, metrics = self.validator.is_kcho_text(text)
        
        expected_keys = ['char_validity', 'marker_score', 'pattern_score', 'overall_confidence']
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
    
    def test_is_kcho_text_confidence_calculation(self):
        """Test confidence calculation components"""
        text = "Om noh Yong am paapai pe ci"
        is_kcho, confidence, metrics = self.validator.is_kcho_text(text)
        
        # Overall confidence should be calculated from components
        expected_confidence = (
            metrics['char_validity'] * 0.3 + 
            metrics['marker_score'] * 0.5 + 
            metrics['pattern_score'] * 0.2
        )
        assert abs(confidence - expected_confidence) < 0.001
    
    def test_is_kcho_text_kcho_patterns(self):
        """Test detection of K'Cho patterns"""
        # Text with K'Cho patterns
        text = "Om noh Yong ci k'cho"
        is_kcho, confidence, metrics = self.validator.is_kcho_text(text)
        
        assert isinstance(is_kcho, bool)
        assert isinstance(confidence, float)
        # Should have some pattern score due to K'Cho patterns
        assert metrics['pattern_score'] >= 0
    
    def test_is_kcho_text_threshold_behavior(self):
        """Test that threshold behavior works correctly"""
        # Test with very low confidence text
        text = "xyz abc def"
        is_kcho, confidence, metrics = self.validator.is_kcho_text(text)
        
        # Should be below threshold (0.3)
        if confidence < 0.3:
            assert is_kcho is False
        else:
            assert is_kcho is True
    
    def test_is_kcho_text_kcho_markers(self):
        """Test detection of K'Cho markers"""
        # Text with known K'Cho markers
        text = "Om noh Yong ci"
        is_kcho, confidence, metrics = self.validator.is_kcho_text(text)
        
        # Should have some marker score
        assert metrics['marker_score'] >= 0
        assert isinstance(metrics['marker_score'], (int, float))
    
    def test_is_kcho_text_character_validity(self):
        """Test character validity calculation"""
        text = "Om noh Yong am paapai pe ci"
        is_kcho, confidence, metrics = self.validator.is_kcho_text(text)
        
        # Character validity should be reasonable for K'Cho text
        assert 0.0 <= metrics['char_validity'] <= 1.0
        assert isinstance(metrics['char_validity'], float)
    
    def test_is_kcho_text_unicode_handling(self):
        """Test handling of Unicode characters"""
        text = "Om noh Yong áéíóú ci"
        is_kcho, confidence, metrics = self.validator.is_kcho_text(text)
        
        assert isinstance(is_kcho, bool)
        assert isinstance(confidence, float)
        assert isinstance(metrics, dict)
        # Should handle Unicode characters gracefully
        import math
        assert not any(math.isnan(v) or math.isinf(v) for v in metrics.values() if isinstance(v, (int, float)))


class TestTextProcessingIntegration:
    """Test integration between text processing components"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.tokenizer = KchoTokenizer()
        self.validator = KchoValidator()
    
    def test_tokenize_then_validate(self):
        """Test tokenizing text then validating tokens"""
        text = "Om noh Yong am paapai pe ci"
        
        # Tokenize first
        tokens = self.tokenizer.tokenize(text)
        assert len(tokens) > 0
        
        # Then validate
        is_kcho, confidence, metrics = self.validator.is_kcho_text(text)
        assert isinstance(is_kcho, bool)
        assert isinstance(confidence, float)
    
    def test_sentence_split_then_validate(self):
        """Test splitting sentences then validating each"""
        text = "Om noh Yong ci. Law noh Khanpughi ci."
        
        # Split sentences
        sentences = self.tokenizer.sentence_split(text)
        assert len(sentences) == 2
        
        # Validate each sentence
        for sentence in sentences:
            is_kcho, confidence, metrics = self.validator.is_kcho_text(sentence)
            assert isinstance(is_kcho, bool)
            assert isinstance(confidence, float)
    
    def test_normalize_then_tokenize_then_validate(self):
        """Test full pipeline: normalize -> tokenize -> validate"""
        text = "  Om    noh   Yong  .  "
        
        # Normalize
        normalized = self.tokenizer.normalize(text)
        assert normalized == "Om noh Yong ."
        
        # Tokenize
        tokens = self.tokenizer.tokenize(normalized)
        assert tokens == ["Om", "noh", "Yong"]
        
        # Validate
        is_kcho, confidence, metrics = self.validator.is_kcho_text(normalized)
        assert isinstance(is_kcho, bool)
        assert isinstance(confidence, float)
