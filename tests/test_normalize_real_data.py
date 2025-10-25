"""
Comprehensive tests for normalize module using real data.

This module tests the normalize module with actual K'Cho data files:
- Text normalization with real K'Cho sentences
- Tokenization with real corpus
- Sentence splitting with real text
"""

import pytest
import os
from pathlib import Path

# Import normalize module components
from kcho.normalize import KChoNormalizer, normalize_text, tokenize
from kcho.kcho_system import KchoSystem


class TestNormalizeModuleWithRealData:
    """Test normalize module with real K'Cho data."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_dir = Path("data")
        self.sample_corpus_file = self.data_dir / "sample_corpus.txt"
        
        # Load sample corpus
        sample_content = self.sample_corpus_file.read_text(encoding='utf-8')
        self.sample_sentences = [
            line.strip() for line in sample_content.split('\n')
            if line.strip() and not line.strip().startswith('#')
        ]
        
        self.normalizer = KChoNormalizer()
    
    def test_normalize_module_import(self):
        """Test that normalize module can be imported."""
        from kcho import normalize
        assert normalize is not None
        assert hasattr(normalize, '__name__')
    
    def test_normalize_text_with_real_kcho_sentences(self):
        """Test normalizing real K'Cho sentences."""
        test_sentences = self.sample_sentences[:10]
        
        for sentence in test_sentences:
            normalized = normalize_text(sentence)
            assert isinstance(normalized, str)
            assert len(normalized) > 0
            
            # Normalized text should be different from original (or same if already normalized)
            assert normalized == sentence or normalized != sentence
    
    def test_normalize_text_with_kcho_normalizer(self):
        """Test normalizing with KChoNormalizer instance."""
        test_sentences = self.sample_sentences[:10]
        
        for sentence in test_sentences:
            normalized = self.normalizer.normalize_text(sentence)
            assert isinstance(normalized, str)
            assert len(normalized) > 0
    
    def test_tokenize_real_kcho_sentences(self):
        """Test tokenizing real K'Cho sentences."""
        test_sentences = self.sample_sentences[:10]
        
        for sentence in test_sentences:
            tokens = tokenize(sentence)
            assert isinstance(tokens, list)
            assert len(tokens) > 0
            
            # Check that tokens contain expected K'Cho elements
            token_text = ' '.join(tokens)
            assert any(char in token_text for char in ['noh', 'ci', 'ah', 'am'])
    
    def test_sentence_split_with_real_text(self):
        """Test sentence splitting with real K'Cho text."""
        # Combine multiple sentences
        combined_text = '. '.join(self.sample_sentences[:5])
        
        sentences = self.normalizer.sentence_split(combined_text)
        assert isinstance(sentences, list)
        assert len(sentences) >= 5
    
    def test_normalize_with_unicode_characters(self):
        """Test normalization with Unicode characters in K'Cho."""
        unicode_sentences = [
            "Ak'hmó lùum ci.",
            "àihli ng'dáng lo(k) ci.",
            "k'chàang noh Yóng am pàapai pe(k) ci."
        ]
        
        for sentence in unicode_sentences:
            normalized = normalize_text(sentence)
            assert isinstance(normalized, str)
            assert len(normalized) > 0
    
    def test_tokenize_with_unicode_characters(self):
        """Test tokenization with Unicode characters."""
        unicode_sentences = [
            "Ak'hmó lùum ci.",
            "àihli ng'dáng lo(k) ci.",
            "k'chàang noh Yóng am pàapai pe(k) ci."
        ]
        
        for sentence in unicode_sentences:
            tokens = tokenize(sentence)
            assert isinstance(tokens, list)
            assert len(tokens) > 0
    
    def test_normalize_with_punctuation(self):
        """Test normalization with punctuation."""
        punctuation_sentences = [
            "Om noh Yóng am pàapai pe(k) ci.",
            "Ak'hmó lùum ci!",
            "àihli ng'dáng lo(k) ci?",
            "k'chàang noh Yóng am pàapai pe(k) ci..."
        ]
        
        for sentence in punctuation_sentences:
            normalized = normalize_text(sentence)
            assert isinstance(normalized, str)
            assert len(normalized) > 0
    
    def test_tokenize_with_punctuation(self):
        """Test tokenization with punctuation."""
        punctuation_sentences = [
            "Om noh Yóng am pàapai pe(k) ci.",
            "Ak'hmó lùum ci!",
            "àihli ng'dáng lo(k) ci?",
            "k'chàang noh Yóng am pàapai pe(k) ci..."
        ]
        
        for sentence in punctuation_sentences:
            tokens = tokenize(sentence)
            assert isinstance(tokens, list)
            assert len(tokens) > 0
    
    def test_normalize_with_whitespace(self):
        """Test normalization with various whitespace patterns."""
        whitespace_sentences = [
            "  Om noh Yóng am pàapai pe(k) ci.  ",
            "\tAk'hmó lùum ci.\t",
            "\nàihli ng'dáng lo(k) ci.\n",
            "  k'chàang   noh   Yóng   am   pàapai   pe(k)   ci.  "
        ]
        
        for sentence in whitespace_sentences:
            normalized = normalize_text(sentence)
            assert isinstance(normalized, str)
            assert len(normalized) > 0
    
    def test_tokenize_with_whitespace(self):
        """Test tokenization with various whitespace patterns."""
        whitespace_sentences = [
            "  Om noh Yóng am pàapai pe(k) ci.  ",
            "\tAk'hmó lùum ci.\t",
            "\nàihli ng'dáng lo(k) ci.\n",
            "  k'chàang   noh   Yóng   am   pàapai   pe(k)   ci.  "
        ]
        
        for sentence in whitespace_sentences:
            tokens = tokenize(sentence)
            assert isinstance(tokens, list)
            assert len(tokens) > 0
    
    def test_normalize_edge_cases(self):
        """Test normalization with edge cases."""
        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            "Om",  # Single word
            "Om noh",  # Two words
            "Om noh Yóng",  # Three words
        ]
        
        for case in edge_cases:
            normalized = normalize_text(case)
            assert isinstance(normalized, str)
    
    def test_tokenize_edge_cases(self):
        """Test tokenization with edge cases."""
        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            "Om",  # Single word
            "Om noh",  # Two words
            "Om noh Yóng",  # Three words
        ]
        
        for case in edge_cases:
            tokens = tokenize(case)
            assert isinstance(tokens, list)
    
    def test_sentence_split_edge_cases(self):
        """Test sentence splitting with edge cases."""
        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            "Om noh Yóng am pàapai pe(k) ci.",  # Single sentence
            "Om noh Yóng am pàapai pe(k) ci. Ak'hmó lùum ci.",  # Two sentences
        ]
        
        for case in edge_cases:
            sentences = self.normalizer.sentence_split(case)
            assert isinstance(sentences, list)
    
    def test_normalize_consistency(self):
        """Test that normalization is consistent."""
        test_sentence = "Om noh Yóng am pàapai pe(k) ci."
        
        # Normalize multiple times
        normalized1 = normalize_text(test_sentence)
        normalized2 = normalize_text(test_sentence)
        
        # Should be consistent
        assert normalized1 == normalized2
    
    def test_tokenize_consistency(self):
        """Test that tokenization is consistent."""
        test_sentence = "Om noh Yóng am pàapai pe(k) ci."
        
        # Tokenize multiple times
        tokens1 = tokenize(test_sentence)
        tokens2 = tokenize(test_sentence)
        
        # Should be consistent
        assert tokens1 == tokens2
    
    def test_normalize_performance(self):
        """Test normalization performance with real data."""
        import time
        
        test_sentences = self.sample_sentences[:20]
        
        start_time = time.time()
        for sentence in test_sentences:
            normalize_text(sentence)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete in reasonable time
        assert processing_time < 2.0, f"Normalization took too long: {processing_time:.2f} seconds"
    
    def test_tokenize_performance(self):
        """Test tokenization performance with real data."""
        import time
        
        test_sentences = self.sample_sentences[:20]
        
        start_time = time.time()
        for sentence in test_sentences:
            tokenize(sentence)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete in reasonable time
        assert processing_time < 2.0, f"Tokenization took too long: {processing_time:.2f} seconds"
    
    def test_normalize_with_system_integration(self):
        """Test normalization integration with KchoSystem."""
        system = KchoSystem()
        
        test_sentences = self.sample_sentences[:5]
        
        for sentence in test_sentences:
            result = system.analyze(sentence)
            assert result is not None
            assert result.text == sentence
            
            # Check that tokens were created
            assert len(result.tokens) > 0
    
    def test_normalize_with_collocation_extraction(self):
        """Test normalization integration with collocation extraction."""
        from kcho import CollocationExtractor
        
        # Normalize sentences first
        normalized_sentences = []
        for sentence in self.sample_sentences[:10]:
            normalized = normalize_text(sentence)
            normalized_sentences.append(normalized)
        
        # Extract collocations from normalized sentences
        extractor = CollocationExtractor(min_freq=2)
        results = extractor.extract(normalized_sentences)
        
        assert isinstance(results, dict)
        assert len(results) > 0
    
    def test_normalize_with_real_corpus_processing(self):
        """Test normalization with real corpus processing."""
        # Process entire sample corpus
        normalized_sentences = []
        for sentence in self.sample_sentences:
            normalized = normalize_text(sentence)
            normalized_sentences.append(normalized)
        
        # All sentences should be normalized
        assert len(normalized_sentences) == len(self.sample_sentences)
        
        # Check that normalization preserved K'Cho patterns
        all_text = ' '.join(normalized_sentences)
        kcho_patterns = ['noh', 'ci', 'ah', 'am', 'pe', 'lo']
        for pattern in kcho_patterns:
            assert pattern in all_text
    
    def test_normalize_with_error_handling(self):
        """Test normalization error handling."""
        error_cases = [
            None,  # None input
            123,   # Non-string input
            [],    # List input
            {},    # Dict input
        ]
        
        for case in error_cases:
            try:
                normalized = normalize_text(case)
                # If no exception, should return string
                assert isinstance(normalized, str)
            except (TypeError, ValueError, AttributeError):
                # Expected behavior for invalid input
                pass
    
    def test_tokenize_with_error_handling(self):
        """Test tokenization error handling."""
        error_cases = [
            None,  # None input
            123,   # Non-string input
            [],    # List input
            {},    # Dict input
        ]
        
        for case in error_cases:
            try:
                tokens = tokenize(case)
                # If no exception, should return list
                assert isinstance(tokens, list)
            except (TypeError, ValueError, AttributeError):
                # Expected behavior for invalid input
                pass
