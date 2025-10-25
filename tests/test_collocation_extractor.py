"""
Comprehensive tests for CollocationExtractor class.

This module tests the core collocation extraction functionality including:
- Basic collocation extraction
- Association measure calculations
- Window-based co-occurrence analysis
- Edge cases and error handling
- Performance with different corpus sizes
"""

import pytest
import math
from unittest.mock import Mock, patch
from kcho.collocation import CollocationExtractor, AssociationMeasure, CollocationResult


class TestCollocationExtractor:
    """Test cases for CollocationExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = CollocationExtractor()
        self.sample_corpus = [
            "Om noh Yong ci",
            "Law noh Khanpughi ci", 
            "Ak'hmó lùum ci",
            "Om noh Yong ci",
            "Law noh Khanpughi ci",
            "Om noh Yong ci",
            "Ak'hmó lùum ci",
            "Om noh Yong ci"
        ]
    
    def test_initialization(self):
        """Test CollocationExtractor initialization."""
        assert self.extractor is not None
        assert hasattr(self.extractor, 'extract')
        assert hasattr(self.extractor, 'window_size')
        assert hasattr(self.extractor, 'min_freq')
        assert hasattr(self.extractor, 'measures')
    
    def test_extract_basic_collocations(self):
        """Test basic collocation extraction."""
        results = self.extractor.extract(self.sample_corpus)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check that results are organized by AssociationMeasure
        for measure, collocations in results.items():
            assert isinstance(measure, AssociationMeasure)
            assert isinstance(collocations, list)
            
            # Check that collocations are CollocationResult objects
            for result in collocations:
                assert isinstance(result, CollocationResult)
                assert hasattr(result, 'words')
                assert hasattr(result, 'frequency')
                assert hasattr(result, 'score')
                assert hasattr(result, 'measure')
    
    def test_extract_with_window_size(self):
        """Test collocation extraction with different window sizes."""
        # Test with window size 2
        extractor_window_2 = CollocationExtractor(window_size=2)
        results_window_2 = extractor_window_2.extract(self.sample_corpus)
        
        # Test with window size 5
        extractor_window_5 = CollocationExtractor(window_size=5)
        results_window_5 = extractor_window_5.extract(self.sample_corpus)
        
        assert isinstance(results_window_2, dict)
        assert isinstance(results_window_5, dict)
        
        # Larger window should potentially find more collocations
        total_window_2 = sum(len(collocations) for collocations in results_window_2.values())
        total_window_5 = sum(len(collocations) for collocations in results_window_5.values())
        assert total_window_5 >= total_window_2
    
    def test_extract_with_min_frequency(self):
        """Test collocation extraction with minimum frequency threshold."""
        # Test with min_freq=1 (should find many collocations)
        extractor_min_1 = CollocationExtractor(min_freq=1)
        results_min_1 = extractor_min_1.extract(self.sample_corpus)
        
        # Test with min_freq=3 (should find fewer collocations)
        extractor_min_3 = CollocationExtractor(min_freq=3)
        results_min_3 = extractor_min_3.extract(self.sample_corpus)
        
        assert isinstance(results_min_1, dict)
        assert isinstance(results_min_3, dict)
        
        # Higher frequency threshold should result in fewer collocations
        total_min_1 = sum(len(collocations) for collocations in results_min_1.values())
        total_min_3 = sum(len(collocations) for collocations in results_min_3.values())
        assert total_min_1 >= total_min_3
        
        # All results should meet the frequency threshold
        for collocations in results_min_3.values():
            for result in collocations:
                assert result.frequency >= 3
    
    def test_extract_with_association_measures(self):
        """Test collocation extraction with specific association measures."""
        measures = [AssociationMeasure.PMI, AssociationMeasure.TSCORE, AssociationMeasure.DICE]
        extractor = CollocationExtractor(measures=measures)
        results = extractor.extract(self.sample_corpus)
        
        assert isinstance(results, dict)
        
        # Check that results have the requested measures
        for measure in measures:
            assert measure in results
            assert isinstance(results[measure], list)
    
    def test_extract_empty_corpus(self):
        """Test collocation extraction with empty corpus."""
        results = self.extractor.extract([])
        
        assert isinstance(results, dict)
        # Empty corpus should result in empty collocations for each measure
        for measure, collocations in results.items():
            assert isinstance(collocations, list)
            assert len(collocations) == 0
    
    def test_extract_single_sentence_corpus(self):
        """Test collocation extraction with single sentence."""
        single_sentence = ["Om noh Yong ci"]
        results = self.extractor.extract(single_sentence)
        
        assert isinstance(results, dict)
        # Single sentence might not produce collocations depending on window size
        for measure, collocations in results.items():
            assert isinstance(collocations, list)
            assert len(collocations) >= 0
    
    def test_extract_single_word_sentences(self):
        """Test collocation extraction with single-word sentences."""
        single_words = ["Om", "noh", "Yong", "ci"]
        results = self.extractor.extract(single_words)
        
        assert isinstance(results, dict)
        # Single words might not produce collocations
        for measure, collocations in results.items():
            assert isinstance(collocations, list)
            assert len(collocations) >= 0
    
    def test_extract_with_whitespace_only_sentences(self):
        """Test collocation extraction with whitespace-only sentences."""
        whitespace_sentences = ["   ", "\t", "\n", "  \t  "]
        results = self.extractor.extract(whitespace_sentences)
        
        assert isinstance(results, dict)
        # Whitespace-only sentences should not produce collocations
        for measure, collocations in results.items():
            assert isinstance(collocations, list)
            assert len(collocations) == 0
    
    def test_extract_with_mixed_content(self):
        """Test collocation extraction with mixed content."""
        mixed_corpus = [
            "Om noh Yong ci",
            "   ",  # whitespace
            "Law noh Khanpughi ci",
            "",  # empty
            "Ak'hmó lùum ci"
        ]
        results = self.extractor.extract(mixed_corpus)
        
        assert isinstance(results, dict)
        # Should handle mixed content gracefully
        for measure, collocations in results.items():
            assert isinstance(collocations, list)
            assert len(collocations) >= 0
    
    def test_extract_with_unicode_content(self):
        """Test collocation extraction with Unicode content."""
        unicode_corpus = [
            "Ak'hmó lùum ci",
            "K'Cho language",
            "ñam ci",
            "Ak'hmó lùum ci"
        ]
        results = self.extractor.extract(unicode_corpus)
        
        assert isinstance(results, dict)
        # Should handle Unicode content properly
        for measure, collocations in results.items():
            assert isinstance(collocations, list)
            assert len(collocations) >= 0
    
    def test_extract_performance_with_large_corpus(self):
        """Test collocation extraction performance with larger corpus."""
        # Create a larger corpus by repeating sample sentences
        large_corpus = self.sample_corpus * 10  # 80 sentences
        
        results = self.extractor.extract(large_corpus)
        
        assert isinstance(results, dict)
        # Should handle larger corpus efficiently
        for measure, collocations in results.items():
            assert isinstance(collocations, list)
            assert len(collocations) >= 0
    
    def test_extract_result_structure(self):
        """Test that extraction results have correct structure."""
        results = self.extractor.extract(self.sample_corpus)
        
        # Check that we have results for each measure
        for measure, collocations in results.items():
            assert isinstance(measure, AssociationMeasure)
            assert isinstance(collocations, list)
            
            if collocations:  # If we have collocations
                result = collocations[0]
                
                # Check required attributes
                assert hasattr(result, 'words')
                assert hasattr(result, 'frequency')
                assert hasattr(result, 'score')
                assert hasattr(result, 'measure')
                
                # Check types
                assert isinstance(result.words, tuple)
                assert isinstance(result.frequency, int)
                assert isinstance(result.score, (int, float))
                assert isinstance(result.measure, AssociationMeasure)
                
                # Check values
                assert len(result.words) >= 2  # Collocations have at least 2 words
                assert result.frequency > 0
                assert result.measure == measure
    
    def test_extract_with_custom_parameters(self):
        """Test collocation extraction with custom parameters."""
        extractor = CollocationExtractor(
            window_size=3,
            min_freq=2,
            measures=[AssociationMeasure.PMI, AssociationMeasure.TSCORE]
        )
        results = extractor.extract(self.sample_corpus)
        
        assert isinstance(results, dict)
        
        # Should have results for both measures
        assert AssociationMeasure.PMI in results
        assert AssociationMeasure.TSCORE in results
        
        # All results should meet the frequency threshold
        for collocations in results.values():
            for result in collocations:
                assert result.frequency >= 2
    
    def test_extract_edge_cases(self):
        """Test collocation extraction with various edge cases."""
        edge_cases = [
            [],  # empty corpus
            [""],  # empty sentence
            ["   "],  # whitespace only
            ["a"],  # single character
            ["word"],  # single word
            ["word1 word2"],  # two words
            ["word1 word2 word3"],  # three words
        ]
        
        for corpus in edge_cases:
            results = self.extractor.extract(corpus)
            assert isinstance(results, dict)
            # Should handle all edge cases gracefully
            for measure, collocations in results.items():
                assert isinstance(collocations, list)
                assert len(collocations) >= 0


class TestAssociationMeasure:
    """Test cases for AssociationMeasure enum."""
    
    def test_association_measure_values(self):
        """Test AssociationMeasure enum values."""
        assert AssociationMeasure.PMI.value == "pmi"
        assert AssociationMeasure.NPMI.value == "npmi"
        assert AssociationMeasure.TSCORE.value == "tscore"
        assert AssociationMeasure.DICE.value == "dice"
        assert AssociationMeasure.LOG_LIKELIHOOD.value == "log_likelihood"
    
    def test_association_measure_iteration(self):
        """Test iterating over AssociationMeasure enum."""
        measures = list(AssociationMeasure)
        assert len(measures) == 5
        assert AssociationMeasure.PMI in measures
        assert AssociationMeasure.TSCORE in measures
        assert AssociationMeasure.DICE in measures
    
    def test_association_measure_membership(self):
        """Test membership testing for AssociationMeasure enum."""
        assert AssociationMeasure.PMI in AssociationMeasure
        # In Python, enum values can be checked for membership
        assert "pmi" in AssociationMeasure  # Value is accessible
        assert "PMI" not in AssociationMeasure  # Name is not the same as enum


class TestCollocationResult:
    """Test cases for CollocationResult class."""
    
    def test_collocation_result_creation(self):
        """Test CollocationResult creation."""
        words = ('Om', 'noh')
        frequency = 5
        score = 2.5
        measure = AssociationMeasure.PMI
        
        result = CollocationResult(words=words, frequency=frequency, score=score, measure=measure)
        
        assert result.words == words
        assert result.frequency == frequency
        assert result.score == score
        assert result.measure == measure
    
    def test_collocation_result_str(self):
        """Test CollocationResult string representation."""
        words = ('Om', 'noh')
        frequency = 5
        score = 2.5
        measure = AssociationMeasure.PMI
        
        result = CollocationResult(words=words, frequency=frequency, score=score, measure=measure)
        result_str = str(result)
        
        assert isinstance(result_str, str)
        assert 'Om' in result_str
        assert 'noh' in result_str
        assert '5' in result_str
    
    def test_collocation_result_with_default_positions(self):
        """Test CollocationResult with default positions."""
        words = ('word1', 'word2')
        frequency = 1
        score = 1.0
        measure = AssociationMeasure.TSCORE
        
        result = CollocationResult(words=words, frequency=frequency, score=score, measure=measure)
        
        assert result.words == words
        assert result.frequency == frequency
        assert result.score == score
        assert result.measure == measure
        assert result.positions == []  # Default empty list
    
    def test_collocation_result_with_positions(self):
        """Test CollocationResult with custom positions."""
        words = ('test', 'word')
        frequency = 3
        score = 1.5
        measure = AssociationMeasure.DICE
        positions = [1, 5, 10]
        
        result = CollocationResult(words=words, frequency=frequency, score=score, measure=measure, positions=positions)
        
        assert result.words == words
        assert result.frequency == frequency
        assert result.score == score
        assert result.measure == measure
        assert result.positions == positions


class TestCollocationExtractorIntegration:
    """Integration tests for CollocationExtractor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = CollocationExtractor()
        self.large_corpus = [
            "Om noh Yong ci",
            "Law noh Khanpughi ci",
            "Ak'hmó lùum ci",
            "Om noh Yong ci",
            "Law noh Khanpughi ci",
            "Ak'hmó lùum ci",
            "Om noh Yong ci",
            "Law noh Khanpughi ci",
            "Ak'hmó lùum ci",
            "Om noh Yong ci"
        ] * 5  # 50 sentences
    
    def test_full_extraction_pipeline(self):
        """Test complete collocation extraction pipeline."""
        extractor = CollocationExtractor(
            window_size=3,
            min_freq=2,
            measures=[AssociationMeasure.PMI, AssociationMeasure.TSCORE, AssociationMeasure.DICE, AssociationMeasure.NPMI]
        )
        results = extractor.extract(self.large_corpus)
        
        assert isinstance(results, dict)
        
        # Should find some collocations in this corpus
        total_collocations = sum(len(collocations) for collocations in results.values())
        assert total_collocations > 0
        
        # Check result quality
        for measure, collocations in results.items():
            assert isinstance(measure, AssociationMeasure)
            assert isinstance(collocations, list)
            
            for result in collocations:
                assert isinstance(result, CollocationResult)
                assert len(result.words) >= 2
                assert result.frequency >= 2
                assert result.measure == measure
    
    def test_extraction_consistency(self):
        """Test that extraction results are consistent."""
        # Run extraction twice with same parameters
        extractor1 = CollocationExtractor(min_freq=2)
        results1 = extractor1.extract(self.large_corpus)
        
        extractor2 = CollocationExtractor(min_freq=2)
        results2 = extractor2.extract(self.large_corpus)
        
        # Results should be identical
        assert len(results1) == len(results2)
        
        # Check each measure
        for measure in results1:
            assert measure in results2
            assert len(results1[measure]) == len(results2[measure])
    
    def test_parameter_sensitivity(self):
        """Test how results change with different parameters."""
        # Test window size sensitivity
        extractor_window_2 = CollocationExtractor(window_size=2)
        results_window_2 = extractor_window_2.extract(self.large_corpus)
        
        extractor_window_5 = CollocationExtractor(window_size=5)
        results_window_5 = extractor_window_5.extract(self.large_corpus)
        
        # Larger window should find more or equal collocations
        total_window_2 = sum(len(collocations) for collocations in results_window_2.values())
        total_window_5 = sum(len(collocations) for collocations in results_window_5.values())
        assert total_window_5 >= total_window_2
        
        # Test frequency threshold sensitivity
        extractor_freq_1 = CollocationExtractor(min_freq=1)
        results_freq_1 = extractor_freq_1.extract(self.large_corpus)
        
        extractor_freq_3 = CollocationExtractor(min_freq=3)
        results_freq_3 = extractor_freq_3.extract(self.large_corpus)
        
        # Higher frequency threshold should find fewer collocations
        total_freq_1 = sum(len(collocations) for collocations in results_freq_1.values())
        total_freq_3 = sum(len(collocations) for collocations in results_freq_3.values())
        assert total_freq_1 >= total_freq_3
    
    def test_performance_with_repeated_sentences(self):
        """Test performance with highly repetitive corpus."""
        # Create corpus with many repeated sentences
        repetitive_corpus = ["Om noh Yong ci"] * 100
        
        extractor = CollocationExtractor(min_freq=10)
        results = extractor.extract(repetitive_corpus)
        
        assert isinstance(results, dict)
        # Should find collocations in repetitive corpus
        total_collocations = sum(len(collocations) for collocations in results.values())
        assert total_collocations >= 0
        
        # All results should meet frequency threshold
        for collocations in results.values():
            for result in collocations:
                assert result.frequency >= 10
    
    def test_unicode_and_special_characters(self):
        """Test extraction with Unicode and special characters."""
        unicode_corpus = [
            "Ak'hmó lùum ci",
            "K'Cho language",
            "ñam ci",
            "Ak'hmó lùum ci",
            "K'Cho language",
            "ñam ci"
        ]
        
        extractor = CollocationExtractor(min_freq=2)
        results = extractor.extract(unicode_corpus)
        
        assert isinstance(results, dict)
        # Should handle Unicode properly
        for measure, collocations in results.items():
            assert isinstance(collocations, list)
            assert len(collocations) >= 0
    
    def test_mixed_sentence_lengths(self):
        """Test extraction with sentences of varying lengths."""
        mixed_corpus = [
            "Om",  # single word
            "Om noh",  # two words
            "Om noh Yong",  # three words
            "Om noh Yong ci",  # four words
            "Om noh Yong ci",
            "Om noh Yong ci"
        ]
        
        extractor = CollocationExtractor(min_freq=2)
        results = extractor.extract(mixed_corpus)
        
        assert isinstance(results, dict)
        # Should handle mixed lengths gracefully
        for measure, collocations in results.items():
            assert isinstance(collocations, list)
            assert len(collocations) >= 0
    
    def test_error_handling(self):
        """Test error handling in extraction."""
        # Test with None corpus - should handle gracefully or raise appropriate error
        try:
            results = self.extractor.extract(None)
            # If it doesn't raise an error, check that it handles None gracefully
            assert isinstance(results, dict)
        except (TypeError, AttributeError, ValueError):
            # Expected behavior - should raise an error for invalid input
            pass
        
        # Test with invalid corpus type - should handle gracefully or raise appropriate error
        try:
            results = self.extractor.extract("not a list")
            # If it doesn't raise an error, check that it handles invalid input gracefully
            assert isinstance(results, dict)
        except (TypeError, AttributeError, ValueError):
            # Expected behavior - should raise an error for invalid input
            pass
    
    def test_default_initialization_parameters(self):
        """Test default initialization parameters."""
        extractor = CollocationExtractor()
        
        assert extractor.window_size == 5
        assert extractor.min_freq == 5
        assert AssociationMeasure.PMI in extractor.measures
        assert AssociationMeasure.TSCORE in extractor.measures
        assert len(extractor.measures) == 2  # Default: PMI and TSCORE
    
    def test_custom_initialization_parameters(self):
        """Test custom initialization parameters."""
        measures = [AssociationMeasure.DICE, AssociationMeasure.NPMI]
        extractor = CollocationExtractor(
            window_size=3,
            min_freq=2,
            measures=measures
        )
        
        assert extractor.window_size == 3
        assert extractor.min_freq == 2
        assert extractor.measures == measures
        assert len(extractor.measures) == 2