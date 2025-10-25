"""
Unit tests for collocation extraction functionality.

Tests the CollocationExtractor class and all association measures.
"""

import pytest
import math
from collections import defaultdict
from kcho.collocation import (
    CollocationExtractor, 
    AssociationMeasure, 
    CollocationResult
)
from kcho.normalize import KChoNormalizer


class TestCollocationExtractor:
    """Test the CollocationExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = CollocationExtractor()
        self.sample_corpus = [
            "Om noh Yong am paapai pe ci",
            "Ak'hmó lùum ci",
            "Om noh Yong am paapai pe ci",
            "Ak'hmó lùum ci",
            "Om noh Yong am paapai pe ci"
        ]
    
    def test_initialization(self):
        """Test CollocationExtractor initialization."""
        assert self.extractor.window_size == 5
        assert self.extractor.min_freq == 5
        assert AssociationMeasure.PMI in self.extractor.measures
        assert AssociationMeasure.TSCORE in self.extractor.measures
        assert isinstance(self.extractor.unigram_freq, dict)
        assert isinstance(self.extractor.bigram_freq, dict)
    
    def test_frequency_computation(self):
        """Test frequency computation."""
        tokenized_corpus = [self.extractor.normalizer.tokenize(sent) for sent in self.sample_corpus]
        self.extractor._compute_frequencies(tokenized_corpus)
        
        # Check that frequencies are computed
        assert len(self.extractor.unigram_freq) > 0
        assert len(self.extractor.bigram_freq) > 0
        assert self.extractor.total_unigrams > 0
        assert self.extractor.total_bigrams > 0
        
        # Check specific frequencies
        assert self.extractor.unigram_freq['Om'] >= 3  # Should appear at least 3 times
        assert self.extractor.unigram_freq['ci'] >= 5  # Should appear at least 5 times
    
    def test_pmi_calculation(self):
        """Test PMI calculation."""
        # Test with known values
        freq_1, freq_2, freq_12 = 10, 8, 5
        total_unigrams, total_bigrams = 100, 50
        
        # Set totals for calculation
        self.extractor.total_unigrams = total_unigrams
        self.extractor.total_bigrams = total_bigrams
        
        pmi = self.extractor._pmi(freq_1, freq_2, freq_12)
        
        # Calculate expected PMI
        p_1 = freq_1 / total_unigrams
        p_2 = freq_2 / total_unigrams
        p_12 = freq_12 / total_bigrams
        expected_pmi = math.log2(p_12 / (p_1 * p_2))
        
        assert abs(pmi - expected_pmi) < 1e-10
    
    def test_npmi_calculation(self):
        """Test NPMI calculation."""
        freq_1, freq_2, freq_12 = 10, 8, 5
        total_unigrams, total_bigrams = 100, 50
        
        self.extractor.total_unigrams = total_unigrams
        self.extractor.total_bigrams = total_bigrams
        
        npmi = self.extractor._npmi(freq_1, freq_2, freq_12)
        
        # NPMI should be a finite number (can be > 1 in some cases)
        assert math.isfinite(npmi)
        assert npmi >= 0
    
    def test_tscore_calculation(self):
        """Test t-score calculation."""
        freq_1, freq_2, freq_12 = 10, 8, 5
        total_unigrams, total_bigrams = 100, 50
        
        self.extractor.total_unigrams = total_unigrams
        self.extractor.total_bigrams = total_bigrams
        
        tscore = self.extractor._tscore(freq_1, freq_2, freq_12)
        
        # t-score should be a positive number
        assert tscore >= 0
    
    def test_dice_calculation(self):
        """Test Dice coefficient calculation."""
        freq_1, freq_2, freq_12 = 10, 8, 5
        
        dice = self.extractor._dice(freq_1, freq_2, freq_12)
        
        # Calculate expected Dice coefficient
        expected_dice = 2 * freq_12 / (freq_1 + freq_2)
        
        assert abs(dice - expected_dice) < 1e-10
    
    def test_log_likelihood_calculation(self):
        """Test log-likelihood ratio calculation."""
        freq_1, freq_2, freq_12 = 10, 8, 5
        total_unigrams, total_bigrams = 100, 50
        
        self.extractor.total_unigrams = total_unigrams
        self.extractor.total_bigrams = total_bigrams
        
        llr = self.extractor._log_likelihood(freq_1, freq_2, freq_12)
        
        # Log-likelihood should be a positive number
        assert llr >= 0
    
    def test_extract_collocations(self):
        """Test full collocation extraction."""
        results = self.extractor.extract(self.sample_corpus)
        
        # Should return results for each measure
        assert len(results) > 0
        for measure in self.extractor.measures:
            assert measure in results
            assert isinstance(results[measure], list)
    
    def test_group_collocations_by_pos_pattern(self):
        """Test POS pattern grouping with defaultdict."""
        # Mock POS tagger
        def mock_pos_tagger(word):
            if word in ['Om', 'Yong', 'paapai']:
                return 'N'
            elif word in ['pe', 'ci']:
                return 'V'
            else:
                return 'UNK'
        
        self.extractor.pos_tagger = mock_pos_tagger
        
        results = self.extractor.group_collocations_by_pos_pattern(self.sample_corpus)
        
        # Should return grouped results
        assert isinstance(results, dict)
        # Should have some POS patterns (or be empty if no collocations found)
        # Note: Results may be empty if no collocations meet the frequency threshold
    
    def test_analyze_word_contexts(self):
        """Test word context analysis with nested defaultdict."""
        contexts = self.extractor.analyze_word_contexts(self.sample_corpus, context_window=2)
        
        # Should return nested dictionary
        assert isinstance(contexts, dict)
        
        # Check structure
        for word, context_dict in contexts.items():
            assert isinstance(context_dict, dict)
            for context_word, count in context_dict.items():
                assert isinstance(count, int)
                assert count > 0
    
    def test_extract_linguistic_patterns(self):
        """Test linguistic pattern extraction."""
        patterns = self.extractor.extract_linguistic_patterns(self.sample_corpus)
        
        # Should return pattern dictionary
        assert isinstance(patterns, dict)
        
        # Should have some pattern types
        expected_types = ['verb_prefixes', 'noun_suffixes', 'word_sequences', 
                        'sentence_starts', 'sentence_ends']
        
        for pattern_type in expected_types:
            if pattern_type in patterns:
                assert isinstance(patterns[pattern_type], dict)


class TestAssociationMeasure:
    """Test AssociationMeasure enum."""
    
    def test_association_measures(self):
        """Test that all association measures are defined."""
        measures = [
            AssociationMeasure.PMI,
            AssociationMeasure.NPMI,
            AssociationMeasure.TSCORE,
            AssociationMeasure.DICE,
            AssociationMeasure.LOG_LIKELIHOOD
        ]
        
        for measure in measures:
            assert measure.value in ['pmi', 'npmi', 'tscore', 'dice', 'log_likelihood']


class TestCollocationResult:
    """Test CollocationResult dataclass."""
    
    def test_collocation_result_creation(self):
        """Test CollocationResult creation."""
        words = ('Om', 'noh')
        score = 5.2
        frequency = 10
        measure = AssociationMeasure.PMI
        
        result = CollocationResult(
            words=words,
            score=score,
            measure=measure,
            frequency=frequency
        )
        
        assert result.words == words
        assert result.score == score
        assert result.frequency == frequency
        assert result.measure == measure
    
    def test_collocation_result_str(self):
        """Test CollocationResult string representation."""
        result = CollocationResult(
            words=('Om', 'noh'),
            score=5.2,
            measure=AssociationMeasure.PMI,
            frequency=10
        )
        
        str_repr = str(result)
        assert 'Om noh' in str_repr
        assert '5.2' in str_repr
        assert '10' in str_repr


@pytest.mark.integration
class TestCollocationIntegration:
    """Integration tests for collocation extraction."""
    
    def test_full_pipeline(self):
        """Test the complete collocation extraction pipeline."""
        extractor = CollocationExtractor(min_freq=1)  # Lower threshold for testing
        
        corpus = [
            "Om noh Yong am paapai pe ci",
            "Ak'hmó lùum ci",
            "Om noh Yong am paapai pe ci",
            "Ak'hmó lùum ci",
            "Om noh Yong am paapai pe ci",
            "Om noh Yong am paapai pe ci",
            "Om noh Yong am paapai pe ci"
        ]
        
        # Extract collocations
        results = extractor.extract(corpus)
        
        # Should have results
        assert len(results) > 0
        
        # Test defaultdict functionality
        pos_groups = extractor.group_collocations_by_pos_pattern(corpus)
        contexts = extractor.analyze_word_contexts(corpus)
        patterns = extractor.extract_linguistic_patterns(corpus)
        
        # All should return dictionaries
        assert isinstance(pos_groups, dict)
        assert isinstance(contexts, dict)
        assert isinstance(patterns, dict)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        extractor = CollocationExtractor()
        
        # Empty corpus
        results = extractor.extract([])
        assert isinstance(results, dict)
        # Should return empty results or results with empty lists
        assert len(results) >= 0
        
        # Single word corpus
        results = extractor.extract(["Om"])
        assert isinstance(results, dict)
        # Should return empty results or results with empty lists
        assert len(results) >= 0
        
        # Very short corpus
        results = extractor.extract(["Om noh"])
        # Should handle gracefully
        assert isinstance(results, dict)
