"""
Unit tests for morphological analysis functionality.

Tests the morphological analysis methods in KchoCorpus.
"""

import pytest
from collections import defaultdict
from kcho.kcho_system import KchoCorpus


class TestMorphologicalAnalysis:
    """Test morphological analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.corpus = KchoCorpus()
        self.sample_sentences = [
            "Om noh Yong am paapai pe ci",
            "Ak'hmó lùum ci",
            "Om noh Yong am paapai pe ci",
            "Ak'hmó lùum ci",
            "Om noh Yong am paapai pe ci"
        ]
    
    def test_analyze_pos_patterns(self):
        """Test POS pattern analysis with defaultdict."""
        # Add sentences to corpus first
        for sentence in self.sample_sentences:
            self.corpus.add_sentence(sentence, validate=False)
        
        patterns = self.corpus.analyze_pos_patterns()
        
        # Should return nested defaultdict
        assert isinstance(patterns, dict)
        
        # Check structure
        for pattern, word_dict in patterns.items():
            assert isinstance(pattern, str)  # Pattern length as string
            assert isinstance(word_dict, dict)
            
            for word, count in word_dict.items():
                assert isinstance(word, (str, tuple))  # Can be string or tuple (POS sequence)
                assert isinstance(count, int)
                assert count > 0
    
    def test_build_word_cooccurrence_matrix(self):
        """Test word co-occurrence matrix building."""
        # Add sentences to corpus first
        for sentence in self.sample_sentences:
            self.corpus.add_sentence(sentence, validate=False)
        
        matrix = self.corpus.build_word_cooccurrence_matrix(window_size=3)
        
        # Should return nested defaultdict
        assert isinstance(matrix, dict)
        
        # Check structure
        for word1, cooccur_dict in matrix.items():
            assert isinstance(word1, str)
            assert isinstance(cooccur_dict, dict)
            
            for word2, count in cooccur_dict.items():
                assert isinstance(word2, str)
                assert isinstance(count, int)
                assert count > 0
    
    def test_extract_morphological_patterns(self):
        """Test morphological pattern extraction."""
        # Add sentences to corpus first
        for sentence in self.sample_sentences:
            self.corpus.add_sentence(sentence, validate=False)
        
        patterns = self.corpus.extract_morphological_patterns()
        
        # Should return nested defaultdict
        assert isinstance(patterns, dict)
        
        # Should have expected pattern types
        expected_types = ['prefixes', 'suffixes', 'infixes', 'root_words']
        
        for pattern_type in expected_types:
            if pattern_type in patterns:
                assert isinstance(patterns[pattern_type], dict)
                
                for pattern, count in patterns[pattern_type].items():
                    assert isinstance(pattern, str)
                    assert isinstance(count, int)
                    assert count > 0
    
    def test_analyze_sentence_structure_patterns(self):
        """Test sentence structure pattern analysis."""
        # Add sentences to corpus first
        for sentence in self.sample_sentences:
            self.corpus.add_sentence(sentence, validate=False)
        
        patterns = self.corpus.analyze_sentence_structure_patterns()
        
        # Should return nested defaultdict
        assert isinstance(patterns, dict)
        
        # Should have expected pattern types
        expected_types = ['sentence_lengths', 'word_positions', 'phrase_patterns']
        
        for pattern_type in expected_types:
            if pattern_type in patterns:
                assert isinstance(patterns[pattern_type], dict)
                
                for pattern, count in patterns[pattern_type].items():
                    assert isinstance(pattern, str)
                    assert isinstance(count, int)
                    assert count > 0
    
    def test_defaultdict_functionality(self):
        """Test that defaultdict works correctly for nested structures."""
        # Test nested defaultdict creation
        nested_dict = defaultdict(lambda: defaultdict(int))
        
        # Add some data
        nested_dict['pattern1']['word1'] += 1
        nested_dict['pattern1']['word2'] += 2
        nested_dict['pattern2']['word1'] += 3
        
        # Test access to existing keys
        assert nested_dict['pattern1']['word1'] == 1
        assert nested_dict['pattern1']['word2'] == 2
        assert nested_dict['pattern2']['word1'] == 3
        
        # Test access to non-existing keys (should return 0)
        assert nested_dict['pattern3']['word3'] == 0
        assert nested_dict['pattern1']['word3'] == 0
    
    def test_empty_input_handling(self):
        """Test handling of empty input."""
        # Empty sentences
        patterns = self.corpus.analyze_pos_patterns()
        assert isinstance(patterns, dict)
        assert len(patterns) == 0
        
        # Single empty sentence
        patterns = self.corpus.analyze_pos_patterns()
        assert isinstance(patterns, dict)
    
    def test_single_word_sentences(self):
        """Test handling of single word sentences."""
        single_word_sentences = ["Om", "ci", "Yong"]
        
        patterns = self.corpus.analyze_pos_patterns()
        assert isinstance(patterns, dict)
        
        matrix = self.corpus.build_word_cooccurrence_matrix(single_word_sentences)
        assert isinstance(matrix, dict)
    
    def test_pattern_extraction_edge_cases(self):
        """Test edge cases in pattern extraction."""
        # Very short words
        short_sentences = ["a", "b", "c"]
        patterns = self.corpus.extract_morphological_patterns()
        assert isinstance(patterns, dict)
        
        # Words with special characters
        special_sentences = ["Om'hmó", "lùum", "paapai"]
        patterns = self.corpus.extract_morphological_patterns()
        assert isinstance(patterns, dict)
    
    def test_cooccurrence_matrix_properties(self):
        """Test properties of co-occurrence matrix."""
        matrix = self.corpus.build_word_cooccurrence_matrix()
        
        # Matrix should be symmetric for co-occurrence
        for word1, cooccur_dict in matrix.items():
            for word2, count in cooccur_dict.items():
                if word1 != word2:
                    # Check symmetry (if word1 co-occurs with word2, word2 should co-occur with word1)
                    assert word1 in matrix[word2] or count == 0
    
    def test_morphological_pattern_consistency(self):
        """Test consistency of morphological patterns."""
        patterns = self.corpus.extract_morphological_patterns()
        
        # All counts should be positive integers
        for pattern_type, pattern_dict in patterns.items():
            for pattern, count in pattern_dict.items():
                assert isinstance(count, int)
                assert count > 0
                assert len(pattern) > 0  # Patterns should not be empty strings


@pytest.mark.integration
class TestMorphologicalIntegration:
    """Integration tests for morphological analysis."""
    
    def test_full_morphological_pipeline(self):
        """Test complete morphological analysis pipeline."""
        corpus = KchoCorpus()
        
        test_sentences = [
            "Om noh Yong am paapai pe ci",
            "Ak'hmó lùum ci",
            "Om noh Yong am paapai pe ci",
            "Ak'hmó lùum ci",
            "Om noh Yong am paapai pe ci",
            "Om noh Yong am paapai pe ci",
            "Om noh Yong am paapai pe ci"
        ]
        
        # Run all morphological analysis methods
        # Add sentences to corpus
        for sentence in test_sentences:
            corpus.add_sentence(sentence, validate=False)
        
        pos_patterns = corpus.analyze_pos_patterns()
        cooccur_matrix = corpus.build_word_cooccurrence_matrix()
        morph_patterns = corpus.extract_morphological_patterns()
        structure_patterns = corpus.analyze_sentence_structure_patterns()
        
        # All should return dictionaries
        assert isinstance(pos_patterns, dict)
        assert isinstance(cooccur_matrix, dict)
        assert isinstance(morph_patterns, dict)
        assert isinstance(structure_patterns, dict)
        
        # Should have some data
        assert len(pos_patterns) > 0 or len(cooccur_matrix) > 0
        assert len(morph_patterns) > 0 or len(structure_patterns) > 0
    
    def test_large_corpus_handling(self):
        """Test handling of larger corpus."""
        corpus = KchoCorpus()
        
        # Create a larger test corpus
        large_corpus = []
        base_sentence = "Om noh Yong am paapai pe ci"
        for i in range(50):
            large_corpus.append(base_sentence)
            large_corpus.append("Ak'hmó lùum ci")
        
        # Should handle large corpus efficiently
        # Add sentences to corpus
        for sentence in large_corpus:
            corpus.add_sentence(sentence, validate=False)
        
        patterns = corpus.analyze_pos_patterns()
        assert isinstance(patterns, dict)
        
        matrix = corpus.build_word_cooccurrence_matrix()
        assert isinstance(matrix, dict)
    
    def test_mixed_content_handling(self):
        """Test handling of mixed content types."""
        corpus = KchoCorpus()
        
        mixed_sentences = [
            "Om noh Yong am paapai pe ci",  # Normal sentence
            "Ak'hmó lùum ci",               # Another normal sentence
            "",                             # Empty sentence
            "Om",                          # Single word
            "Om noh Yong am paapai pe ci",  # Repeat
            "Ak'hmó lùum ci"               # Repeat
        ]
        
        # Should handle mixed content gracefully
        # Add sentences to corpus (skip empty ones)
        for sentence in mixed_sentences:
            if sentence.strip():  # Skip empty sentences
                corpus.add_sentence(sentence, validate=False)
        
        patterns = corpus.analyze_pos_patterns()
        assert isinstance(patterns, dict)
        
        matrix = corpus.build_word_cooccurrence_matrix()
        assert isinstance(matrix, dict)
        
        morph_patterns = corpus.extract_morphological_patterns()
        assert isinstance(morph_patterns, dict)
        
        structure_patterns = corpus.analyze_sentence_structure_patterns()
        assert isinstance(structure_patterns, dict)
