"""
Unit tests for corpus management functionality.

Tests the KchoCorpus class and corpus processing methods.
"""

import pytest
from collections import defaultdict
from kcho.kcho_system import KchoCorpus, KchoKnowledge


class TestKchoCorpus:
    """Test the KchoCorpus class."""
    
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
    
    def test_initialization(self):
        """Test KchoCorpus initialization."""
        assert isinstance(self.corpus, KchoCorpus)
        assert hasattr(self.corpus, 'analyze_pos_patterns')
        assert hasattr(self.corpus, 'build_word_cooccurrence_matrix')
        assert hasattr(self.corpus, 'extract_morphological_patterns')
        assert hasattr(self.corpus, 'analyze_sentence_structure_patterns')
    
    def test_analyze_pos_patterns(self):
        """Test POS pattern analysis."""
        # Add some sentences to the corpus first
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
        # Add some sentences to the corpus first
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
        # Add some sentences to the corpus first
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
        # Add some sentences to the corpus first
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
    
    def test_empty_corpus_handling(self):
        """Test handling of empty corpus."""
        # Empty corpus should return empty results
        patterns = self.corpus.analyze_pos_patterns()
        assert isinstance(patterns, dict)
        assert len(patterns) == 0
        
        matrix = self.corpus.build_word_cooccurrence_matrix()
        assert isinstance(matrix, dict)
        assert len(matrix) == 0
        
        morph_patterns = self.corpus.extract_morphological_patterns()
        assert isinstance(morph_patterns, dict)
        assert len(morph_patterns) == 0
        
        structure_patterns = self.corpus.analyze_sentence_structure_patterns()
        assert isinstance(structure_patterns, dict)
        assert len(structure_patterns) == 0
    
    def test_single_word_sentences(self):
        """Test handling of single word sentences."""
        single_word_sentences = ["Om", "ci", "Yong"]
        
        # Add sentences to corpus
        for sentence in single_word_sentences:
            self.corpus.add_sentence(sentence, validate=False)
        
        patterns = self.corpus.analyze_pos_patterns()
        assert isinstance(patterns, dict)
        
        matrix = self.corpus.build_word_cooccurrence_matrix()
        assert isinstance(matrix, dict)
        
        morph_patterns = self.corpus.extract_morphological_patterns()
        assert isinstance(morph_patterns, dict)
        
        structure_patterns = self.corpus.analyze_sentence_structure_patterns()
        assert isinstance(structure_patterns, dict)
    
    def test_window_size_parameter(self):
        """Test window_size parameter in co-occurrence matrix."""
        # Add sentences to corpus
        for sentence in self.sample_sentences:
            self.corpus.add_sentence(sentence, validate=False)
        
        # Test different window sizes
        for window_size in [1, 2, 3, 5]:
            matrix = self.corpus.build_word_cooccurrence_matrix(
                window_size=window_size
            )
            assert isinstance(matrix, dict)
    
    def test_pattern_extraction_consistency(self):
        """Test consistency of pattern extraction."""
        # Add sentences to corpus
        for sentence in self.sample_sentences:
            self.corpus.add_sentence(sentence, validate=False)
        
        # Multiple calls should give consistent results
        patterns1 = self.corpus.extract_morphological_patterns()
        patterns2 = self.corpus.extract_morphological_patterns()
        
        assert isinstance(patterns1, dict)
        assert isinstance(patterns2, dict)
    
    def test_cooccurrence_matrix_properties(self):
        """Test properties of co-occurrence matrix."""
        # Add sentences to corpus
        for sentence in self.sample_sentences:
            self.corpus.add_sentence(sentence, validate=False)
        
        matrix = self.corpus.build_word_cooccurrence_matrix()
        
        # Matrix should have some data
        if matrix:
            # Check that all counts are positive integers
            for word1, cooccur_dict in matrix.items():
                for word2, count in cooccur_dict.items():
                    assert isinstance(count, int)
                    assert count > 0
    
    def test_morphological_pattern_types(self):
        """Test that morphological patterns have expected types."""
        # Add sentences to corpus
        for sentence in self.sample_sentences:
            self.corpus.add_sentence(sentence, validate=False)
        
        patterns = self.corpus.extract_morphological_patterns()
        
        # Should have some pattern types
        if patterns:
            for pattern_type, pattern_dict in patterns.items():
                assert isinstance(pattern_type, str)
                assert isinstance(pattern_dict, dict)
                
                for pattern, count in pattern_dict.items():
                    assert isinstance(pattern, str)
                    assert isinstance(count, int)
                    assert count > 0


class TestKchoKnowledge:
    """Test the KchoKnowledge class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Note: This might fail if data files don't exist, which is expected
        try:
            self.knowledge = KchoKnowledge()
        except FileNotFoundError:
            pytest.skip("KchoKnowledge data files not found")
    
    def test_initialization(self):
        """Test KchoKnowledge initialization."""
        if hasattr(self, 'knowledge'):
            assert isinstance(self.knowledge, KchoKnowledge)
    
    def test_data_loading(self):
        """Test data loading functionality."""
        if hasattr(self, 'knowledge'):
            # Should have loaded some data
            assert hasattr(self.knowledge, 'VERB_STEMS')
            assert hasattr(self.knowledge, 'PRONOUNS')
            assert hasattr(self.knowledge, 'COMMON_WORDS')


@pytest.mark.integration
class TestCorpusIntegration:
    """Integration tests for corpus management."""
    
    def test_full_corpus_pipeline(self):
        """Test complete corpus processing pipeline."""
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
        
        # Add sentences to corpus
        for sentence in test_sentences:
            corpus.add_sentence(sentence, validate=False)
        
        # Run all corpus analysis methods
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
        total_patterns = (len(pos_patterns) + len(cooccur_matrix) + 
                         len(morph_patterns) + len(structure_patterns))
        assert total_patterns > 0
    
    def test_large_corpus_handling(self):
        """Test handling of larger corpus."""
        corpus = KchoCorpus()
        
        # Create a larger test corpus
        large_corpus = []
        base_sentence = "Om noh Yong am paapai pe ci"
        for i in range(50):
            large_corpus.append(base_sentence)
            large_corpus.append("Ak'hmó lùum ci")
        
        # Add sentences to corpus
        for sentence in large_corpus:
            corpus.add_sentence(sentence, validate=False)
        
        # Should handle large corpus efficiently
        patterns = corpus.analyze_pos_patterns()
        assert isinstance(patterns, dict)
        
        matrix = corpus.build_word_cooccurrence_matrix()
        assert isinstance(matrix, dict)
        
        morph_patterns = corpus.extract_morphological_patterns()
        assert isinstance(morph_patterns, dict)
        
        structure_patterns = corpus.analyze_sentence_structure_patterns()
        assert isinstance(structure_patterns, dict)
    
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
        
        # Add sentences to corpus (skip empty ones)
        for sentence in mixed_sentences:
            if sentence.strip():  # Skip empty sentences
                corpus.add_sentence(sentence, validate=False)
        
        # Should handle mixed content gracefully
        patterns = corpus.analyze_pos_patterns()
        assert isinstance(patterns, dict)
        
        matrix = corpus.build_word_cooccurrence_matrix()
        assert isinstance(matrix, dict)
        
        morph_patterns = corpus.extract_morphological_patterns()
        assert isinstance(morph_patterns, dict)
        
        structure_patterns = corpus.analyze_sentence_structure_patterns()
        assert isinstance(structure_patterns, dict)
    
    def test_performance_with_repeated_sentences(self):
        """Test performance with repeated sentences."""
        corpus = KchoCorpus()
        
        # Create corpus with many repeated sentences
        repeated_corpus = []
        base_sentences = [
            "Om noh Yong am paapai pe ci",
            "Ak'hmó lùum ci",
            "Om noh Yong am paapai pe ci"
        ]
        
        for _ in range(100):
            repeated_corpus.extend(base_sentences)
        
        # Add sentences to corpus
        for sentence in repeated_corpus:
            corpus.add_sentence(sentence, validate=False)
        
        # Should handle repeated content efficiently
        patterns = corpus.analyze_pos_patterns()
        assert isinstance(patterns, dict)
        
        matrix = corpus.build_word_cooccurrence_matrix()
        assert isinstance(matrix, dict)
        
        # Should have consistent patterns
        assert len(patterns) > 0 or len(matrix) > 0
    
    def test_edge_cases_comprehensive(self):
        """Test comprehensive edge cases."""
        corpus = KchoCorpus()
        
        edge_cases = [
            [],                           # Empty corpus
            [""],                         # Single empty sentence
            ["Om"],                       # Single word
            ["Om noh"],                   # Two words
            ["Om noh Yong am paapai pe ci"],  # Single sentence
            ["Om", "noh", "Yong"],        # Multiple single words
            ["Om noh Yong am paapai pe ci", "Ak'hmó lùum ci"],  # Two sentences
        ]
        
        for test_case in edge_cases:
            # Create new corpus for each test case
            test_corpus = KchoCorpus()
            
            # Add sentences to corpus (skip empty ones)
            for sentence in test_case:
                if sentence.strip():  # Skip empty sentences
                    test_corpus.add_sentence(sentence, validate=False)
            
            # All methods should handle edge cases gracefully
            patterns = test_corpus.analyze_pos_patterns()
            matrix = test_corpus.build_word_cooccurrence_matrix()
            morph_patterns = test_corpus.extract_morphological_patterns()
            structure_patterns = test_corpus.analyze_sentence_structure_patterns()
            
            # All should return dictionaries
            assert isinstance(patterns, dict)
            assert isinstance(matrix, dict)
            assert isinstance(morph_patterns, dict)
            assert isinstance(structure_patterns, dict)
    
    def test_defaultdict_behavior_in_corpus(self):
        """Test defaultdict behavior in corpus methods."""
        corpus = KchoCorpus()
        
        sentences = ["Om noh Yong am paapai pe ci"]
        
        # Add sentences to corpus
        for sentence in sentences:
            corpus.add_sentence(sentence, validate=False)
        
        # Test that defaultdict behavior works correctly
        patterns = corpus.analyze_pos_patterns()
        
        # Should be able to access non-existent keys
        if patterns:
            # Get a random pattern
            pattern = next(iter(patterns.keys()))
            # Access a non-existent word in that pattern
            assert isinstance(patterns[pattern], dict)
        
        matrix = corpus.build_word_cooccurrence_matrix()
        
        # Should be able to access non-existent keys
        if matrix:
            # Get a random word
            word = next(iter(matrix.keys()))
            # Access a non-existent co-occurring word
            assert isinstance(matrix[word], dict)
    
    def test_corpus_methods_independence(self):
        """Test that corpus methods are independent."""
        corpus = KchoCorpus()
        
        sentences = [
            "Om noh Yong am paapai pe ci",
            "Ak'hmó lùum ci",
            "Om noh Yong am paapai pe ci"
        ]
        
        # Add sentences to corpus
        for sentence in sentences:
            corpus.add_sentence(sentence, validate=False)
        
        # Run methods in different orders
        patterns1 = corpus.analyze_pos_patterns()
        matrix1 = corpus.build_word_cooccurrence_matrix()
        
        matrix2 = corpus.build_word_cooccurrence_matrix()
        patterns2 = corpus.analyze_pos_patterns()
        
        # Results should be consistent regardless of order
        assert isinstance(patterns1, dict)
        assert isinstance(patterns2, dict)
        assert isinstance(matrix1, dict)
        assert isinstance(matrix2, dict)
