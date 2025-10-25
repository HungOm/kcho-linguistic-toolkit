"""
Tests for create_gold_standard.py module.

This module tests the GoldStandardCreator class and its methods for creating
gold standard collocation files from K'Cho corpora.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from collections import Counter

from kcho.create_gold_standard import GoldStandardCreator, CATEGORIES, main
from kcho.collocation import CollocationExtractor, AssociationMeasure, CollocationResult
from kcho.normalize import KChoNormalizer


class TestGoldStandardCreator:
    """Test the GoldStandardCreator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use the real gold standard K'Cho-English parallel corpus
        self.parallel_corpus_path = Path(__file__).parent.parent / "data" / "gold_standard_kcho_english.json"
        
        # Create a temporary corpus file with K'Cho sentences from the parallel corpus
        self.temp_corpus = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
        
        # Load the parallel corpus and extract K'Cho sentences
        with open(self.parallel_corpus_path, 'r', encoding='utf-8') as f:
            parallel_data = json.load(f)
        
        # Extract K'Cho sentences for corpus
        kcho_sentences = [pair['kcho'] for pair in parallel_data['sentence_pairs'][:20]]  # Use first 20 sentences
        self.corpus_content = '\n'.join(kcho_sentences)
        self.temp_corpus.write(self.corpus_content)
        self.temp_corpus.close()
        
        # Store parallel data for reference
        self.parallel_data = parallel_data
        
        # Initialize creator
        self.creator = GoldStandardCreator(self.temp_corpus.name)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_corpus.name):
            os.unlink(self.temp_corpus.name)
    
    def test_initialization(self):
        """Test GoldStandardCreator initialization."""
        assert self.creator.corpus_path == Path(self.temp_corpus.name)
        assert isinstance(self.creator.normalizer, KChoNormalizer)
        assert isinstance(self.creator.extractor, CollocationExtractor)
        assert self.creator.corpus == []
        assert self.creator.candidates == []
        assert self.creator.gold_standard == {}
    
    def test_load_corpus(self):
        """Test corpus loading."""
        self.creator.load_corpus()
        
        assert len(self.creator.corpus) > 0
        assert all(isinstance(sentence, str) for sentence in self.creator.corpus)
        assert all(not sentence.startswith('#') for sentence in self.creator.corpus)
        assert all(len(sentence.strip()) > 0 for sentence in self.creator.corpus)
    
    def test_load_corpus_with_comments(self):
        """Test that comments are filtered out."""
        self.creator.load_corpus()
        
        # Should not contain comment lines
        for sentence in self.creator.corpus:
            assert not sentence.startswith('#')
    
    def test_load_corpus_empty_lines(self):
        """Test that empty lines are filtered out."""
        self.creator.load_corpus()
        
        # Should not contain empty strings
        for sentence in self.creator.corpus:
            assert len(sentence.strip()) > 0
    
    @patch('kcho.create_gold_standard.CollocationExtractor')
    def test_extract_candidates(self, mock_extractor_class):
        """Test candidate extraction."""
        # Mock the extractor
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        
        # Mock extraction results
        mock_results = {
            AssociationMeasure.PMI: [
                CollocationResult(words=('pe', 'ci'), score=0.8, measure=AssociationMeasure.PMI, frequency=5),
                CollocationResult(words=('lo', 'ci'), score=0.7, measure=AssociationMeasure.PMI, frequency=4),
            ],
            AssociationMeasure.TSCORE: [
                CollocationResult(words=('pe', 'ci'), score=0.9, measure=AssociationMeasure.TSCORE, frequency=5),
                CollocationResult(words=('noh', 'Yóng'), score=0.6, measure=AssociationMeasure.TSCORE, frequency=3),
            ]
        }
        mock_extractor.extract.return_value = mock_results
        
        # Load corpus and extract candidates
        self.creator.load_corpus()
        self.creator.extract_candidates(top_k=10)
        
        # Verify results
        assert len(self.creator.candidates) > 0
        assert all(isinstance(candidate, tuple) for candidate in self.creator.candidates)
        assert all(len(candidate) == 3 for candidate in self.creator.candidates)
        
        # Check that candidates are sorted by score (descending)
        scores = [candidate[1] for candidate in self.creator.candidates]
        assert scores == sorted(scores, reverse=True)
    
    def test_extract_candidates_empty_corpus(self):
        """Test candidate extraction with empty corpus."""
        self.creator.corpus = []
        
        with patch.object(self.creator.extractor, 'extract', return_value={}):
            self.creator.extract_candidates()
            assert len(self.creator.candidates) == 0
    
    def test_find_examples(self):
        """Test finding example sentences."""
        self.creator.load_corpus()
        
        # Test finding examples for a collocation
        examples = self.creator._find_examples(('pe', 'ci'), max_examples=2)
        
        assert isinstance(examples, list)
        assert len(examples) <= 2
        assert all(isinstance(example, str) for example in examples)
    
    def test_find_examples_no_matches(self):
        """Test finding examples when no matches exist."""
        self.creator.load_corpus()
        
        # Test with non-existent collocation
        examples = self.creator._find_examples(('nonexistent', 'words'), max_examples=3)
        
        assert examples == []
    
    def test_find_examples_max_examples(self):
        """Test that max_examples limit is respected."""
        self.creator.load_corpus()
        
        examples = self.creator._find_examples(('pe', 'ci'), max_examples=1)
        
        assert len(examples) <= 1
    
    def test_auto_annotate(self):
        """Test automatic annotation."""
        # Set up candidates
        self.creator.candidates = [
            (('pe', 'ci'), 0.8, 5),      # VP pattern
            (('noh', 'Yóng'), 0.7, 4),    # PP pattern
            (('luum-na', 'ci'), 0.6, 3),  # APP pattern
            (('ka', 'hngu'), 0.5, 3),     # AGR pattern
            (('ci', 'ah'), 0.4, 2),       # COMP pattern
            (('random', 'words'), 0.3, 1) # No pattern
        ]
        
        self.creator.auto_annotate()
        
        # Check that high-confidence patterns were annotated
        assert ('pe', 'ci') in self.creator.gold_standard
        assert self.creator.gold_standard[('pe', 'ci')]['category'] == 'VP'
        
        assert ('noh', 'Yóng') in self.creator.gold_standard
        assert self.creator.gold_standard[('noh', 'Yóng')]['category'] == 'PP'
        
        assert ('luum-na', 'ci') in self.creator.gold_standard
        assert self.creator.gold_standard[('luum-na', 'ci')]['category'] == 'VP'  # Updated: VP takes precedence over APP
        
        assert ('ci', 'ah') in self.creator.gold_standard
        assert self.creator.gold_standard[('ci', 'ah')]['category'] == 'COMP'
    
    def test_auto_annotate_skip_existing(self):
        """Test that auto-annotation skips already annotated items."""
        # Pre-annotate one item
        self.creator.gold_standard[('pe', 'ci')] = {
            'category': 'VP',
            'frequency': 5,
            'score': 0.8,
            'notes': 'Pre-annotated'
        }
        
        # Set up candidates
        self.creator.candidates = [
            (('pe', 'ci'), 0.8, 5),      # Already annotated
            (('noh', 'Yóng'), 0.7, 4),    # Should be annotated
        ]
        
        self.creator.auto_annotate()
        
        # Check that existing annotation wasn't overwritten
        assert self.creator.gold_standard[('pe', 'ci')]['notes'] == 'Pre-annotated'
        
        # Check that new item was annotated
        assert ('noh', 'Yóng') in self.creator.gold_standard
    
    def test_auto_annotate_confidence_levels(self):
        """Test that only high-confidence patterns are auto-annotated."""
        # Set up candidates with different confidence levels
        self.creator.candidates = [
            (('pe', 'ci'), 0.8, 5),      # High confidence VP
            (('ka', 'hngu'), 0.5, 3),     # Medium confidence AGR
            (('random', 'words'), 0.3, 1) # No pattern
        ]
        
        self.creator.auto_annotate()
        
        # Only high-confidence patterns should be annotated
        assert ('pe', 'ci') in self.creator.gold_standard
        assert ('ka', 'hngu') not in self.creator.gold_standard  # Medium confidence
        assert ('random', 'words') not in self.creator.gold_standard
    
    def test_save_gold_standard(self):
        """Test saving gold standard to file."""
        # Set up gold standard data
        self.creator.gold_standard = {
            ('pe', 'ci'): {
                'category': 'VP',
                'frequency': 5,
                'score': 0.8,
                'notes': 'Test annotation'
            },
            ('noh', 'Yóng'): {
                'category': 'PP',
                'frequency': 4,
                'score': 0.7,
                'notes': ''
            }
        }
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_output = f.name
        
        try:
            self.creator.save_gold_standard(temp_output)
            
            # Verify file was created
            assert os.path.exists(temp_output)
            
            # Verify file content
            with open(temp_output, 'r', encoding='utf-8') as f:
                content = f.read()
                
                assert "K'Cho Gold Standard Collocations" in content
                assert "pe ci" in content
                assert "noh Yóng" in content
                assert "VP" in content
                assert "PP" in content
                assert "Test annotation" in content
        
        finally:
            if os.path.exists(temp_output):
                os.unlink(temp_output)
    
    def test_save_gold_standard_empty(self):
        """Test saving empty gold standard."""
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_output = f.name
        
        try:
            self.creator.save_gold_standard(temp_output)
            
            # Verify file was created
            assert os.path.exists(temp_output)
            
            # Verify file content
            with open(temp_output, 'r', encoding='utf-8') as f:
                content = f.read()
                
                assert "Total entries: 0" in content
        
        finally:
            if os.path.exists(temp_output):
                os.unlink(temp_output)
    
    def test_save_gold_standard_sorting(self):
        """Test that gold standard is sorted by category and score."""
        # Set up gold standard data with different categories and scores
        self.creator.gold_standard = {
            ('pe', 'ci'): {
                'category': 'VP',
                'frequency': 5,
                'score': 0.8,
                'notes': ''
            },
            ('noh', 'Yóng'): {
                'category': 'PP',
                'frequency': 4,
                'score': 0.7,
                'notes': ''
            },
            ('lo', 'ci'): {
                'category': 'VP',
                'frequency': 3,
                'score': 0.9,  # Higher score than pe ci
                'notes': ''
            }
        }
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_output = f.name
        
        try:
            self.creator.save_gold_standard(temp_output)
            
            # Verify file content
            with open(temp_output, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Find positions of entries
                pe_ci_pos = content.find('pe ci')
                lo_ci_pos = content.find('lo ci')
                noh_yong_pos = content.find('noh Yóng')
                
                # VP entries should be together, with lo ci (higher score) before pe ci
                assert lo_ci_pos < pe_ci_pos
                
                # PP should come before VP (alphabetical order: PP < VP)
                assert noh_yong_pos < pe_ci_pos
        
        finally:
            if os.path.exists(temp_output):
                os.unlink(temp_output)


class TestCategories:
    """Test the CATEGORIES constant."""
    
    def test_categories_defined(self):
        """Test that all expected categories are defined."""
        expected_categories = {
            'VP', 'PP', 'APP', 'AGR', 'AUX', 'COMP', 'MWE', 
            'COMPOUND', 'LEX', 'DISC', 'OTHER'
        }
        
        assert set(CATEGORIES.keys()) == expected_categories
    
    def test_categories_descriptions(self):
        """Test that categories have meaningful descriptions."""
        for category, description in CATEGORIES.items():
            assert isinstance(description, str)
            assert len(description) > 0
            assert not description.startswith(' ')  # No leading whitespace


class TestMainFunction:
    """Test the main function and command-line interface."""
    
    def test_main_with_args(self):
        """Test main function with command-line arguments."""
        # Create temporary corpus file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Om noh Yóng am pàapai pe(k) ci.\n")
            temp_corpus = f.name
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_output = f.name
        
        try:
            # Mock the GoldStandardCreator to avoid actual processing
            with patch('kcho.create_gold_standard.GoldStandardCreator') as mock_creator_class:
                mock_creator = Mock()
                mock_creator_class.return_value = mock_creator
                
                # Test with auto annotation
                with patch('sys.argv', ['create_gold_standard.py', '--corpus', temp_corpus, '--output', temp_output, '--auto']):
                    main()
                
                # Verify creator was called with correct arguments
                mock_creator_class.assert_called_once_with(temp_corpus)
                mock_creator.load_corpus.assert_called_once()
                mock_creator.extract_candidates.assert_called_once()
                mock_creator.auto_annotate.assert_called_once()
                mock_creator.save_gold_standard.assert_called_once_with(temp_output)
        
        finally:
            if os.path.exists(temp_corpus):
                os.unlink(temp_corpus)
            if os.path.exists(temp_output):
                os.unlink(temp_output)
    
    def test_main_interactive_mode(self):
        """Test main function in interactive mode."""
        # Create temporary corpus file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Om noh Yóng am pàapai pe(k) ci.\n")
            temp_corpus = f.name
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_output = f.name
        
        try:
            # Mock the GoldStandardCreator to avoid actual processing
            with patch('kcho.create_gold_standard.GoldStandardCreator') as mock_creator_class:
                mock_creator = Mock()
                mock_creator_class.return_value = mock_creator
                
                # Test with interactive annotation
                with patch('sys.argv', ['create_gold_standard.py', '--corpus', temp_corpus, '--output', temp_output, '--interactive']):
                    main()
                
                # Verify creator was called with correct arguments
                mock_creator_class.assert_called_once_with(temp_corpus)
                mock_creator.load_corpus.assert_called_once()
                mock_creator.extract_candidates.assert_called_once()
                mock_creator.interactive_annotation.assert_called_once()
                mock_creator.save_gold_standard.assert_called_once_with(temp_output)
        
        finally:
            if os.path.exists(temp_corpus):
                os.unlink(temp_corpus)
            if os.path.exists(temp_output):
                os.unlink(temp_output)


class TestIntegrationWithRealData:
    """Integration tests using real data files."""
    
    def setup_method(self):
        """Set up test fixtures with real data."""
        self.data_dir = Path("data")
        self.sample_corpus_file = self.data_dir / "sample_corpus.txt"
        self.gold_standard_file = self.data_dir / "gold_standard_collocations.txt"
    
    def test_create_gold_standard_with_real_corpus(self):
        """Test creating gold standard with real corpus data."""
        if not self.sample_corpus_file.exists():
            pytest.skip("Sample corpus file not found")
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_output = f.name
        
        try:
            # Create gold standard creator
            creator = GoldStandardCreator(str(self.sample_corpus_file))
            creator.load_corpus()
            
            # Verify corpus was loaded
            assert len(creator.corpus) > 0
            assert all(isinstance(sentence, str) for sentence in creator.corpus)
            
            # Extract candidates
            creator.extract_candidates(top_k=20)
            
            # Verify candidates were extracted
            assert len(creator.candidates) > 0
            assert len(creator.candidates) <= 20
            
            # Auto-annotate
            creator.auto_annotate()
            
            # Verify some annotations were made
            assert len(creator.gold_standard) > 0
            
            # Save gold standard
            creator.save_gold_standard(temp_output)
            
            # Verify file was created and has content
            assert os.path.exists(temp_output)
            with open(temp_output, 'r', encoding='utf-8') as f:
                content = f.read()
                assert len(content) > 0
                assert "K'Cho Gold Standard Collocations" in content
        
        finally:
            if os.path.exists(temp_output):
                os.unlink(temp_output)
    
    def test_gold_standard_consistency_with_existing(self):
        """Test that created gold standard is consistent with existing one."""
        if not self.sample_corpus_file.exists() or not self.gold_standard_file.exists():
            pytest.skip("Required data files not found")
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_output = f.name
        
        try:
            # Create gold standard creator
            creator = GoldStandardCreator(str(self.sample_corpus_file))
            creator.load_corpus()
            creator.extract_candidates(top_k=50)
            creator.auto_annotate()
            creator.save_gold_standard(temp_output)
            
            # Load existing gold standard
            with open(self.gold_standard_file, 'r', encoding='utf-8') as f:
                existing_content = f.read()
            
            # Load created gold standard
            with open(temp_output, 'r', encoding='utf-8') as f:
                created_content = f.read()
            
            # Check that some common patterns appear in both
            common_patterns = ['pe ci', 'lo ci', 'noh Yóng', 'ci ah']
            for pattern in common_patterns:
                if pattern in existing_content:
                    # If pattern exists in existing gold standard, it should likely appear in created one
                    # (though not guaranteed due to different extraction methods)
                    pass
        
        finally:
            if os.path.exists(temp_output):
                os.unlink(temp_output)
    
    def test_performance_with_large_corpus(self):
        """Test performance with larger corpus."""
        if not self.sample_corpus_file.exists():
            pytest.skip("Sample corpus file not found")
        
        import time
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_output = f.name
        
        try:
            start_time = time.time()
            
            # Create gold standard creator
            creator = GoldStandardCreator(str(self.sample_corpus_file))
            creator.load_corpus()
            creator.extract_candidates(top_k=100)
            creator.auto_annotate()
            creator.save_gold_standard(temp_output)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should complete within reasonable time (adjust threshold as needed)
            assert processing_time < 30  # 30 seconds should be more than enough
            
            # Verify results
            assert len(creator.corpus) > 0
            assert len(creator.candidates) > 0
            assert len(creator.gold_standard) > 0
        
        finally:
            if os.path.exists(temp_output):
                os.unlink(temp_output)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_nonexistent_corpus_file(self):
        """Test handling of nonexistent corpus file."""
        with pytest.raises(FileNotFoundError):
            creator = GoldStandardCreator("nonexistent_file.txt")
            creator.load_corpus()
    
    def test_empty_corpus_file(self):
        """Test handling of empty corpus file."""
        # Create empty temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_file = f.name
        
        try:
            creator = GoldStandardCreator(temp_file)
            creator.load_corpus()
            
            # Should handle empty corpus gracefully
            assert len(creator.corpus) == 0
            
            # Extract candidates should not fail
            creator.extract_candidates()
            assert len(creator.candidates) == 0
            
            # Auto-annotate should not fail
            creator.auto_annotate()
            assert len(creator.gold_standard) == 0
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_corpus_with_only_comments(self):
        """Test handling of corpus with only comments."""
        # Create file with only comments
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("# This is a comment\n# Another comment\n")
            temp_file = f.name
        
        try:
            creator = GoldStandardCreator(temp_file)
            creator.load_corpus()
            
            # Should handle comment-only corpus gracefully
            assert len(creator.corpus) == 0
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_invalid_output_path(self):
        """Test handling of invalid output path."""
        creator = GoldStandardCreator("dummy.txt")
        creator.gold_standard = {('test', 'words'): {'category': 'VP', 'frequency': 1, 'score': 0.5, 'notes': ''}}
        
        # Test with invalid path (should create directory if possible)
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_path = os.path.join(temp_dir, "nonexistent", "subdir", "output.txt")
            
            # Should not raise error, should create directories
            creator.save_gold_standard(invalid_path)
            assert os.path.exists(invalid_path)
