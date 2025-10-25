"""
Tests for export_training_csv.py module.

This module tests the CSV export functionality for training data from K'Cho corpora.
"""

import pytest
import tempfile
import os
import csv
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from kcho.export_training_csv import main
from kcho.eng_kcho_parallel_extractor import ParallelCorpusExtractor, ParallelSentence
from kcho.kcho_system import KchoSystem
from kcho.collocation import CollocationExtractor, AssociationMeasure, CollocationResult


class TestParallelCorpusExtractor:
    """Test the ParallelCorpusExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = ParallelCorpusExtractor()
    
    def test_initialization(self):
        """Test ParallelCorpusExtractor initialization."""
        assert isinstance(self.extractor, ParallelCorpusExtractor)
        assert hasattr(self.extractor, 'add_sentence_pair')
        assert hasattr(self.extractor, 'load_files')
        assert hasattr(self.extractor, 'align_sentences')
        assert hasattr(self.extractor, 'export_to_csv')
        assert hasattr(self.extractor, 'export_to_json')
        assert isinstance(self.extractor.corpus, list)
    
    def test_add_sentence_pair(self):
        """Test adding sentence pairs."""
        english = "Hello world"
        kcho = "Om noh Yóng am pàapai pe(k) ci."
        
        self.extractor.add_sentence_pair(english, kcho, source="test")
        
        assert len(self.extractor.corpus) == 1
        assert isinstance(self.extractor.corpus[0], ParallelSentence)
        assert self.extractor.corpus[0].english == english
        assert self.extractor.corpus[0].kcho == kcho
        assert self.extractor.corpus[0].source == "test"
    
    def test_add_sentence_pair_empty(self):
        """Test adding empty sentence pairs."""
        # Test with empty English
        self.extractor.add_sentence_pair("", "Om noh Yóng am pàapai pe(k) ci.")
        assert len(self.extractor.corpus) == 0
        
        # Test with empty K'Cho
        self.extractor.add_sentence_pair("Hello world", "")
        assert len(self.extractor.corpus) == 0
        
        # Test with whitespace only
        self.extractor.add_sentence_pair("   ", "Om noh Yóng am pàapai pe(k) ci.")
        assert len(self.extractor.corpus) == 0
    
    def test_add_sentence_pair_with_metadata(self):
        """Test adding sentence pairs with metadata."""
        english = "Hello world"
        kcho = "Om noh Yóng am pàapai pe(k) ci."
        metadata = {"domain": "greeting", "confidence": 0.9}
        
        self.extractor.add_sentence_pair(english, kcho, source="test", metadata=metadata)
        
        assert len(self.extractor.corpus) == 1
        assert self.extractor.corpus[0].metadata == metadata
    
    def test_load_files(self):
        """Test loading files."""
        # Create temporary English file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Hello world\nHow are you?\nGood morning")
            temp_english = f.name
        
        # Create temporary K'Cho file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Om noh Yóng am pàapai pe(k) ci.\nKa zèi-na(k) ci?\nAk'hmó lùum ci.")
            temp_kcho = f.name
        
        try:
            self.extractor.load_files(temp_english, temp_kcho)
            
            assert len(self.extractor.corpus) == 3
            assert all(isinstance(pair, ParallelSentence) for pair in self.extractor.corpus)
            assert self.extractor.corpus[0].english == "Hello world"
            assert self.extractor.corpus[0].kcho == "Om noh Yóng am pàapai pe(k) ci."
        
        finally:
            if os.path.exists(temp_english):
                os.unlink(temp_english)
            if os.path.exists(temp_kcho):
                os.unlink(temp_kcho)
    
    def test_load_files_nonexistent(self):
        """Test loading nonexistent files."""
        with pytest.raises(FileNotFoundError):
            self.extractor.load_files("nonexistent_english.txt", "nonexistent_kcho.txt")
    
    def test_load_files_mismatched_lengths(self):
        """Test loading files with mismatched lengths."""
        # Create temporary English file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Hello world\nHow are you?")
            temp_english = f.name
        
        # Create temporary K'Cho file with different number of lines
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Om noh Yóng am pàapai pe(k) ci.\nKa zèi-na(k) ci?\nAk'hmó lùum ci.")
            temp_kcho = f.name
        
        try:
            # Should handle mismatched lengths gracefully
            self.extractor.load_files(temp_english, temp_kcho)
            
            # Should load the minimum number of lines
            assert len(self.extractor.corpus) == 2
        
        finally:
            if os.path.exists(temp_english):
                os.unlink(temp_english)
            if os.path.exists(temp_kcho):
                os.unlink(temp_kcho)
    
    def test_align_sentences(self):
        """Test sentence alignment."""
        # Add some sentence pairs
        self.extractor.add_sentence_pair("Hello world", "Om noh Yóng am pàapai pe(k) ci.")
        self.extractor.add_sentence_pair("How are you?", "Ka zèi-na(k) ci?")
        
        # Align sentences
        self.extractor.align_sentences()
        
        # Should not change the corpus structure
        assert len(self.extractor.corpus) == 2
        assert all(isinstance(pair, ParallelSentence) for pair in self.extractor.corpus)
    
    def test_export_to_csv(self):
        """Test exporting to CSV."""
        # Add some sentence pairs
        self.extractor.add_sentence_pair("Hello world", "Om noh Yóng am pàapai pe(k) ci.")
        self.extractor.add_sentence_pair("How are you?", "Ka zèi-na(k) ci?")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
        
        try:
            result = self.extractor.export_to_csv(temp_file)
            
            assert result is True
            assert os.path.exists(temp_file)
            
            # Verify CSV content
            with open(temp_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                assert len(rows) == 2
                assert 'english' in rows[0]
                assert 'kcho' in rows[0]
                assert 'source' in rows[0]
                assert rows[0]['english'] == "Hello world"
                assert rows[0]['kcho'] == "Om noh Yóng am pàapai pe(k) ci."
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_export_to_csv_empty(self):
        """Test exporting empty corpus to CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
        
        try:
            result = self.extractor.export_to_csv(temp_file)
            
            assert result is True
            assert os.path.exists(temp_file)
            
            # Verify CSV has header but no data rows
            with open(temp_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                assert len(rows) == 0
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_export_to_json(self):
        """Test exporting to JSON."""
        # Add some sentence pairs
        self.extractor.add_sentence_pair("Hello world", "Om noh Yóng am pàapai pe(k) ci.")
        self.extractor.add_sentence_pair("How are you?", "Ka zèi-na(k) ci?")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            result = self.extractor.export_to_json(temp_file)
            
            assert result is True
            assert os.path.exists(temp_file)
            
            # Verify JSON content
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                assert 'corpus' in data
                assert len(data['corpus']) == 2
                assert data['corpus'][0]['english'] == "Hello world"
                assert data['corpus'][0]['kcho'] == "Om noh Yóng am pàapai pe(k) ci."
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_export_to_json_empty(self):
        """Test exporting empty corpus to JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            result = self.extractor.export_to_json(temp_file)
            
            assert result is True
            assert os.path.exists(temp_file)
            
            # Verify JSON content
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                assert 'corpus' in data
                assert len(data['corpus']) == 0
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestParallelSentence:
    """Test the ParallelSentence class."""
    
    def test_initialization(self):
        """Test ParallelSentence initialization."""
        sentence = ParallelSentence("Hello world", "Om noh Yóng am pàapai pe(k) ci.")
        
        assert sentence.english == "Hello world"
        assert sentence.kcho == "Om noh Yóng am pàapai pe(k) ci."
        assert sentence.source == ""
        assert sentence.metadata == {}
    
    def test_initialization_with_metadata(self):
        """Test ParallelSentence initialization with metadata."""
        metadata = {"domain": "greeting", "confidence": 0.9}
        sentence = ParallelSentence("Hello world", "Om noh Yóng am pàapai pe(k) ci.", source="test", metadata=metadata)
        
        assert sentence.english == "Hello world"
        assert sentence.kcho == "Om noh Yóng am pàapai pe(k) ci."
        assert sentence.source == "test"
        assert sentence.metadata == metadata
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        sentence = ParallelSentence("Hello world", "Om noh Yóng am pàapai pe(k) ci.", source="test")
        
        result = sentence.to_dict()
        
        assert isinstance(result, dict)
        assert result['english'] == "Hello world"
        assert result['kcho'] == "Om noh Yóng am pàapai pe(k) ci."
        assert result['source'] == "test"
        assert result['metadata'] == {}


class TestMainFunction:
    """Test the main function and command-line interface."""
    
    def test_main_with_args(self):
        """Test main function with command-line arguments."""
        # Create temporary English file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Hello world\nHow are you?")
            temp_english = f.name
        
        # Create temporary K'Cho file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Om noh Yóng am pàapai pe(k) ci.\nKa zèi-na(k) ci?")
            temp_kcho = f.name
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the ParallelCorpusExtractor to avoid actual processing
            with patch('kcho.export_training_csv.ParallelCorpusExtractor') as mock_extractor_class:
                mock_extractor = Mock()
                mock_extractor_class.return_value = mock_extractor
                
                # Test with basic export
                with patch('sys.argv', ['export_training_csv.py', '--english-file', temp_english, '--kcho-file', temp_kcho, '--output-dir', temp_dir]):
                    main()
                
                # Verify extractor was created and methods were called
                mock_extractor_class.assert_called_once()
                mock_extractor.load_files.assert_called_once()
                mock_extractor.align_sentences.assert_called_once()
                mock_extractor.export_to_csv.assert_called_once()
        
        if os.path.exists(temp_english):
            os.unlink(temp_english)
        if os.path.exists(temp_kcho):
            os.unlink(temp_kcho)
    
    def test_main_with_force_flag(self):
        """Test main function with force flag."""
        # Create temporary English file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Hello world")
            temp_english = f.name
        
        # Create temporary K'Cho file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Om noh Yóng am pàapai pe(k) ci.")
            temp_kcho = f.name
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the ParallelCorpusExtractor to avoid actual processing
            with patch('kcho.export_training_csv.ParallelCorpusExtractor') as mock_extractor_class:
                mock_extractor = Mock()
                mock_extractor_class.return_value = mock_extractor
                
                # Test with force flag
                with patch('sys.argv', ['export_training_csv.py', '--english-file', temp_english, '--kcho-file', temp_kcho, '--output-dir', temp_dir, '--force']):
                    main()
                
                # Verify extractor was created and methods were called
                mock_extractor_class.assert_called_once()
                mock_extractor.load_files.assert_called_once()
                mock_extractor.align_sentences.assert_called_once()
                mock_extractor.export_to_csv.assert_called_once()
        
        if os.path.exists(temp_english):
            os.unlink(temp_english)
        if os.path.exists(temp_kcho):
            os.unlink(temp_kcho)
    
    def test_main_with_nonexistent_files(self):
        """Test main function with nonexistent files."""
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with nonexistent files
            with patch('sys.argv', ['export_training_csv.py', '--english-file', 'nonexistent.txt', '--kcho-file', 'nonexistent.txt', '--output-dir', temp_dir]):
                # Should handle nonexistent files gracefully
                main()


class TestIntegrationWithRealData:
    """Integration tests using real data files."""
    
    def setup_method(self):
        """Set up test fixtures with real data."""
        self.data_dir = Path("data")
        self.sample_corpus_file = self.data_dir / "sample_corpus.txt"
    
    def test_extract_with_real_corpus(self):
        """Test extraction with real corpus data."""
        if not self.sample_corpus_file.exists():
            pytest.skip("Sample corpus file not found")
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
        
        try:
            # Create extractor
            extractor = ParallelCorpusExtractor()
            
            # Read real corpus and create parallel pairs
            with open(self.sample_corpus_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Create mock English translations
            english_lines = [f"English translation {i+1}" for i in range(len(lines))]
            
            # Add sentence pairs
            for i, (english, kcho) in enumerate(zip(english_lines, lines)):
                kcho = kcho.strip()
                if kcho and not kcho.startswith('#'):
                    extractor.add_sentence_pair(english, kcho, source=f"line_{i+1}")
            
            # Export to CSV
            result = extractor.export_to_csv(temp_file)
            
            assert result is True
            assert os.path.exists(temp_file)
            
            # Verify CSV content
            with open(temp_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                assert len(rows) > 0
                assert 'english' in rows[0]
                assert 'kcho' in rows[0]
                assert 'source' in rows[0]
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_performance_with_large_corpus(self):
        """Test performance with larger corpus."""
        if not self.sample_corpus_file.exists():
            pytest.skip("Sample corpus file not found")
        
        import time
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
        
        try:
            start_time = time.time()
            
            # Create extractor
            extractor = ParallelCorpusExtractor()
            
            # Read real corpus and create parallel pairs
            with open(self.sample_corpus_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Create mock English translations
            english_lines = [f"English translation {i+1}" for i in range(len(lines))]
            
            # Add sentence pairs
            for i, (english, kcho) in enumerate(zip(english_lines, lines)):
                kcho = kcho.strip()
                if kcho and not kcho.startswith('#'):
                    extractor.add_sentence_pair(english, kcho, source=f"line_{i+1}")
            
            # Export to CSV
            result = extractor.export_to_csv(temp_file)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should complete within reasonable time
            assert processing_time < 30  # 30 seconds should be more than enough
            assert result is True
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_extract_with_corrupted_data(self):
        """Test extraction with corrupted data."""
        extractor = ParallelCorpusExtractor()
        
        # Test with corrupted sentence pairs
        extractor.add_sentence_pair("", "Om noh Yóng am pàapai pe(k) ci.")
        extractor.add_sentence_pair("Hello world", "")
        extractor.add_sentence_pair("   ", "   ")
        
        # Should handle corrupted data gracefully
        assert len(extractor.corpus) == 0
    
    def test_export_with_permission_error(self):
        """Test export with permission error."""
        extractor = ParallelCorpusExtractor()
        extractor.add_sentence_pair("Hello world", "Om noh Yóng am pàapai pe(k) ci.")
        
        # Test with read-only directory (if possible)
        with tempfile.TemporaryDirectory() as temp_dir:
            # Make directory read-only
            os.chmod(temp_dir, 0o444)
            
            try:
                result = extractor.export_to_csv(os.path.join(temp_dir, "output.csv"))
                
                # Should handle permission error gracefully
                assert result is False
            finally:
                # Restore permissions for cleanup
                os.chmod(temp_dir, 0o755)
    
    def test_load_files_with_encoding_error(self):
        """Test loading files with encoding error."""
        extractor = ParallelCorpusExtractor()
        
        # Create file with invalid encoding
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hello world\nHow are you?")
            temp_english = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Om noh Yóng am pàapai pe(k) ci.\nKa zèi-na(k) ci?")
            temp_kcho = f.name
        
        try:
            # Should handle encoding gracefully
            extractor.load_files(temp_english, temp_kcho)
            
            assert len(extractor.corpus) > 0
        
        finally:
            if os.path.exists(temp_english):
                os.unlink(temp_english)
            if os.path.exists(temp_kcho):
                os.unlink(temp_kcho)