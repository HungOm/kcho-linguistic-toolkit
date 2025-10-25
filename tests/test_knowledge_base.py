"""
Tests for KchoKnowledgeBase - SQLite Backend Implementation
==========================================================

Comprehensive test suite for the SQLite-based knowledge base system.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, mock_open

from kcho.knowledge_base import KchoKnowledgeBase, init_database, VerbStemModel, CollocationModel


class TestKchoKnowledgeBase:
    """Test cases for KchoKnowledgeBase class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Create temporary data directory
        self.temp_data_dir = tempfile.mkdtemp()
        
        # Sample linguistic data
        self.sample_linguistic_data = {
            "verb_stems": {
                "pe": {"stem2": "peit", "gloss": "give", "pattern": "suffix_it"},
                "lo": {"stem2": "loo", "gloss": "come", "pattern": "vowel_change"},
                "that": {"stem2": "thah", "gloss": "beat", "pattern": "t_to_h"}
            },
            "collocations": {
                "pe ci": {"category": "VP", "frequency": "high"},
                "lo ci": {"category": "VP", "frequency": "high"}
            }
        }
        
        # Sample gold standard patterns
        self.sample_gold_standard = """# Gold Standard Collocations
pe ci          # VP, high_freq, give + non-future
lo ci          # VP, high_freq, come + non-future
noh Yóng       # PP, high_freq, postposition + proper_noun
"""
        
        # Sample word frequencies
        self.sample_word_frequencies = """word,frequency
pe,100
lo,80
ci,150
noh,60
"""
        
        # Sample parallel data
        self.sample_parallel_data = {
            "sentence_pairs": [
                {
                    "id": "s1",
                    "kcho": "Om noh Yóng am pàapai pe ci.",
                    "english": "Om gave flowers to Yong.",
                    "source": "test",
                    "linguistic_features": ["VP", "PP"]
                }
            ]
        }
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Remove temporary database
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        
        # Remove temporary data directory
        import shutil
        shutil.rmtree(self.temp_data_dir, ignore_errors=True)
    
    def _create_sample_files(self):
        """Create sample data files for testing."""
        # Create linguistic data file
        linguistic_file = Path(self.temp_data_dir) / 'linguistic_data.json'
        with open(linguistic_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_linguistic_data, f)
        
        # Create gold standard file
        gold_standard_file = Path(self.temp_data_dir) / 'gold_standard_collocations.txt'
        with open(gold_standard_file, 'w', encoding='utf-8') as f:
            f.write(self.sample_gold_standard)
        
        # Create word frequency file
        freq_file = Path(self.temp_data_dir) / 'word_frequency_top_1000.csv'
        with open(freq_file, 'w', encoding='utf-8') as f:
            f.write(self.sample_word_frequencies)
        
        # Create parallel data file
        parallel_file = Path(self.temp_data_dir) / 'gold_standard_kcho_english.json'
        with open(parallel_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_parallel_data, f)
    
    def test_database_creation(self):
        """Test database schema creation."""
        kb = KchoKnowledgeBase(db_path=self.db_path, data_dir=self.temp_data_dir)
        
        # Check that tables were created
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Check tables exist
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        
        expected_tables = ['verb_stems', 'collocations', 'word_frequencies', 
                          'parallel_sentences', 'raw_texts', 'word_categories']
        for table in expected_tables:
            assert table in tables, f"Table {table} not found"
        
        # Check indexes exist
        cur.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [row[0] for row in cur.fetchall()]
        
        expected_indexes = ['idx_colloc_category', 'idx_colloc_source', 'idx_word_freq']
        for index in expected_indexes:
            assert any(index in idx for idx in indexes), f"Index {index} not found"
        
        conn.close()
        kb.close()
    
    def test_data_loading(self):
        """Test loading data from files."""
        self._create_sample_files()
        
        kb = KchoKnowledgeBase(db_path=self.db_path, data_dir=self.temp_data_dir)
        
        # Check verb stems were loaded
        verb_stem = kb.get_verb_stem('pe')
        assert verb_stem is not None
        assert verb_stem['verb'] == 'pe'
        assert verb_stem['stem2'] == 'peit'
        assert verb_stem['gloss'] == 'give'
        
        # Check collocations were loaded
        vp_collocations = kb.get_collocations_by_category('VP')
        assert len(vp_collocations) >= 2  # At least pe ci and lo ci
        
        # Check word frequencies were loaded
        freq = kb.get_word_frequency('pe')
        assert freq == 100
        
        # Check parallel sentences were loaded
        sentences = kb.get_parallel_sentences()
        assert len(sentences) >= 1
        
        kb.close()
    
    def test_crud_operations(self):
        """Test CRUD operations."""
        kb = KchoKnowledgeBase(db_path=self.db_path, data_dir=self.temp_data_dir)
        
        # Test adding collocation
        success = kb.add_collocation(
            words='test pattern',
            category='VP',
            frequency='medium',
            notes='Test collocation',
            source='test',
            confidence=0.8
        )
        assert success
        
        # Verify it was added
        collocations = kb.search_collocations('test pattern')
        assert len(collocations) == 1
        assert collocations[0]['words'] == 'test pattern'
        assert collocations[0]['category'] == 'VP'
        assert collocations[0]['confidence'] == 0.8
        
        # Test inserting raw text
        success = kb.insert_raw_text(
            text='Test raw text',
            source='test',
            metadata={'type': 'test'}
        )
        assert success
        
        # Verify it was added
        stats = kb.get_statistics()
        assert stats['raw_texts'] >= 1
        
        kb.close()
    
    def test_query_methods(self):
        """Test query methods."""
        self._create_sample_files()
        kb = KchoKnowledgeBase(db_path=self.db_path, data_dir=self.temp_data_dir)
        
        # Test get_verb_stem
        verb_stem = kb.get_verb_stem('pe')
        assert verb_stem is not None
        assert verb_stem['verb'] == 'pe'
        
        # Test get_collocations_by_category
        vp_collocations = kb.get_collocations_by_category('VP')
        assert len(vp_collocations) > 0
        
        # Test get_collocations_by_source
        gold_collocations = kb.get_collocations_by_source('gold_standard_txt')
        assert len(gold_collocations) > 0
        
        # Test search_collocations
        search_results = kb.search_collocations('pe')
        assert len(search_results) > 0
        
        # Test get_word_frequency
        freq = kb.get_word_frequency('pe')
        assert freq == 100
        
        # Test get_word_categories
        categories = kb.get_word_categories('pe')
        assert 'verb' in categories
        
        kb.close()
    
    def test_caching(self):
        """Test LRU caching functionality."""
        kb = KchoKnowledgeBase(db_path=self.db_path, data_dir=self.temp_data_dir)
        
        # First call should hit database
        verb_stem1 = kb.get_verb_stem('pe')
        
        # Second call should hit cache
        verb_stem2 = kb.get_verb_stem('pe')
        
        assert verb_stem1 == verb_stem2
        
        kb.close()
    
    def test_statistics(self):
        """Test statistics method."""
        self._create_sample_files()
        kb = KchoKnowledgeBase(db_path=self.db_path, data_dir=self.temp_data_dir)
        
        stats = kb.get_statistics()
        
        assert 'database_path' in stats
        assert 'verb_stems' in stats
        assert 'collocations' in stats
        assert 'word_frequencies' in stats
        assert 'parallel_sentences' in stats
        assert 'raw_texts' in stats
        assert 'word_categories' in stats
        
        # Check that we have some data
        assert stats['verb_stems'] > 0
        assert stats['collocations'] > 0
        assert stats['word_frequencies'] > 0
        
        kb.close()
    
    def test_context_manager(self):
        """Test context manager functionality."""
        self._create_sample_files()
        
        with KchoKnowledgeBase(db_path=self.db_path, data_dir=self.temp_data_dir) as kb:
            # Should be able to use the knowledge base
            stats = kb.get_statistics()
            assert stats['verb_stems'] > 0
        
        # Database should be closed after context exit
        # (We can't easily test this without accessing private attributes)
    
    def test_in_memory_database(self):
        """Test in-memory database functionality."""
        kb = KchoKnowledgeBase(in_memory=True, data_dir=self.temp_data_dir)
        
        # Should work with in-memory database
        stats = kb.get_statistics()
        assert 'database_path' in stats
        assert stats['database_path'] == ':memory:'
        
        kb.close()
    
    def test_pattern_discovery_from_raw(self):
        """Test pattern discovery from raw texts."""
        kb = KchoKnowledgeBase(db_path=self.db_path, data_dir=self.temp_data_dir)
        
        # Add some raw texts
        kb.insert_raw_text("Om noh Yóng am pàapai pe ci.", source="test")
        kb.insert_raw_text("Yóng am pàapai pe ci ah k'chàang.", source="test")
        
        # Mock the collocation extractor to avoid complex dependencies
        with patch('kcho.knowledge_base.CollocationExtractor') as mock_extractor:
            mock_result = type('CollocationResult', (), {
                'collocation': 'pe ci',
                'category': 'VP',
                'score': 0.8,
                'notes': 'Test pattern'
            })()
            
            mock_instance = mock_extractor.return_value
            mock_instance.extract_collocations.return_value = [mock_result]
            
            # Run pattern discovery
            kb.discover_patterns_from_raw()
            
            # Check that patterns were added
            collocations = kb.get_collocations_by_source('discovered')
            assert len(collocations) > 0
        
        kb.close()
    
    def test_error_handling(self):
        """Test error handling for invalid data."""
        kb = KchoKnowledgeBase(db_path=self.db_path, data_dir=self.temp_data_dir)
        
        # Test adding invalid collocation (should not crash)
        success = kb.add_collocation(
            words='',  # Empty words
            category='',  # Empty category
            source='test'
        )
        # Should handle gracefully (may return False or True depending on validation)
        
        # Test inserting raw text with invalid metadata
        success = kb.insert_raw_text(
            text='Test text',
            source='test',
            metadata={'invalid': object()}  # Non-serializable object
        )
        # Should handle gracefully
        
        kb.close()


class TestPydanticModels:
    """Test Pydantic validation models."""
    
    def test_verb_stem_model(self):
        """Test VerbStemModel validation."""
        # Valid data
        valid_data = {
            'verb': 'pe',
            'stem2': 'peit',
            'gloss': 'give',
            'pattern': 'suffix_it'
        }
        
        model = VerbStemModel(**valid_data)
        assert model.verb == 'pe'
        assert model.stem2 == 'peit'
        assert model.gloss == 'give'
        assert model.pattern == 'suffix_it'
        
        # Test with minimal data
        minimal_data = {'verb': 'lo'}
        model = VerbStemModel(**minimal_data)
        assert model.verb == 'lo'
        assert model.stem2 is None
        assert model.gloss is None
        assert model.pattern is None
    
    def test_collocation_model(self):
        """Test CollocationModel validation."""
        # Valid data
        valid_data = {
            'words': 'pe ci',
            'category': 'VP',
            'frequency': 'high',
            'notes': 'Test pattern',
            'source': 'gold_standard'
        }
        
        model = CollocationModel(**valid_data)
        assert model.words == 'pe ci'
        assert model.category == 'VP'
        assert model.frequency == 'high'
        assert model.notes == 'Test pattern'
        assert model.source == 'gold_standard'
        
        # Test with minimal data
        minimal_data = {
            'words': 'lo ci',
            'category': 'VP'
        }
        model = CollocationModel(**minimal_data)
        assert model.words == 'lo ci'
        assert model.category == 'VP'
        assert model.frequency is None
        assert model.notes is None
        assert model.source == 'unknown'  # Default value


class TestInitDatabase:
    """Test init_database helper function."""
    
    def test_init_database(self):
        """Test init_database helper function."""
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db.close()
        
        try:
            kb = init_database(db_path=temp_db.name)
            
            # Should return a KchoKnowledgeBase instance
            assert isinstance(kb, KchoKnowledgeBase)
            
            # Should be able to get statistics
            stats = kb.get_statistics()
            assert 'database_path' in stats
            
            kb.close()
            
        finally:
            os.unlink(temp_db.name)


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_comprehensive_knowledge_compatibility(self):
        """Test that ComprehensiveKchoKnowledge maintains backward compatibility."""
        from kcho.kcho_system import ComprehensiveKchoKnowledge
        
        # Should be able to create instance
        kb = ComprehensiveKchoKnowledge(in_memory=True)
        
        # Should have expected attributes
        assert hasattr(kb, 'VERB_STEMS')
        assert hasattr(kb, 'gold_standard_patterns')
        assert hasattr(kb, 'word_frequency_data')
        assert hasattr(kb, 'all_words')
        assert hasattr(kb, 'word_categories')
        
        # Should have expected methods
        assert hasattr(kb, 'is_gold_standard_pattern')
        assert hasattr(kb, 'get_pattern_confidence')
        
        # Should be able to use new methods
        assert hasattr(kb, 'add_collocation')
        assert hasattr(kb, 'insert_raw_text')
        assert hasattr(kb, 'get_statistics')
        
        kb.close()
    
    def test_api_layer_compatibility(self):
        """Test that KchoAPILayer works with both old and new knowledge bases."""
        from kcho.api_layer import KchoAPILayer
        from kcho.kcho_system import ComprehensiveKchoKnowledge
        
        # Test with new knowledge base
        kb = ComprehensiveKchoKnowledge(in_memory=True)
        api = KchoAPILayer(kb)
        
        # Should be able to process queries
        response = api.process_query("What are the verb patterns?")
        assert 'query' in response
        assert 'response_type' in response
        assert 'data' in response
        
        # Should be able to get knowledge summary
        summary = api.get_knowledge_summary()
        assert 'metadata' in summary
        
        kb.close()


if __name__ == '__main__':
    pytest.main([__file__])
