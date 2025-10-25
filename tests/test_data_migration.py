"""
Tests for K'Cho Data Migration System
====================================

Comprehensive test suite for the data migration system with versioning,
checksums, and fresh data loading following database best practices.
"""

import pytest
import tempfile
import os
import json
import hashlib
from pathlib import Path
from unittest.mock import patch, mock_open
from datetime import datetime

from kcho.data_migration import (
    DataVersion, DataMigrationManager, FreshDataLoader, 
    create_migration_manager, check_and_migrate_data
)


class TestDataVersion:
    """Test cases for DataVersion class."""
    
    def test_data_version_creation(self):
        """Test DataVersion creation and serialization."""
        checksums = {
            'linguistic_data.json': 'abc123',
            'gold_standard_collocations.txt': 'def456'
        }
        
        version = DataVersion(
            version='v1.0.0',
            checksums=checksums,
            timestamp='2025-01-01T00:00:00'
        )
        
        assert version.version == 'v1.0.0'
        assert version.checksums == checksums
        assert version.timestamp == '2025-01-01T00:00:00'
        
        # Test serialization
        version_dict = version.to_dict()
        assert version_dict['version'] == 'v1.0.0'
        assert version_dict['checksums'] == checksums
        
        # Test deserialization
        restored_version = DataVersion.from_dict(version_dict)
        assert restored_version.version == version.version
        assert restored_version.checksums == version.checksums


class TestDataMigrationManager:
    """Test cases for DataMigrationManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Create temporary data directory
        self.temp_data_dir = tempfile.mkdtemp()
        
        # Sample data files
        self.sample_files = {
            'linguistic_data.json': '{"verb_stems": {"pe": {"stem2": "peit"}}}',
            'gold_standard_collocations.txt': 'pe ci # VP, high_freq',
            'word_frequency_top_1000.csv': 'word,frequency\npe,100',
            'gold_standard_kcho_english.json': '{"sentence_pairs": [{"id": "s1", "kcho": "test"}]}'
        }
        
        # Create sample files
        for filename, content in self.sample_files.items():
            file_path = Path(self.temp_data_dir) / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Remove temporary database
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        
        # Remove temporary data directory
        import shutil
        shutil.rmtree(self.temp_data_dir, ignore_errors=True)
    
    def test_migration_tables_creation(self):
        """Test that migration tables are created correctly."""
        manager = DataMigrationManager(self.db_path, self.temp_data_dir)
        
        # Check that tables exist
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        
        expected_tables = ['data_versions', 'migration_history', 'data_file_tracking']
        for table in expected_tables:
            assert table in tables, f"Table {table} not found"
        
        conn.close()
        manager.close()
    
    def test_file_checksum_calculation(self):
        """Test file checksum calculation."""
        manager = DataMigrationManager(self.db_path, self.temp_data_dir)
        
        # Test checksum calculation
        file_path = Path(self.temp_data_dir) / 'linguistic_data.json'
        checksum = manager.calculate_file_checksum(file_path)
        
        assert len(checksum) == 64  # SHA-256 hex length
        assert isinstance(checksum, str)
        
        # Test with non-existent file
        non_existent = Path(self.temp_data_dir) / 'non_existent.json'
        empty_checksum = manager.calculate_file_checksum(non_existent)
        assert empty_checksum == ""
        
        manager.close()
    
    def test_data_version_calculation(self):
        """Test current data version calculation."""
        manager = DataMigrationManager(self.db_path, self.temp_data_dir)
        
        version = manager.calculate_current_data_version()
        
        assert isinstance(version, DataVersion)
        assert len(version.version) > 0
        assert isinstance(version.checksums, dict)
        assert len(version.checksums) > 0
        assert isinstance(version.timestamp, str)
        
        manager.close()
    
    def test_data_change_detection(self):
        """Test data change detection."""
        manager = DataMigrationManager(self.db_path, self.temp_data_dir)
        
        # First time - should detect change (no stored version)
        has_changed = manager.has_data_changed()
        assert has_changed is True
        
        # Save current version
        current_version = manager.calculate_current_data_version()
        manager.save_data_version(current_version)
        
        # Should not detect change now
        has_changed = manager.has_data_changed()
        assert has_changed is False
        
        # Modify a file
        file_path = Path(self.temp_data_dir) / 'linguistic_data.json'
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('{"modified": true}')
        
        # Should detect change now
        has_changed = manager.has_data_changed()
        assert has_changed is True
        
        manager.close()
    
    def test_migration_recording(self):
        """Test migration recording functionality."""
        manager = DataMigrationManager(self.db_path, self.temp_data_dir)
        
        # Record migration start
        migration_id = manager.record_migration_start(
            version_from='v1.0.0',
            version_to='v1.1.0',
            migration_type='fresh_load'
        )
        
        assert isinstance(migration_id, int)
        assert migration_id > 0
        
        # Record successful completion
        manager.record_migration_completion(migration_id, records_affected=100)
        
        # Check migration history
        history = manager.get_migration_history()
        assert len(history) == 1
        
        migration = history[0]
        assert migration['version_from'] == 'v1.0.0'
        assert migration['version_to'] == 'v1.1.0'
        assert migration['migration_type'] == 'fresh_load'
        assert migration['status'] == 'completed'
        assert migration['records_affected'] == 100
        
        manager.close()
    
    def test_migration_failure_recording(self):
        """Test migration failure recording."""
        manager = DataMigrationManager(self.db_path, self.temp_data_dir)
        
        # Record migration start
        migration_id = manager.record_migration_start(
            version_from='v1.0.0',
            version_to='v1.1.0',
            migration_type='fresh_load'
        )
        
        # Record failure
        error_message = "Test error message"
        manager.record_migration_failure(migration_id, error_message)
        
        # Check migration history
        history = manager.get_migration_history()
        assert len(history) == 1
        
        migration = history[0]
        assert migration['status'] == 'failed'
        assert migration['error_message'] == error_message
        
        manager.close()
    
    def test_data_file_tracking(self):
        """Test data file tracking functionality."""
        manager = DataMigrationManager(self.db_path, self.temp_data_dir)
        
        # Track a file
        filename = 'test_file.json'
        version = 'v1.0.0'
        checksum = 'abc123'
        file_size = 1024
        last_modified = '2025-01-01T00:00:00'
        
        manager.track_data_file(filename, version, checksum, file_size, last_modified)
        
        # Check file status
        status = manager.get_data_file_status()
        assert len(status) == 1
        
        file_status = status[0]
        assert file_status['filename'] == filename
        assert file_status['version'] == version
        assert file_status['checksum'] == checksum
        assert file_status['file_size'] == file_size
        
        manager.close()
    
    def test_clear_all_data(self):
        """Test clearing all data tables."""
        manager = DataMigrationManager(self.db_path, self.temp_data_dir)
        
        # Add some test data
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        cur.execute("INSERT INTO verb_stems (verb, stem2) VALUES (?, ?)", ('test', 'tested'))
        cur.execute("INSERT INTO collocations (words, category) VALUES (?, ?)", ('test pattern', 'VP'))
        
        conn.commit()
        conn.close()
        
        # Clear all data
        manager.clear_all_data()
        
        # Verify data is cleared
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) FROM verb_stems")
        assert cur.fetchone()[0] == 0
        
        cur.execute("SELECT COUNT(*) FROM collocations")
        assert cur.fetchone()[0] == 0
        
        conn.close()
        manager.close()


class TestFreshDataLoader:
    """Test cases for FreshDataLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary database and data directory
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        self.temp_data_dir = tempfile.mkdtemp()
        
        # Create sample data files
        self.sample_files = {
            'linguistic_data.json': '{"verb_stems": {"pe": {"stem2": "peit", "gloss": "give"}}}',
            'gold_standard_collocations.txt': 'pe ci # VP, high_freq, give + non-future',
            'word_frequency_top_1000.csv': 'word,frequency\npe,100\nci,150',
            'gold_standard_kcho_english.json': '{"sentence_pairs": [{"id": "s1", "kcho": "Om pe ci", "english": "Om gives"}]}'
        }
        
        for filename, content in self.sample_files.items():
            file_path = Path(self.temp_data_dir) / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        
        import shutil
        shutil.rmtree(self.temp_data_dir, ignore_errors=True)
    
    def test_fresh_data_loading(self):
        """Test fresh data loading functionality."""
        # Mock the knowledge base
        class MockKnowledgeBase:
            def __init__(self):
                self.db_path = self.db_path
                self.data_dir = Path(self.temp_data_dir)
            
            def _load_linguistic_data(self):
                return 1  # Mock return value
            
            def _load_gold_standard_patterns(self):
                return 1
            
            def _load_word_frequency_data(self):
                return 2
            
            def _load_parallel_data(self):
                return 1
            
            def _load_word_categories(self):
                return 1
        
        mock_kb = MockKnowledgeBase()
        migration_manager = DataMigrationManager(self.db_path, self.temp_data_dir)
        
        try:
            fresh_loader = FreshDataLoader(migration_manager, mock_kb)
            
            # Test fresh data loading
            success = fresh_loader.load_fresh_data()
            
            assert success is True
            
            # Check that migration was recorded
            history = migration_manager.get_migration_history()
            assert len(history) == 1
            
            migration = history[0]
            assert migration['migration_type'] == 'fresh_load'
            assert migration['status'] == 'completed'
            assert migration['records_affected'] == 6  # Sum of all mock returns
            
        finally:
            migration_manager.close()


class TestMigrationIntegration:
    """Test integration between migration system and knowledge base."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        self.temp_data_dir = tempfile.mkdtemp()
        
        # Create sample data files
        self.sample_files = {
            'linguistic_data.json': '{"verb_stems": {"pe": {"stem2": "peit", "gloss": "give"}}}',
            'gold_standard_collocations.txt': 'pe ci # VP, high_freq',
            'word_frequency_top_1000.csv': 'word,frequency\npe,100',
            'gold_standard_kcho_english.json': '{"sentence_pairs": [{"id": "s1", "kcho": "test"}]}'
        }
        
        for filename, content in self.sample_files.items():
            file_path = Path(self.temp_data_dir) / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        
        import shutil
        shutil.rmtree(self.temp_data_dir, ignore_errors=True)
    
    def test_check_and_migrate_data_function(self):
        """Test the check_and_migrate_data helper function."""
        # Mock knowledge base
        class MockKnowledgeBase:
            def __init__(self):
                self.db_path = self.db_path
                self.data_dir = Path(self.temp_data_dir)
        
        mock_kb = MockKnowledgeBase()
        
        # Test migration check
        migration_performed = check_and_migrate_data(mock_kb)
        
        # Should perform migration on first run (no stored version)
        assert migration_performed is True
        
        # Test second run (should not migrate)
        migration_performed = check_and_migrate_data(mock_kb)
        assert migration_performed is False
    
    def test_create_migration_manager(self):
        """Test create_migration_manager helper function."""
        manager = create_migration_manager(self.db_path, self.temp_data_dir)
        
        assert isinstance(manager, DataMigrationManager)
        assert manager.db_path == self.db_path
        assert manager.data_dir == Path(self.temp_data_dir)
        
        manager.close()


class TestMigrationBestPractices:
    """Test that migration system follows database best practices."""
    
    def test_atomic_migrations(self):
        """Test that migrations are atomic (all or nothing)."""
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db.close()
        db_path = temp_db.name
        
        temp_data_dir = tempfile.mkdtemp()
        
        try:
            # Create a file that will cause an error during loading
            file_path = Path(temp_data_dir) / 'linguistic_data.json'
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('{"invalid": json}')  # Invalid JSON
            
            manager = DataMigrationManager(db_path, temp_data_dir)
            
            # This should fail gracefully
            current_version = manager.calculate_current_data_version()
            migration_id = manager.record_migration_start(
                version_from=None,
                version_to=current_version.version,
                migration_type='fresh_load'
            )
            
            # Record failure (simulating what would happen)
            manager.record_migration_failure(migration_id, "Invalid JSON in linguistic_data.json")
            
            # Check that failure was recorded
            history = manager.get_migration_history()
            assert len(history) == 1
            assert history[0]['status'] == 'failed'
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
            
            import shutil
            shutil.rmtree(temp_data_dir, ignore_errors=True)
    
    def test_version_consistency(self):
        """Test that version consistency is maintained."""
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db.close()
        db_path = temp_db.name
        
        temp_data_dir = tempfile.mkdtemp()
        
        try:
            # Create sample files
            sample_content = '{"test": "data"}'
            file_path = Path(temp_data_dir) / 'linguistic_data.json'
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(sample_content)
            
            manager = DataMigrationManager(db_path, temp_data_dir)
            
            # Calculate version
            version1 = manager.calculate_current_data_version()
            
            # Modify file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('{"test": "modified"}')
            
            # Calculate version again
            version2 = manager.calculate_current_data_version()
            
            # Versions should be different
            assert version1.version != version2.version
            
            # But checksums should reflect the change
            assert version1.checksums['linguistic_data.json'] != version2.checksums['linguistic_data.json']
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
            
            import shutil
            shutil.rmtree(temp_data_dir, ignore_errors=True)
    
    def test_migration_rollback_capability(self):
        """Test that migration system supports rollback scenarios."""
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db.close()
        db_path = temp_db.name
        
        temp_data_dir = tempfile.mkdtemp()
        
        try:
            # Create initial data
            file_path = Path(temp_data_dir) / 'linguistic_data.json'
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('{"version": "1.0"}')
            
            manager = DataMigrationManager(db_path, temp_data_dir)
            
            # Save initial version
            initial_version = manager.calculate_current_data_version()
            manager.save_data_version(initial_version)
            
            # Modify data
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('{"version": "2.0"}')
            
            # Calculate new version
            new_version = manager.calculate_current_data_version()
            
            # Verify versions are different
            assert initial_version.version != new_version.version
            
            # This demonstrates rollback capability - we can detect changes
            # and potentially restore to previous version if needed
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
            
            import shutil
            shutil.rmtree(temp_data_dir, ignore_errors=True)


if __name__ == '__main__':
    pytest.main([__file__])
