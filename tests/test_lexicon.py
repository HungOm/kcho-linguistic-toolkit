"""
Test suite for Lexicon Management in kcho_system.py

Tests LexiconEntry dataclass and KchoLexicon database operations
"""

import pytest
import tempfile
import os
import sqlite3
from unittest.mock import patch, MagicMock
from kcho.kcho_system import LexiconEntry, KchoLexicon


class TestLexiconEntry:
    """Test LexiconEntry dataclass functionality"""
    
    def test_lexicon_entry_creation(self):
        """Test basic lexicon entry creation"""
        entry = LexiconEntry(
            headword="om",
            pos="V",
            stem1="om",
            stem2="law",
            gloss_en="go",
            gloss_my="သွား",
            definition="To move from one place to another",
            examples=["Om noh Yong ci", "Om noh paapai ci"],
            frequency=25,
            semantic_field="motion"
        )
        
        assert entry.headword == "om"
        assert entry.pos == "V"
        assert entry.stem1 == "om"
        assert entry.stem2 == "law"
        assert entry.gloss_en == "go"
        assert entry.gloss_my == "သွား"
        assert entry.definition == "To move from one place to another"
        assert len(entry.examples) == 2
        assert entry.examples[0] == "Om noh Yong ci"
        assert entry.frequency == 25
        assert entry.semantic_field == "motion"
    
    def test_lexicon_entry_minimal(self):
        """Test lexicon entry with minimal required fields"""
        entry = LexiconEntry(
            headword="noh",
            pos="P"
        )
        
        assert entry.headword == "noh"
        assert entry.pos == "P"
        assert entry.stem1 is None
        assert entry.stem2 is None
        assert entry.gloss_en == ""
        assert entry.gloss_my == ""
        assert entry.definition == ""
        assert entry.examples == []
        assert entry.frequency == 0
        assert entry.semantic_field == ""
    
    def test_lexicon_entry_to_dict(self):
        """Test lexicon entry to_dict conversion"""
        entry = LexiconEntry(
            headword="ci",
            pos="T",
            gloss_en="PAST",
            definition="Past tense marker",
            examples=["Om ci", "Law ci"],
            frequency=100
        )
        
        entry_dict = entry.to_dict()
        
        assert isinstance(entry_dict, dict)
        assert entry_dict["headword"] == "ci"
        assert entry_dict["pos"] == "T"
        assert entry_dict["gloss_en"] == "PAST"
        assert entry_dict["definition"] == "Past tense marker"
        assert entry_dict["examples"] == ["Om ci", "Law ci"]
        assert entry_dict["frequency"] == 100
    
    def test_lexicon_entry_equality(self):
        """Test lexicon entry equality"""
        entry1 = LexiconEntry("om", "V", "om", "law", "go")
        entry2 = LexiconEntry("om", "V", "om", "law", "go")
        entry3 = LexiconEntry("law", "V", "law", "om", "come")
        
        assert entry1 == entry2
        assert entry1 != entry3


class TestKchoLexicon:
    """Test KchoLexicon database functionality"""
    
    def setup_method(self):
        """Set up test fixtures with temporary database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.lexicon = KchoLexicon(self.temp_db.name)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.lexicon.close()
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_initialization(self):
        """Test lexicon initialization"""
        assert hasattr(self.lexicon, 'db_path')
        assert hasattr(self.lexicon, 'conn')
        assert isinstance(self.lexicon.conn, sqlite3.Connection)
        assert self.lexicon.db_path == self.temp_db.name
    
    def test_database_schema_creation(self):
        """Test that database schema is created correctly"""
        cursor = self.lexicon.conn.cursor()
        
        # Check entries table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='entries'")
        assert cursor.fetchone() is not None
        
        # Check examples table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='examples'")
        assert cursor.fetchone() is not None
        
        # Check index exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_headword'")
        assert cursor.fetchone() is not None
    
    def test_add_entry_basic(self):
        """Test adding basic lexicon entry"""
        entry = LexiconEntry(
            headword="om",
            pos="V",
            stem1="om",
            stem2="law",
            gloss_en="go",
            definition="To move from one place to another",
            examples=["Om noh Yong ci"],
            frequency=25,
            semantic_field="motion"
        )
        
        entry_id = self.lexicon.add_entry(entry)
        
        assert isinstance(entry_id, int)
        assert entry_id > 0
        
        # Verify entry was added
        retrieved_entry = self.lexicon.get_entry("om")
        assert retrieved_entry is not None
        assert retrieved_entry.headword == "om"
        assert retrieved_entry.pos == "V"
        assert retrieved_entry.stem1 == "om"
        assert retrieved_entry.stem2 == "law"
        assert retrieved_entry.gloss_en == "go"
        assert retrieved_entry.frequency == 25
        assert retrieved_entry.examples == ["Om noh Yong ci"]
    
    def test_add_entry_with_examples(self):
        """Test adding entry with multiple examples"""
        entry = LexiconEntry(
            headword="noh",
            pos="P",
            gloss_en="OBJ",
            definition="Object marker",
            examples=["Om noh Yong ci", "Law noh paapai ci", "Ak'hmó noh Khanpughi ci"],
            frequency=50
        )
        
        entry_id = self.lexicon.add_entry(entry)
        
        retrieved_entry = self.lexicon.get_entry("noh")
        assert retrieved_entry is not None
        assert len(retrieved_entry.examples) == 3
        assert "Om noh Yong ci" in retrieved_entry.examples
        assert "Law noh paapai ci" in retrieved_entry.examples
        assert "Ak'hmó noh Khanpughi ci" in retrieved_entry.examples
    
    def test_add_entry_replace_existing(self):
        """Test that adding existing entry replaces it"""
        entry1 = LexiconEntry("om", "V", "om", "law", "go", frequency=10)
        entry2 = LexiconEntry("om", "V", "om", "law", "go", frequency=20)
        
        # Add first entry
        entry_id1 = self.lexicon.add_entry(entry1)
        
        # Add second entry (should replace)
        entry_id2 = self.lexicon.add_entry(entry2)
        
        # Should get the updated entry
        retrieved_entry = self.lexicon.get_entry("om")
        assert retrieved_entry.frequency == 20  # Updated frequency
    
    def test_get_entry_existing(self):
        """Test retrieving existing entry"""
        entry = LexiconEntry(
            headword="ci",
            pos="T",
            gloss_en="PAST",
            definition="Past tense marker",
            examples=["Om ci"],
            frequency=100
        )
        
        self.lexicon.add_entry(entry)
        retrieved_entry = self.lexicon.get_entry("ci")
        
        assert retrieved_entry is not None
        assert retrieved_entry.headword == "ci"
        assert retrieved_entry.pos == "T"
        assert retrieved_entry.gloss_en == "PAST"
        assert retrieved_entry.definition == "Past tense marker"
        assert retrieved_entry.examples == ["Om ci"]
        assert retrieved_entry.frequency == 100
    
    def test_get_entry_nonexistent(self):
        """Test retrieving non-existent entry"""
        retrieved_entry = self.lexicon.get_entry("nonexistent")
        assert retrieved_entry is None
    
    def test_update_frequency(self):
        """Test updating word frequency"""
        entry = LexiconEntry("om", "V", frequency=10)
        self.lexicon.add_entry(entry)
        
        # Update frequency
        self.lexicon.update_frequency("om", 5)
        
        retrieved_entry = self.lexicon.get_entry("om")
        assert retrieved_entry.frequency == 15  # 10 + 5
    
    def test_update_frequency_nonexistent(self):
        """Test updating frequency for non-existent word"""
        # Should not raise error
        self.lexicon.update_frequency("nonexistent", 5)
        
        # Verify no entry was created
        retrieved_entry = self.lexicon.get_entry("nonexistent")
        assert retrieved_entry is None
    
    def test_export_json_basic(self):
        """Test exporting lexicon to JSON"""
        # Add some entries
        entries = [
            LexiconEntry("om", "V", "om", "law", "go", frequency=25),
            LexiconEntry("noh", "P", gloss_en="OBJ", frequency=50),
            LexiconEntry("ci", "T", gloss_en="PAST", frequency=100)
        ]
        
        for entry in entries:
            self.lexicon.add_entry(entry)
        
        # Export to temporary file
        temp_json = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        temp_json.close()
        
        try:
            self.lexicon.export_json(temp_json.name)
            
            # Verify file was created and has content
            assert os.path.exists(temp_json.name)
            
            import json
            with open(temp_json.name, 'r', encoding='utf-8') as f:
                exported_data = json.load(f)
            
            assert isinstance(exported_data, list)
            assert len(exported_data) == 3
            
            # Check that entries are sorted by frequency (highest first)
            frequencies = [entry['frequency'] for entry in exported_data]
            assert frequencies == [100, 50, 25]  # Sorted descending
            
        finally:
            if os.path.exists(temp_json.name):
                os.unlink(temp_json.name)
    
    def test_export_json_with_examples(self):
        """Test exporting lexicon with examples"""
        entry = LexiconEntry(
            headword="om",
            pos="V",
            examples=["Om noh Yong ci", "Om noh paapai ci"],
            frequency=25
        )
        
        self.lexicon.add_entry(entry)
        
        temp_json = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        temp_json.close()
        
        try:
            self.lexicon.export_json(temp_json.name)
            
            import json
            with open(temp_json.name, 'r', encoding='utf-8') as f:
                exported_data = json.load(f)
            
            assert len(exported_data) == 1
            exported_entry = exported_data[0]
            assert exported_entry['headword'] == "om"
            assert exported_entry['examples'] == ["Om noh Yong ci", "Om noh paapai ci"]
            
        finally:
            if os.path.exists(temp_json.name):
                os.unlink(temp_json.name)
    
    def test_export_json_sort_by_frequency(self):
        """Test that export sorts by frequency when requested"""
        entries = [
            LexiconEntry("low", "V", frequency=10),
            LexiconEntry("high", "V", frequency=50),
            LexiconEntry("medium", "V", frequency=25)
        ]
        
        for entry in entries:
            self.lexicon.add_entry(entry)
        
        temp_json = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        temp_json.close()
        
        try:
            self.lexicon.export_json(temp_json.name, sort_by_frequency=True)
            
            import json
            with open(temp_json.name, 'r', encoding='utf-8') as f:
                exported_data = json.load(f)
            
            # Should be sorted by frequency descending
            headwords = [entry['headword'] for entry in exported_data]
            assert headwords == ["high", "medium", "low"]
            
        finally:
            if os.path.exists(temp_json.name):
                os.unlink(temp_json.name)
    
    def test_export_json_no_sort(self):
        """Test that export doesn't sort when requested"""
        entries = [
            LexiconEntry("first", "V", frequency=10),
            LexiconEntry("second", "V", frequency=50),
            LexiconEntry("third", "V", frequency=25)
        ]
        
        for entry in entries:
            self.lexicon.add_entry(entry)
        
        temp_json = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        temp_json.close()
        
        try:
            self.lexicon.export_json(temp_json.name, sort_by_frequency=False)
            
            import json
            with open(temp_json.name, 'r', encoding='utf-8') as f:
                exported_data = json.load(f)
            
            # Should maintain insertion order
            headwords = [entry['headword'] for entry in exported_data]
            assert headwords == ["first", "second", "third"]
            
        finally:
            if os.path.exists(temp_json.name):
                os.unlink(temp_json.name)
    
    def test_close(self):
        """Test closing the lexicon database"""
        # Create a new lexicon
        temp_db2 = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db2.close()
        
        lexicon2 = KchoLexicon(temp_db2.name)
        
        # Verify connection is open
        assert lexicon2.conn is not None
        
        # Close the lexicon
        lexicon2.close()
        
        # Verify connection is closed (should raise error on use)
        with pytest.raises(sqlite3.ProgrammingError):
            lexicon2.conn.execute("SELECT 1")
        
        # Clean up
        if os.path.exists(temp_db2.name):
            os.unlink(temp_db2.name)


class TestLexiconIntegration:
    """Test integration scenarios for lexicon management"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.lexicon = KchoLexicon(self.temp_db.name)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.lexicon.close()
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_complex_lexicon_workflow(self):
        """Test complex workflow with multiple operations"""
        # Add multiple entries
        entries = [
            LexiconEntry("om", "V", "om", "law", "go", examples=["Om noh Yong ci"], frequency=25),
            LexiconEntry("noh", "P", gloss_en="OBJ", examples=["Om noh Yong ci"], frequency=50),
            LexiconEntry("ci", "T", gloss_en="PAST", examples=["Om ci"], frequency=100),
            LexiconEntry("Yong", "N", gloss_en="Yong", examples=["Om noh Yong ci"], frequency=15)
        ]
        
        for entry in entries:
            self.lexicon.add_entry(entry)
        
        # Update frequencies
        self.lexicon.update_frequency("om", 5)
        self.lexicon.update_frequency("Yong", 10)
        
        # Retrieve and verify
        om_entry = self.lexicon.get_entry("om")
        assert om_entry.frequency == 30  # 25 + 5
        
        yong_entry = self.lexicon.get_entry("Yong")
        assert yong_entry.frequency == 25  # 15 + 10
        
        # Export and verify sorting
        temp_json = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        temp_json.close()
        
        try:
            self.lexicon.export_json(temp_json.name, sort_by_frequency=True)
            
            import json
            with open(temp_json.name, 'r', encoding='utf-8') as f:
                exported_data = json.load(f)
            
            # Should be sorted by frequency descending
            frequencies = [entry['frequency'] for entry in exported_data]
            assert frequencies == [100, 50, 30, 25]
            
        finally:
            if os.path.exists(temp_json.name):
                os.unlink(temp_json.name)
    
    def test_unicode_handling(self):
        """Test handling of Unicode characters in lexicon entries"""
        entry = LexiconEntry(
            headword="သွား",
            pos="V",
            gloss_en="go",
            gloss_my="သွား",
            definition="သွားရန်လိုအပ်သည်",
            examples=["သွားပါ", "သွားကြပါ"],
            frequency=10
        )
        
        entry_id = self.lexicon.add_entry(entry)
        retrieved_entry = self.lexicon.get_entry("သွား")
        
        assert retrieved_entry is not None
        assert retrieved_entry.headword == "သွား"
        assert retrieved_entry.gloss_my == "သွား"
        assert retrieved_entry.definition == "သွားရန်လိုအပ်သည်"
        assert retrieved_entry.examples == ["သွားပါ", "သွားကြပါ"]
    
    def test_empty_lexicon_export(self):
        """Test exporting empty lexicon"""
        temp_json = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        temp_json.close()
        
        try:
            self.lexicon.export_json(temp_json.name)
            
            import json
            with open(temp_json.name, 'r', encoding='utf-8') as f:
                exported_data = json.load(f)
            
            assert exported_data == []
            
        finally:
            if os.path.exists(temp_json.name):
                os.unlink(temp_json.name)
    
    def test_large_lexicon_performance(self):
        """Test performance with larger lexicon"""
        # Add many entries
        entries = []
        for i in range(100):
            entry = LexiconEntry(
                headword=f"word{i}",
                pos="V",
                gloss_en=f"meaning{i}",
                examples=[f"example{i}a", f"example{i}b"],
                frequency=i
            )
            entries.append(entry)
        
        # Add all entries
        for entry in entries:
            self.lexicon.add_entry(entry)
        
        # Verify all entries were added
        for i in range(100):
            retrieved_entry = self.lexicon.get_entry(f"word{i}")
            assert retrieved_entry is not None
            assert retrieved_entry.frequency == i
            assert len(retrieved_entry.examples) == 2
