"""
K'Cho Data Migration System
==========================

Robust data migration system for handling changes to kcho/data files.
Implements versioning, checksums, and fresh data loading following database best practices.

Author: Based on K'Cho linguistic research (Bedell & Mang 2012)
Version: 2.0.0
"""

import sqlite3
import json
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataVersion:
    """Represents a version of data files with checksums."""
    
    def __init__(self, version: str, checksums: Dict[str, str], timestamp: str):
        self.version = version
        self.checksums = checksums
        self.timestamp = timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'checksums': self.checksums,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataVersion':
        return cls(
            version=data['version'],
            checksums=data['checksums'],
            timestamp=data['timestamp']
        )


class DataMigrationManager:
    """Manages data migrations and versioning for K'Cho knowledge base."""
    
    def __init__(self, db_path: str, data_dir: str):
        self.db_path = db_path
        self.data_dir = Path(data_dir)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Initialize migration tables
        self._create_migration_tables()
    
    def _create_migration_tables(self):
        """Create tables for tracking data versions and migrations."""
        cur = self.conn.cursor()
        
        # Data versions table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS data_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT UNIQUE NOT NULL,
                checksums TEXT NOT NULL,  -- JSON string
                timestamp TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Migration history table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS migration_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_from TEXT,
                version_to TEXT NOT NULL,
                migration_type TEXT NOT NULL,  -- 'fresh_load', 'incremental', 'schema_change'
                status TEXT NOT NULL,  -- 'started', 'completed', 'failed'
                error_message TEXT,
                records_affected INTEGER,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        ''')
        
        # Data file tracking table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS data_file_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                version TEXT NOT NULL,
                checksum TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                last_modified TIMESTAMP NOT NULL,
                loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        cur.execute('CREATE INDEX IF NOT EXISTS idx_data_versions_version ON data_versions(version)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_migration_history_version ON migration_history(version_to)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_data_file_tracking_filename ON data_file_tracking(filename)')
        
        self.conn.commit()
        logger.info("📋 Migration tables created")
    
    def calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        if not file_path.exists():
            return ""
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def get_current_data_version(self) -> Optional[DataVersion]:
        """Get the current data version from database."""
        cur = self.conn.cursor()
        cur.execute('''
            SELECT version, checksums, timestamp 
            FROM data_versions 
            ORDER BY created_at DESC 
            LIMIT 1
        ''')
        row = cur.fetchone()
        
        if row:
            return DataVersion(
                version=row['version'],
                checksums=json.loads(row['checksums']),
                timestamp=row['timestamp']
            )
        return None
    
    def calculate_current_data_version(self) -> DataVersion:
        """Calculate current data version based on file checksums."""
        data_files = [
            'linguistic_data.json',
            'gold_standard_collocations.txt',
            'word_frequency_top_1000.csv',
            'gold_standard_kcho_english.json',
            'sample_corpus.txt'
        ]
        
        checksums = {}
        for filename in data_files:
            file_path = self.data_dir / filename
            checksums[filename] = self.calculate_file_checksum(file_path)
        
        # Create version string from checksums
        version_string = hashlib.sha256(
            json.dumps(checksums, sort_keys=True).encode()
        ).hexdigest()[:16]  # Use first 16 chars as version
        
        return DataVersion(
            version=version_string,
            checksums=checksums,
            timestamp=datetime.now().isoformat()
        )
    
    def has_data_changed(self) -> bool:
        """Check if data files have changed since last migration."""
        current_version = self.calculate_current_data_version()
        stored_version = self.get_current_data_version()
        
        if not stored_version:
            return True
        
        return current_version.version != stored_version.version
    
    def record_migration_start(self, version_from: Optional[str], version_to: str, 
                              migration_type: str) -> int:
        """Record the start of a migration."""
        cur = self.conn.cursor()
        cur.execute('''
            INSERT INTO migration_history 
            (version_from, version_to, migration_type, status)
            VALUES (?, ?, ?, ?)
        ''', (version_from, version_to, migration_type, 'started'))
        
        migration_id = cur.lastrowid
        self.conn.commit()
        
        logger.info(f"🔄 Migration {migration_id} started: {version_from} → {version_to}")
        return migration_id
    
    def record_migration_completion(self, migration_id: int, records_affected: int = 0):
        """Record successful completion of a migration."""
        cur = self.conn.cursor()
        cur.execute('''
            UPDATE migration_history 
            SET status = 'completed', 
                records_affected = ?,
                completed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (records_affected, migration_id))
        
        self.conn.commit()
        logger.info(f"✅ Migration {migration_id} completed ({records_affected} records)")
    
    def record_migration_failure(self, migration_id: int, error_message: str):
        """Record failed migration."""
        cur = self.conn.cursor()
        cur.execute('''
            UPDATE migration_history 
            SET status = 'failed', 
                error_message = ?,
                completed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (error_message, migration_id))
        
        self.conn.commit()
        logger.error(f"❌ Migration {migration_id} failed: {error_message}")
    
    def save_data_version(self, version: DataVersion):
        """Save data version to database."""
        cur = self.conn.cursor()
        cur.execute('''
            INSERT OR REPLACE INTO data_versions 
            (version, checksums, timestamp)
            VALUES (?, ?, ?)
        ''', (version.version, json.dumps(version.checksums), version.timestamp))
        
        self.conn.commit()
        logger.info(f"💾 Data version {version.version} saved")
    
    def clear_all_data(self):
        """Clear all data tables (for fresh migration)."""
        cur = self.conn.cursor()
        
        # Clear all data tables
        tables_to_clear = [
            'verb_stems', 'collocations', 'word_frequencies', 
            'parallel_sentences', 'raw_texts', 'word_categories'
        ]
        
        for table in tables_to_clear:
            cur.execute(f'DELETE FROM {table}')
            logger.info(f"🗑️  Cleared table: {table}")
        
        self.conn.commit()
        logger.info("🧹 All data tables cleared for fresh migration")
    
    def get_migration_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent migration history."""
        cur = self.conn.cursor()
        cur.execute('''
            SELECT * FROM migration_history 
            ORDER BY started_at DESC 
            LIMIT ?
        ''', (limit,))
        
        return [dict(row) for row in cur.fetchall()]
    
    def get_data_file_status(self) -> List[Dict[str, Any]]:
        """Get status of all tracked data files."""
        cur = self.conn.cursor()
        cur.execute('''
            SELECT filename, version, checksum, file_size, last_modified, loaded_at
            FROM data_file_tracking 
            ORDER BY filename
        ''')
        
        return [dict(row) for row in cur.fetchall()]
    
    def track_data_file(self, filename: str, version: str, checksum: str, 
                       file_size: int, last_modified: str):
        """Track a data file in the database."""
        cur = self.conn.cursor()
        cur.execute('''
            INSERT OR REPLACE INTO data_file_tracking 
            (filename, version, checksum, file_size, last_modified)
            VALUES (?, ?, ?, ?, ?)
        ''', (filename, version, checksum, file_size, last_modified))
        
        self.conn.commit()
    
    def close(self):
        """Close database connection."""
        self.conn.close()


class FreshDataLoader:
    """Handles fresh data loading with proper migration tracking."""
    
    def __init__(self, migration_manager: DataMigrationManager, knowledge_base):
        self.migration_manager = migration_manager
        self.knowledge_base = knowledge_base
    
    def load_fresh_data(self) -> bool:
        """
        Load fresh data from files, replacing all existing data.
        
        Returns:
            True if successful, False otherwise
        """
        current_version = self.migration_manager.calculate_current_data_version()
        stored_version = self.migration_manager.get_current_data_version()
        
        # Record migration start
        migration_id = self.migration_manager.record_migration_start(
            version_from=stored_version.version if stored_version else None,
            version_to=current_version.version,
            migration_type='fresh_load'
        )
        
        try:
            logger.info("🔄 Starting fresh data migration...")
            
            # Clear all existing data
            self.migration_manager.clear_all_data()
            
            # Load data from files
            records_loaded = self._load_all_data_files()
            
            # Save new data version
            self.migration_manager.save_data_version(current_version)
            
            # Record successful completion
            self.migration_manager.record_migration_completion(migration_id, records_loaded)
            
            logger.info(f"✅ Fresh data migration completed successfully ({records_loaded} records)")
            return True
            
        except Exception as e:
            error_msg = str(e)
            self.migration_manager.record_migration_failure(migration_id, error_msg)
            logger.error(f"❌ Fresh data migration failed: {error_msg}")
            return False
    
    def _load_all_data_files(self) -> int:
        """Load all data files and return total records loaded."""
        total_records = 0
        
        # Load linguistic data
        records = self._load_linguistic_data()
        total_records += records
        logger.info(f"📥 Loaded {records} records from linguistic_data.json")
        
        # Load gold standard patterns
        records = self._load_gold_standard_patterns()
        total_records += records
        logger.info(f"📥 Loaded {records} records from gold_standard_collocations.txt")
        
        # Load word frequencies
        records = self._load_word_frequency_data()
        total_records += records
        logger.info(f"📥 Loaded {records} records from word_frequency_top_1000.csv")
        
        # Load parallel data
        records = self._load_parallel_data()
        total_records += records
        logger.info(f"📥 Loaded {records} records from gold_standard_kcho_english.json")
        
        # Load word categories
        records = self._load_word_categories()
        total_records += records
        logger.info(f"📥 Loaded {records} word-category mappings")
        
        return total_records
    
    def _load_linguistic_data(self) -> int:
        """Load linguistic data and return number of records."""
        file_path = self.migration_manager.data_dir / 'linguistic_data.json'
        if not file_path.exists():
            return 0
        
        # Track file
        checksum = self.migration_manager.calculate_file_checksum(file_path)
        file_size = file_path.stat().st_size
        last_modified = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        
        self.migration_manager.track_data_file(
            'linguistic_data.json', 
            self.migration_manager.calculate_current_data_version().version,
            checksum, file_size, last_modified
        )
        
        # Load data using knowledge base method
        return self.knowledge_base._load_linguistic_data()
    
    def _load_gold_standard_patterns(self) -> int:
        """Load gold standard patterns and return number of records."""
        file_path = self.migration_manager.data_dir / 'gold_standard_collocations.txt'
        if not file_path.exists():
            return 0
        
        # Track file
        checksum = self.migration_manager.calculate_file_checksum(file_path)
        file_size = file_path.stat().st_size
        last_modified = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        
        self.migration_manager.track_data_file(
            'gold_standard_collocations.txt',
            self.migration_manager.calculate_current_data_version().version,
            checksum, file_size, last_modified
        )
        
        # Load data using knowledge base method
        return self.knowledge_base._load_gold_standard_patterns()
    
    def _load_word_frequency_data(self) -> int:
        """Load word frequency data and return number of records."""
        file_path = self.migration_manager.data_dir / 'word_frequency_top_1000.csv'
        if not file_path.exists():
            return 0
        
        # Track file
        checksum = self.migration_manager.calculate_file_checksum(file_path)
        file_size = file_path.stat().st_size
        last_modified = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        
        self.migration_manager.track_data_file(
            'word_frequency_top_1000.csv',
            self.migration_manager.calculate_current_data_version().version,
            checksum, file_size, last_modified
        )
        
        # Load data using knowledge base method
        return self.knowledge_base._load_word_frequency_data()
    
    def _load_parallel_data(self) -> int:
        """Load parallel data and return number of records."""
        file_path = self.migration_manager.data_dir / 'gold_standard_kcho_english.json'
        if not file_path.exists():
            return 0
        
        # Track file
        checksum = self.migration_manager.calculate_file_checksum(file_path)
        file_size = file_path.stat().st_size
        last_modified = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        
        self.migration_manager.track_data_file(
            'gold_standard_kcho_english.json',
            self.migration_manager.calculate_current_data_version().version,
            checksum, file_size, last_modified
        )
        
        # Load data using knowledge base method
        return self.knowledge_base._load_parallel_data()
    
    def _load_word_categories(self) -> int:
        """Load word categories and return number of records."""
        return self.knowledge_base._load_word_categories()


def create_migration_manager(db_path: str, data_dir: str) -> DataMigrationManager:
    """Create a migration manager instance."""
    return DataMigrationManager(db_path, data_dir)


def check_and_migrate_data(knowledge_base) -> bool:
    """
    Check if data has changed and perform migration if needed.
    
    Args:
        knowledge_base: KchoKnowledgeBase instance
        
    Returns:
        True if migration was performed, False if no migration needed
    """
    migration_manager = DataMigrationManager(knowledge_base.db_path, knowledge_base.data_dir)
    
    try:
        if migration_manager.has_data_changed():
            logger.info("🔄 Data files have changed, performing fresh migration...")
            
            fresh_loader = FreshDataLoader(migration_manager, knowledge_base)
            success = fresh_loader.load_fresh_data()
            
            if success:
                logger.info("✅ Data migration completed successfully")
                return True
            else:
                logger.error("❌ Data migration failed")
                return False
        else:
            logger.info("📊 Data files unchanged, no migration needed")
            return False
            
    finally:
        migration_manager.close()
