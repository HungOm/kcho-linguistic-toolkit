"""
K'Cho Knowledge Base - SQLite Backend Implementation
==================================================

Efficient SQLite-based knowledge base for K'Cho linguistic data with support for:
- Fast indexed queries
- Persistent storage
- Incremental updates
- Pattern discovery
- Data validation

Author: Based on K'Cho linguistic research (Bedell & Mang 2012)
Version: 2.0.0
"""

import sqlite3
import json
import os
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Optional, Any, Set, Tuple
import logging
from datetime import datetime

from .data_migration import DataMigrationManager, check_and_migrate_data, FreshDataLoader

# Optional Pydantic import for validation
try:
    from pydantic import BaseModel, Field, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Create dummy classes for when Pydantic is not available
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class Field:
        def __init__(self, default=None, **kwargs):
            self.default = default
    
    class ValidationError(Exception):
        pass

logger = logging.getLogger(__name__)


class VerbStemModel(BaseModel):
    """Pydantic model for verb stem validation"""
    verb: str
    stem2: Optional[str] = None
    gloss: Optional[str] = None
    pattern: Optional[str] = None


class CollocationModel(BaseModel):
    """Pydantic model for collocation validation"""
    words: str
    category: str
    frequency: Optional[str] = None
    notes: Optional[str] = None
    source: str = "unknown"


class KchoKnowledgeBase:
    """
    Efficient K'Cho knowledge base using SQLite backend for persistence and querying.
    Loads data from files into DB; supports expansion and discovery.
    """
    
    def __init__(self, db_path: Optional[str] = None, data_dir: Optional[str] = None, 
                 in_memory: bool = False):
        """
        Initialize knowledge base with SQLite backend.
        
        Args:
            db_path: Path to SQLite database file. If None, uses default location.
            data_dir: Path to data directory. If None, uses default location.
            in_memory: If True, use in-memory database (faster but non-persistent)
        """
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
        
        self.data_dir = Path(data_dir)
        
        # Set up database connection
        if in_memory:
            self.db_path = ':memory:'
        elif db_path is None:
            # Default to package directory
            package_dir = Path(__file__).parent
            self.db_path = str(package_dir / 'kcho_lexicon.db')
        else:
            self.db_path = db_path
        
        # Create directory if needed
        if not in_memory and self.db_path != ':memory:':
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        
        # Initialize database
        self._create_tables()
        
        # Check for data changes and migrate if needed
        migration_performed = check_and_migrate_data(self)
        
        # Load data if no migration was performed (empty database)
        if not migration_performed:
            self._load_data_if_needed()
        
        # Cache for frequently accessed data
        self._cache = {}
        
        logger.info(f"✅ KchoKnowledgeBase initialized with DB at {self.db_path}")
        logger.info(f"📊 Pydantic validation: {'enabled' if PYDANTIC_AVAILABLE else 'disabled'}")
        logger.info(f"🔄 Migration performed: {'Yes' if migration_performed else 'No'}")
    
    def _create_tables(self):
        """Create standard DB schema with indexes for efficient querying."""
        cur = self.conn.cursor()
        
        # Verb stems table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS verb_stems (
                verb TEXT PRIMARY KEY,
                stem2 TEXT,
                gloss TEXT,
                pattern TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Collocations table (from gold standard and discovered)
        cur.execute('''
            CREATE TABLE IF NOT EXISTS collocations (
                words TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                frequency TEXT,
                notes TEXT,
                source TEXT DEFAULT 'unknown',
                confidence REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Word frequencies table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS word_frequencies (
                word TEXT PRIMARY KEY,
                frequency INTEGER NOT NULL,
                source TEXT DEFAULT 'unknown',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Parallel sentences table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS parallel_sentences (
                id TEXT PRIMARY KEY,
                kcho TEXT NOT NULL,
                english TEXT,
                source TEXT DEFAULT 'unknown',
                features TEXT,  -- JSON string
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Raw texts for discovery/research
        cur.execute('''
            CREATE TABLE IF NOT EXISTS raw_texts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                source TEXT DEFAULT 'user',
                processed BOOLEAN DEFAULT FALSE,
                metadata TEXT,  -- JSON string
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Linguistic categories table (for word classification)
        cur.execute('''
            CREATE TABLE IF NOT EXISTS word_categories (
                word TEXT,
                category TEXT,
                PRIMARY KEY (word, category)
            )
        ''')
        
        # Create indexes for efficient querying
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_colloc_category ON collocations(category)',
            'CREATE INDEX IF NOT EXISTS idx_colloc_source ON collocations(source)',
            'CREATE INDEX IF NOT EXISTS idx_word_freq ON word_frequencies(word)',
            'CREATE INDEX IF NOT EXISTS idx_word_freq_freq ON word_frequencies(frequency)',
            'CREATE INDEX IF NOT EXISTS idx_raw_text_source ON raw_texts(source)',
            'CREATE INDEX IF NOT EXISTS idx_raw_text_processed ON raw_texts(processed)',
            'CREATE INDEX IF NOT EXISTS idx_word_cat_word ON word_categories(word)',
            'CREATE INDEX IF NOT EXISTS idx_word_cat_category ON word_categories(category)',
        ]
        
        for index_sql in indexes:
            cur.execute(index_sql)
        
        self.conn.commit()
        logger.info("📋 Database schema created with indexes")
    
    def _load_data_if_needed(self):
        """Load data from files into DB if tables are empty."""
        cur = self.conn.cursor()
        
        # Check if verb_stems table is empty
        cur.execute("SELECT COUNT(*) FROM verb_stems")
        if cur.fetchone()[0] == 0:
            logger.info("📥 Loading data from files into database...")
            total_records = 0
            
            total_records += self._load_linguistic_data()
            total_records += self._load_gold_standard_patterns()
            total_records += self._load_word_frequency_data()
            total_records += self._load_parallel_data()
            total_records += self._load_word_categories()
            
            logger.info(f"✅ Data loading completed ({total_records} total records)")
        else:
            logger.info("📊 Database already contains data")
    
    def _load_linguistic_data(self) -> int:
        """Load from linguistic_data.json. Returns number of records loaded."""
        file_path = self.data_dir / 'linguistic_data.json'
        if not file_path.exists():
            logger.warning(f"⚠️  Linguistic data file not found: {file_path}")
            return 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cur = self.conn.cursor()
        records_loaded = 0
        
        # Load verb stems
        verb_stems = data.get('verb_stems', {})
        for verb, info in verb_stems.items():
            try:
                if PYDANTIC_AVAILABLE:
                    VerbStemModel(verb=verb, **info)
                
                cur.execute('''
                    INSERT OR IGNORE INTO verb_stems (verb, stem2, gloss, pattern)
                    VALUES (?, ?, ?, ?)
                ''', (verb, info.get('stem2'), info.get('gloss'), info.get('pattern')))
                records_loaded += 1
            except ValidationError as e:
                logger.warning(f"Invalid verb stem data for '{verb}': {e}")
        
        # Load collocations from linguistic data
        collocations = data.get('collocations', {})
        for words, info in collocations.items():
            try:
                if PYDANTIC_AVAILABLE:
                    CollocationModel(words=words, **info)
                
                cur.execute('''
                    INSERT OR IGNORE INTO collocations (words, category, source)
                    VALUES (?, ?, ?)
                ''', (words, info.get('category', 'unknown'), 'linguistic_json'))
                records_loaded += 1
            except ValidationError as e:
                logger.warning(f"Invalid collocation data for '{words}': {e}")
        
        self.conn.commit()
        logger.info(f"📥 Loaded {len(verb_stems)} verb stems and {len(collocations)} collocations from linguistic_data.json")
        return records_loaded
    
    def _load_gold_standard_patterns(self) -> int:
        """Load from gold_standard_collocations.txt. Returns number of records loaded."""
        file_path = self.data_dir / 'gold_standard_collocations.txt'
        if not file_path.exists():
            logger.warning(f"⚠️  Gold standard file not found: {file_path}")
            return 0
        
        cur = self.conn.cursor()
        patterns_loaded = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    # Parse format: words # category, frequency, notes
                    if '#' in line:
                        words, info_part = line.split('#', 1)
                        words = words.strip()
                        info_parts = [p.strip() for p in info_part.split(',')]
                        
                        category = info_parts[0] if info_parts else 'unknown'
                        frequency = info_parts[1] if len(info_parts) > 1 else ''
                        notes = ','.join(info_parts[2:]) if len(info_parts) > 2 else ''
                        
                        if PYDANTIC_AVAILABLE:
                            CollocationModel(words=words, category=category, 
                                           frequency=frequency, notes=notes, source='gold_standard_txt')
                        
                        cur.execute('''
                            INSERT OR REPLACE INTO collocations 
                            (words, category, frequency, notes, source)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (words, category, frequency, notes, 'gold_standard_txt'))
                        patterns_loaded += 1
                        
                except Exception as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
        
        self.conn.commit()
        logger.info(f"📥 Loaded {patterns_loaded} gold standard patterns")
        return patterns_loaded
    
    def _load_word_frequency_data(self) -> int:
        """Load word frequencies from CSV file. Returns number of records loaded."""
        file_path = self.data_dir / 'word_frequency_top_1000.csv'
        if not file_path.exists():
            logger.warning(f"⚠️  Word frequency file not found: {file_path}")
            return 0
        
        cur = self.conn.cursor()
        frequencies_loaded = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            import csv
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    word = row.get('word', '').strip()
                    frequency = int(row.get('frequency', 0))
                    
                    if word and frequency > 0:
                        cur.execute('''
                            INSERT OR REPLACE INTO word_frequencies (word, frequency, source)
                            VALUES (?, ?, ?)
                        ''', (word, frequency, 'frequency_csv'))
                        frequencies_loaded += 1
                        
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error parsing frequency row: {e}")
        
        self.conn.commit()
        logger.info(f"📥 Loaded {frequencies_loaded} word frequencies")
        return frequencies_loaded
    
    def _load_parallel_data(self) -> int:
        """Load from gold_standard_kcho_english.json. Returns number of records loaded."""
        file_path = self.data_dir / 'gold_standard_kcho_english.json'
        if not file_path.exists():
            logger.warning(f"⚠️  Parallel data file not found: {file_path}")
            return 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cur = self.conn.cursor()
        sentences_loaded = 0
        
        for pair in data.get('sentence_pairs', []):
            try:
                features = json.dumps(pair.get('linguistic_features', []))
                cur.execute('''
                    INSERT OR IGNORE INTO parallel_sentences 
                    (id, kcho, english, source, features)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    pair.get('id', f'sentence_{sentences_loaded}'),
                    pair.get('kcho', ''),
                    pair.get('english', ''),
                    pair.get('source', 'gold_standard_json'),
                    features
                ))
                sentences_loaded += 1
                
            except Exception as e:
                logger.warning(f"Error loading parallel sentence: {e}")
        
        self.conn.commit()
        logger.info(f"📥 Loaded {sentences_loaded} parallel sentences")
        return sentences_loaded
    
    def _load_word_categories(self) -> int:
        """Load word categories from linguistic data. Returns number of records loaded."""
        file_path = self.data_dir / 'linguistic_data.json'
        if not file_path.exists():
            return 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cur = self.conn.cursor()
        categories_loaded = 0
        
        # Define category mappings
        category_mappings = {
            'pronouns': 'pronoun',
            'agreement_particles': 'agreement',
            'postpositions': 'postposition',
            'tense_aspect': 'tense_aspect',
            'applicatives': 'applicative',
            'connectives': 'connective',
            'common_nouns': 'noun',
            'demonstratives': 'demonstrative',
            'quantifiers': 'quantifier',
            'adjectives': 'adjective',
            'directionals': 'directional',
            'verb_stems': 'verb'
        }
        
        for data_key, category in category_mappings.items():
            words = data.get(data_key, {})
            if isinstance(words, dict):
                words = words.keys()
            
            for word in words:
                cur.execute('''
                    INSERT OR IGNORE INTO word_categories (word, category)
                    VALUES (?, ?)
                ''', (word.lower(), category))
                categories_loaded += 1
        
        self.conn.commit()
        logger.info(f"📥 Loaded {categories_loaded} word-category mappings")
        return categories_loaded
    
    # Query methods with caching
    @lru_cache(maxsize=128)
    def get_verb_stem(self, verb: str) -> Optional[Dict]:
        """Get verb stem information."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM verb_stems WHERE verb = ?", (verb,))
        row = cur.fetchone()
        if row:
            return dict(row)
        return None
    
    def get_collocations_by_category(self, category: str) -> List[Dict]:
        """Get all collocations in a category."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM collocations WHERE category = ? ORDER BY confidence DESC", (category,))
        return [dict(row) for row in cur.fetchall()]
    
    def get_collocations_by_source(self, source: str) -> List[Dict]:
        """Get all collocations from a specific source."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM collocations WHERE source = ? ORDER BY confidence DESC", (source,))
        return [dict(row) for row in cur.fetchall()]
    
    def search_collocations(self, pattern: str) -> List[Dict]:
        """Search collocations containing a pattern."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM collocations WHERE words LIKE ? ORDER BY confidence DESC", (f'%{pattern}%',))
        return [dict(row) for row in cur.fetchall()]
    
    def get_word_frequency(self, word: str) -> Optional[int]:
        """Get frequency of a word."""
        cur = self.conn.cursor()
        cur.execute("SELECT frequency FROM word_frequencies WHERE word = ?", (word.lower(),))
        row = cur.fetchone()
        return row[0] if row else None
    
    def get_all_word_frequencies(self) -> List[Dict]:
        """Get all word frequencies."""
        cur = self.conn.cursor()
        cur.execute("SELECT word, frequency, source FROM word_frequencies ORDER BY frequency DESC")
        return [dict(row) for row in cur.fetchall()]
    
    def get_all_collocations(self) -> List[Dict]:
        """Get all collocations."""
        cur = self.conn.cursor()
        cur.execute("SELECT words, category, frequency, notes, source FROM collocations ORDER BY words")
        return [dict(row) for row in cur.fetchall()]
    
    def get_all_parallel_sentences(self) -> List[Dict]:
        """Get all parallel sentences."""
        cur = self.conn.cursor()
        cur.execute("SELECT id, kcho, english, source, features FROM parallel_sentences ORDER BY id")
        return [dict(row) for row in cur.fetchall()]
    
    def get_all_word_categories(self) -> List[Dict]:
        """Get all word categories."""
        cur = self.conn.cursor()
        cur.execute("SELECT word, category FROM word_categories ORDER BY word")
        return [dict(row) for row in cur.fetchall()]
    
    def get_all_verb_stems(self) -> List[Dict]:
        """Get all verb stems."""
        cur = self.conn.cursor()
        cur.execute("SELECT verb, stem2, gloss, pattern FROM verb_stems ORDER BY verb")
        return [dict(row) for row in cur.fetchall()]
    
    def get_word_categories(self, word: str) -> List[str]:
        """Get all categories for a word."""
        cur = self.conn.cursor()
        cur.execute("SELECT category FROM word_categories WHERE word = ?", (word.lower(),))
        return [row[0] for row in cur.fetchall()]
    
    def get_parallel_sentences(self, limit: int = 100) -> List[Dict]:
        """Get parallel sentences."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM parallel_sentences ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(row) for row in cur.fetchall()]
    
    # CRUD operations
    def add_collocation(self, words: str, category: str, frequency: str = '', 
                       notes: str = '', source: str = 'user', confidence: float = 0.0):
        """Add a new collocation."""
        try:
            if PYDANTIC_AVAILABLE:
                CollocationModel(words=words, category=category, frequency=frequency, 
                               notes=notes, source=source)
            
            cur = self.conn.cursor()
            cur.execute('''
                INSERT OR REPLACE INTO collocations 
                (words, category, frequency, notes, source, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (words, category, frequency, notes, source, confidence))
            self.conn.commit()
            
            # Clear cache
            self.get_collocations_by_category.cache_clear()
            
            logger.info(f"➕ Added collocation: {words} ({category})")
            return True
            
        except ValidationError as e:
            logger.error(f"Invalid collocation data: {e}")
            return False
        except Exception as e:
            logger.error(f"Error adding collocation: {e}")
            return False
    
    def insert_raw_text(self, text: str, source: str = 'user', metadata: Optional[Dict] = None):
        """Add raw text for future processing."""
        try:
            metadata_json = json.dumps(metadata or {})
            cur = self.conn.cursor()
            cur.execute('''
                INSERT INTO raw_texts (text, source, metadata)
                VALUES (?, ?, ?)
            ''', (text, source, metadata_json))
            self.conn.commit()
            
            logger.info(f"📄 Added raw text from {source}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting raw text: {e}")
            return False
    
    def discover_patterns_from_raw(self):
        """Process unprocessed raw texts to discover patterns."""
        cur = self.conn.cursor()
        cur.execute("SELECT id, text FROM raw_texts WHERE processed = FALSE")
        
        processed_count = 0
        for row in cur.fetchall():
            text_id, text = row
            
            try:
                # Import here to avoid circular imports
                from .normalize import normalize_text, tokenize
                from .collocation import CollocationExtractor
                
                # Normalize and tokenize
                normalized = normalize_text(text)
                tokens = tokenize(normalized)
                
                # Extract collocations
                extractor = CollocationExtractor()
                results = extractor.extract_collocations(tokens)
                
                # Add discovered patterns
                for result in results:
                    self.add_collocation(
                        words=result.collocation,
                        category=result.category,
                        frequency=str(result.score),
                        notes=result.notes,
                        source='discovered',
                        confidence=result.score
                    )
                
                # Mark as processed
                cur.execute("UPDATE raw_texts SET processed = TRUE WHERE id = ?", (text_id,))
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing raw text {text_id}: {e}")
        
        self.conn.commit()
        logger.info(f"🔍 Processed {processed_count} raw texts for pattern discovery")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        cur = self.conn.cursor()
        
        stats = {
            'database_path': self.db_path,
            'verb_stems': cur.execute("SELECT COUNT(*) FROM verb_stems").fetchone()[0],
            'collocations': cur.execute("SELECT COUNT(*) FROM collocations").fetchone()[0],
            'word_frequencies': cur.execute("SELECT COUNT(*) FROM word_frequencies").fetchone()[0],
            'parallel_sentences': cur.execute("SELECT COUNT(*) FROM parallel_sentences").fetchone()[0],
            'raw_texts': cur.execute("SELECT COUNT(*) FROM raw_texts").fetchone()[0],
            'unprocessed_texts': cur.execute("SELECT COUNT(*) FROM raw_texts WHERE processed = FALSE").fetchone()[0],
            'word_categories': cur.execute("SELECT COUNT(*) FROM word_categories").fetchone()[0],
        }
        
        # Get collocation sources
        cur.execute("SELECT source, COUNT(*) FROM collocations GROUP BY source")
        stats['collocation_sources'] = dict(cur.fetchall())
        
        return stats
    
    def get_migration_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent migration history."""
        migration_manager = DataMigrationManager(self.db_path, str(self.data_dir))
        try:
            return migration_manager.get_migration_history(limit)
        finally:
            migration_manager.close()
    
    def get_data_file_status(self) -> List[Dict[str, Any]]:
        """Get status of all tracked data files."""
        migration_manager = DataMigrationManager(self.db_path, str(self.data_dir))
        try:
            return migration_manager.get_data_file_status()
        finally:
            migration_manager.close()
    
    def force_fresh_migration(self) -> bool:
        """Force a fresh data migration, replacing all existing data."""
        migration_manager = DataMigrationManager(self.db_path, str(self.data_dir))
        try:
            fresh_loader = FreshDataLoader(migration_manager, self)
            return fresh_loader.load_fresh_data()
        finally:
            migration_manager.close()
    
    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
            logger.info("🔒 Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def init_database(db_path: Optional[str] = None, data_dir: Optional[str] = None) -> KchoKnowledgeBase:
    """
    Initialize database with data loading.
    
    Args:
        db_path: Path to database file
        data_dir: Path to data directory
        
    Returns:
        Initialized KchoKnowledgeBase instance
    """
    return KchoKnowledgeBase(db_path=db_path, data_dir=data_dir)
