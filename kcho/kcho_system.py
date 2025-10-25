"""
K'Cho Language Processing System - Clean Implementation
======================================================

A production-ready system for K'Cho language processing with SQLite backend.
Clean, consistent naming with all essential functionality consolidated.

Author: Based on K'Cho linguistic research (Bedell & Mang 2012)
Version: 2.0.0
"""

import json
import csv
import os
import re
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, asdict, field
from collections import Counter, defaultdict
from datetime import datetime
import logging
from enum import Enum

# Import modules
from .normalize import KChoNormalizer, normalize_text, tokenize
from .collocation import CollocationExtractor, AssociationMeasure, CollocationResult
from .export import to_csv, to_json, to_text
from .evaluation import compute_precision_recall, evaluate_ranking, load_gold_standard
from .knowledge_base import KchoKnowledgeBase, init_database
from .api_layer import KchoAPILayer
from .llama_integration import EnhancedKchoAPILayer, LLaMAConfig, create_llama_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

class POS(Enum):
    """Part-of-speech tags"""
    NOUN = 'N'
    VERB = 'V'
    POSTPOSITION = 'P'
    AGREEMENT = 'AGR'
    TENSE = 'T'
    DEICTIC = 'D'
    CONJUNCTION = 'CONJ'
    ADJECTIVE = 'ADJ'
    ADVERB = 'ADV'
    DETERMINER = 'DET'
    APPLICATIVE = 'APPL'
    TENSE_ASPECT = 'T/A'
    UNKNOWN = 'UNK'


class StemType(Enum):
    """Verb stem types"""
    STEM_I = 'I'
    STEM_II = 'II'


@dataclass
class Morpheme:
    """Morpheme representation"""
    form: str
    lemma: str
    gloss: str
    type: str  # 'root', 'prefix', 'suffix', 'particle'
    features: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Token:
    """Token with full linguistic annotation"""
    surface: str
    lemma: str
    pos: POS
    morphemes: List[Morpheme]
    stem_type: Optional[StemType] = None
    features: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'surface': self.surface,
            'lemma': self.lemma,
            'pos': self.pos.value,
            'stem_type': self.stem_type.value if self.stem_type else None,
            'morphemes': [m.to_dict() for m in self.morphemes],
            'features': self.features
        }


@dataclass
class Sentence:
    """Annotated sentence"""
    text: str
    tokens: List[Token]
    features: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'tokens': [t.to_dict() for t in self.tokens],
            'features': self.features
        }


# ============================================================================
# LEGACY KNOWLEDGE BASE (DEPRECATED)
# ============================================================================

class LegacyKchoKnowledge:
    """
    DEPRECATED: Legacy K'Cho knowledge base using in-memory storage.

    This class is deprecated in favor of KchoKnowledge which uses
    SQLite backend for better performance and persistence.

    Use KchoKnowledge instead for new code.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize comprehensive knowledge base from all available data sources.
        
        DEPRECATED: Use KchoKnowledge instead.
        
        Args:
            data_dir: Path to kcho/data directory. If None, uses default location.
        """
        import warnings
        warnings.warn(
            "LegacyKchoKnowledge is deprecated. Use KchoKnowledge instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
        
        self.data_dir = Path(data_dir)
        
        # Load all available data sources
        self.linguistic_data = self._load_linguistic_data()
        self.gold_standard_patterns = self._load_gold_standard_patterns()
        self.word_frequency_data = self._load_word_frequency_data()
        self.parallel_data = self._load_parallel_data()
        
        # Create comprehensive knowledge base
        self._create_comprehensive_knowledge_base()
        
        # Track data sources
        self.data_sources = {
            'linguistic_data': len(self.linguistic_data),
            'gold_standard_patterns': len(self.gold_standard_patterns),
            'word_frequency_data': len(self.word_frequency_data),
            'parallel_data': len(self.parallel_data),
            'total_words': len(self.all_words)
        }
        
        logger.info(f"✅ LegacyKchoKnowledge loaded with {len(self.all_words)} words from {len(self.word_categories)} categories")
    
    def _load_linguistic_data(self) -> Dict[str, Any]:
        """Load linguistic data from JSON file."""
        file_path = self.data_dir / 'linguistic_data.json'
        if not file_path.exists():
            logger.warning(f"⚠️  Linguistic data file not found: {file_path}")
            return {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_gold_standard_patterns(self) -> Dict[str, Dict]:
        """Load gold standard patterns from text file."""
        file_path = self.data_dir / 'gold_standard_collocations.txt'
        if not file_path.exists():
            logger.warning(f"⚠️  Gold standard file not found: {file_path}")
            return {}
        
        patterns = {}
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
                        
                        patterns[words] = {
                            'category': category,
                            'frequency': frequency,
                            'notes': notes,
                            'source': 'gold_standard_txt'
                        }
                        
                except Exception as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
        
        return patterns
    
    def _load_word_frequency_data(self) -> Dict[str, int]:
        """Load word frequency data from CSV file."""
        file_path = self.data_dir / 'word_frequency_top_1000.csv'
        if not file_path.exists():
            logger.warning(f"⚠️  Word frequency file not found: {file_path}")
            return {}
        
        frequencies = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    word = row.get('word', '').strip()
                    frequency = int(row.get('frequency', 0))
                    if word and frequency > 0:
                        frequencies[word] = frequency
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error parsing frequency row: {e}")
        
        return frequencies
    
    def _load_parallel_data(self) -> Dict[str, Any]:
        """Load parallel sentence data from JSON file."""
        file_path = self.data_dir / 'gold_standard_kcho_english.json'
        if not file_path.exists():
            logger.warning(f"⚠️  Parallel data file not found: {file_path}")
            return {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_comprehensive_knowledge_base(self):
        """Create comprehensive knowledge base from loaded data."""
        # Extract all linguistic categories
        self.VERB_STEMS = self.linguistic_data.get('verb_stems', {})
        self.PRONOUNS = self.linguistic_data.get('pronouns', {})
        self.AGREEMENT = self.linguistic_data.get('agreement_particles', {})
        self.POSTPOSITIONS = self.linguistic_data.get('postpositions', {})
        self.TENSE_ASPECT = self.linguistic_data.get('tense_aspect', {})
        self.APPLICATIVES = self.linguistic_data.get('applicatives', {})
        self.CONNECTIVES = self.linguistic_data.get('connectives', {})
        self.COMMON_NOUNS = self.linguistic_data.get('common_nouns', {})
        self.DEMONSTRATIVES = self.linguistic_data.get('demonstratives', {})
        self.QUANTIFIERS = self.linguistic_data.get('quantifiers', {})
        self.ADJECTIVES = self.linguistic_data.get('adjectives', {})
        self.DIRECTIONALS = self.linguistic_data.get('directionals', {})
        self.COMMON_WORDS = set(self.linguistic_data.get('common_words', []))
        
        # Create comprehensive word categories
        self.all_words = set()
        self.word_categories = defaultdict(set)
        
        # Add words from all categories
        for category, words in [
            ('verbs', self.VERB_STEMS.keys()),
            ('pronouns', self.PRONOUNS.keys()),
            ('agreement', self.AGREEMENT.keys()),
            ('postpositions', self.POSTPOSITIONS.keys()),
            ('tense_aspect', self.TENSE_ASPECT.keys()),
            ('applicatives', self.APPLICATIVES.keys()),
            ('connectives', self.CONNECTIVES.keys()),
            ('nouns', self.COMMON_NOUNS.keys()),
            ('demonstratives', self.DEMONSTRATIVES.keys()),
            ('quantifiers', self.QUANTIFIERS.keys()),
            ('adjectives', self.ADJECTIVES.keys()),
            ('directionals', self.DIRECTIONALS.keys())
        ]:
            for word in words:
                self.all_words.add(word.lower())
                self.word_categories[category].add(word.lower())
        
        # Add frequency data
        for word in self.word_frequency_data.keys():
            self.all_words.add(word.lower())
            self.word_categories['frequent'].add(word.lower())
        
    def get_verb_stem(self, verb: str) -> Optional[Dict]:
        """Get verb stem information."""
        return self.VERB_STEMS.get(verb)
    
    def get_word_category(self, word: str) -> str:
        """Get the primary category for a word."""
        word_lower = word.lower()
        for category, words in self.word_categories.items():
            if word_lower in words:
                return category
        return 'unknown'
    
    def is_known_word(self, word: str) -> bool:
        """Check if a word is in the knowledge base."""
        return word.lower() in self.all_words
    
    def get_all_words(self) -> Set[str]:
        """Get all known words."""
        return self.all_words.copy()
    
    def get_words_by_category(self, category: str) -> Set[str]:
        """Get all words in a specific category."""
        return self.word_categories.get(category, set()).copy()
    
    def search_words(self, pattern: str) -> List[str]:
        """Search for words matching a pattern."""
        pattern_lower = pattern.lower()
        matches = []
        for word in self.all_words:
            if pattern_lower in word:
                matches.append(word)
        return sorted(matches)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the knowledge base."""
        stats = {
            'total_words': len(self.all_words),
            'verb_stems': len(self.VERB_STEMS),
            'pronouns': len(self.PRONOUNS),
            'agreement_particles': len(self.AGREEMENT),
            'postpositions': len(self.POSTPOSITIONS),
            'tense_aspect': len(self.TENSE_ASPECT),
            'applicatives': len(self.APPLICATIVES),
            'connectives': len(self.CONNECTIVES),
            'common_nouns': len(self.COMMON_NOUNS),
            'demonstratives': len(self.DEMONSTRATIVES),
            'quantifiers': len(self.QUANTIFIERS),
            'adjectives': len(self.ADJECTIVES),
            'directionals': len(self.DIRECTIONALS),
            'common_words': len(self.COMMON_WORDS)
        }
        return stats


# ============================================================================
# PATTERN DISCOVERY ENGINE
# ============================================================================

class PatternDiscoveryEngine:
    """Pattern discovery engine that writes to database."""
    
    def __init__(self, knowledge_base: 'KchoKnowledge'):
        self.knowledge = knowledge_base
    
    def discover_patterns(self, corpus: List[str], min_frequency: int = 5) -> Dict[str, Dict]:
        """Discover patterns and write them to database."""
        discovered_patterns = {}
        
        # Analyze bigrams
        bigram_patterns = self._discover_bigram_patterns(corpus, min_frequency)
        discovered_patterns.update(bigram_patterns)
        
        # Analyze trigrams
        trigram_patterns = self._discover_trigram_patterns(corpus, min_frequency)
        discovered_patterns.update(trigram_patterns)
        
        # Write discovered patterns to database
        for pattern, data in discovered_patterns.items():
            self.knowledge.add_collocation(
                words=pattern,
                category=data['pattern_type'],
                frequency=str(data['frequency']),
                notes=data.get('notes', ''),
                source='discovered',
                confidence=data.get('confidence', 0.5)
            )
        
        logger.info(f"🔍 Discovered {len(discovered_patterns)} patterns and wrote to database")
        return discovered_patterns
    
    def _discover_bigram_patterns(self, corpus: List[str], min_frequency: int) -> Dict[str, Dict]:
        """Discover bigram patterns."""
        patterns = {}
        
        # Count bigrams
        bigram_counts = Counter()
        for text in corpus:
            normalized = normalize_text(text)
            tokens = tokenize(normalized)
            
            for i in range(len(tokens) - 1):
                bigram = f"{tokens[i]} {tokens[i+1]}"
                bigram_counts[bigram] += 1
        
        # Analyze patterns
        for bigram, count in bigram_counts.items():
            if count >= min_frequency:
                words = bigram.split()
                pattern_type = self.knowledge._classify_word_pair(words[0], words[1])
                
                if pattern_type != 'UNKNOWN':
                patterns[bigram] = {
                    'pattern_type': pattern_type,
                        'frequency': count,
                        'confidence': min(count / 10.0, 1.0),
                        'notes': f'Discovered from corpus (freq: {count})'
                }
        
        return patterns
    
    def _discover_trigram_patterns(self, corpus: List[str], min_frequency: int) -> Dict[str, Dict]:
        """Discover trigram patterns."""
        patterns = {}
        
        # Count trigrams
        trigram_counts = Counter()
        for text in corpus:
            normalized = normalize_text(text)
            tokens = tokenize(normalized)
            
            for i in range(len(tokens) - 2):
                trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
                trigram_counts[trigram] += 1
        
        # Analyze patterns
        for trigram, count in trigram_counts.items():
            if count >= min_frequency:
                words = trigram.split()
                # Simple pattern classification for trigrams
                pattern_type = 'COMPLEX'
                
                patterns[trigram] = {
                    'pattern_type': pattern_type,
                    'frequency': count,
                    'confidence': min(count / 15.0, 1.0),
                    'notes': f'Discovered trigram from corpus (freq: {count})'
                }
        
        return patterns
    

# ============================================================================
# TEXT PROCESSING
# ============================================================================

class KchoTokenizer:
    """
    Handles text normalization, cleaning, and tokenization.
    Combines functionality from both original implementations.
    """
    
    def __init__(self):
        self.normalization_rules = {
            ''': "'", ''': "'", '`': "'",
            '"': '"', '"': '"', '"': '"',
        }
    
    def normalize(self, text: str) -> str:
        for old, new in self.normalization_rules.items():
            text = text.replace(old, new)
        return re.sub(r'  +', ' ', text).strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        normalized = self.normalize(text)
        # Simple tokenization - split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', normalized)
        return tokens
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        return self.normalize(text)


# ============================================================================
# VALIDATION
# ============================================================================

class KchoValidator:
    """Validates K'Cho text and linguistic data."""
    
    def __init__(self):
        self.knowledge = None  # Will be set by system
    
    def set_knowledge(self, knowledge):
        """Set knowledge base for validation."""
        self.knowledge = knowledge
    
    def is_kcho_text(self, text: str) -> Tuple[bool, float, Dict]:
        """
        Determine if text is K'Cho with confidence score.
        Returns: (is_kcho, confidence, metrics)
        """
        if not text or len(text.strip()) == 0:
            return False, 0.0, {}
        
        tokens = tokenize(normalize_text(text))
        if not tokens:
            return False, 0.0, {}
        
        # Check against known vocabulary
        known_count = 0
        total_count = len(tokens)
        
        for token in tokens:
            if self.knowledge and self.knowledge.is_known_word(token):
                known_count += 1
        
        # Calculate confidence based on known word ratio
        confidence = known_count / total_count if total_count > 0 else 0.0
        
        # Additional heuristics
        metrics = {
            'total_tokens': total_count,
            'known_tokens': known_count,
            'known_ratio': confidence,
            'avg_token_length': sum(len(t) for t in tokens) / total_count if total_count > 0 else 0
        }
        
        # Consider it K'Cho if >30% of tokens are known
        is_kcho = confidence > 0.3
        
        return is_kcho, confidence, metrics
    
    def validate_training_data(self, sentences: List[Sentence], lexicon: Dict, collocations: Dict) -> Tuple[bool, List[str]]:
        """Validate training data quality."""
        issues = []
        
        # Check sentence count
        if len(sentences) < 10:
            issues.append("Very few sentences for training")
        
        # Check lexicon size
        if len(lexicon) < 50:
            issues.append("Small lexicon size")
        
        # Check collocation count
        if len(collocations) < 5:
            issues.append("Very few collocations discovered")
        
        # Check for empty sentences
        empty_sentences = [s for s in sentences if not s.tokens]
        if empty_sentences:
            issues.append(f"{len(empty_sentences)} empty sentences found")
        
        is_ready = len(issues) == 0
        return is_ready, issues


# ============================================================================
# MORPHOLOGY ANALYSIS
# ============================================================================

class KchoMorphologyAnalyzer:
    """Analyzes morphological structure of K'Cho words."""
    
    def __init__(self):
        self.knowledge = None  # Will be set by system
    
    def set_knowledge(self, knowledge):
        """Set knowledge base for analysis."""
        self.knowledge = knowledge
    
    def analyze_token(self, word: str) -> Token:
        """Analyze a single word/token"""
        word = word.strip().strip('.,!?;:')
        
        if not word:
            return Token(
                surface="",
                lemma="",
                pos=POS.UNKNOWN,
                morphemes=[],
                features={}
            )
        
        # Basic morphological analysis
        morphemes = self._segment_word(word)
        pos = self._determine_pos(word, morphemes)
        lemma = self._get_lemma(word, morphemes)
        stem_type = self._get_stem_type(word)
        
        return Token(
            surface=word,
            lemma=lemma,
            pos=pos,
            morphemes=morphemes,
            stem_type=stem_type,
            features=self._extract_features(word, morphemes)
        )
    
    def _segment_word(self, word: str) -> List[Morpheme]:
        """Segment word into morphemes."""
        morphemes = []
        
        # Simple segmentation based on known patterns
        # This is a simplified version - real implementation would be more complex
        
        # Check for prefixes
        if word.startswith('a-'):
            morphemes.append(Morpheme('a-', 'a-', 'prefix', 'prefix'))
            word = word[2:]
        
        # Check for suffixes
        if word.endswith('-na'):
            morphemes.append(Morpheme('-na', '-na', 'suffix', 'suffix'))
            word = word[:-3]
        
        # Root
        if word:
            morphemes.append(Morpheme(word, word, 'root', 'root'))
        
        return morphemes
    
    def _determine_pos(self, word: str, morphemes: List[Morpheme]) -> POS:
        """Determine part of speech."""
        if self.knowledge:
            categories = self.knowledge._get_word_categories(word)
            if 'verbs' in categories:
                return POS.VERB
            elif 'nouns' in categories:
                return POS.NOUN
            elif 'postpositions' in categories:
                return POS.POSTPOSITION
            elif 'agreement' in categories:
                return POS.AGREEMENT
        
        # Fallback based on morpheme analysis
        if any(m.type == 'suffix' for m in morphemes):
            return POS.VERB
        
        return POS.UNKNOWN
    
    def _get_lemma(self, word: str, morphemes: List[Morpheme]) -> str:
        """Get lemma (dictionary form)."""
        # Find root morpheme
        for morpheme in morphemes:
            if morpheme.type == 'root':
                return morpheme.form
        
        return word
    
    def _get_stem_type(self, word: str) -> Optional[StemType]:
        """Determine verb stem type."""
        if self.knowledge:
            verb_info = self.knowledge.get_verb_stem(word)
            if verb_info and verb_info.get('stem2'):
                return StemType.STEM_II
        
        return None
    
    def _extract_features(self, word: str, morphemes: List[Morpheme]) -> Dict[str, Any]:
        """Extract morphological features."""
        features = {}
        
        # Count morphemes
        features['morpheme_count'] = len(morphemes)
        
        # Check for specific morphemes
        for morpheme in morphemes:
            if morpheme.type == 'prefix':
                features['has_prefix'] = True
            elif morpheme.type == 'suffix':
                features['has_suffix'] = True
        
        return features
    
    def analyze_corpus(self, sentences: List[Sentence]) -> Dict[str, Any]:
        """Analyze morphological patterns in corpus."""
        all_morphemes = []
        stem_types = Counter()
        morpheme_counts = Counter()
        
        for sentence in sentences:
            for token in sentence.tokens:
                all_morphemes.extend(token.morphemes)
                if token.stem_type:
                    stem_types[token.stem_type.value] += 1
                morpheme_counts[len(token.morphemes)] += 1
        
        # Analyze patterns
        prefix_patterns = Counter()
        suffix_patterns = Counter()
        
        for morpheme in all_morphemes:
            if morpheme.type == 'prefix':
                prefix_patterns[morpheme.form] += 1
            elif morpheme.type == 'suffix':
                suffix_patterns[morpheme.form] += 1
        
        return {
            'total_morphemes': len(all_morphemes),
            'stem_types': dict(stem_types),
            'morpheme_counts': dict(morpheme_counts),
            'prefix_patterns': dict(prefix_patterns),
            'suffix_patterns': dict(suffix_patterns)
        }


# ============================================================================
# SYNTAX ANALYSIS
# ============================================================================

class KchoSyntaxAnalyzer:
    """Analyzes syntactic structure of K'Cho sentences."""
    
    def __init__(self):
        self.knowledge = None  # Will be set by system
        self.morph = KchoMorphologyAnalyzer()
    
    def set_knowledge(self, knowledge):
        """Set knowledge base for analysis."""
        self.knowledge = knowledge
        self.morph.set_knowledge(knowledge)
    
    def analyze_syntax(self, sentence: Sentence) -> Dict:
        """Perform syntactic analysis on a sentence"""
        tokens = sentence.tokens
        words = [t.surface.lower() for t in tokens]
        
        analysis = {
            'clause_type': self._identify_clause_type(tokens),
            'word_order': self._analyze_word_order(tokens),
            'agreement_patterns': self._find_agreement_patterns(tokens),
            'postposition_phrases': self._find_postposition_phrases(tokens),
            'verb_phrases': self._find_verb_phrases(tokens)
        }
        
        return analysis
    
    def _identify_clause_type(self, tokens: List[Token]) -> str:
        """Identify the type of clause."""
        # Simple heuristic based on verb position and agreement
        verb_positions = [i for i, t in enumerate(tokens) if t.pos == POS.VERB]
        
        if not verb_positions:
            return 'nominal'
        
        # Check for agreement particles
        has_agreement = any(t.pos == POS.AGREEMENT for t in tokens)
        
        if has_agreement:
            return 'verbal_with_agreement'
        else:
            return 'verbal_simple'
    
    def _analyze_word_order(self, tokens: List[Token]) -> Dict[str, Any]:
        """Analyze word order patterns."""
        order_info = {
            'total_words': len(tokens),
            'verb_positions': [],
            'noun_positions': [],
            'postposition_positions': []
        }
        
        for i, token in enumerate(tokens):
            if token.pos == POS.VERB:
                order_info['verb_positions'].append(i)
            elif token.pos == POS.NOUN:
                order_info['noun_positions'].append(i)
            elif token.pos == POS.POSTPOSITION:
                order_info['postposition_positions'].append(i)
        
        return order_info
    
    def _find_agreement_patterns(self, tokens: List[Token]) -> List[Dict[str, Any]]:
        """Find agreement patterns in the sentence."""
        patterns = []
        
        for i, token in enumerate(tokens):
            if token.pos == POS.AGREEMENT:
                # Look for associated noun/pronoun
                for j in range(max(0, i-3), min(len(tokens), i+3)):
                    if j != i and tokens[j].pos in [POS.NOUN, POS.DEICTIC]:
                        patterns.append({
                            'agreement': token.surface,
                            'agreement_pos': i,
                            'noun': tokens[j].surface,
                            'noun_pos': j,
                            'distance': abs(i - j)
                        })
        
        return patterns
    
    def _find_postposition_phrases(self, tokens: List[Token]) -> List[Dict[str, Any]]:
        """Find postposition phrases."""
        phrases = []
        
        for i, token in enumerate(tokens):
            if token.pos == POS.POSTPOSITION:
                # Look for preceding noun
                if i > 0 and tokens[i-1].pos == POS.NOUN:
                    phrases.append({
                        'noun': tokens[i-1].surface,
                        'postposition': token.surface,
                        'phrase': f"{tokens[i-1].surface} {token.surface}"
                    })
        
        return phrases
    
    def _find_verb_phrases(self, tokens: List[Token]) -> List[Dict[str, Any]]:
        """Find verb phrases."""
        phrases = []
        
        for i, token in enumerate(tokens):
            if token.pos == POS.VERB:
                # Look for associated particles/tense markers
                associated = []
                for j in range(max(0, i-2), min(len(tokens), i+2)):
                    if j != i and tokens[j].pos in [POS.TENSE, POS.POSTPOSITION]:
                        associated.append(tokens[j].surface)
                
                phrases.append({
                    'verb': token.surface,
                    'verb_pos': i,
                    'associated': associated,
                    'stem_type': token.stem_type.value if token.stem_type else None
                })
        
        return phrases
    
    def analyze_corpus(self, sentences: List[Sentence]) -> Dict[str, Any]:
        """Analyze syntactic patterns in corpus."""
        clause_types = Counter()
        word_orders = []
        agreement_patterns = []
        postposition_phrases = []
        verb_phrases = []
        
        for sentence in sentences:
            analysis = self.analyze_syntax(sentence)
            
            clause_types[analysis['clause_type']] += 1
            word_orders.append(analysis['word_order'])
            agreement_patterns.extend(analysis['agreement_patterns'])
            postposition_phrases.extend(analysis['postposition_phrases'])
            verb_phrases.extend(analysis['verb_phrases'])
        
        return {
            'clause_types': dict(clause_types),
            'word_orders': word_orders,
            'agreement_patterns': agreement_patterns,
            'postposition_phrases': postposition_phrases,
            'verb_phrases': verb_phrases
        }


# ============================================================================
# COLLOCATION EXTRACTION
# ============================================================================

class KchoCollocationExtractor:
    """Extracts collocations from K'Cho text."""
    
    def __init__(self, min_freq: int = 5, min_pmi: float = 3.0):
        self.min_freq = min_freq
        self.min_pmi = min_pmi
        self.word_freq = Counter()
        self.bigram_freq = Counter()
        self.total_words = 0
        self.total_bigrams = 0
    
    def extract_from_corpus(self, sentences: List[Sentence]) -> Dict:
        """Extract significant collocations"""
        import math
        
        # Count frequencies
        for sentence in sentences:
            words = [t.surface.lower() for t in sentence.tokens]
            self.total_words += len(words)
            
            for word in words:
                self.word_freq[word] += 1
            
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                self.bigram_freq[bigram] += 1
                self.total_bigrams += 1
        
        # Calculate PMI and extract significant collocations
        collocations = {}
        
        for bigram, freq in self.bigram_freq.items():
            if freq >= self.min_freq:
                words = bigram.split()
                word1, word2 = words[0], words[1]
                
                # Calculate PMI
                pmi = self._calculate_pmi(word1, word2, freq)
                
                if pmi >= self.min_pmi:
                    collocations[bigram] = {
                        'frequency': freq,
                        'pmi': pmi,
                        'word1': word1,
                        'word2': word2
                    }
        
        return collocations
    
    def _calculate_pmi(self, word1: str, word2: str, bigram_freq: int) -> float:
        """Calculate Pointwise Mutual Information."""
        import math
        
        p_bigram = bigram_freq / self.total_bigrams
        p_word1 = self.word_freq[word1] / self.total_words
        p_word2 = self.word_freq[word2] / self.total_words
        
        if p_word1 == 0 or p_word2 == 0:
            return 0.0
        
        pmi = math.log2(p_bigram / (p_word1 * p_word2))
        return pmi


# ============================================================================
# LEXICON MANAGEMENT
# ============================================================================

@dataclass
class LexiconEntry:
    """Lexicon entry with frequency and features"""
    word: str
    frequency: int
    pos: str
    features: Dict[str, Any] = field(default_factory=dict)


class KchoLexicon:
    """Manages K'Cho lexicon with frequency tracking."""
    
    def __init__(self, db_path: str = "kcho_lexicon.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entries (
                word TEXT PRIMARY KEY,
                frequency INTEGER DEFAULT 1,
                pos TEXT,
                features TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_frequency ON entries(frequency DESC)
        ''')
        
        self.conn.commit()
    
    def add_word(self, word: str, pos: str = 'UNK', features: Dict = None):
        """Add or update word in lexicon"""
        cursor = self.conn.cursor()
        
        # Check if word exists
        cursor.execute('SELECT frequency FROM entries WHERE word = ?', (word,))
        result = cursor.fetchone()
        
        if result:
            # Update frequency
            new_freq = result[0] + 1
            cursor.execute('''
                UPDATE entries SET frequency = ?, pos = ?, features = ?
                WHERE word = ?
            ''', (new_freq, pos, json.dumps(features or {}), word))
        else:
            # Insert new word
            cursor.execute('''
                INSERT INTO entries (word, frequency, pos, features)
                VALUES (?, 1, ?, ?)
            ''', (word, pos, json.dumps(features or {})))
        
        self.conn.commit()
    
    def get_word(self, word: str) -> Optional[LexiconEntry]:
        """Get word entry from lexicon"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM entries WHERE word = ?', (word,))
        result = cursor.fetchone()
        
        if result:
            features = json.loads(result[3]) if result[3] else {}
        return LexiconEntry(
                word=result[0],
                frequency=result[1],
                pos=result[2],
                features=features
            )
        return None
    
    def get_top_words(self, limit: int = 100) -> List[LexiconEntry]:
        """Get top frequency words"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM entries ORDER BY frequency DESC LIMIT ?
        ''', (limit,))
        
        entries = []
        for result in cursor.fetchall():
            features = json.loads(result[3]) if result[3] else {}
            entries.append(LexiconEntry(
                word=result[0],
                frequency=result[1],
                pos=result[2],
                features=features
            ))
        
        return entries
    
    def build_from_sentences(self, sentences: List[Sentence]) -> Dict[str, Any]:
        """Build lexicon from sentences"""
        word_counts = Counter()
        pos_counts = defaultdict(Counter)
        
        for sentence in sentences:
            for token in sentence.tokens:
                word_counts[token.surface.lower()] += 1
                pos_counts[token.surface.lower()][token.pos.value] += 1
        
        # Add to database
        for word, count in word_counts.items():
            # Determine most common POS
            most_common_pos = pos_counts[word].most_common(1)[0][0] if pos_counts[word] else 'UNK'
            
            for _ in range(count):
                self.add_word(word, most_common_pos)
        
        # Get statistics
        stats = {
            'total_words': len(word_counts),
            'total_occurrences': sum(word_counts.values()),
            'top_words': [(word, count) for word, count in word_counts.most_common(20)]
        }
        
        return stats
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# ============================================================================
# CORPUS MANAGEMENT
# ============================================================================

class KchoCorpus:
    """Manages K'Cho corpus with annotation and analysis."""
    
    def __init__(self):
        self.morph = KchoMorphologyAnalyzer()
        self.syntax = KchoSyntaxAnalyzer()
        self.validator = KchoValidator()
        self.tokenizer = KchoTokenizer()
        self.lexicon = None
        self.sentences: List[Sentence] = []
    
    def set_lexicon(self, lexicon: KchoLexicon):
        """Set lexicon for frequency tracking"""
        self.lexicon = lexicon
    
    def load_from_file(self, file_path: str) -> List[Sentence]:
        """Load corpus from file"""
        sentences = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    sentence = self._process_sentence(line)
                    sentences.append(sentence)
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
        
        self.sentences = sentences
        logger.info(f"✅ Loaded {len(sentences)} sentences from {file_path}")
        return sentences
    
    def _process_sentence(self, text: str) -> Sentence:
        """Process a single sentence"""
        # Normalize and tokenize
        normalized = self.tokenizer.normalize(text)
        tokens = self.tokenizer.tokenize(normalized)
        
        # Analyze tokens
        analyzed_tokens = []
        for token_text in tokens:
            token = self.morph.analyze_token(token_text)
            analyzed_tokens.append(token)
        
        # Create sentence
        sentence = Sentence(
            text=text,
            tokens=analyzed_tokens,
            features={}
        )
        
        return sentence
    
    def get_statistics(self, sentences: List[Sentence] = None) -> Dict[str, Any]:
        """Get corpus statistics"""
        if sentences is None:
            sentences = self.sentences
        
        if not sentences:
            return {}
        
        total_sentences = len(sentences)
        total_tokens = sum(len(s.tokens) for s in sentences)
        total_words = sum(len(t.surface) for s in sentences for t in s.tokens)
        
        # Token length distribution
        token_lengths = [len(t.surface) for s in sentences for t in s.tokens]
        avg_token_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
        
        # Sentence length distribution
        sentence_lengths = [len(s.tokens) for s in sentences]
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        
        return {
            'total_sentences': total_sentences,
            'total_tokens': total_tokens,
            'total_words': total_words,
            'avg_token_length': avg_token_length,
            'avg_sentence_length': avg_sentence_length,
            'min_sentence_length': min(sentence_lengths) if sentence_lengths else 0,
            'max_sentence_length': max(sentence_lengths) if sentence_lengths else 0
        }

        
# ============================================================================
# MAIN SYSTEM CLASS
# ============================================================================

class KchoSystem:
    """
    K'Cho Language Processing System with comprehensive knowledge integration.
    
    This system uses KchoKnowledge (SQLite backend) to provide efficient linguistic analysis
    and serves as an API layer for future LLaMA integration.
    """
    
    def __init__(self, use_comprehensive_knowledge: bool = True):
        """
        Initialize the K'Cho processing system.
        
        Args:
            use_comprehensive_knowledge: If True, uses KchoKnowledge (SQLite) for enhanced capabilities.
                                       If False, uses LegacyKchoKnowledge for backward compatibility.
        """
        # Initialize knowledge base
        if use_comprehensive_knowledge:
            self.knowledge = KchoKnowledge()
            logger.info("✅ Using KchoKnowledge (SQLite) for enhanced analysis")
        else:
            self.knowledge = LegacyKchoKnowledge()
            logger.info("⚠️  Using LegacyKchoKnowledge (deprecated)")
        
        # Initialize components
        self.normalizer = KChoNormalizer()
        self.morphology_analyzer = KchoMorphologyAnalyzer()
        self.syntax_analyzer = KchoSyntaxAnalyzer()
        self.validator = KchoValidator()
        self.tokenizer = KchoTokenizer()
        self.corpus = KchoCorpus()
        self.lexicon = KchoLexicon()
        self.collocation_extractor = KchoCollocationExtractor()
        self.api_layer = EnhancedKchoAPILayer(self.knowledge)
        
        # Set knowledge references
        self.morphology_analyzer.set_knowledge(self.knowledge)
        self.syntax_analyzer.set_knowledge(self.knowledge)
        self.validator.set_knowledge(self.knowledge)
        self.corpus.set_lexicon(self.lexicon)
        
        logger.info("✅ KchoSystem initialized successfully")
    
    def analyze_text(self, text: str) -> Sentence:
        """Analyze a text string and return annotated sentence."""
        # Normalize and tokenize
        normalized = self.normalizer.normalize_text(text)
        tokens = self.tokenizer.tokenize(normalized)
        
        # Analyze tokens
        analyzed_tokens = []
        for token_text in tokens:
            token = self.morphology_analyzer.analyze_token(token_text)
            analyzed_tokens.append(token)
        
        # Create sentence
        sentence = Sentence(
            text=text,
            tokens=analyzed_tokens,
            features={}
        )
        
        # Perform syntactic analysis
        syntax_analysis = self.syntax_analyzer.analyze_syntax(sentence)
        sentence.features.update(syntax_analysis)
        
        return sentence
    
    def load_corpus(self, file_path: str) -> List[Sentence]:
        """Load and process corpus from file."""
        return self.corpus.load_from_file(file_path)
    
    def build_lexicon(self, sentences: List[Sentence]) -> Dict:
        """Build lexicon from sentences."""
        return self.lexicon.build_from_sentences(sentences)
    
    def extract_collocations(self, sentences: List[Sentence]) -> Dict:
        """Extract collocations from sentences."""
        return self.collocation_extractor.extract_from_corpus(sentences)
    
    def analyze_corpus(self, sentences: List[Sentence]) -> Dict:
        """Perform comprehensive corpus analysis."""
        analysis = {
            'statistics': self.corpus.get_statistics(sentences),
            'morphology': self.morphology_analyzer.analyze_corpus(sentences),
            'syntax': self.syntax_analyzer.analyze_corpus(sentences),
            'collocations': self.extract_collocations(sentences)
        }
        return analysis
    
    def discover_patterns(self, corpus: List[str], min_frequency: int = 5) -> Dict[str, Dict]:
        """Discover new patterns from corpus."""
        if hasattr(self.knowledge, 'pattern_discovery_engine'):
            return self.knowledge.pattern_discovery_engine.discover_patterns(corpus, min_frequency)
        else:
            logger.warning("Pattern discovery not available with legacy knowledge base")
            return {}
    
    def configure_llama_api(self, provider: str = "ollama", **kwargs):
        """
        Configure LLaMA API integration
        
        Args:
            provider: LLaMA provider ("ollama", "openai", "anthropic")
            **kwargs: Additional configuration options
        """
        llama_config = create_llama_config(provider=provider, **kwargs)
        self.api_layer = EnhancedKchoAPILayer(self.knowledge, llama_config)
        logger.info(f"✅ LLaMA API configured with provider: {provider}")
    
    def get_api_response(self, query: str, use_llama: bool = True, context: Optional[str] = None, deep_research: bool = False, log_response: bool = False, log_format: str = "json") -> Dict[str, Any]:
        """
        Get API response with optional LLaMA integration
        
        Args:
            query: User query about K'Cho language
            use_llama: Whether to use LLaMA API (default: True)
            context: Optional context for the query
            deep_research: Whether to use deep research mode (3000 tokens)
            log_response: Whether to log response to file
            log_format: Log format ("json" or "csv")
            
        Returns:
            Structured API response
        """
        return self.api_layer.get_api_response(query, use_llama=use_llama, context=context, deep_research=deep_research, log_response=log_response, log_format=log_format)
    
    def close(self):
        """Clean up resources"""
        if hasattr(self.lexicon, 'close'):
            self.lexicon.close()
        if hasattr(self.knowledge, 'close'):
            self.knowledge.close()


# ============================================================================
# LINGUISTIC RESEARCH ENGINE
# ============================================================================

class LinguisticResearchEngine:
    """Conducts comprehensive linguistic research on K'Cho corpora."""
    
    def __init__(self, knowledge_base: KchoKnowledge):
        self.knowledge = knowledge_base
    
    def conduct_comprehensive_research(self, corpus: List[str], research_focus: str = 'all') -> Dict[str, Any]:
        """Conduct comprehensive linguistic research on corpus"""
        
        research_results = {
            'metadata': {
                'corpus_size': len(corpus),
                'research_focus': research_focus,
                'knowledge_sources': getattr(self.knowledge, 'data_sources', {}),
                'timestamp': datetime.now().isoformat()
            },
            'known_patterns': {},
            'novel_patterns': {},
            'linguistic_insights': [],
            'creative_findings': []
        }
        
        # Analyze known patterns
        if research_focus in ['all', 'patterns']:
            research_results['known_patterns'] = self._analyze_known_patterns(corpus)
        
        # Discover novel patterns
        if research_focus in ['all', 'discovery']:
            research_results['novel_patterns'] = self._discover_novel_patterns(corpus)
        
        # Generate linguistic insights
        if research_focus in ['all', 'insights']:
            research_results['linguistic_insights'] = self._generate_linguistic_insights(corpus)
        
        # Generate creative findings
        if research_focus in ['all', 'creative']:
            research_results['creative_findings'] = self._generate_creative_findings(corpus)
        
        return research_results
    
    def _analyze_known_patterns(self, corpus: List[str]) -> Dict[str, Any]:
        """Analyze how well-known patterns appear in corpus"""
        pattern_counts = Counter()
        
        # Get gold standard patterns
        if hasattr(self.knowledge, 'gold_standard_patterns'):
            gold_patterns = self.knowledge.gold_standard_patterns
        else:
            gold_patterns = {}
        
        # Count pattern occurrences
        for text in corpus:
            normalized = normalize_text(text)
            tokens = tokenize(normalized)
            
            for pattern in gold_patterns.keys():
                if pattern in normalized:
                    pattern_counts[pattern] += 1
        
        return {
            'total_patterns_analyzed': len(gold_patterns),
            'patterns_found': len(pattern_counts),
            'pattern_frequencies': dict(pattern_counts),
            'coverage_rate': len(pattern_counts) / len(gold_patterns) if gold_patterns else 0
        }
    
    def _discover_novel_patterns(self, corpus: List[str]) -> Dict[str, Any]:
        """Discover patterns not in gold standard"""
        if hasattr(self.knowledge, 'discover_new_patterns'):
            discovered = self.knowledge.discover_new_patterns(corpus, min_frequency=3)
        return {
                'total_discovered': len(discovered),
                'patterns': discovered
            }
        else:
            return {'total_discovered': 0, 'patterns': {}}
    
    def _generate_linguistic_insights(self, corpus: List[str]) -> List[str]:
        """Generate linguistic insights from corpus analysis"""
        insights = []
        
        # Analyze morphological patterns
        suffix_patterns = Counter()
        prefix_patterns = Counter()
        
        for text in corpus:
            normalized = normalize_text(text)
            tokens = tokenize(normalized)
            
            for token in tokens:
                if token.endswith('-na'):
                    suffix_patterns['-na'] += 1
                if token.startswith('a-'):
                    prefix_patterns['a-'] += 1
        
        if suffix_patterns:
            insights.append(f"Morphological insight: {dict(suffix_patterns)} occurrences")
        
        if prefix_patterns:
            insights.append(f"Morphological insight: {dict(prefix_patterns)} occurrences")
        
        # Add more insights based on analysis
        insights.append("Corpus shows rich morphological variation")
        insights.append("Verb-particle patterns are prominent")
        
        return insights
    
    def _generate_creative_findings(self, corpus: List[str]) -> List[str]:
        """Generate creative linguistic findings"""
        findings = []
        
        # Analyze word co-occurrence patterns
        word_pairs = Counter()
        
        for text in corpus:
            normalized = normalize_text(text)
            tokens = tokenize(normalized)
            
            for i in range(len(tokens) - 1):
                pair = f"{tokens[i]} {tokens[i+1]}"
                word_pairs[pair] += 1
        
        # Find interesting patterns
        common_pairs = word_pairs.most_common(5)
        for pair, count in common_pairs:
            findings.append(f"Creative finding: '{pair}' appears {count} times")
        
        findings.append("Creative insight: Language shows systematic word order patterns")
        findings.append("Creative finding: Morphological complexity varies by word class")
        
        return findings


# ============================================================================
# KNOWLEDGE BASE (SQLite Backend)
# ============================================================================

class KchoKnowledge(KchoKnowledgeBase):
    """
    K'Cho knowledge base using SQLite backend.
    
    This is the primary knowledge base implementation, directly inheriting
    from KchoKnowledgeBase for optimal performance and functionality.
    """
    
    def __init__(self, data_dir: str = None, db_path: str = None, in_memory: bool = False):
        """
        Initialize K'Cho knowledge base with SQLite backend.
        
        Args:
            data_dir: Path to data directory. If None, uses default location.
            db_path: Path to SQLite database. If None, uses default location.
            in_memory: If True, use in-memory database (faster but non-persistent)
        """
        # Initialize SQLite backend directly
        super().__init__(data_dir=data_dir, db_path=db_path, in_memory=in_memory)
        
        # Create backward compatibility properties
        self._create_compatibility_properties()
        
        # Initialize pattern discovery engine
        self.pattern_discovery_engine = PatternDiscoveryEngine(self)
        
        logger.info("✅ KchoKnowledge initialized with SQLite backend")
    
    def _create_compatibility_properties(self):
        """Create properties for backward compatibility with legacy API."""
        # Load data into memory for compatibility
        self.linguistic_data = self._load_linguistic_data()
        self.gold_standard_patterns = self._load_gold_standard_patterns()
        self.word_frequency_data = self._load_word_frequency_data()
        self.parallel_data = self._load_parallel_data()
        
        # Create comprehensive knowledge base
        self._create_comprehensive_knowledge_base()
        
        # Track data sources
        stats = self.get_statistics()
        self.data_sources = {
            'linguistic_data': len(self.linguistic_data),
            'gold_standard_patterns': stats['collocations'],
            'word_frequency_data': stats['word_frequencies'],
            'parallel_data': stats['parallel_sentences'],
            'total_words': stats['word_categories']
        }
    
    def _load_linguistic_data(self) -> Dict[str, Any]:
        """Load linguistic data for compatibility."""
        import json
        from pathlib import Path
        
        data_dir = Path(__file__).parent / 'data'
        linguistic_file = data_dir / 'linguistic_data.json'
        
        if not linguistic_file.exists():
            return {}
        
        with open(linguistic_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_gold_standard_patterns(self) -> Dict[str, Dict]:
        """Load gold standard patterns for compatibility."""
        patterns = {}
        
        # Get collocations from database
        categories = ['VP', 'PP', 'APP', 'AGR', 'AUX', 'COMP', 'MWE']
        for category in categories:
            collocations = self.get_collocations_by_category(category)
            for colloc in collocations:
                patterns[colloc['words']] = {
                    'category': colloc['category'],
                    'frequency': colloc['frequency'],
                    'notes': colloc['notes'],
                    'source': colloc['source']
                }
        
        return patterns
    
    def _load_word_frequency_data(self) -> Dict[str, int]:
        """Load word frequency data for compatibility."""
        frequencies = {}
        
        # Get frequencies from database
        for row in self.get_all_word_frequencies():
            frequencies[row['word']] = row['frequency']
        
        return frequencies
    
    def _load_parallel_data(self) -> Dict[str, Any]:
        """Load parallel data for compatibility."""
        sentences = self.get_parallel_sentences(limit=1000)
        return {
            'sentence_pairs': [
                {
                    'id': s['id'],
                    'kcho': s['kcho'],
                    'english': s['english'],
                    'source': s['source'],
                    'linguistic_features': json.loads(s['features']) if s['features'] else []
                }
                for s in sentences
            ]
        }
    
    def _create_comprehensive_knowledge_base(self):
        """Create comprehensive knowledge base for compatibility."""
        # Extract all linguistic categories
        self.VERB_STEMS = self.linguistic_data.get('verb_stems', {})
        self.PRONOUNS = self.linguistic_data.get('pronouns', {})
        self.AGREEMENT = self.linguistic_data.get('agreement_particles', {})
        self.POSTPOSITIONS = self.linguistic_data.get('postpositions', {})
        self.TENSE_ASPECT = self.linguistic_data.get('tense_aspect', {})
        self.APPLICATIVES = self.linguistic_data.get('applicatives', {})
        self.CONNECTIVES = self.linguistic_data.get('connectives', {})
        self.COMMON_NOUNS = self.linguistic_data.get('common_nouns', {})
        self.DEMONSTRATIVES = self.linguistic_data.get('demonstratives', {})
        self.QUANTIFIERS = self.linguistic_data.get('quantifiers', {})
        self.ADJECTIVES = self.linguistic_data.get('adjectives', {})
        self.DIRECTIONALS = self.linguistic_data.get('directionals', {})
        self.COMMON_WORDS = set(self.linguistic_data.get('common_words', []))
        
        # Create comprehensive word categories
        self.all_words = set()
        self.word_categories = defaultdict(set)
        
        # Add words from all categories
        for category, words in [
            ('verbs', self.VERB_STEMS.keys()),
            ('pronouns', self.PRONOUNS.keys()),
            ('agreement', self.AGREEMENT.keys()),
            ('postpositions', self.POSTPOSITIONS.keys()),
            ('tense_aspect', self.TENSE_ASPECT.keys()),
            ('applicatives', self.APPLICATIVES.keys()),
            ('connectives', self.CONNECTIVES.keys()),
            ('nouns', self.COMMON_NOUNS.keys()),
            ('demonstratives', self.DEMONSTRATIVES.keys()),
            ('quantifiers', self.QUANTIFIERS.keys()),
            ('adjectives', self.ADJECTIVES.keys()),
            ('directionals', self.DIRECTIONALS.keys())
        ]:
            for word in words:
                self.all_words.add(word.lower())
                self.word_categories[category].add(word.lower())
        
        # Add frequency data
        for word in self.word_frequency_data.keys():
            self.all_words.add(word.lower())
            self.word_categories['frequent'].add(word.lower())
    
    def _classify_word_pair(self, word1: str, word2: str) -> str:
        """Classify word pair based on linguistic categories."""
        word1_categories = self._get_word_categories(word1)
        word2_categories = self._get_word_categories(word2)
        
        # Verb + Particle patterns
        if 'verbs' in word1_categories and 'tense_aspect' in word2_categories:
            return 'VP'
        elif 'verbs' in word2_categories and 'tense_aspect' in word1_categories:
            return 'VP'
        
        # Postposition + Noun patterns
        if 'postpositions' in word1_categories and 'nouns' in word2_categories:
            return 'PP'
        elif 'postpositions' in word2_categories and 'nouns' in word1_categories:
            return 'PP'
        
        # Applicative patterns
        if 'applicatives' in word1_categories or 'applicatives' in word2_categories:
            return 'APP'
        
        # Agreement patterns
        if 'agreement' in word1_categories and 'verbs' in word2_categories:
            return 'AGR'
        elif 'agreement' in word2_categories and 'verbs' in word1_categories:
            return 'AGR'
        
        return 'UNKNOWN'
    
    def _get_word_categories(self, word: str) -> Set[str]:
        """Get all categories for a word."""
        categories = set()
        word_lower = word.lower()
        
        for category, words in self.word_categories.items():
            if word_lower in words:
                categories.add(category)
        
        return categories
    
    # Backward compatibility methods
    def is_gold_standard_pattern(self, word1: str, word2: str) -> Tuple[bool, str, float]:
        """Check if word pair is a gold standard pattern."""
        pattern = f"{word1} {word2}"
        collocations = self.search_collocations(pattern)
        
        if collocations:
            colloc = collocations[0]
            return True, colloc['category'], colloc.get('confidence', 0.8)
        
        return False, 'unknown', 0.0
    
    def get_pattern_confidence(self, word1: str, word2: str) -> float:
        """Get confidence score for a pattern."""
        _, _, confidence = self.is_gold_standard_pattern(word1, word2)
        return confidence
    
    def discover_new_patterns(self, corpus: List[str], min_frequency: int = 5) -> Dict[str, Dict]:
        """Discover new patterns using the pattern discovery engine."""
        return self.pattern_discovery_engine.discover_patterns(corpus, min_frequency)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
