"""
K'Cho Language Processing System - Unified Implementation
==========================================================

A production-ready system for K'Cho language processing, combining morphological
analysis, syntactic parsing, corpus management, and ML data preparation.

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
# Import new modules
from .normalize import KChoNormalizer, normalize_text, tokenize
from .collocation import CollocationExtractor, AssociationMeasure, CollocationResult
from .export import to_csv, to_json, to_text
from .evaluation import compute_precision_recall, evaluate_ranking, load_gold_standard

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
    gloss: str
    translation: Optional[str] = None
    syntax: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'tokens': [t.to_dict() for t in self.tokens],
            'gloss': self.gloss,
            'translation': self.translation,
            'syntax': self.syntax,
            'metadata': self.metadata
        }


# ============================================================================
# LINGUISTIC KNOWLEDGE BASE
# ============================================================================
# """
# K'CHO LINGUISTIC KNOWLEDGE BASE - JSON-Based Version
# Loads linguistic data from JSON files for better maintainability
# """



class KchoKnowledge:
    """
    K'Cho linguistic knowledge base that loads data from JSON files.
    This approach separates data from code for better maintainability.
    """
    
    def __init__(self, json_path: Optional[str] = None):
        """
        Initialize the knowledge base by loading data from JSON.
        
        Args:
            json_path: Path to the JSON file. If None, looks for 'kcho_linguistic_data.json'
                      in the current directory or same directory as this script.
        """
        if json_path is None:
            # Try to find the JSON file in common locations
            possible_paths = [
                'kcho/data/linguistic_data.json',  # New package data location
                os.path.join(os.path.dirname(__file__), 'data', 'linguistic_data.json'),  # Relative to script location
                os.path.join(os.path.dirname(__file__), 'linguistic_data.json'),  # Keep original as fallback
                '/Users/hungom/Desktop/KChoCaToolKit/kcho/data/linguistic_data.json',  # Absolute path
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    json_path = path
                    break
            
            if json_path is None:
                raise FileNotFoundError(
                    "Could not find linguistic_data.json. "
                    "Please specify the path explicitly or ensure the file is in kcho/data/."
                )
        
        # Load the JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Store metadata
        self.metadata = data.get('metadata', {})
        
        # Load all linguistic categories
        self.VERB_STEMS = data.get('verb_stems', {})
        self.PRONOUNS = data.get('pronouns', {})
        self.AGREEMENT = data.get('agreement_particles', {})
        self.POSTPOSITIONS = data.get('postpositions', {})
        self.TENSE_ASPECT = data.get('tense_aspect', {})
        self.APPLICATIVES = data.get('applicatives', {})
        self.CONNECTIVES = data.get('connectives', {})
        self.COMMON_NOUNS = data.get('common_nouns', {})
        self.DEMONSTRATIVES = data.get('demonstratives', {})
        self.QUANTIFIERS = data.get('quantifiers', {})
        self.ADJECTIVES = data.get('adjectives', {})
        self.DIRECTIONALS = data.get('directionals', {})
        self.COMMON_WORDS = set(data.get('common_words', []))
        
        # Character set
        self.KCHO_CHARS = set("abcdefghijklmnopqrstuvwyzáàéèíìóòúùÜäö'ΫӪ ")
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def is_postposition(self, word: str) -> bool:
        """Check if word is a postposition/case marker"""
        return word.lower() in self.POSTPOSITIONS
    
    def is_tense_marker(self, word: str) -> bool:
        """Check if word is a tense/aspect marker"""
        return word.lower() in self.TENSE_ASPECT
    
    def is_agreement(self, word: str) -> bool:
        """Check if word is an agreement particle"""
        return word.lower() in self.AGREEMENT
    
    def is_pronoun(self, word: str) -> bool:
        """Check if word is a pronoun"""
        return word.lower() in self.PRONOUNS
    
    def is_connective(self, word: str) -> bool:
        """Check if word is a connective/conjunction"""
        return word.lower() in self.CONNECTIVES
    
    def is_verb(self, word: str) -> bool:
        """Check if word is a known verb"""
        return word.lower() in self.VERB_STEMS
    
    def get_stem_ii(self, stem_i: str) -> Optional[str]:
        """Get Stem II form from Stem I"""
        verb_data = self.VERB_STEMS.get(stem_i.lower())
        return verb_data.get('stem2') if verb_data else None
    
    def get_stem_i(self, stem_ii: str) -> Optional[str]:
        """Get Stem I form from Stem II"""
        for stem1, info in self.VERB_STEMS.items():
            if info['stem2'] == stem_ii.lower():
                return stem1
        return None
    
    def get_word_info(self, word: str) -> Optional[Dict]:
        """Get comprehensive information about a word"""
        word_lower = word.lower()
        
        # Check all categories
        if word_lower in self.VERB_STEMS:
            return {'category': 'verb', 'data': self.VERB_STEMS[word_lower]}
        elif word_lower in self.PRONOUNS:
            return {'category': 'pronoun', 'data': self.PRONOUNS[word_lower]}
        elif word_lower in self.POSTPOSITIONS:
            return {'category': 'postposition', 'data': self.POSTPOSITIONS[word_lower]}
        elif word_lower in self.TENSE_ASPECT:
            return {'category': 'tense_aspect', 'data': self.TENSE_ASPECT[word_lower]}
        elif word_lower in self.COMMON_NOUNS:
            return {'category': 'noun', 'data': self.COMMON_NOUNS[word_lower]}
        elif word_lower in self.CONNECTIVES:
            return {'category': 'connective', 'data': self.CONNECTIVES[word_lower]}
        elif word_lower in self.DEMONSTRATIVES:
            return {'category': 'demonstrative', 'data': self.DEMONSTRATIVES[word_lower]}
        elif word_lower in self.QUANTIFIERS:
            return {'category': 'quantifier', 'data': self.QUANTIFIERS[word_lower]}
        elif word_lower in self.ADJECTIVES:
            return {'category': 'adjective', 'data': self.ADJECTIVES[word_lower]}
        elif word_lower in self.DIRECTIONALS:
            return {'category': 'directional', 'data': self.DIRECTIONALS[word_lower]}
        
        return None
    
    def get_verb_pattern(self, stem_i: str) -> Optional[str]:
        """Get the morphological pattern for a verb"""
        verb_data = self.VERB_STEMS.get(stem_i.lower())
        return verb_data.get('pattern') if verb_data else None
    
    def is_negative(self, word: str) -> bool:
        """Check if word is negative marker"""
        return word.lower() == 'kä'
    
    def get_all_verbs(self) -> List[str]:
        """Get list of all known verbs"""
        return list(self.VERB_STEMS.keys())
    
    def get_all_nouns(self) -> List[str]:
        """Get list of all known nouns"""
        return list(self.COMMON_NOUNS.keys())
    
    def analyze_word(self, word: str) -> str:
        """Provide detailed analysis of a word"""
        info = self.get_word_info(word)
        if info:
            category = info['category']
            data = info['data']
            gloss = data.get('gloss', 'N/A')
            return f"{word} → {category.upper()}: {gloss}"
        return f"{word} → UNKNOWN"
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the knowledge base"""
        return {
            'verbs': len(self.VERB_STEMS),
            'nouns': len(self.COMMON_NOUNS),
            'pronouns': len(self.PRONOUNS),
            'postpositions': len(self.POSTPOSITIONS),
            'tense_aspect': len(self.TENSE_ASPECT),
            'connectives': len(self.CONNECTIVES),
            'demonstratives': len(self.DEMONSTRATIVES),
            'quantifiers': len(self.QUANTIFIERS),
            'adjectives': len(self.ADJECTIVES),
            'directionals': len(self.DIRECTIONALS),
            'total': sum([
                len(self.VERB_STEMS),
                len(self.COMMON_NOUNS),
                len(self.PRONOUNS),
                len(self.POSTPOSITIONS),
                len(self.TENSE_ASPECT),
                len(self.CONNECTIVES),
                len(self.DEMONSTRATIVES),
                len(self.QUANTIFIERS),
                len(self.ADJECTIVES),
                len(self.DIRECTIONALS),
            ])
        }
    
    def export_to_json(self, output_path: str):
        """Export the current knowledge base back to JSON"""
        data = {
            'metadata': self.metadata,
            'verb_stems': self.VERB_STEMS,
            'pronouns': self.PRONOUNS,
            'agreement_particles': self.AGREEMENT,
            'postpositions': self.POSTPOSITIONS,
            'tense_aspect': self.TENSE_ASPECT,
            'applicatives': self.APPLICATIVES,
            'connectives': self.CONNECTIVES,
            'common_nouns': self.COMMON_NOUNS,
            'demonstratives': self.DEMONSTRATIVES,
            'quantifiers': self.QUANTIFIERS,
            'adjectives': self.ADJECTIVES,
            'directionals': self.DIRECTIONALS,
            'common_words': list(self.COMMON_WORDS)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def print_statistics(self):
        """Print comprehensive statistics"""
        stats = self.get_statistics()
        print("=== K'CHO KNOWLEDGE BASE STATISTICS ===\n")
        print(f"Version: {self.metadata.get('version', 'N/A')}")
        print(f"Last Updated: {self.metadata.get('last_updated', 'N/A')}")
        print(f"\nCategories:")
        print(f"  Verbs:          {stats['verbs']:3d}")
        print(f"  Nouns:          {stats['nouns']:3d}")
        print(f"  Pronouns:       {stats['pronouns']:3d}")
        print(f"  Postpositions:  {stats['postpositions']:3d}")
        print(f"  Tense/Aspect:   {stats['tense_aspect']:3d}")
        print(f"  Connectives:    {stats['connectives']:3d}")
        print(f"  Demonstratives: {stats['demonstratives']:3d}")
        print(f"  Quantifiers:    {stats['quantifiers']:3d}")
        print(f"  Adjectives:     {stats['adjectives']:3d}")
        print(f"  Directionals:   {stats['directionals']:3d}")
        print(f"\nTotal Entries:  {stats['total']:3d}")


# ========================================================================
# USAGE EXAMPLE
# ========================================================================
if __name__ == "__main__":
    # Load the knowledge base
    kb = KchoKnowledge()
    
    # Print statistics
    kb.print_statistics()
    
    # Test some lookups
    print("\n=== Word Analysis Examples ===")
    test_words = ['om', 'kei', 'naw', 'ci', 'law', 'Khanpughi']
    for word in test_words:
        print(kb.analyze_word(word))
    
    # Test verb stems
    print("\n=== Verb Stem Examples ===")
    for verb in ['om', 'law', 'pyein', 'hmu'][:4]:
        stem2 = kb.get_stem_ii(verb)
        if stem2:
            print(f"Stem I: {verb:10s} → Stem II: {stem2}")

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
        """
        Tokenize text into words.
        Properly handles K'Cho orthography including:
        - k', K' (e.g., k'cho, K'Cho)
        - m', M' (e.g., m'htak)
        - ng', Ng' (e.g., ng'thu, Ng'laamihta)
        """
        text = self.normalize(text)
        
        # K'Cho consonants that use apostrophe
        KCHO_APOSTROPHE_PREFIXES = ("k'", "m'", "ng'", "K'", "M'", "Ng'", "NG'")
        
        tokens = []
        for word in text.split():
            # Check if word starts with K'Cho apostrophe pattern
            has_kcho_apostrophe = any(word.startswith(prefix) for prefix in KCHO_APOSTROPHE_PREFIXES)
            
            if has_kcho_apostrophe:
                # Apostrophe is part of the word - only strip trailing punctuation
                word_clean = word.rstrip('.,!?;:"')
            else:
                # Normal word - strip both leading and trailing punctuation
                # But preserve internal apostrophes
                word_clean = word.strip('.,!?;:"')
            
            if word_clean:
                tokens.append(word_clean)
        
        return tokens
        
    def sentence_split(self, text: str) -> List[str]:
        """
        Split text into sentences.
        CRITICAL: Semicolons (;) do NOT split K'Cho sentences!
        """
        text = self.normalize(text)
        import re
        
        # Only split on period, !, ? - NOT semicolon
        text = re.sub(r'\.\s+([A-Z])', r'.|SENT_BOUNDARY|\1', text)
        text = re.sub(r'\.$', r'.|SENT_BOUNDARY|', text)
        text = re.sub(r'[!?]\s+', r'|SENT_BOUNDARY|', text)
        text = re.sub(r'[!?]$', r'|SENT_BOUNDARY|', text)
        
        sentences = text.split('|SENT_BOUNDARY|')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences and text.strip():
            sentences = [text.strip()]
        
        return sentences


class KchoValidator:
    """Text validation and K'Cho language detection"""
    
    def __init__(self):
        self.knowledge = KchoKnowledge()
    
    def is_kcho_text(self, text: str) -> Tuple[bool, float, Dict]:
        """
        Determine if text is K'Cho with confidence score.
        Returns: (is_kcho, confidence, metrics)
        """
        if not text or len(text.strip()) == 0:
            return False, 0.0, {}
        
        metrics = {}
        
        # Check character set
        text_chars = set(text.lower())
        valid_chars = text_chars.intersection(self.knowledge.KCHO_CHARS)
        char_ratio = len(valid_chars) / len(text_chars) if text_chars else 0
        metrics['char_validity'] = char_ratio
        
        # Check for K'Cho markers
        words = text.lower().split()
        kcho_markers = 0
        
        for word in words:
            if self.knowledge.is_postposition(word):
                kcho_markers += 2
            elif self.knowledge.is_tense_marker(word):
                kcho_markers += 2
            elif self.knowledge.is_agreement(word):
                kcho_markers += 1
            elif word in self.knowledge.COMMON_WORDS:
                kcho_markers += 1
        
        marker_score = min(kcho_markers / len(words), 1.0) if words else 0
        metrics['marker_score'] = marker_score
        
        # Check for K'Cho patterns
        patterns = [
            r"\bnoh\b.*\b(ci|khai)\b",
            r"\b(ka|na|a)\b\s+\w+",
            r"\w+\s+(am|on)\b",
            r"k'[a-z]+",
        ]
        
        pattern_matches = sum(1 for p in patterns if re.search(p, text.lower()))
        pattern_score = pattern_matches / len(patterns)
        metrics['pattern_score'] = pattern_score
        
        # Overall confidence
        confidence = (char_ratio * 0.3 + marker_score * 0.5 + pattern_score * 0.2)
        metrics['overall_confidence'] = confidence
        
        is_kcho = confidence > 0.3
        
        return is_kcho, confidence, metrics


# ============================================================================
# MORPHOLOGICAL ANALYSIS
# ============================================================================

class KchoMorphologyAnalyzer:
    """
    Comprehensive morphological analysis including verb stem alternation,
    applicatives, and agreement marking.
    """
    
    def __init__(self):
        self.knowledge = KchoKnowledge()
    
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
        
        morphemes = []
        lemma = word
        pos = POS.UNKNOWN
        stem_type = None
        features = {}
        
        # Check postpositions
        if self.knowledge.is_postposition(word):
            pos = POS.POSTPOSITION
            post_info = self.knowledge.POSTPOSITIONS[word]
            morphemes.append(Morpheme(
                form=word,
                lemma=word,
                gloss=post_info['gloss'],
                type='postposition',
                features=post_info
            ))
            return Token(word, lemma, pos, morphemes, stem_type, features)
        
        # Check tense markers
        if self.knowledge.is_tense_marker(word):
            pos = POS.TENSE
            tense_info = self.knowledge.TENSE_ASPECT[word]
            morphemes.append(Morpheme(
                form=word,
                lemma=word,
                gloss=tense_info['gloss'],
                type='particle',
                features=tense_info
            ))
            return Token(word, lemma, pos, morphemes, stem_type, features)
        
        # Check agreement
        if self.knowledge.is_agreement(word):
            pos = POS.AGREEMENT
            agr_info = self.knowledge.AGREEMENT[word]
            morphemes.append(Morpheme(
                form=word,
                lemma=word,
                gloss=f"{agr_info.get('person', '')}{agr_info.get('number', '')}",
                type='agreement',
                features=agr_info
            ))
            return Token(word, lemma, pos, morphemes, stem_type, features)
        
        # Analyze verb with applicative
        if word.endswith('nák') or word.endswith('na'):
            suffix = 'nák' if word.endswith('nák') else 'na'
            root = word[:-len(suffix)]
            pos = POS.VERB
            stem_type = StemType.STEM_II if suffix == 'nák' else StemType.STEM_I
            
            lemma = self._get_lemma(root)
            
            morphemes.append(Morpheme(
                form=root,
                lemma=lemma,
                gloss='V',
                type='root',
                features={'stem': stem_type.value}
            ))
            
            morphemes.append(Morpheme(
                form=suffix,
                lemma='na',
                gloss='APPL',
                type='suffix',
                features=self.knowledge.APPLICATIVES[suffix]
            ))
        else:
            # Simple word
            lemma = self._get_lemma(word)
            
            if lemma in self.knowledge.VERB_STEMS:
                pos = POS.VERB
                stem_type = StemType.STEM_I
            elif lemma in self.knowledge.COMMON_NOUNS:
                pos = POS.NOUN
            elif word and (word[0].isupper() or word.startswith('k\'')):
                pos = POS.NOUN
            
            morphemes.append(Morpheme(
                form=word,
                lemma=lemma,
                gloss=self.knowledge.VERB_STEMS.get(lemma, {}).get('gloss', word),
                type='root',
                features={}
            ))
        
        return Token(word, lemma, pos, morphemes, stem_type, features)
    
    def _get_lemma(self, surface: str) -> str:
        """Get citation form (Stem I) from any verb form"""
        # Check if it's Stem II
        stem_i = self.knowledge.get_stem_i(surface)
        if stem_i:
            return stem_i
        return surface
    
    def analyze_sentence(self, text: str) -> Sentence:
        """Analyze complete sentence"""
        tokenizer = KchoTokenizer()
        words = tokenizer.tokenize(text)
        
        tokens = [self.analyze_token(word) for word in words]
        gloss = self._generate_gloss(tokens)
        
        return Sentence(
            text=text,
            tokens=tokens,
            gloss=gloss,
            metadata={'timestamp': datetime.now().isoformat()}
        )
    
    def _generate_gloss(self, tokens: List[Token]) -> str:
        """Generate interlinear gloss"""
        gloss_parts = []
        
        for token in tokens:
            if len(token.morphemes) == 1:
                gloss_parts.append(token.morphemes[0].gloss)
            else:
                morpheme_glosses = [m.gloss for m in token.morphemes]
                gloss_parts.append('-'.join(morpheme_glosses))
        
        return ' '.join(gloss_parts)


# ============================================================================
# SYNTACTIC ANALYSIS
# ============================================================================

class KchoSyntaxAnalyzer:
    """
    Syntactic analysis: clause type identification, argument structure,
    relative clauses, and verb stem context.
    """
    
    def __init__(self):
        self.knowledge = KchoKnowledge()
        self.morph = KchoMorphologyAnalyzer()
    
    def analyze_syntax(self, sentence: Sentence) -> Dict:
        """Perform syntactic analysis on a sentence"""
        tokens = sentence.tokens
        words = [t.surface.lower() for t in tokens]
        
        analysis = {
            'clause_type': self._identify_clause_type(tokens),
            'has_applicative': any(t.pos == POS.APPLICATIVE for t in tokens),
            'has_relative_clause': self._has_relative_clause(words),
            'verb_stem_form': self._identify_stem_form(tokens),
            'arguments': self._extract_arguments(tokens),
        }
        
        return analysis
    
    def _identify_clause_type(self, tokens: List[Token]) -> str:
        """Identify transitivity: intransitive/transitive/ditransitive"""
        has_noh = any(t.surface.lower() == 'noh' for t in tokens)
        noun_count = sum(1 for t in tokens if t.pos == POS.NOUN)
        
        if has_noh:
            if noun_count >= 3:
                return 'ditransitive'
            return 'transitive'
        return 'intransitive'
    
    def _has_relative_clause(self, words: List[str]) -> bool:
        """Detect relative clause marked by 'ah'"""
        for i in range(len(words) - 1):
            if words[i] == 'ah' and i > 2:
                return True
        return False
    
    def _identify_stem_form(self, tokens: List[Token]) -> Optional[str]:
        """Identify verb stem form (I or II)"""
        words = [t.surface.lower() for t in tokens]
        
        # Check for tense markers
        if 'ci' in words or 'khai' in words:
            has_3rd_agr = 'a' in words[:words.index('ci') if 'ci' in words else words.index('khai')]
            if not has_3rd_agr:
                return 'I'
        
        if 'ung' in words:
            return 'II'
        
        # Check for applicative
        for token in tokens:
            for morph in token.morphemes:
                if morph.gloss == 'APPL':
                    return morph.features.get('stem')
        
        return None
    
    def _extract_arguments(self, tokens: List[Token]) -> Dict[str, List[str]]:
        """Extract sentence arguments"""
        arguments = {
            'subject': [],
            'object': [],
            'oblique': []
        }
        
        words = [t.surface for t in tokens]
        
        for i, token in enumerate(tokens):
            # Subject: NP followed by 'noh'
            if token.pos == POS.NOUN and i < len(words) - 1:
                if words[i+1].lower() == 'noh':
                    arguments['subject'].append(token.surface)
            
            # Object: N before verb
            if token.pos == POS.NOUN:
                if i < len(words) - 1:
                    next_token = tokens[i+1]
                    if next_token.pos == POS.VERB or next_token.pos == POS.TENSE:
                        if words[i+1].lower() not in ['noh', 'ah', 'am', 'on']:
                            arguments['object'].append(token.surface)
            
            # Oblique: NP with postposition
            if i > 0 and token.surface.lower() in ['on', 'am']:
                if tokens[i-1].pos == POS.NOUN:
                    arguments['oblique'].append(words[i-1])
        
        return arguments
#============================================================================
# COLLOCATION EXTRACTION
#============================================================================

class KchoCollocationExtractor:
    """
    Extract collocations for low-resource K'Cho language.
    Uses PMI (Pointwise Mutual Information) - standard for low-resource NLP.
    """
    
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
        
        # Collect statistics
        for sentence in sentences:
            tokens = [t.lemma.lower() for t in sentence.tokens]
            self.word_freq.update(tokens)
            self.total_words += len(tokens)
            
            bigrams = list(zip(tokens[:-1], tokens[1:]))
            self.bigram_freq.update(bigrams)
            self.total_bigrams += len(bigrams)
        
        # Extract significant bigrams using PMI
        collocations = []
        for (w1, w2), count in self.bigram_freq.most_common():
            if count < self.min_freq:
                continue
            
            p_bigram = count / self.total_bigrams
            p_w1 = self.word_freq[w1] / self.total_words
            p_w2 = self.word_freq[w2] / self.total_words
            
            if p_w1 > 0 and p_w2 > 0:
                pmi = math.log2(p_bigram / (p_w1 * p_w2))
                
                if pmi >= self.min_pmi:
                    collocations.append({
                        'bigram': f"{w1} {w2}",
                        'word1': w1,
                        'word2': w2,
                        'frequency': count,
                        'pmi': round(pmi, 2)
                    })
        
        # Sort by frequency (highest first)
        collocations.sort(key=lambda x: x['frequency'], reverse=True)
        return collocations


# ============================================================================
# LEXICON MANAGEMENT
# ============================================================================

@dataclass
class LexiconEntry:
    """Dictionary entry"""
    headword: str
    pos: str
    stem1: Optional[str] = None
    stem2: Optional[str] = None
    gloss_en: str = ""
    gloss_my: str = ""
    definition: str = ""
    examples: List[str] = field(default_factory=list)
    frequency: int = 0
    semantic_field: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


class KchoLexicon:
    """Lexicon database manager"""
    
    def __init__(self, db_path: str = "kcho_lexicon.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                headword TEXT UNIQUE NOT NULL,
                pos TEXT,
                stem1 TEXT,
                stem2 TEXT,
                gloss_en TEXT,
                gloss_my TEXT,
                definition TEXT,
                semantic_field TEXT,
                frequency INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id INTEGER,
                text TEXT,
                gloss TEXT,
                translation TEXT,
                FOREIGN KEY (entry_id) REFERENCES entries(id)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_headword ON entries(headword)')
        
        self.conn.commit()
    
    def add_entry(self, entry: LexiconEntry) -> int:
        """Add or update lexicon entry"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO entries 
            (headword, pos, stem1, stem2, gloss_en, gloss_my, definition, semantic_field, frequency)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry.headword, entry.pos, entry.stem1, entry.stem2,
            entry.gloss_en, entry.gloss_my, entry.definition,
            entry.semantic_field, entry.frequency
        ))
        
        entry_id = cursor.lastrowid
        
        for example in entry.examples:
            cursor.execute('''
                INSERT INTO examples (entry_id, text) VALUES (?, ?)
            ''', (entry_id, example))
        
        self.conn.commit()
        return entry_id
    
    def get_entry(self, headword: str) -> Optional[LexiconEntry]:
        """Retrieve entry by headword"""
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT * FROM entries WHERE headword = ?', (headword,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        cursor.execute('SELECT text FROM examples WHERE entry_id = ?', (row[0],))
        examples = [r[0] for r in cursor.fetchall()]
        
        return LexiconEntry(
            headword=row[1], pos=row[2], stem1=row[3], stem2=row[4],
            gloss_en=row[5], gloss_my=row[6], definition=row[7],
            semantic_field=row[8], frequency=row[9], examples=examples
        )
    
    def update_frequency(self, headword: str, increment: int = 1):
        """Update word frequency"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE entries SET frequency = frequency + ? WHERE headword = ?
        ''', (increment, headword))
        self.conn.commit()
    
    def export_json(self, output_path: str, sort_by_frequency: bool = True):
        """Export lexicon to JSON, optionally sorted by frequency"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM entries')
        
        entries = []
        for row in cursor.fetchall():
            cursor.execute('SELECT text FROM examples WHERE entry_id = ?', (row[0],))
            examples = [r[0] for r in cursor.fetchall()]
            
            entries.append({
                'headword': row[1], 'pos': row[2], 'stem1': row[3], 'stem2': row[4],
                'gloss_en': row[5], 'gloss_my': row[6], 'definition': row[7],
                'semantic_field': row[8], 'frequency': row[9], 'examples': examples
            })
        
        # CRITICAL FIX: Sort by frequency (highest first)
        if sort_by_frequency:
            entries.sort(key=lambda x: x.get('frequency', 0), reverse=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
    
    def close(self):
        self.conn.close()


# ============================================================================
# CORPUS MANAGEMENT
# ============================================================================

class KchoCorpus:
    """
    Unified corpus management combining building, annotation, and statistics.
    Optimized for ML training data preparation.
    """
    
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
    
    def add_sentence(self, text: str, translation: str = None,
                    metadata: Dict = None, validate: bool = True) -> Optional[Sentence]:
        """Add sentence to corpus with analysis"""
        
        if validate:
            is_kcho, confidence, _ = self.validator.is_kcho_text(text)
            if not is_kcho:
                logger.warning(f"Text rejected (confidence: {confidence:.2f}): {text[:50]}...")
                return None
        
        sentence = self.morph.analyze_sentence(text)
        sentence.translation = translation
        sentence.syntax = self.syntax.analyze_syntax(sentence)
        
        if metadata:
            sentence.metadata.update(metadata)
        
        # Update lexicon frequencies
        if self.lexicon:
            for token in sentence.tokens:
                self.lexicon.update_frequency(token.lemma)
        
        self.sentences.append(sentence)
        logger.info(f"Added sentence: {text}")
        
        return sentence
    
    def add_parallel_text(self, kcho_file: str, translation_file: str):
        """Add parallel corpus"""
        with open(kcho_file, 'r', encoding='utf-8') as f:
            kcho_lines = f.readlines()
        
        with open(translation_file, 'r', encoding='utf-8') as f:
            trans_lines = f.readlines()
        
        for kcho, trans in zip(kcho_lines, trans_lines):
            kcho = kcho.strip()
            trans = trans.strip()
            if kcho and trans:
                self.add_sentence(kcho, trans)
    
    def get_statistics(self) -> Dict:
        """Comprehensive corpus statistics"""
        total_sentences = len(self.sentences)
        total_tokens = sum(len(s.tokens) for s in self.sentences)
        
        pos_counts = Counter()
        for sentence in self.sentences:
            for token in sentence.tokens:
                pos_counts[token.pos.value] += 1
        
        vocabulary = set()
        for sentence in self.sentences:
            for token in sentence.tokens:
                vocabulary.add(token.lemma)
        
        avg_length = total_tokens / total_sentences if total_sentences > 0 else 0
        
        return {
            'total_sentences': total_sentences,
            'total_tokens': total_tokens,
            'vocabulary_size': len(vocabulary),
            'avg_sentence_length': round(avg_length, 2),
            'pos_distribution': dict(pos_counts),
            'unique_lemmas': len(vocabulary)
        }
    
    def export_json(self, output_path: str):
        """Export corpus to JSON"""
        data = [s.to_dict() for s in self.sentences]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def export_conllu(self, output_path: str):
        """Export to CoNLL-U format for NLP tools"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for sent_id, sentence in enumerate(self.sentences, 1):
                f.write(f"# sent_id = {sent_id}\n")
                f.write(f"# text = {sentence.text}\n")
                if sentence.translation:
                    f.write(f"# translation = {sentence.translation}\n")
                f.write(f"# gloss = {sentence.gloss}\n")
                
                for tok_id, token in enumerate(sentence.tokens, 1):
                    morphs = '|'.join([f"{m.type}={m.form}" for m in token.morphemes])
                    f.write(f"{tok_id}\t{token.surface}\t{token.lemma}\t{token.pos.value}\t_\t{morphs}\t_\t_\t_\t_\n")
                
                f.write("\n")
    
    def export_parallel_corpus(self, output_path: str):
        """Export parallel corpus for translation models"""
        with open(output_path, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'target', 'gloss', 'tokens', 'pos_tags'])
            
            for sentence in self.sentences:
                if sentence.translation:
                    tokens = ' '.join([t.surface for t in sentence.tokens])
                    pos = ' '.join([t.pos.value for t in sentence.tokens])
                    writer.writerow([
                        sentence.text,
                        sentence.translation,
                        sentence.gloss,
                        tokens,
                        pos
                    ])

    def add_sentences_batch(self, texts: List[str], translations: List[str] = None, batch_size: int = 100) -> Tuple[int, int]:
        """
        Add multiple sentences efficiently.
        
        Returns:
            (processed_count, error_count)
        """
        if translations is None:
            translations = [None] * len(texts)
        
        processed = 0
        errors = 0
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_trans = translations[i:i+batch_size]
            
            for text, trans in zip(batch_texts, batch_trans):
                try:
                    if self.add_sentence(text, trans, validate=True):
                        processed += 1
                    else:
                        errors += 1
                except:
                    errors += 1
        
        return processed, errors
    
    def analyze_pos_patterns(self) -> Dict[str, Dict[str, int]]:
        """
        Analyze POS patterns using defaultdict for efficient pattern grouping.
        
        This method identifies common part-of-speech sequences in K'Cho text,
        which is crucial for understanding syntactic patterns in the language.
        
        Returns:
            Dictionary mapping pattern lengths to pattern frequencies
            
        Example:
            {
                '2': {'N-V': 45, 'V-N': 32, 'ADJ-N': 28, ...},
                '3': {'N-V-N': 15, 'V-N-V': 12, ...},
                '4': {'N-V-N-V': 5, ...}
            }
        """
        logger.info("Analyzing POS patterns in corpus")
        
        # Use defaultdict for automatic pattern grouping
        pos_patterns = defaultdict(lambda: defaultdict(int))
        
        for sentence in self.sentences:
            if not sentence.tokens:
                continue
                
            # Extract POS sequence
            pos_sequence = tuple(token.pos.value for token in sentence.tokens)
            
            # Group by pattern length
            pattern_length = str(len(pos_sequence))
            pos_patterns[pattern_length][pos_sequence] += 1
        
        # Convert to regular nested dict
        result = {}
        for length, patterns in pos_patterns.items():
            result[length] = dict(patterns)
        
        logger.info(f"Analyzed POS patterns for {len(result)} different lengths")
        return result
    
    def build_word_cooccurrence_matrix(self, window_size: int = 5) -> Dict[str, Dict[str, int]]:
        """
        Build word co-occurrence matrix using nested defaultdict for efficient counting.
        
        This method creates a comprehensive co-occurrence matrix showing how often
        words appear together within a specified window size.
        
        Args:
            window_size: Size of co-occurrence window
            
        Returns:
            Nested dictionary: word1 -> word2 -> co-occurrence_count
            
        Example:
            {
                'kcho': {'language': 15, 'people': 8, 'culture': 3, ...},
                'language': {'kcho': 15, 'speak': 12, 'learn': 5, ...}
            }
        """
        logger.info(f"Building word co-occurrence matrix with window size {window_size}")
        
        # Use nested defaultdict for automatic dictionary creation
        cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for sentence in self.sentences:
            tokens = [token.lemma.lower() for token in sentence.tokens]
            
            for i, word1 in enumerate(tokens):
                # Get co-occurring words within window
                start = max(0, i - window_size)
                end = min(len(tokens), i + window_size + 1)
                
                for j in range(start, end):
                    if i != j:  # Don't count word with itself
                        word2 = tokens[j]
                        cooccurrence[word1][word2] += 1
        
        # Convert to regular nested dict
        result = {}
        for word1, cooccur_dict in cooccurrence.items():
            result[word1] = dict(cooccur_dict)
        
        logger.info(f"Built co-occurrence matrix for {len(result)} unique words")
        return result
    
    def extract_morphological_patterns(self) -> Dict[str, Dict[str, int]]:
        """
        Extract morphological patterns using defaultdict for pattern grouping.
        
        This method identifies common morphological patterns in K'Cho text,
        such as prefixes, suffixes, and morphological combinations.
        
        Returns:
            Dictionary mapping pattern types to pattern frequencies
            
        Example:
            {
                'prefixes': {'ka-': 25, 'ma-': 18, 'ta-': 12, ...},
                'suffixes': {'-ng': 20, '-k': 15, '-t': 8, ...},
                'combinations': {'ka-...-ng': 5, 'ma-...-k': 3, ...}
            }
        """
        logger.info("Extracting morphological patterns from corpus")
        
        # Use defaultdict for automatic pattern grouping
        morph_patterns = defaultdict(lambda: defaultdict(int))
        
        for sentence in self.sentences:
            for token in sentence.tokens:
                # Extract prefixes
                if token.surface.startswith(('ka-', 'ma-', 'ta-', 'pa-', 'sa-')):
                    prefix = token.surface[:3]
                    morph_patterns['prefixes'][prefix] += 1
                
                # Extract suffixes
                if token.surface.endswith(('-ng', '-k', '-t', '-m', '-n')):
                    suffix = token.surface[-3:] if len(token.surface) > 3 else token.surface[-2:]
                    morph_patterns['suffixes'][suffix] += 1
                
                # Extract prefix-suffix combinations
                if (token.surface.startswith(('ka-', 'ma-', 'ta-', 'pa-')) and 
                    token.surface.endswith(('-ng', '-k', '-t', '-m'))):
                    prefix = token.surface[:3]
                    suffix = token.surface[-3:] if len(token.surface) > 3 else token.surface[-2:]
                    combination = f"{prefix}...{suffix}"
                    morph_patterns['combinations'][combination] += 1
        
        # Convert to regular nested dict
        result = {}
        for pattern_type, patterns in morph_patterns.items():
            result[pattern_type] = dict(patterns)
        
        logger.info(f"Extracted {len(result)} morphological pattern types")
        return result
    
    def analyze_sentence_structure_patterns(self) -> Dict[str, Dict[str, int]]:
        """
        Analyze sentence structure patterns using defaultdict for grouping.
        
        This method identifies common sentence structures in K'Cho text,
        which helps understand the syntactic organization of the language.
        
        Returns:
            Dictionary mapping structure types to pattern frequencies
            
        Example:
            {
                'length_distribution': {'short': 45, 'medium': 32, 'long': 15},
                'pos_sequences': {'N-V': 25, 'V-N': 18, 'N-V-N': 12},
                'clause_types': {'simple': 40, 'complex': 25, 'compound': 10}
            }
        """
        logger.info("Analyzing sentence structure patterns")
        
        # Use defaultdict for automatic pattern grouping
        structure_patterns = defaultdict(lambda: defaultdict(int))
        
        for sentence in self.sentences:
            if not sentence.tokens:
                continue
            
            # Analyze sentence length
            length = len(sentence.tokens)
            if length <= 5:
                structure_patterns['length_distribution']['short'] += 1
            elif length <= 15:
                structure_patterns['length_distribution']['medium'] += 1
            else:
                structure_patterns['length_distribution']['long'] += 1
            
            # Analyze clause complexity (simplified)
            verb_count = sum(1 for token in sentence.tokens if token.pos == POS.VERB)
            if verb_count == 1:
                structure_patterns['clause_types']['simple'] += 1
            elif verb_count == 2:
                structure_patterns['clause_types']['complex'] += 1
            else:
                structure_patterns['clause_types']['compound'] += 1
        
        # Convert to regular nested dict
        result = {}
        for structure_type, patterns in structure_patterns.items():
            result[structure_type] = dict(patterns)
        
        logger.info(f"Analyzed {len(result)} sentence structure pattern types")
        return result
        
# ============================================================================
# UNIFIED SYSTEM
# ============================================================================

class KchoSystem:
    """
    Main unified system integrating all components.
    Production-ready interface for K'Cho language processing.
    """
    
    def __init__(self, project_dir: str = "./kcho_project"):
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.knowledge = KchoKnowledge()
        self.tokenizer = KchoTokenizer()
        self.validator = KchoValidator()
        self.morph = KchoMorphologyAnalyzer()
        self.syntax = KchoSyntaxAnalyzer()
        self.lexicon = KchoLexicon(str(self.project_dir / "lexicon.db"))
        self.corpus = KchoCorpus()
        self.corpus.set_lexicon(self.lexicon)

        """Initialize K'Cho system with default components."""
        self.normalizer = KChoNormalizer()
        self.collocation_extractor = CollocationExtractor(normalizer=self.normalizer)
        
        # Create directories
        self.dirs = {
            'corpus': self.project_dir / 'corpus',
            'exports': self.project_dir / 'exports',
            'reports': self.project_dir / 'reports',
            'models': self.project_dir / 'models',
        }
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        logger.info(f"K'Cho System initialized: {project_dir}")
        self._populate_base_lexicon()
    
    def _populate_base_lexicon(self):
        """Populate lexicon with base vocabulary"""
        for stem1, info in self.knowledge.VERB_STEMS.items():
            entry = LexiconEntry(
                headword=stem1,
                pos='V',
                stem1=stem1,
                stem2=info['stem2'],
                gloss_en=info['gloss'],
                definition=f"Verb: {info['gloss']}"
            )
            self.lexicon.add_entry(entry)
        
        for word, info in self.knowledge.POSTPOSITIONS.items():
            entry = LexiconEntry(
                headword=word,
                pos='P',
                gloss_en=info['function'],
                definition=f"Postposition: {info['function']}"
            )
            self.lexicon.add_entry(entry)
    
    # High-level API
    def analyze(self, text: str) -> Sentence:
        """Quick analysis of K'Cho text"""
        return self.morph.analyze_sentence(text)
    
    def validate(self, text: str) -> Tuple[bool, float, Dict]:
        """Validate if text is K'Cho"""
        return self.validator.is_kcho_text(text)
    
    def add_to_corpus(self, text: str, translation: str = None, **kwargs):
        """Add text to corpus"""
        return self.corpus.add_sentence(text, translation, **kwargs)
    
    def process_parallel_corpus(self, kcho_file: str, translation_file: str):
        """Process parallel corpus files"""
        self.corpus.add_parallel_text(kcho_file, translation_file)
    
    def corpus_stats(self) -> Dict:
        """Get corpus statistics"""
        return self.corpus.get_statistics()
    
    def validate_export_readiness(self) -> Tuple[bool, List[str]]:
        """Validate system has sufficient data for ML export"""
        issues = []
        
        # Check minimum corpus size
        if len(self.corpus.sentences) < 100:
            issues.append(f"Corpus too small: {len(self.corpus.sentences)} sentences (min: 100)")
        
        # Check vocabulary coverage
        stats = self.corpus_stats()
        if stats['vocabulary_size'] < 500:
            issues.append(f"Vocabulary too small: {stats['vocabulary_size']} words (min: 500)")
        
        # Check translation coverage
        has_translation = sum(1 for s in self.corpus.sentences if s.translation)
        if has_translation < len(self.corpus.sentences) * 0.5:
            pct = has_translation / len(self.corpus.sentences) * 100 if self.corpus.sentences else 0
            issues.append(f"Low translation coverage: {pct:.1f}% (min: 50%)")
        
        return len(issues) == 0, issues
   # ADD NEW METHOD: Extract collocations from corpus
    def extract_collocations(self,
                            corpus: List[str],
                            window_size: int = 5,
                            min_freq: int = 5,
                            measures: List[str] = None) -> Dict:
        """
        Extract collocations from corpus.
        
        NEW: Now delegates to collocation module with enhanced functionality.
        
        Args:
            corpus: List of K'Cho sentences
            window_size: Co-occurrence window size
            min_freq: Minimum frequency threshold
            measures: List of measure names (pmi, npmi, tscore, dice, log_likelihood)
        
        Returns:
            Dictionary mapping measure names to CollocationResult lists
        """
        # Convert measure names to enum
        measure_enums = []
        if measures:
            for m in measures:
                try:
                    measure_enums.append(AssociationMeasure(m.lower()))
                except ValueError:
                    logger.warning(f"Unknown measure: {m}, skipping")
        else:
            measure_enums = [AssociationMeasure.PMI, AssociationMeasure.TSCORE]
        
        # Configure and run extractor
        self.collocation_extractor.window_size = window_size
        self.collocation_extractor.min_freq = min_freq
        self.collocation_extractor.measures = measure_enums
        
        return self.collocation_extractor.extract(corpus)
    def export_collocations(self,
                           results: Dict,
                           output_path: str,
                           format: str = 'csv',
                           top_k: int = None) -> None:
        """
        Export collocation results to file.
        
        NEW: Convenience wrapper for export module.
        
        Args:
            results: Collocation results from extract_collocations()
            output_path: Output file path
            format: Output format (csv, json, text)
            top_k: Limit to top K results per measure
        """
        if format == 'csv':
            to_csv(results, output_path, top_k=top_k)
        elif format == 'json':
            to_json(results, output_path, top_k=top_k)
        elif format == 'text':
            to_text(results, output_path, top_k=top_k)
        else:
            raise ValueError(f"Unknown format: {format}")
    def evaluate_collocations(self,
                              predicted: List[CollocationResult],
                              gold_standard_file: str) -> Dict:
        """
        Evaluate collocation extraction against gold standard.
        
        NEW: Convenience wrapper for evaluation module.
        
        Args:
            predicted: Predicted collocations
            gold_standard_file: Path to gold standard file
        
        Returns:
            Dict with evaluation metrics
        """
        gold_set = load_gold_standard(gold_standard_file)
        return evaluate_ranking(predicted, gold_set)
    
    # ADD NEW METHOD: Normalize text
    def normalize(self, text: str, **kwargs) -> str:
        """
        Normalize K'Cho text.
        
        Args:
            text: Input text
            **kwargs: Override normalizer settings
            
        Returns:
            Normalized text
        """
        if kwargs:
            normalizer = KChoNormalizer(**kwargs)
            return normalizer.normalize_text(text)
        return self.normalizer.normalize_text(text)
    
    # ADD NEW METHOD: Tokenize
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize K'Cho text respecting morphology.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return self.normalizer.tokenize(text)
    def export_training_data(self, force: bool = False):
        """Export all data for ML training with validation"""
        
        # VALIDATE FIRST
        is_ready, issues = self.validate_export_readiness()
        
        if not is_ready and not force:
            print("\n⚠️  Export validation failed:")
            for issue in issues:
                print(f"   • {issue}")
            print("\nFix issues or use force=True to export anyway.")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export corpus in multiple formats
        self.corpus.export_json(str(self.dirs['exports'] / f'corpus_{timestamp}.json'))
        self.corpus.export_conllu(str(self.dirs['exports'] / f'corpus_{timestamp}.conllu'))
        self.corpus.export_parallel_corpus(str(self.dirs['exports'] / f'parallel_{timestamp}.csv'))
        
        # Export lexicon (SORTED!)
        self.lexicon.export_json(
            str(self.dirs['exports'] / f'lexicon_{timestamp}.json'),
            sort_by_frequency=True  # ← FIX APPLIED
        )
        
        # Extract collocations
        colloc_extractor = KchoCollocationExtractor(min_freq=5, min_pmi=3.0)
        collocations = colloc_extractor.extract_from_corpus(self.corpus.sentences)
        
        colloc_path = self.dirs['exports'] / f'collocations_{timestamp}.json'
        with open(colloc_path, 'w', encoding='utf-8') as f:
            json.dump(collocations, f, ensure_ascii=False, indent=2)
        
        # Generate report
        stats = self.corpus_stats()
        report = {
            'timestamp': timestamp,
            'statistics': stats,
            'validation': {'passed': is_ready, 'issues': issues},
            'collocations_count': len(collocations)
        }
        
        with open(self.dirs['reports'] / f'report_{timestamp}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Exported training data: {timestamp}")
        logger.info(f"  • Corpus: {stats['total_sentences']} sentences")
        logger.info(f"  • Lexicon: sorted by frequency ✓")
        logger.info(f"  • Collocations: {len(collocations)} bigrams ✓")
        
        return report
        
    def close(self):
        """Clean up resources"""
        self.lexicon.close()
# Module-level convenience API
def extract_collocations(corpus: List[str], **kwargs) -> Dict:
    """Module-level convenience function for collocation extraction."""
    system = KChoSystem()
    return system.extract_collocations(corpus, **kwargs)
