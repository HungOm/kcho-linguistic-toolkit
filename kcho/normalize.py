# FILE: normalize.py (renamed from normaliz.py - fix typo)
# PURPOSE: K'Cho text normalization with enhanced lemmatization
# CHANGES: Enhanced lemmatization rules based on Mang & Bedell research

"""
K'Cho text normalization module.

Based on:
- Nolan (2002) on K'Cho orthography
- Mang & Bedell (2006-2012) on morphology
- Mang (2006) MA Thesis on verb stem alternation
"""

import re
import unicodedata
from typing import List, Dict, Optional, Tuple

class KChoNormalizer:
    """Normalize K'Cho text while preserving linguistic features."""
    
    # K'Cho specific patterns (from research)
    POSTPOSITIONS = {'noh', 'ah', 'am', 'on', 'ung', 'tu', 'bà', 'bâ'}
    PARTICLES = {'ci', 'khai', 'ne', 'cuh', 'gui', 'goi', 'ni', 'te'}
    PREFIXED_GLOTTALS = r"[km]'|ng'"
    
    # Enhanced suffix patterns (Mang 2006)
    VERB_SUFFIXES = {'-nák', '-na', '-nak', '(k)', 'k', 'ci', 'khai'}
    AGREEMENT_PREFIXES = {'ka-', 'a-', 'ani-', 'ami-', 'an-'}
    
    def __init__(self, 
                 preserve_tones: bool = True,
                 preserve_length: bool = True,
                 lowercase: bool = False):
        """
        Initialize normalizer.
        
        Args:
            preserve_tones: Keep tone marks (default True for research quality)
            preserve_length: Keep vowel length distinctions (critical for K'Cho)
            lowercase: Convert to lowercase (use cautiously)
        """
        self.preserve_tones = preserve_tones
        self.preserve_length = preserve_length
        self.lowercase = lowercase
        
    def normalize_text(self, text: str) -> str:
        """
        Normalize K'Cho text.
        
        Process:
        1. Unicode normalization (NFC)
        2. Whitespace cleanup
        3. Optional tone/case transformations
        
        PRESERVES: Linguistic distinctions critical for K'Cho morphology
        """
        if not text:
            return ""
        
        # Unicode normalize (NFC: composed characters)
        text = unicodedata.normalize('NFC', text)
        
        # Whitespace cleanup
        text = ' '.join(text.split())
        
        # Optional lowercase (use with caution - may affect proper nouns)
        if self.lowercase:
            text = text.lower()
        
        # Optional tone removal (NOT recommended for research)
        if not self.preserve_tones:
            # Remove combining diacritics
            text = ''.join(c for c in unicodedata.normalize('NFD', text)
                          if unicodedata.category(c) != 'Mn')
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize K'Cho text respecting morphology.
        
        K'Cho morphology considerations:
        - Verb stem alternation (stem I vs stem II)
        - Agreement prefixes (ka-, a-, ani-, etc.)
        - Applicative suffix -na/-nák
        - Postpositions attach to NPs
        
        Strategy: White-space + punctuation tokenization with
        glottal preservation and suffix awareness.
        """
        text = self.normalize_text(text)
        
        # Split on whitespace and punctuation, preserve glottals
        # Pattern: word characters + glottal stops + hyphens
        pattern = r"[\w''-]+"
        tokens = re.findall(pattern, text, re.UNICODE)
        
        return [t for t in tokens if t]
    
    def lemmatize_simple(self, token: str) -> str:
        """
        Simple lemmatization for K'Cho.
        
        ENHANCEMENT: Added more suffix/prefix patterns from Mang (2006)
        
        WARNING: This is rule-based approximation. K'Cho has complex
        morphology (stem alternation, agreement, applicatives) that
        requires full morphological analysis for proper lemmatization.
        
        Current implementation:
        - Remove common suffixes (-na, -nák, -ci, -khai, etc.)
        - Remove agreement prefixes (ka-, a-, ani-, etc.)
        - Preserve stem
        
        For production: Consider training a morphological analyzer
        or using annotated lexicon (Mang 2006 thesis).
        """
        token = token.lower().strip()
        
        # Remove verb suffixes (tense/aspect/applicative markers)
        for suffix in ['-nák', '-na', '-nak', '(k)', 'k', 'ci', 'khai', 'ne', 'te']:
            if token.endswith(suffix):
                token = token[:-len(suffix)]
                break
        
        # Remove agreement prefixes (person markers)
        for prefix in ['ka-', 'a-', 'ani-', 'ami-', 'an-']:
            if token.startswith(prefix):
                token = token[len(prefix):]
                break
        
        return token if token else ""
    
    def get_stem_variants(self, verb: str) -> Dict[str, any]:
        """
        Return possible stem I / stem II variants.
        
        ENHANCEMENT: Added basic alternation rules from Nolan (2003) and Mang (2006)
        
        Based on verb stem alternation patterns:
        - Stem I: used with tense particles (ci, khai)
        - Stem II: used in relative clauses, applicatives
        
        Common patterns:
        - Tone change (low -> high)
        - Vowel lengthening
        - Final consonant alternation (t <-> h)
        
        Example: lùum (stem I) <-> luum (stem II) 'play'
                 that (stem I) <-> thah (stem II) 'beat'
        
        NOTE: This requires a full morphological lexicon for
        accurate implementation. Current version provides basic patterns.
        """
        verb = verb.lower().strip()
        
        # Basic alternation rules (approximation)
        stem_i = verb
        stem_ii = verb
        
        # Pattern 1: Final t -> h alternation
        if verb.endswith('t'):
            stem_ii = verb[:-1] + 'h'
        elif verb.endswith('h'):
            stem_i = verb[:-1] + 't'
        
        # Pattern 2: Vowel lengthening (heuristic: double last vowel)
        # This is a simplification; real alternation is lexically conditioned
        vowels = 'aeiouàèìòùáéíóú'
        for i in range(len(verb)-1, -1, -1):
            if verb[i] in vowels:
                if i+1 < len(verb) and verb[i] != verb[i+1]:
                    stem_ii = verb[:i+1] + verb[i] + verb[i+1:]
                break
        
        return {
            'stem_i': stem_i,
            'stem_ii': stem_ii,
            'confidence': 0.3  # Low confidence - needs lexicon
        }


# Convenience functions for module-level API
_default_normalizer = KChoNormalizer()

def normalize_text(text: str, **kwargs) -> str:
    """Module-level convenience function for text normalization."""
    if kwargs:
        normalizer = KChoNormalizer(**kwargs)
        return normalizer.normalize_text(text)
    return _default_normalizer.normalize_text(text)

def tokenize(text: str, **kwargs) -> List[str]:
    """Module-level convenience function for tokenization."""
    if kwargs:
        normalizer = KChoNormalizer(**kwargs)
        return normalizer.tokenize(text)
    return _default_normalizer.tokenize(text)

def lemmatize(token: str, **kwargs) -> str:
    """Module-level convenience function for lemmatization."""
    if kwargs:
        normalizer = KChoNormalizer(**kwargs)
        return normalizer.lemmatize_simple(token)
    return _default_normalizer.lemmatize_simple(token)