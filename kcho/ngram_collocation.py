#!/usr/bin/env python3
"""
Enhanced Collocation Extractor with N-gram Support

This module extends the basic collocation extractor to support trigrams and longer n-grams.
It includes specialized methods for extracting and analyzing multi-word expressions.
"""

import math
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Callable
from enum import Enum

from .normalize import KChoNormalizer
from .collocation import AssociationMeasure, LinguisticPattern, CollocationResult

logger = logging.getLogger(__name__)

@dataclass
class NGramCollocationResult(CollocationResult):
    """Enhanced collocation result for n-grams."""
    ngram_size: int = 2  # Size of the n-gram (2=bigram, 3=trigram, etc.)
    sub_ngrams: List[Tuple[str, ...]] = field(default_factory=list)  # Sub-ngrams for analysis
    linguistic_pattern: Optional[LinguisticPattern] = None
    pattern_confidence: float = 0.0
    pos_tags: Optional[List[str]] = None
    context_examples: List[str] = field(default_factory=list)

class NGramCollocationExtractor:
    """
    Enhanced collocation extractor supporting n-grams of any size.
    
    Features:
    - Extract bigrams, trigrams, and longer n-grams
    - Multiple association measures
    - Linguistic pattern classification
    - Context example extraction
    """
    
    def __init__(self,
                 normalizer: Optional[KChoNormalizer] = None,
                 window_size: int = 5,
                 min_freq: int = 5,
                 max_ngram_size: int = 4,
                 measures: List[AssociationMeasure] = None,
                 pos_tagger: Optional[Callable] = None):
        """
        Initialize n-gram collocation extractor.
        
        Args:
            normalizer: Text normalizer (uses default if None)
            window_size: Co-occurrence window size
            min_freq: Minimum frequency threshold
            max_ngram_size: Maximum n-gram size to extract (2=bigrams, 3=trigrams, etc.)
            measures: List of association measures to compute
            pos_tagger: Optional POS tagger function
        """
        self.normalizer = normalizer or KChoNormalizer()
        self.window_size = window_size
        self.min_freq = min_freq
        self.max_ngram_size = max_ngram_size
        self.measures = measures or [AssociationMeasure.PMI, AssociationMeasure.TSCORE]
        self.pos_tagger = pos_tagger
        
        # Frequency counters for different n-gram sizes
        self.ngram_freqs: Dict[int, Counter] = {
            n: Counter() for n in range(1, max_ngram_size + 1)
        }
        self.total_ngrams: Dict[int, int] = {
            n: 0 for n in range(1, max_ngram_size + 1)
        }
        
        # Context storage for examples
        self.ngram_contexts: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    
    def extract_ngrams(self, corpus: List[str]) -> Dict[int, Dict[AssociationMeasure, List[NGramCollocationResult]]]:
        """
        Extract n-grams of all sizes from corpus.
        
        Args:
            corpus: List of K'Cho sentences
            
        Returns:
            Dictionary mapping n-gram size to results by measure
        """
        logger.info(f"Extracting n-grams (1-{self.max_ngram_size}) from {len(corpus)} sentences")
        
        # Tokenize corpus
        tokenized_corpus = [self.normalizer.tokenize(sent) for sent in corpus]
        
        # Compute frequencies for all n-gram sizes
        self._compute_ngram_frequencies(tokenized_corpus)
        
        # Extract results for each n-gram size
        results = {}
        for n in range(2, self.max_ngram_size + 1):  # Skip unigrams
            results[n] = self._extract_ngrams_of_size(n)
        
        total_collocations = sum(
            len(measure_results)
            for ngram_results in results.values()
            for measure_results in ngram_results.values()
        )
        logger.info(f"Extracted {total_collocations} n-gram collocations")
        
        return results
    
    def _compute_ngram_frequencies(self, tokenized_corpus: List[List[str]]):
        """Compute frequencies for all n-gram sizes."""
        # Clear previous counts
        for counter in self.ngram_freqs.values():
            counter.clear()
        for key in self.ngram_contexts:
            self.ngram_contexts[key].clear()
        
        for tokens in tokenized_corpus:
            if len(tokens) < 2:
                continue
                
            # Extract n-grams of all sizes
            for n in range(1, min(len(tokens) + 1, self.max_ngram_size + 1)):
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i:i+n])
                    self.ngram_freqs[n][ngram] += 1
                    
                    # Store context for examples
                    if n >= 2:  # Only store context for multi-word n-grams
                        context = " ".join(tokens[max(0, i-2):i+n+2])
                        self.ngram_contexts[ngram].append(context)
        
        # Update totals
        for n in range(1, self.max_ngram_size + 1):
            self.total_ngrams[n] = sum(self.ngram_freqs[n].values())
        
        logger.debug(f"N-gram counts: {dict(self.total_ngrams)}")
    
    def _extract_ngrams_of_size(self, n: int) -> Dict[AssociationMeasure, List[NGramCollocationResult]]:
        """Extract collocations for n-grams of specific size."""
        results = {}
        
        for measure in self.measures:
            measure_results = []
            
            for ngram, freq in self.ngram_freqs[n].items():
                if freq < self.min_freq:
                    continue
                
                # Compute association score
                score = self._compute_ngram_score(ngram, freq, measure)
                
                if not math.isfinite(score):
                    continue
                
                # Create result
                result = NGramCollocationResult(
                    words=ngram,
                    score=score,
                    measure=measure,
                    frequency=freq,
                    ngram_size=n,
                    positions=[],
                    context_examples=self.ngram_contexts[ngram][:3]  # Top 3 examples
                )
                
                # Classify linguistic pattern
                pattern, confidence = self._classify_ngram_pattern(ngram)
                result.linguistic_pattern = pattern
                result.pattern_confidence = confidence
                
                # Extract POS tags if available
                if self.pos_tagger:
                    result.pos_tags = [self.pos_tagger(word) for word in ngram]
                
                measure_results.append(result)
            
            # Sort by frequency (descending) - largest at top
            measure_results.sort(key=lambda x: x.frequency, reverse=True)
            results[measure] = measure_results
        
        return results
    
    def _compute_ngram_score(self, ngram: Tuple[str, ...], freq: int, measure: AssociationMeasure) -> float:
        """Compute association score for n-gram."""
        n = len(ngram)
        
        if measure == AssociationMeasure.PMI:
            return self._pmi_ngram(ngram, freq)
        elif measure == AssociationMeasure.NPMI:
            return self._npmi_ngram(ngram, freq)
        elif measure == AssociationMeasure.TSCORE:
            return self._tscore_ngram(ngram, freq)
        elif measure == AssociationMeasure.DICE:
            return self._dice_ngram(ngram, freq)
        elif measure == AssociationMeasure.LOG_LIKELIHOOD:
            return self._log_likelihood_ngram(ngram, freq)
        else:
            return 0.0
    
    def _pmi_ngram(self, ngram: Tuple[str, ...], freq: int) -> float:
        """Compute PMI for n-gram."""
        n = len(ngram)
        total = self.total_ngrams[n]
        
        # P(ngram)
        p_ngram = freq / total
        
        # P(w1) * P(w2) * ... * P(wn) - independence assumption
        p_independent = 1.0
        for word in ngram:
            word_freq = self.ngram_freqs[1][(word,)]
            p_independent *= word_freq / self.total_ngrams[1]
        
        if p_independent == 0:
            return 0.0
        
        return math.log2(p_ngram / p_independent)
    
    def _npmi_ngram(self, ngram: Tuple[str, ...], freq: int) -> float:
        """Compute normalized PMI for n-gram."""
        pmi = self._pmi_ngram(ngram, freq)
        n = len(ngram)
        total = self.total_ngrams[n]
        p_ngram = freq / total
        
        if p_ngram == 0:
            return 0.0
        
        return pmi / (-math.log2(p_ngram))
    
    def _tscore_ngram(self, ngram: Tuple[str, ...], freq: int) -> float:
        """Compute t-score for n-gram."""
        n = len(ngram)
        total = self.total_ngrams[n]
        
        # Expected frequency under independence
        expected = 1.0
        for word in ngram:
            word_freq = self.ngram_freqs[1][(word,)]
            expected *= word_freq / self.total_ngrams[1]
        expected *= total
        
        if expected == 0:
            return 0.0
        
        return (freq - expected) / math.sqrt(freq)
    
    def _dice_ngram(self, ngram: Tuple[str, ...], freq: int) -> float:
        """Compute Dice coefficient for n-gram."""
        n = len(ngram)
        
        # Sum of individual word frequencies
        sum_freqs = sum(self.ngram_freqs[1][(word,)] for word in ngram)
        
        if sum_freqs == 0:
            return 0.0
        
        return (n * freq) / sum_freqs
    
    def _log_likelihood_ngram(self, ngram: Tuple[str, ...], freq: int) -> float:
        """Compute log-likelihood ratio for n-gram."""
        n = len(ngram)
        total = self.total_ngrams[n]
        
        # Expected frequency under independence
        expected = 1.0
        for word in ngram:
            word_freq = self.ngram_freqs[1][(word,)]
            expected *= word_freq / self.total_ngrams[1]
        expected *= total
        
        if expected == 0:
            return 0.0
        
        # Log-likelihood ratio
        if freq > expected:
            return 2 * (freq * math.log(freq / expected) - (freq - expected))
        else:
            return 0.0
    
    def _classify_ngram_pattern(self, ngram: Tuple[str, ...]) -> Tuple[Optional[LinguisticPattern], float]:
        """Classify n-gram into linguistic pattern."""
        n = len(ngram)
        
        # Load gold standard patterns for classification
        gold_patterns = self._load_gold_standard_patterns()
        
        # Convert ngram to string for pattern matching
        ngram_str = " ".join(ngram)
        
        # Check against gold standard patterns
        for pattern_type, patterns in gold_patterns.items():
            if ngram_str in patterns:
                return pattern_type, 1.0
        
        # Heuristic classification based on n-gram size and content
        if n == 2:
            return self._classify_bigram_pattern(ngram)
        elif n == 3:
            return self._classify_trigram_pattern(ngram)
        elif n >= 4:
            return LinguisticPattern.MWE, 0.7  # Multi-word expression
        else:
            return None, 0.0
    
    def _classify_bigram_pattern(self, bigram: Tuple[str, str]) -> Tuple[Optional[LinguisticPattern], float]:
        """Classify bigram pattern."""
        w1, w2 = bigram
        
        # Verb + Particle patterns
        if self._is_verb_particle_pattern(w1, w2):
            return LinguisticPattern.VP, 0.9
        
        # Postposition + Noun patterns
        if self._is_postposition_noun_pattern(w1, w2):
            return LinguisticPattern.PP, 0.9
        
        # Agreement + Verb patterns
        if self._is_agreement_verb_pattern(w1, w2):
            return LinguisticPattern.AGR, 0.9
        
        # Complementizer patterns
        if self._is_complementizer_pattern(w1, w2):
            return LinguisticPattern.COMP, 0.9
        
        # Compound noun patterns
        if self._is_compound_noun_pattern(w1, w2):
            return LinguisticPattern.COMPOUND, 0.8
        
        # Discourse marker patterns
        if self._is_discourse_marker_pattern(w1, w2):
            return LinguisticPattern.DISC, 0.8
        
        # Default to lexical collocation
        return LinguisticPattern.LEX, 0.5
    
    def _classify_trigram_pattern(self, trigram: Tuple[str, str, str]) -> Tuple[Optional[LinguisticPattern], float]:
        """Classify trigram pattern."""
        w1, w2, w3 = trigram
        
        # Multi-word expressions (3+ words)
        if self._is_multi_word_expression(trigram):
            return LinguisticPattern.MWE, 0.8
        
        # Applicative constructions (verb-na + particle)
        if self._is_applicative_trigram(trigram):
            return LinguisticPattern.APP, 0.9
        
        # Auxiliary constructions
        if self._is_auxiliary_trigram(trigram):
            return LinguisticPattern.AUX, 0.8
        
        # Default to multi-word expression
        return LinguisticPattern.MWE, 0.6
    
    def _is_verb_particle_pattern(self, w1: str, w2: str) -> bool:
        """Check if bigram is verb + particle pattern."""
        # Common K'Cho particles
        particles = {'ci', 'khai', 'te', 'ne', 'ah', 'goi', 'gui'}
        return w2 in particles
    
    def _is_postposition_noun_pattern(self, w1: str, w2: str) -> bool:
        """Check if bigram is postposition + noun pattern."""
        # Common K'Cho postpositions
        postpositions = {'noh', 'am', 'ah', 'ung', 'cuh'}
        return w1 in postpositions
    
    def _is_agreement_verb_pattern(self, w1: str, w2: str) -> bool:
        """Check if bigram is agreement + verb pattern."""
        # Common K'Cho agreement markers
        agreement_markers = {'a', 'ka', 'ani', 'ami', 'ke', 'keu'}
        return w1 in agreement_markers
    
    def _is_complementizer_pattern(self, w1: str, w2: str) -> bool:
        """Check if bigram is complementizer pattern."""
        complementizers = {'ah', 'ci', 'khai'}
        return w2 in complementizers
    
    def _is_compound_noun_pattern(self, w1: str, w2: str) -> bool:
        """Check if bigram is compound noun pattern."""
        # Look for k'- prefix patterns
        return w1.startswith("k'") and not w2.startswith("k'")
    
    def _is_discourse_marker_pattern(self, w1: str, w2: str) -> bool:
        """Check if bigram is discourse marker pattern."""
        discourse_markers = {'cun', 'sin', 'te', 'ne'}
        return w1 in discourse_markers
    
    def _is_multi_word_expression(self, trigram: Tuple[str, str, str]) -> bool:
        """Check if trigram is multi-word expression."""
        # Common MWE patterns in K'Cho
        mwe_patterns = [
            ('pe', 'ci', 'ah'),  # give NF COMP
            ('lo', 'ci', 'ah'),  # come NF COMP
            ('noh', 'Yóng', 'am'),  # P Yong to
        ]
        return trigram in mwe_patterns
    
    def _is_applicative_trigram(self, trigram: Tuple[str, str, str]) -> bool:
        """Check if trigram is applicative construction."""
        w1, w2, w3 = trigram
        return w1.endswith('-na') and w3 in {'ci', 'khai', 'te'}
    
    def _is_auxiliary_trigram(self, trigram: Tuple[str, str, str]) -> bool:
        """Check if trigram is auxiliary construction."""
        w1, w2, w3 = trigram
        auxiliaries = {'lo', 'loo', 'na', 'nák'}
        return w2 in auxiliaries
    
    def _load_gold_standard_patterns(self) -> Dict[LinguisticPattern, Set[str]]:
        """Load gold standard patterns for classification."""
        patterns = {
            LinguisticPattern.VP: {
                'pe ci', 'lo ci', 'that ci', 'htui ci', 'hngu ci',
                'zòi ci', 'ni ci', 'pe khai', 'lo khai', 'sì ci'
            },
            LinguisticPattern.PP: {
                'noh Yóng', 'noh Om', 'am pàapai', 'noh k\'chàang',
                'noh k\'hngumí', 'noh ui', 'ah k\'chàang', 'am k\'am'
            },
            LinguisticPattern.APP: {
                'luum-na ci', 'thah-na ci', 'zèi-na ci', 'k\'chú-na ci',
                'ng\'mìng-na ci', 'khò-na ci', 'ka-na ci'
            },
            LinguisticPattern.AGR: {
                'a péit', 'ka hngu', 'a hnguh', 'ani thah-nák',
                'ami hnguh', 'ka zèi'
            },
            LinguisticPattern.AUX: {
                'lo lo', 'pe lo', 'that lo', 'luum loo-na'
            },
            LinguisticPattern.COMP: {
                'ci ah', 'khai ah', 'te ah', 'ne ah'
            },
            LinguisticPattern.MWE: {
                'pe ci ah', 'lo ci ah', 'noh Yóng am', 'Om noh Yóng',
                'am pàapai pe', 'a péit ah', 'ka hngu ci'
            },
            LinguisticPattern.COMPOUND: {
                'k\'am-k\'zòi', 'k\'chàang-k\'ní'
            },
            LinguisticPattern.LEX: {
                'ui htui', 'khò na', 'àihli ng\'dáng'
            },
            LinguisticPattern.DISC: {
                'cun ah', 'sin ah'
            }
        }
        return patterns
    
    def get_top_ngrams_by_size(self, results: Dict[int, Dict[AssociationMeasure, List[NGramCollocationResult]]], 
                              top_k: int = 10) -> Dict[int, List[NGramCollocationResult]]:
        """Get top n-grams by size across all measures."""
        top_by_size = {}
        
        for n, measure_results in results.items():
            all_ngrams = []
            for measure, ngrams in measure_results.items():
                all_ngrams.extend(ngrams)
            
            # Sort by score and take top k
            all_ngrams.sort(key=lambda x: x.score, reverse=True)
            top_by_size[n] = all_ngrams[:top_k]
        
        return top_by_size
    
    def print_ngram_summary(self, results: Dict[int, Dict[AssociationMeasure, List[NGramCollocationResult]]]):
        """Print summary of n-gram extraction results."""
        print("🔍 N-Gram Collocation Analysis Summary")
        print("=" * 50)
        
        for n, measure_results in results.items():
            total_ngrams = sum(len(ngrams) for ngrams in measure_results.values())
            print(f"\n📊 {n}-grams: {total_ngrams} collocations found")
            
            # Show top n-grams by measure
            for measure, ngrams in measure_results.items():
                if ngrams:
                    print(f"\n  {measure.value.upper()} (top 5):")
                    for i, ngram in enumerate(ngrams[:5], 1):
                        pattern = ngram.linguistic_pattern.value if ngram.linguistic_pattern else 'UNK'
                        print(f"    {i}. {' '.join(ngram.words)} (score: {ngram.score:.3f}, freq: {ngram.frequency}, pattern: {pattern})")
        
        # Pattern distribution
        print(f"\n📈 Linguistic Pattern Distribution:")
        pattern_counts = defaultdict(int)
        for n, measure_results in results.items():
            for measure, ngrams in measure_results.items():
                for ngram in ngrams:
                    if ngram.linguistic_pattern:
                        pattern_counts[ngram.linguistic_pattern.value] += 1
        
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern}: {count}")
