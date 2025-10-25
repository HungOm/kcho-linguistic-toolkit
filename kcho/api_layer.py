"""
K'Cho API Layer - LLaMA Integration and External Access
=====================================================

API layer for integrating K'Cho linguistic knowledge with LLaMA and other external systems.
Provides structured responses for natural language queries about K'Cho language.

Author: Based on K'Cho linguistic research (Bedell & Mang 2012)
Version: 2.0.0
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class KchoAPILayer:
    """API layer for LLaMA integration and external access"""
    
    def __init__(self, knowledge_base):
        """
        Initialize API layer with knowledge base.
        
        Args:
            knowledge_base: KchoKnowledgeBase or LegacyKchoKnowledge instance
        """
        self.knowledge = knowledge_base
    
    def process_query(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Process query for LLaMA integration.
        
        Args:
            query: User query about K'Cho language
            context: Optional context for the query
            
        Returns:
            Structured API response
        """
        response = {
            'query': query,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'response_type': 'linguistic_analysis',
            'data': {}
        }
        
        # Get knowledge sources (handle both old and new knowledge base)
        if hasattr(self.knowledge, 'data_sources'):
            response['knowledge_sources'] = self.knowledge.data_sources
        elif hasattr(self.knowledge, 'get_statistics'):
            stats = self.knowledge.get_statistics()
            response['knowledge_sources'] = {
                'verb_stems': stats['verb_stems'],
                'collocations': stats['collocations'],
                'word_frequencies': stats['word_frequencies'],
                'parallel_sentences': stats['parallel_sentences']
            }
        
        # Process different types of queries
        query_lower = query.lower()
        
        if 'pattern' in query_lower or 'collocation' in query_lower:
            response['data'] = self._handle_pattern_query(query, context)
        elif 'morphology' in query_lower or 'morpheme' in query_lower:
            response['data'] = self._handle_morphology_query(query, context)
        elif 'syntax' in query_lower or 'grammar' in query_lower:
            response['data'] = self._handle_syntax_query(query, context)
        elif 'word' in query_lower or 'vocabulary' in query_lower:
            response['data'] = self._handle_vocabulary_query(query, context)
        else:
            response['data'] = self._handle_general_query(query, context)
        
        return response
    
    def _handle_pattern_query(self, query: str, context: Optional[str]) -> Dict[str, Any]:
        """Handle pattern-related queries"""
        # Use optimized DB queries if available
        if hasattr(self.knowledge, 'get_collocations_by_category'):
            # New SQLite-based knowledge base
            categories = ['VP', 'PP', 'APP', 'AGR', 'AUX', 'COMP', 'MWE']
            pattern_categories = []
            total_patterns = 0
            
            for category in categories:
                patterns = self.knowledge.get_collocations_by_category(category)
                if patterns:
                    pattern_categories.append(category)
                    total_patterns += len(patterns)
            
            sample_patterns = {}
            for category in pattern_categories[:3]:  # Show first 3 categories
                patterns = self.knowledge.get_collocations_by_category(category)
                sample_patterns[category] = [p['words'] for p in patterns[:3]]
            
            return {
                'query_type': 'pattern_analysis',
                'total_patterns': total_patterns,
                'pattern_categories': pattern_categories,
                'sample_patterns': sample_patterns,
                'discovered_patterns': len(self.knowledge.get_collocations_by_source('discovered'))
            }
        else:
            # Legacy knowledge base
            return {
                'query_type': 'pattern_analysis',
                'gold_standard_patterns': len(self.knowledge.gold_standard_patterns),
                'pattern_categories': list(set(data['category'] for data in self.knowledge.gold_standard_patterns.values())),
                'sample_patterns': dict(list(self.knowledge.gold_standard_patterns.items())[:5])
            }
    
    def _handle_morphology_query(self, query: str, context: Optional[str]) -> Dict[str, Any]:
        """Handle morphology-related queries"""
        if hasattr(self.knowledge, 'get_verb_stem'):
            # New SQLite-based knowledge base
            stats = self.knowledge.get_statistics()
            return {
                'query_type': 'morphology_analysis',
                'verb_stems': stats['verb_stems'],
                'sample_verbs': self._get_sample_verbs(),
                'morphological_categories': ['verb', 'noun', 'pronoun', 'agreement', 'postposition']
            }
        else:
            # Legacy knowledge base
            return {
                'query_type': 'morphology_analysis',
                'verb_stems': len(self.knowledge.VERB_STEMS),
                'sample_verbs': dict(list(self.knowledge.VERB_STEMS.items())[:5]),
                'morphological_categories': list(self.knowledge.word_categories.keys())
            }
    
    def _handle_syntax_query(self, query: str, context: Optional[str]) -> Dict[str, Any]:
        """Handle syntax-related queries"""
        if hasattr(self.knowledge, 'get_collocations_by_category'):
            # New SQLite-based knowledge base
            pp_patterns = self.knowledge.get_collocations_by_category('PP')
            agr_patterns = self.knowledge.get_collocations_by_category('AGR')
            
            return {
                'query_type': 'syntax_analysis',
                'postposition_patterns': len(pp_patterns),
                'sample_postpositions': [p['words'] for p in pp_patterns[:5]],
                'agreement_patterns': len(agr_patterns),
                'sample_agreement': [p['words'] for p in agr_patterns[:5]]
            }
        else:
            # Legacy knowledge base
            return {
                'query_type': 'syntax_analysis',
                'postpositions': len(self.knowledge.POSTPOSITIONS),
                'sample_postpositions': dict(list(self.knowledge.POSTPOSITIONS.items())[:5]),
                'agreement_particles': len(self.knowledge.AGREEMENT)
            }
    
    def _handle_vocabulary_query(self, query: str, context: Optional[str]) -> Dict[str, Any]:
        """Handle vocabulary-related queries"""
        if hasattr(self.knowledge, 'get_word_frequency'):
            # New SQLite-based knowledge base
            stats = self.knowledge.get_statistics()
            return {
                'query_type': 'vocabulary_analysis',
                'total_words': stats['word_frequencies'],
                'word_categories': stats['word_categories'],
                'frequency_data_available': stats['word_frequencies'] > 0,
                'top_words': self._get_top_frequent_words()
            }
        else:
            # Legacy knowledge base
            return {
                'query_type': 'vocabulary_analysis',
                'total_words': len(self.knowledge.all_words),
                'word_categories': {cat: len(words) for cat, words in self.knowledge.word_categories.items()},
                'frequency_data_available': len(self.knowledge.word_frequency_data) > 0
            }
    
    def _handle_general_query(self, query: str, context: Optional[str]) -> Dict[str, Any]:
        """Handle general queries"""
        if hasattr(self.knowledge, 'get_statistics'):
            # New SQLite-based knowledge base
            stats = self.knowledge.get_statistics()
            return {
                'query_type': 'general_analysis',
                'knowledge_base_summary': {
                    'verb_stems': stats['verb_stems'],
                    'collocations': stats['collocations'],
                    'word_frequencies': stats['word_frequencies'],
                    'parallel_sentences': stats['parallel_sentences'],
                    'raw_texts': stats['raw_texts']
                },
                'available_analyses': [
                    'pattern_discovery',
                    'morphological_analysis',
                    'syntactic_analysis',
                    'semantic_analysis',
                    'frequency_analysis'
                ]
            }
        else:
            # Legacy knowledge base
            return {
                'query_type': 'general_analysis',
                'knowledge_base_summary': {
                    'total_words': len(self.knowledge.all_words),
                    'gold_standard_patterns': len(self.knowledge.gold_standard_patterns),
                    'word_categories': len(self.knowledge.word_categories),
                    'data_sources': self.knowledge.data_sources
                },
                'available_analyses': [
                    'pattern_discovery',
                    'morphological_analysis',
                    'syntactic_analysis',
                    'semantic_analysis',
                    'frequency_analysis'
                ]
            }
    
    def _get_sample_verbs(self) -> Dict[str, str]:
        """Get sample verbs for display."""
        if hasattr(self.knowledge, 'get_verb_stem'):
            # Get first few verbs from database
            import sqlite3
            conn = sqlite3.connect(self.knowledge.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT verb, gloss FROM verb_stems LIMIT 5")
            rows = cur.fetchall()
            conn.close()
            return {row['verb']: row['gloss'] for row in rows}
        return {}
    
    def _get_top_frequent_words(self) -> Dict[str, int]:
        """Get top frequent words."""
        if hasattr(self.knowledge, 'get_word_frequency'):
            import sqlite3
            conn = sqlite3.connect(self.knowledge.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT word, frequency FROM word_frequencies ORDER BY frequency DESC LIMIT 10")
            rows = cur.fetchall()
            conn.close()
            return {row['word']: row['frequency'] for row in rows}
        return {}
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of available knowledge"""
        if hasattr(self.knowledge, 'get_statistics'):
            # New SQLite-based knowledge base
            stats = self.knowledge.get_statistics()
            return {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'knowledge_base_version': '2.0.0',
                    'database_path': stats['database_path']
                },
                'data_counts': {
                    'verb_stems': stats['verb_stems'],
                    'collocations': stats['collocations'],
                    'word_frequencies': stats['word_frequencies'],
                    'parallel_sentences': stats['parallel_sentences'],
                    'raw_texts': stats['raw_texts']
                },
                'collocation_sources': stats.get('collocation_sources', {}),
                'pattern_categories': self._get_pattern_categories()
            }
        else:
            # Legacy knowledge base
            return {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'knowledge_base_version': '2.0.0'
                },
                'data_sources': self.knowledge.data_sources,
                'linguistic_categories': {
                    'verbs': len(self.knowledge.VERB_STEMS),
                    'nouns': len(self.knowledge.COMMON_NOUNS),
                    'postpositions': len(self.knowledge.POSTPOSITIONS),
                    'agreement_particles': len(self.knowledge.AGREEMENT),
                    'pronouns': len(self.knowledge.PRONOUNS)
                },
                'pattern_data': {
                    'gold_standard_patterns': len(self.knowledge.gold_standard_patterns),
                    'pattern_categories': list(set(data['category'] for data in self.knowledge.gold_standard_patterns.values()))
                },
                'frequency_data': {
                    'word_frequency_entries': len(self.knowledge.word_frequency_data),
                    'top_frequent_words': dict(list(self.knowledge.word_frequency_data.items())[:10])
                }
            }
    
    def _get_pattern_categories(self) -> List[str]:
        """Get available pattern categories."""
        if hasattr(self.knowledge, 'get_collocations_by_category'):
            categories = ['VP', 'PP', 'APP', 'AGR', 'AUX', 'COMP', 'MWE']
            available_categories = []
            for category in categories:
                patterns = self.knowledge.get_collocations_by_category(category)
                if patterns:
                    available_categories.append(category)
            return available_categories
        return []
    
    def query_collocations_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Query collocations by category using optimized DB queries."""
        if hasattr(self.knowledge, 'get_collocations_by_category'):
            return self.knowledge.get_collocations_by_category(category)
        else:
            # Fallback for legacy knowledge base
            return [
                {'words': words, 'category': data['category'], 'frequency': data.get('frequency', ''),
                 'notes': data.get('notes', ''), 'source': 'gold_standard'}
                for words, data in self.knowledge.gold_standard_patterns.items()
                if data['category'] == category
            ]
    
    def search_patterns(self, pattern: str) -> List[Dict[str, Any]]:
        """Search for patterns containing a specific string."""
        if hasattr(self.knowledge, 'search_collocations'):
            return self.knowledge.search_collocations(pattern)
        else:
            # Fallback for legacy knowledge base
            return [
                {'words': words, 'category': data['category'], 'frequency': data.get('frequency', ''),
                 'notes': data.get('notes', ''), 'source': 'gold_standard'}
                for words, data in self.knowledge.gold_standard_patterns.items()
                if pattern.lower() in words.lower()
            ]
    
    def get_word_info(self, word: str) -> Dict[str, Any]:
        """Get comprehensive information about a word."""
        info = {'word': word}
        
        if hasattr(self.knowledge, 'get_word_categories'):
            info['categories'] = self.knowledge.get_word_categories(word)
        
        if hasattr(self.knowledge, 'get_word_frequency'):
            freq = self.knowledge.get_word_frequency(word)
            if freq:
                info['frequency'] = freq
        
        if hasattr(self.knowledge, 'get_verb_stem'):
            verb_info = self.knowledge.get_verb_stem(word)
            if verb_info:
                info['verb_stem'] = verb_info
        
        return info
