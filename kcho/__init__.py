# FILE: __init__.py (NEW)
# PURPOSE: Package initialization and public API

"""
K'Cho Linguistic Processing Toolkit

Author: Hung Om
Research Foundation: Based on foundational K'Cho linguistic research and Austroasiatic language studies
Version: 0.2.0
Date: 2024-10-25

Abstract:
A comprehensive computational toolkit for K'Cho language processing, featuring 
SQLite-backed knowledge management, pattern discovery, linguistic analysis, and 
API integration capabilities. The system implements statistical collocation 
extraction, morphological analysis, syntactic parsing, and pattern discovery 
algorithms optimized for linguistic research applications.

Features:
- Text normalization and tokenization
- Collocation extraction with multiple association measures
- Parallel corpus processing
- Evaluation utilities
- SQLite-backed knowledge management
- Pattern discovery and linguistic analysis

Basic usage:
    >>> from kcho import normalize, collocation, export
    >>> normalized = normalize.normalize_text(text)
    >>> results = collocation.extract(corpus, measures=['pmi', 'tscore'])
    >>> export.to_csv(results, 'collocations.csv')
"""

__version__ = "0.2.0"

# Import key modules for public API
from . import normalize
from . import collocation
from . import export
from . import evaluation
from .kcho_system import KchoSystem, KchoCorpus, KchoTokenizer, KchoValidator, KchoMorphologyAnalyzer, KchoSyntaxAnalyzer, KchoLexicon, LegacyKchoKnowledge, KchoKnowledge, LinguisticResearchEngine, PatternDiscoveryEngine
from .knowledge_base import KchoKnowledgeBase, init_database
from .data_migration import DataMigrationManager, FreshDataLoader, check_and_migrate_data
from .api_layer import KchoAPILayer
from .llama_integration import EnhancedKchoAPILayer, LLaMAConfig, create_llama_config, load_env_file
from .collocation import CollocationExtractor, AssociationMeasure, CollocationResult, LinguisticPattern
from .ngram_collocation import NGramCollocationExtractor, NGramCollocationResult
from .text_loader import TextLoader
from .config import load_config, save_config_template
from .cli import cli

__all__ = [
    'normalize',
    'collocation', 
    'export',
    'evaluation',
    'KchoSystem',
    'KchoCorpus',
    'KchoTokenizer',
    'KchoValidator',
    'KchoMorphologyAnalyzer',
    'KchoSyntaxAnalyzer',
    'LegacyKchoKnowledge',
    'KchoKnowledge',
    'LinguisticResearchEngine',
    'PatternDiscoveryEngine',
    'KchoKnowledgeBase',
    'init_database',
    'DataMigrationManager',
    'FreshDataLoader',
    'check_and_migrate_data',
    'KchoAPILayer',
    'EnhancedKchoAPILayer',
    'LLaMAConfig',
    'load_env_file',
    'CollocationExtractor',
    'AssociationMeasure',
    'CollocationResult',
    'LinguisticPattern',
    'NGramCollocationExtractor',
    'NGramCollocationResult',
    'TextLoader',
    'load_config',
    'save_config_template',
    'cli',
    '__version__',
]