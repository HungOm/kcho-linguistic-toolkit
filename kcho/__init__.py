# FILE: __init__.py (NEW)
# PURPOSE: Package initialization and public API

"""
K'Cho Language Toolkit

A comprehensive toolkit for low-resource language processing with K'Cho.

Features:
- Text normalization and tokenization
- Collocation extraction with multiple association measures
- Parallel corpus processing
- Evaluation utilities

Basic usage:
    >>> from kcho import normalize, collocation, export
    >>> normalized = normalize.normalize_text(text)
    >>> results = collocation.extract(corpus, measures=['pmi', 'tscore'])
    >>> export.to_csv(results, 'collocations.csv')
"""

__version__ = "0.1.0"

# Import key modules for public API
from . import normalize
from . import collocation
from . import export
from . import evaluation
from .kcho_system import KchoSystem, KchoCorpus, KchoKnowledge, KchoTokenizer, KchoValidator, KchoMorphologyAnalyzer, KchoSyntaxAnalyzer, KchoLexicon
from .collocation import CollocationExtractor, AssociationMeasure, CollocationResult

__all__ = [
    'normalize',
    'collocation', 
    'export',
    'evaluation',
    'KchoSystem',
    'KchoCorpus',
    'KchoKnowledge',
    'KchoTokenizer',
    'KchoValidator',
    'KchoMorphologyAnalyzer',
    'KchoSyntaxAnalyzer',
    'KchoLexicon',
    'CollocationExtractor',
    'AssociationMeasure',
    'CollocationResult',
    '__version__',
]