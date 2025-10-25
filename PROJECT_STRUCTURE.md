# Project Structure

**Author**: Hung Om  
**Research Foundation**: Based on foundational K'Cho linguistic research and Austroasiatic language studies  
**Version**: 0.2.0  
**Date**: 2025-01-25

## Abstract

This document describes the architecture and organization of the K'Cho Linguistic Processing Toolkit, a computational system for analyzing K'Cho language structure, morphology, and syntax using modern software engineering practices.

## Directory Organization

```
KChoCaToolKit/
├── README.md                    # Main documentation
├── LICENSE                      # MIT License
├── pyproject.toml              # Package configuration
├── MIGRATION_CLI_GUIDE.md       # Migration commands guide
│
├── kcho/                        # Main package
│   ├── __init__.py             # Package exports
│   ├── kcho_system.py          # Core system classes
│   ├── knowledge_base.py        # SQLite backend
│   ├── api_layer.py             # API integration layer
│   ├── data_migration.py        # Data migration system
│   ├── kcho_app.py             # CLI application
│   ├── kcho_lexicon.db         # SQLite database
│   │
│   ├── data/                    # Package data
│   │   ├── linguistic_data.json
│   │   ├── gold_standard_collocations.txt
│   │   ├── word_frequency_top_1000.csv
│   │   ├── gold_standard_kcho_english.json
│   │   └── sample_corpus.txt
│   │
│   ├── normalize.py             # Text normalization
│   ├── collocation.py           # Collocation extraction
│   ├── ngram_collocation.py     # N-gram analysis
│   ├── evaluation.py            # Evaluation metrics
│   ├── export.py                # Data export utilities
│   └── text_loader.py           # Text loading utilities
│
├── examples/                    # Usage examples
│   ├── README.md               # Examples documentation
│   ├── enhanced_kcho_system_demo.py
│   ├── practical_demonstration.py
│   ├── generate_research_data.py
│   └── tutorials/
│       ├── collocation_extraction_tutorial.py
│       └── ngram_collocation_test.py
│
├── tests/                       # Test suite
│   ├── test_knowledge_base.py   # Database tests
│   ├── test_data_migration.py  # Migration tests
│   ├── test_collocation.py     # Collocation tests
│   ├── test_corpus.py          # Corpus tests
│   └── test_main_system.py     # System tests
│
├── docs/                        # Additional documentation
│   ├── README.md               # Documentation index
│   ├── COLLOCATION_GUIDE.md    # Collocation analysis guide
│   └── RELEASE_NOTES_v0.1.0.md # Release notes
│
├── data/                        # Research data
│   ├── README.md               # Data documentation
│   ├── bible_versions/         # Bible translations
│   └── parallel_corpora/       # Parallel texts
│
├── research/                    # Research outputs
│   ├── cli_bible_analysis.csv
│   ├── cli_sample_analysis.csv
│   └── cli_sample_ngrams.csv
│
└── venv/                       # Virtual environment
```

## 🏗️ Architecture Overview

### Core Components

1. **Knowledge Base Layer** (`knowledge_base.py`)
   - SQLite database backend
   - Data migration and versioning
   - CRUD operations for linguistic data

2. **System Layer** (`kcho_system.py`)
   - Main orchestrator classes
   - Pattern discovery engines
   - Linguistic analysis components

3. **API Layer** (`api_layer.py`)
   - External integration interface
   - Query processing and response formatting
   - LLaMA integration support

4. **CLI Layer** (`kcho_app.py`)
   - Command-line interface
   - Migration management commands
   - Analysis and research tools

### Data Flow

```
Raw Text → Normalization → Tokenization → Analysis → Database Storage
    ↓
Pattern Discovery → Research Engine → API Layer → External Systems
```

### Database Schema

- **verb_stems**: Verb morphology data
- **collocations**: Word co-occurrence patterns
- **word_frequencies**: Statistical frequency data
- **parallel_sentences**: Translation pairs
- **raw_texts**: Unprocessed text for discovery
- **word_categories**: Lexical categorization

## 🔧 Development Guidelines

### Code Organization
- **Single Responsibility**: Each module has a clear purpose
- **Separation of Concerns**: Database, business logic, and API layers separated
- **Dependency Injection**: Components receive dependencies rather than creating them

### Naming Conventions
- **Classes**: PascalCase (e.g., `KchoKnowledge`, `PatternDiscoveryEngine`)
- **Functions**: snake_case (e.g., `get_verb_stem`, `discover_patterns`)
- **Constants**: UPPER_CASE (e.g., `VERB_STEMS`, `GOLD_STANDARD_PATTERNS`)

### Testing Strategy
- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component functionality
- **Performance Tests**: Database query optimization
- **Compatibility Tests**: Backward compatibility verification

## 📦 Package Structure

### Public API (`__init__.py`)
```python
# Core classes
from kcho import KchoKnowledge, KchoSystem
from kcho import PatternDiscoveryEngine, LinguisticResearchEngine
from kcho import KchoAPILayer

# Database utilities
from kcho import KchoKnowledgeBase, init_database

# Migration system
from kcho import DataMigrationManager, check_and_migrate_data
```

### Internal Modules
- **normalize.py**: Text preprocessing utilities
- **collocation.py**: Statistical collocation analysis
- **evaluation.py**: Performance metrics and evaluation
- **export.py**: Data export and serialization

## 🚀 Deployment

### Production Setup
1. Install dependencies: `pip install -e .`
2. Initialize database: `python -m kcho.kcho_app migrate init`
3. Verify installation: `python examples/enhanced_kcho_system_demo.py`

### Development Setup
1. Clone repository
2. Create virtual environment: `python -m venv venv`
3. Activate environment: `source venv/bin/activate`
4. Install in development mode: `pip install -e .[dev]`
5. Run tests: `pytest tests/`

## 📊 Performance Considerations

### Database Optimization
- **Indexes**: Created on frequently queried fields
- **Caching**: LRU cache for hot data paths
- **Batch Operations**: Efficient bulk inserts and updates

### Memory Management
- **Lazy Loading**: Data loaded on-demand
- **Connection Pooling**: Optimized database connections
- **Garbage Collection**: Proper resource cleanup

### Scalability
- **Modular Design**: Easy to extend with new components
- **Plugin Architecture**: Support for custom analyzers
- **API Layer**: Ready for distributed deployment
