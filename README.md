# K'Cho Linguistic Processing Toolkit

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)]()

A comprehensive computational toolkit for K'Cho language processing, featuring SQLite-backed knowledge management, pattern discovery, linguistic analysis, and API integration capabilities.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{om2025kcho,
  title={K'Cho Linguistic Processing Toolkit},
  author={Om, Hung},
  year={2025},
  url={https://github.com/HungOm/kcho-linguistic-toolkit},
  note={Computational implementation based on foundational K'Cho linguistic research}
}
```

## Abstract

This toolkit provides computational methods for analyzing K'Cho, a low-resource Austroasiatic language. The system implements statistical collocation extraction, morphological analysis, syntactic parsing, and pattern discovery algorithms optimized for linguistic research applications.

## 🚀 Features

### Core Capabilities
- **SQLite Knowledge Base**: Efficient, persistent storage with indexed queries
- **Pattern Discovery**: Automatic detection of linguistic patterns from corpora
- **Morphological Analysis**: Token analysis with morpheme segmentation
- **Syntactic Analysis**: Sentence structure and word order analysis
- **Collocation Extraction**: Statistical analysis of word co-occurrences
- **API Layer**: Ready for integration with external systems (LLaMA, etc.)

### Data Management
- **Migration System**: CLI commands for data versioning and updates
- **Multiple Data Sources**: JSON, TXT, CSV file support
- **Backward Compatibility**: Legacy support with deprecation warnings
- **Validation**: Optional Pydantic schema validation

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- SQLite3 (included with Python)

### Install Dependencies
```bash
# Core dependencies (minimal)
pip install click numpy pandas scikit-learn nltk

# Optional validation support
pip install pydantic>=2.0
```

### Development Setup
```bash
git clone <repository-url>
cd KChoCaToolKit
pip install -e .
```

## 🏗️ Architecture

### Core Components

#### `KchoKnowledge` (SQLite Backend)
The primary knowledge base using SQLite for efficient storage and querying:

```python
from kcho import KchoKnowledge

# Initialize with SQLite backend
knowledge = KchoKnowledge()

# Query verb stems
verb_info = knowledge.get_verb_stem("pàapai")

# Get collocations by category
vp_patterns = knowledge.get_collocations_by_category("VP")

# Add new discoveries
knowledge.add_collocation("new pattern", "VP", "high_freq", "discovered")
```

#### `KchoSystem` (Main Orchestrator)
The main system integrating all components:

```python
from kcho import KchoSystem

# Initialize system
system = KchoSystem(use_comprehensive_knowledge=True)

# Analyze text
sentence = system.analyze_text("Om noh Yóng am pàapai pe ci.")

# Discover patterns
patterns = system.discover_patterns(corpus, min_frequency=5)

# Get API response
response = system.get_api_response("What are common verb patterns?")
```

#### `PatternDiscoveryEngine`
Automatic pattern discovery from corpora:

```python
from kcho import PatternDiscoveryEngine

engine = PatternDiscoveryEngine(knowledge)
discovered = engine.discover_patterns(corpus, min_frequency=3)
```

#### `LinguisticResearchEngine`
Comprehensive linguistic research capabilities:

```python
from kcho import LinguisticResearchEngine

research = LinguisticResearchEngine(knowledge)
results = research.conduct_comprehensive_research(corpus)
```

### Database Schema

The SQLite backend includes these tables:

- **`verb_stems`**: Verb stem information with patterns
- **`collocations`**: Word co-occurrence patterns by category
- **`word_frequencies`**: Word frequency data
- **`parallel_sentences`**: K'Cho-English sentence pairs
- **`raw_texts`**: Raw text for pattern discovery
- **`word_categories`**: Word categorization data

All tables include proper indexing for efficient queries.

## 🛠️ Usage

### Basic Text Analysis

```python
from kcho import KchoSystem

# Initialize system
system = KchoSystem()

# Analyze a sentence
text = "Om noh Yóng am pàapai pe ci."
sentence = system.analyze_text(text)

print(f"Tokens: {len(sentence.tokens)}")
for token in sentence.tokens:
    print(f"  {token.surface}: {token.pos.value}")
```

### Pattern Discovery

```python
# Load corpus
corpus = [
    "Om noh Yóng am pàapai pe ci.",
    "Cun ah k'chàang noh Yóng am pàapai pe ci.",
    # ... more sentences
]

# Discover patterns
patterns = system.discover_patterns(corpus, min_frequency=3)
print(f"Discovered {len(patterns)} patterns")
```

### Research Analysis

```python
# Conduct comprehensive research
research_engine = LinguisticResearchEngine(system.knowledge)
results = research_engine.conduct_comprehensive_research(
    corpus, 
    research_focus='all'
)

print(f"Known patterns: {results['known_patterns']['total_patterns_analyzed']}")
print(f"Novel patterns: {results['novel_patterns']['total_discovered']}")
```

### API Integration

```python
# Get structured API response
api_layer = KchoAPILayer(system.knowledge)
response = api_layer.process_query("What are the common verb-particle patterns?")

print(f"Response type: {response['response_type']}")
print(f"Data: {response['data']}")
```

## 🔧 CLI Commands

### Data Migration Management

```bash
# Check migration status
python -m kcho.cli migrate check

# Run migration
python -m kcho.cli migrate run

# View migration history
python -m kcho.cli migrate history

# Force fresh migration
python -m kcho.cli migrate force --confirm

# Initialize database
python -m kcho.cli migrate init
```

### Pattern Analysis

```bash
# Analyze collocations
python -m kcho.cli collocation --corpus corpus.txt --output results.csv

# Extract n-grams
python -m kcho.cli extract-ngrams --input corpus.txt --output ngrams --format json

# Generate research data
python -m kcho.cli generate research-data --corpus corpus.txt --output research/ --include-research --include-collocations
```

### Research Commands

```bash
# Conduct comprehensive research analysis
python -m kcho.cli research analyze --corpus corpus.txt --output analysis.json --focus all

# Discover linguistic patterns
python -m kcho.cli research discover-patterns --corpus corpus.txt --output patterns.json --min-freq 3

# Analyze text structure
python -m kcho.cli research analyze-text --corpus corpus.txt --output text_analysis.json --format json
```

## 📊 Data Sources

The system integrates data from multiple sources:

### Core Data Files
- **`linguistic_data.json`**: Verb stems, pronouns, agreement particles, etc.
- **`gold_standard_collocations.txt`**: Manually curated collocation patterns
- **`word_frequency_top_1000.csv`**: Word frequency data
- **`gold_standard_kcho_english.json`**: Parallel sentence pairs
- **`sample_corpus.txt`**: Sample K'Cho texts

### Research Data
- **Bible versions**: Multiple K'Cho Bible translations
- **Parallel corpora**: K'Cho-English aligned texts
- **Frequency analysis**: Statistical word frequency data

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_knowledge_base.py
pytest tests/test_data_migration.py
pytest tests/test_collocation.py
```

### Demo System
```bash
# Run comprehensive demo
python examples/enhanced_kcho_system_demo.py
```

## 📈 Performance

### Database Performance
- **Indexed queries**: O(log n) lookups on common fields
- **LRU caching**: Frequently accessed data cached in memory
- **Batch operations**: Efficient bulk inserts and updates
- **Connection pooling**: Optimized database connections

### Memory Usage
- **SQLite backend**: Persistent storage reduces memory footprint
- **Lazy loading**: Data loaded on-demand
- **Efficient data structures**: Optimized for linguistic processing

## 🔄 Migration Guide

### From Legacy System
The system maintains full backward compatibility:

```python
# Old way (still works)
from kcho import LegacyKchoKnowledge
knowledge = LegacyKchoKnowledge()

# New way (recommended)
from kcho import KchoKnowledge
knowledge = KchoKnowledge()  # Uses SQLite backend
```

### Data Migration
```bash
# Check if migration is needed
python -m kcho.kcho_app migrate check

# Run migration if needed
python -m kcho.kcho_app migrate run
```

## 🤝 Contributing

### Development Setup
1. Clone the repository
2. Install development dependencies: `pip install -e .[dev]`
3. Run tests: `pytest tests/`
4. Follow PEP 8 style guidelines

### Code Organization
- **Core logic**: `kcho/kcho_system.py`
- **Database backend**: `kcho/knowledge_base.py`
- **API layer**: `kcho/api_layer.py`
- **Data migration**: `kcho/data_migration.py`
- **CLI interface**: `kcho/kcho_app.py`

## 📚 API Reference

### Core Classes

#### `KchoKnowledge`
- `get_verb_stem(verb: str) -> Dict`
- `get_collocations_by_category(category: str) -> List[Dict]`
- `add_collocation(words: str, category: str, ...)`
- `discover_patterns_from_raw() -> Dict`
- `get_statistics() -> Dict[str, int]`

#### `KchoSystem`
- `analyze_text(text: str) -> Sentence`
- `discover_patterns(corpus: List[str], min_frequency: int) -> Dict`
- `get_api_response(query: str) -> Dict`
- `load_corpus(file_path: str) -> List[Sentence]`

#### `PatternDiscoveryEngine`
- `discover_patterns(corpus: List[str], min_frequency: int) -> Dict`
- `_discover_bigram_patterns(corpus: List[str], min_frequency: int) -> Dict`
- `_discover_trigram_patterns(corpus: List[str], min_frequency: int) -> Dict`

#### `LinguisticResearchEngine`
- `conduct_comprehensive_research(corpus: List[str], research_focus: str) -> Dict`
- `_analyze_known_patterns(corpus: List[str]) -> Dict`
- `_discover_novel_patterns(corpus: List[str]) -> Dict`

## 📚 Documentation

- **CLI Guide**: Complete command reference in [CLI_GUIDE.md](CLI_GUIDE.md)
- **Migration Guide**: Data migration commands in [MIGRATION_CLI_GUIDE.md](MIGRATION_CLI_GUIDE.md)
- **Project Structure**: Architecture details in [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **Examples**: Usage examples in [examples/README.md](examples/README.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

**Author**: Hung Om

**Research Foundation**: 
This implementation builds upon foundational linguistic research on K'Cho language structure, morphology, and syntax. The computational methods are based on established linguistic theories and empirical research in Austroasiatic language studies.

**Academic References**:
- Bedell, G. & Mang, S. (2012). *K'Cho linguistic research and documentation*. [Original foundational research]
- Additional K'Cho language researchers and Austroasiatic linguistics community
- Statistical methods for collocation analysis in low-resource languages
- Computational approaches to morphological and syntactic analysis
- Corpus linguistics methodologies for endangered language documentation

**Software Dependencies**:
- Python 3.8+ (van Rossum & Drake, 2009)
- SQLite3 (Hipp, 2020)
- NLTK (Bird et al., 2009)
- scikit-learn (Pedregosa et al., 2011)
- Click (Krekel, 2014)
- NumPy (Harris et al., 2020)

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation in `docs/`
- Review the examples in `examples/`

---

**Version**: 2.0.0  
**Last Updated**: 2025-10-25  
**Status**: Production Ready