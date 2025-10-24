# K'Cho Linguistic Toolkit

> **A comprehensive toolkit for K'Cho language processing with collocation extraction, morphological analysis, and corpus processing.**

Based on linguistic research by George Bedell and Kee Shein Mang (2012), this toolkit is developed by Hung Om, an enthusiastic K'Cho speaker and independent developer to provide essential tools for working with K'Cho, a Kuki-Chin language spoken by 10,000-20,000 people in southern Chin State, Myanmar.

## 🎯 What This Toolkit Does

This is a **single, integrated package** that provides:

- ✅ **Collocation Extraction** - Extract meaningful word combinations using multiple association measures
- ✅ **Morphological Analysis** - Analyze K'Cho word structure (stems, affixes, particles)
- ✅ **Text Normalization** - Clean and normalize K'Cho text for analysis
- ✅ **Corpus Building** - Create annotated datasets with quality control
- ✅ **Lexicon Management** - Build and manage digital K'Cho dictionaries
- ✅ **Data Export** - Export to standard formats (JSON, CoNLL-U, CSV)
- ✅ **Evaluation Tools** - Evaluate collocation extraction quality
- ✅ **Parallel Corpus Processing** - Process aligned K'Cho-English texts
- ✅ **ML-Ready Output** - Prepare data for machine learning training

## 🚀 Quick Start

### Installation

```bash
# Install in development mode
pip install -e .

# Or install from PyPI (when published)
pip install kcho-linguistic-toolkit
```

### Basic Usage

```python
from kcho import CollocationExtractor, KChoSystem

# Initialize the system
system = KChoSystem()

# Extract collocations from corpus
extractor = CollocationExtractor()
corpus = ["Om noh Yong am paapai pe ci", "Ak'hmó lùum ci"]
results = extractor.extract(corpus)

# Use advanced defaultdict functionality
pos_patterns = system.corpus.analyze_pos_patterns()
word_contexts = extractor.analyze_word_contexts(corpus)
```

### Command Line Interface

```bash
# Run collocation extraction
python -m kcho.create_gold_standard --corpus data/sample_corpus.txt --output gold_standard.txt

# Use the main CLI
kcho analyze --corpus data/sample_corpus.txt --output results/
```

## 📦 Installation

Install the package in development mode:

```bash
# Clone the repository
git clone https://github.com/hungom/kcho-linguistic-toolkit.git
cd kcho-linguistic-toolkit

# Install in development mode
pip install -e .

# Verify installation
python -c "from kcho import CollocationExtractor; print('✅ Installation successful!')"
```

## 📁 Project Structure

The toolkit is organized following Python packaging best practices:

```
KchoLinguisticToolkit/
├── kcho/                           # Main package
│   ├── __init__.py                 # Package initialization
│   ├── collocation.py              # Collocation extraction
│   ├── kcho_system.py              # Core system
│   ├── normalize.py                # Text normalization
│   ├── evaluation.py               # Evaluation utilities
│   ├── export.py                   # Export functions
│   ├── eng_kcho_parallel_extractor.py
│   ├── export_training_csv.py
│   ├── create_gold_standard.py     # Gold standard helper
│   ├── kcho_app.py                 # CLI entry point
│   └── data/                       # Package data
│       ├── linguistic_data.json
│       └── word_frequency_top_1000.csv
├── examples/                       # Example scripts
│   └── defaultdict_usage.py
├── data/                           # External data (not in package)
│   ├── README.md                   # Data documentation
│   ├── sample_corpus.txt           # Small, keep in git
│   ├── gold_standard_collocations.txt
│   ├── bible_versions/             # Large, .gitignored
│   ├── parallel_corpora/           # Medium, .gitignored
│   └── research_outputs/           # Generated, .gitignored
├── .gitignore                      # Comprehensive ignore rules
├── pyproject.toml                  # Package configuration
└── README.md                       # This file
```

## 🌟 Key Features

### 1. Collocation Extraction

Advanced collocation extraction with multiple association measures:

- **PMI (Pointwise Mutual Information)** - Classical measure for word association
- **NPMI (Normalized PMI)** - Bounded [0,1] variant for comparison
- **t-score** - Statistical significance testing
- **Dice Coefficient** - Symmetric association measure
- **Log-likelihood Ratio (G²)** - Asymptotic significance testing

```python
from kcho import CollocationExtractor

extractor = CollocationExtractor()
results = extractor.extract(corpus)

# Group by POS patterns using defaultdict
pos_groups = extractor.group_collocations_by_pos_pattern(corpus)

# Analyze word contexts
contexts = extractor.analyze_word_contexts(corpus, context_window=3)
```

### 2. Morphological Analysis

Based on K'Cho linguistic research, the toolkit understands:

- **Applicative Suffix** (-na/-nák)
  - `luum-na` = "play with"
  - Automatically detects and analyzes

- **Agreement Particles** (ka, na, a)
- **Postpositions** (noh, ah, am, on)
- **Tense Markers** (ci, khai)

Example:
```python
sentence = toolkit.analyze("Ak'hmó noh k'khìm luum-na ci")
# Automatically identifies: subject + postposition + instrument + verb-APPL + tense
```

### 2. Text Validation

Automatically detects K'cho text with confidence scoring:

```python
is_kcho, confidence, metrics = toolkit.validate("Om noh Yong am paapai pe ci")
# Returns: (True, 0.875, {...detailed metrics...})
```

**Validation Features:**
- Character set validation
- K'cho marker detection (postpositions, particles)
- Pattern matching for K'cho structures
- Confidence scoring (0-100%)

### 3. Corpus Building

Build clean, annotated K'cho datasets:

```python
# Add with automatic analysis
toolkit.add_to_corpus(
    "Om noh Yong am paapai pe ci",
    translation="Om gave Yong flowers"
)

# Get statistics
stats = toolkit.corpus_stats()
# Returns: total_sentences, vocabulary_size, POS distribution, etc.

# Create ML splits
splits = toolkit.corpus.create_splits(train_ratio=0.8)
# Returns: {'train': [...], 'dev': [...], 'test': [...]}
```

### 4. Lexicon Management

SQLite-based dictionary with full search:

```python
from kcho_toolkit import LexiconEntry

# Add words
entry = LexiconEntry(
    headword="paapai",
    pos="N",
    gloss_en="flower",
    gloss_my="ပန်း",  # Myanmar translation
    examples=["Om noh Yong am paapai pe ci"]
)
toolkit.lexicon.add_entry(entry)

# Search
results = toolkit.search_lexicon("flower")

# Get frequency list
top_words = toolkit.lexicon.get_frequency_list(100)
```

### 5. Data Export

Export to multiple standard formats:

```python
# JSON (for ML training)
toolkit.corpus.export_json("corpus.json")

# CoNLL-U (for linguistic research)
toolkit.corpus.export_conllu("corpus.conllu")

# CSV (for spreadsheet analysis)
toolkit.corpus.export_csv("corpus.csv")

# Or export everything at once
toolkit.export_all()
```

## 📊 Use Cases

### Machine Translation Training

```python
# Build parallel corpus
for kcho, english in parallel_sentences:
    toolkit.add_to_corpus(kcho, translation=english)

# Create splits
splits = toolkit.corpus.create_splits()

# Export for training
for split_name, sentences in splits.items():
    data = [{'source': s.text, 'target': s.translation} for s in sentences]
    # Use with Hugging Face, Fairseq, etc.
```

### Linguistic Research

```python
# Analyze corpus
stats = toolkit.corpus_stats()
print(f"POS distribution: {stats['pos_distribution']}")

# Study verb paradigms
paradigm = toolkit.get_verb_forms('lùum')
# Returns complete conjugation tables

# Export to CoNLL-U for dependency parsing research
toolkit.corpus.export_conllu("research_corpus.conllu")
```

### Dictionary Application Backend

```python
# Search API
results = toolkit.search_lexicon(query)

# Morphological analysis API
analysis = toolkit.analyze(user_input)

# Validation API
is_valid, confidence, _ = toolkit.validate(user_text)
```

## 📁 File Structure

The toolkit creates this organized structure:

```
your_project/
├── kcho_lexicon.db          # SQLite dictionary
├── corpus/                  # Raw corpus data
├── exports/                 # Exported datasets
│   ├── corpus_*.json
│   ├── corpus_*.conllu
│   ├── corpus_*.csv
│   └── lexicon_*.json
└── reports/                 # Quality reports
    └── report_*.json
```

## 🎓 Examples

See `kcho_examples.py` for 8 complete examples:

1. **Basic Analysis** - Analyze K'cho sentences
2. **Build Corpus** - Create annotated corpus
3. **Validate Text** - Detect K'cho text
4. **Lexicon Management** - Work with dictionary
5. **Verb Paradigms** - Generate conjugation tables
6. **Data Export** - Export to different formats
7. **Quality Control** - Validate corpus quality
8. **ML Preparation** - Prepare training data

Run examples:
```bash
python kcho_examples.py
```

## 📖 Documentation

- **[KCHO_TOOLKIT_DOCS.md](KCHO_TOOLKIT_DOCS.md)** - Complete API reference and usage guide
- **[kcho_examples.py](kcho_examples.py)** - 8 practical examples
- **[kcho_toolkit.py](kcho_toolkit.py)** - Main source code (well-documented)

## 📊 Data Organization

The toolkit includes several types of data:

### Package Data (included in installation)
- `kcho/data/linguistic_data.json` - Core linguistic knowledge base
- `kcho/data/word_frequency_top_1000.csv` - High-frequency word list

### External Data (not in package)
- `data/sample_corpus.txt` - Small sample corpus for testing
- `data/gold_standard_collocations.txt` - Gold standard annotations
- `data/bible_versions/` - Bible translations (public domain, large files)
- `data/parallel_corpora/` - Aligned parallel texts
- `data/research_outputs/` - Generated analysis results

**Note**: Large data files are not included in the package to keep it lightweight. See `data/README.md` for details on data sources and copyright information.

## 🔬 Based on Research

This toolkit implements findings from:

- **Bedell, G. & Mang, K. S. (2012)**. "The Applicative Suffix -na in K'cho"
- **Jordan, M. (1969)**. "Chin Dictionary and Grammar"
- **K'cho linguistic research** on verb stem alternation and morphology

## 🎯 What You Can Build

With this toolkit, you can create:

1. **K'cho-English Machine Translation**
   - Generate parallel corpus
   - Export in ML-ready format
   - Train transformer models

2. **K'cho Dictionary App**
   - SQLite backend ready
   - Full-text search
   - Multi-lingual support

3. **Text Analysis Tools**
   - Morphological analyzer
   - Grammar checker
   - Spell checker (with lexicon validation)

4. **Linguistic Research Tools**
   - Annotated corpus
   - Statistical analysis
   - Pattern discovery

5. **Language Learning Apps**
   - Verb conjugation practice
   - Example sentence database
   - Vocabulary lists by frequency

## 📈 Data Quality

Built-in quality control:

- ✅ **Text validation** with confidence scoring
- ✅ **Morphological validation** (checks grammatical structure)
- ✅ **Character set validation** (ensures K'cho characters)
- ✅ **Quality reports** (identifies issues in corpus)

Example:
```python
quality = toolkit.corpus.quality_report()
print(f"Validated: {quality['validated_sentences']}/{quality['total_sentences']}")
print(f"Avg confidence: {quality['avg_confidence']:.2%}")
```

## 🚦 Project Status

**Status**: Production Ready ✅

- ✅ Core features complete
- ✅ Fully documented
- ✅ Example code provided
- ✅ Based on peer-reviewed research
- ✅ No external dependencies

## 🤝 Contributing

To extend the toolkit:

1. **Add vocabulary**: Extend `KchoConfig.VERB_STEMS`
2. **Add patterns**: Update validation patterns
3. **Add languages**: Add more gloss languages to `LexiconEntry`
4. **Report issues**: Document any K'cho linguistic features not yet handled

## 📝 Citation

If you use this toolkit in research, please cite:

```bibtex
@misc{kcho_toolkit_2025,
  title={K'cho Language Toolkit: A Unified Package for K'cho Language Processing},
  author={Based on research by Bedell, George and Mang, Kee Shein},
  year={2025},
  note={Linguistic analysis based on "The Applicative Suffix -na in K'cho" (2012)}
}
```

## ⚠️ Important Notes

- K'cho has **no standard orthography** - this toolkit handles common variants
- The toolkit focuses on **Mindat Township dialect** (southern Chin State)
- Based on research from early 2000s - contemporary usage may vary
- Speaker population: approximately 10,000-20,000

## 🔮 Future Enhancements

Potential additions (not yet implemented):

- [ ] Audio processing (speech recognition/synthesis)
- [ ] Neural morphological analyzer
- [ ] Automatic tokenization improvements
- [ ] More comprehensive verb stem database
- [ ] Integration with existing Chin language tools

## 📞 Support

For K'cho linguistic questions, refer to:
- Published papers by George Bedell and Kee Shein Mang
- Jordan's Chin Dictionary and Grammar (1969)
- K'cho community language documentation

## 📄 License

This toolkit is provided for K'cho language research, documentation, and preservation.

---

**Version**: 1.0.0  
**Language**: K'cho (Kuki-Chin family)  
**Region**: Mindat Township, Southern Chin State, Myanmar  
**Speakers**: ~10,000-20,000

---

## Quick Links

- [Complete Documentation](KCHO_TOOLKIT_DOCS.md)
- [Example Scripts](kcho_examples.py)
- [Source Code](kcho_toolkit.py)

---

*"Preserving K'cho for future generations through technology"*
