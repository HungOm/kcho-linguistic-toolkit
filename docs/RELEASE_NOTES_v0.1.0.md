# K'Cho Linguistic Toolkit v0.1.0 Release Notes

## 🎉 First Official Release: K'Cho Linguistic Toolkit

This is the first official release of the K'Cho Linguistic Toolkit, a comprehensive Python package for linguistic analysis of the K'Cho language. Developed by Hung Om, an enthusiastic K'Cho speaker and independent developer, this toolkit provides essential tools for working with K'Cho, a Kuki-Chin language spoken by 10,000-20,000 people in southern Chin State, Myanmar.

## ✨ What's New

### 🏗️ **Professional Python Package**
- **Clean Package Structure**: Organized into proper `kcho/` package following Python best practices
- **Easy Installation**: Install with `pip install -e .` for development or distribute via PyPI
- **Comprehensive Documentation**: Complete README, API docs, and usage examples
- **Professional .gitignore**: Excludes large files while keeping essential references

### 🚀 **Core Linguistic Analysis Features**
- **Collocation Extraction**: Multiple association measures (PMI, NPMI, t-score, Dice, Log-likelihood)
- **Morphological Analysis**: Analyze K'Cho word structure (stems, affixes, particles)
- **Text Normalization**: Clean and normalize K'Cho text for analysis
- **Corpus Building**: Create annotated datasets with quality control
- **Lexicon Management**: Build and manage digital K'Cho dictionaries
- **Data Export**: Export to standard formats (JSON, CoNLL-U, CSV)

### 🔧 **Advanced defaultdict Functionality**
- **POS Pattern Analysis**: Group collocations by part-of-speech patterns
- **Word Context Analysis**: Analyze word co-occurrence with nested structures
- **Morphological Pattern Extraction**: Identify prefixes, suffixes, and combinations
- **Sentence Structure Analysis**: Analyze syntactic patterns and complexity
- **Efficient Data Grouping**: Automatic list/dict creation without manual initialization

### 📦 **Package Data Organization**
- **Package Data**: Core linguistic knowledge base included in installation
- **External Data**: Large datasets organized separately (Bible translations, parallel corpora)
- **Sample Data**: Small reference files for testing and examples
- **Research Outputs**: Generated analysis results (not included in package)

## 🔧 **Installation & Usage**

### Installation
```bash
# Clone the repository
git clone https://github.com/HungOmungom/kcho-linguistic-toolkit.git
cd kcho-linguistic-toolkit

# Install in development mode
pip install -e .

# Verify installation
python -c "from kcho import CollocationExtractor; print('✅ Success!')"
```

### Basic Usage
```python
from kcho import CollocationExtractor, KchoSystem

# Initialize the system
system = KchoSystem()

# Extract collocations
extractor = CollocationExtractor()
corpus = ["Om noh Yong am paapai pe ci", "Ak'hmó lùum ci"]
results = extractor.extract(corpus)

# Use advanced defaultdict functionality
pos_patterns = system.corpus.analyze_pos_patterns()
word_contexts = extractor.analyze_word_contexts(corpus)
```

### Command Line Interface
```bash
# Extract collocations
python -m kcho.create_gold_standard --corpus data/sample_corpus.txt --output gold_standard.txt

# Run examples
python examples/defaultdict_usage.py
```

## 📁 **Project Structure**

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
├── LICENSE                         # MIT License
└── README.md                       # Documentation
```

## 📊 **Data Organization**

### Package Data (included in installation)
- `kcho/data/linguistic_data.json` - Core linguistic knowledge base
- `kcho/data/word_frequency_top_1000.csv` - High-frequency word list

### External Data (not in package)
- `data/sample_corpus.txt` - Small sample corpus for testing
- `data/gold_standard_collocations.txt` - Gold standard annotations
- `data/bible_versions/` - Bible translations (public domain, large files)
- `data/parallel_corpora/` - Aligned parallel texts
- `data/research_outputs/` - Generated analysis results

**Note**: Large data files are not included in the package to keep it lightweight. See `data/README.md` for details.

## 🧪 **Testing**

The package has been thoroughly tested:

```bash
# Test imports
python -c "from kcho import CollocationExtractor, KchoSystem; print('✅ Imports work!')"

# Test CLI
python -m kcho.create_gold_standard --corpus data/sample_corpus.txt --output test.txt --auto

# Test examples
python examples/defaultdict_usage.py
```

## 📚 **Documentation**

- **[README.md](README.md)** - Complete installation and usage guide
- **[GUIDE.md](GUIDE.md)** - Detailed user guide
- **[KCHO_TOOLKIT_DOCS.md](KCHO_TOOLKIT_DOCS.md)** - API documentation
- **[DEFAULTDICT_INTEGRATION.md](DEFAULTDICT_INTEGRATION.md)** - defaultdict functionality guide
- **[data/README.md](data/README.md)** - Data organization and copyright information

## 🔬 **Research Foundation**

This toolkit implements findings from:
- **Bedell, G. & Mang, K. S. (2012)**. "The Applicative Suffix -na in K'Cho"
- **Jordan, M. (1969)**. "Chin Dictionary and Grammar"
- **Mang, K. S. & Bedell, G. (2006)**. "Relative Clauses in K'Cho"

## 🛠️ **Tools and Libraries Used**

This toolkit builds upon several excellent open-source tools:

- **Python Standard Library** - Core functionality
- **NLTK** - Natural language processing utilities
- **scikit-learn** - Machine learning algorithms
- **NumPy** - Numerical computing
- **Click** - Command-line interface framework
- **Python collections** - defaultdict and Counter for efficient data structures

## 🤝 **Contributing**

We welcome contributions! Please see the documentation for:
- Code style guidelines
- Testing requirements
- Data contribution guidelines
- Issue reporting

## 📄 **License**

MIT License - see LICENSE file for details.

## 🙏 **Acknowledgments**

- **K'Cho language community** - For preserving and sharing the language
- **Linguistic researchers** - Bedell & Mang (2012) for foundational research
- **Open source community** - For the excellent tools and libraries
- **Public domain Bible translation providers** - For making translations available
- **Python community** - For the robust ecosystem and packaging tools

---

**Full Changelog**: This is the first official release of the K'Cho Linguistic Toolkit.

**Download**: [Source Code (zip)](https://github.com/HungOm/kcho-linguistic-toolkit/archive/refs/tags/v0.1.0.zip) | [Source Code (tar.gz)](https://github.com/hungom/kcho-linguistic-toolkit/archive/refs/tags/v0.1.0.tar.gz)