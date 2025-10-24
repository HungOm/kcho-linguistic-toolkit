# K'Cho Language Toolkit - Complete System Guide

**Version 0.2.0**  
**A Comprehensive Toolkit for Low-Resource Language Processing**

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [System Architecture](#system-architecture)
5. [Core Modules](#core-modules)
6. [API Reference](#api-reference)
7. [CLI Commands](#cli-commands)
8. [Collocation Extraction](#collocation-extraction)
9. [Gold Standard Creation](#gold-standard-creation)
10. [Evaluation](#evaluation)
11. [Parallel Corpus Processing](#parallel-corpus-processing)
12. [Research Background](#research-background)
13. [Advanced Usage](#advanced-usage)
14. [Troubleshooting](#troubleshooting)
15. [Contributing](#contributing)
16. [Citation](#citation)
17. [Appendix](#appendix)

---

## Overview

The K'Cho Language Toolkit is a comprehensive Python package for processing K'Cho, a low-resource Tibeto-Burman language spoken in Myanmar. The toolkit provides:

- **Text Normalization**: Unicode handling, diacritics, orthographic variants
- **Tokenization**: Morphology-aware segmentation
- **Collocation Extraction**: Multiple association measures (PMI, t-score, Dice, etc.)
- **Parallel Corpus Processing**: English-K'Cho alignment and training data export
- **Evaluation Framework**: Precision/recall, ranking metrics
- **CLI Tools**: Command-line interface for all major functions

### Key Features

✓ **Low-Resource Optimized**: Works with small corpora (thousands of tokens)  
✓ **Research-Based**: Implements methods from Mang & Bedell (2006-2012)  
✓ **Modular Design**: Import only what you need  
✓ **Extensible**: Pluggable POS taggers, parsers, lemmatizers  
✓ **Production-Ready**: Type hints, logging, error handling  
✓ **PyPI-Ready**: Standard packaging, easy installation  

### Linguistic Background

K'Cho (also spelled K'cho, Khumi, Mro-Khimi) is spoken by 10,000-20,000 people in Mindat Township, southern Chin State, Myanmar. Key features:

- **Verb Stem Alternation**: Stem I (with tense particles) vs. Stem II (in relative clauses)
- **Agreement System**: Subject/object/genitive agreement particles
- **Applicative Suffix**: -na/-nák adds arguments to verbs
- **Postpositions**: Mark grammatical relations (noh, ah, am, etc.)
- **Tonal System**: High, low, rising tones (marked with diacritics)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### From PyPI (Recommended)

```bash
pip install kcho
```

### From Source

```bash
# Clone repository
git clone https://github.com/yourusername/kcho.git
cd kcho

# Install in development mode
pip install -e .

# Or install with dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Check version
python -c "import kcho; print(kcho.__version__)"

# Test CLI
kcho --help
```

### Dependencies

Core dependencies (automatically installed):
- `nltk>=3.7` - Statistical NLP utilities
- `scikit-learn>=1.0` - Machine learning tools
- `click>=8.0` - CLI framework
- `numpy>=1.20` - Numerical computing

Optional dependencies:
- `stanza` - For future POS tagging/parsing
- `sentencepiece` - For subword tokenization
- `pytest` - For running tests

---

## Quick Start

### 30-Second Example

```python
from kcho import normalize, collocation, export

# Sample K'Cho sentences
corpus = [
    "Om noh Yóng am pàapai pe(k) ci.",
    "Yóng am pàapai pe(k) ci ah k'chàang.",
    "Om noh Yóng am a péit ah pàapai.",
]

# Extract collocations
results = collocation.extract(corpus, measures=['pmi', 'tscore'], min_freq=2)

# Export to CSV
export.to_csv(results, 'collocations.csv')

# View results
for measure, collocations in results.items():
    print(f"\n{measure.value} - Top 5:")
    for coll in collocations[:5]:
        print(f"  {' '.join(coll.words)}: {coll.score:.3f}")
```

### CLI Quick Start

```bash
# Extract collocations from corpus
kcho extract-collocations corpus.txt -o collocations.csv

# Normalize text
kcho normalize input.txt -o normalized.txt

# Tokenize text
kcho tokenize input.txt -o tokens.txt

# Evaluate against gold standard
kcho evaluate-collocations predicted.csv gold_standard.txt
```

### Complete Workflow Example

```python
from kcho import KChoSystem

# Initialize system
system = KChoSystem()

# Load corpus
with open('kcho_corpus.txt', 'r', encoding='utf-8') as f:
    corpus = [line.strip() for line in f if line.strip()]

# Extract collocations with multiple measures
results = system.extract_collocations(
    corpus,
    window_size=5,
    min_freq=5,
    measures=['pmi', 'npmi', 'tscore', 'dice']
)

# Export results
system.export_collocations(results, 'output.csv', format='csv', top_k=50)
system.export_collocations(results, 'output.json', format='json')

# Evaluate (if gold standard available)
from kcho.collocation import AssociationMeasure
predicted = results[AssociationMeasure.PMI]
metrics = system.evaluate_collocations(predicted, 'gold_standard.txt')
print(f"Precision: {metrics['precision@10']:.3f}")
print(f"Recall: {metrics['recall@10']:.3f}")
```

---

## System Architecture

### Package Structure

```
kcho/
├── __init__.py              # Package initialization, public API
├── normalize.py             # Text normalization and tokenization
├── collocation.py           # Collocation extraction (core)
├── export.py                # CSV/JSON/text export
├── evaluation.py            # Evaluation metrics
├── kcho_system.py           # High-level system wrapper
├── kcho_app.py              # CLI application
└── eng_kcho_parallel_extractor.py  # Parallel corpus processing

scripts/
├── create_gold_standard.py  # Gold standard creation tool
├── validate_gold_standard.py # Validation tool
├── export_training_csv.py   # Parallel corpus export
└── quickstart_example.py    # Demo script

data/
├── sample_corpus_kcho.txt   # Sample corpus
├── gold_standard_kcho_collocations.txt  # Gold standard
└── parallel/
    ├── english.txt          # English side of parallel corpus
    └── kcho.txt             # K'Cho side of parallel corpus

tests/
├── test_normalize.py
├── test_collocation.py
└── test_integration.py
```

### Data Flow

```
Input Corpus
    ↓
[normalize.py] Text Normalization
    ↓
[normalize.py] Tokenization
    ↓
[collocation.py] Frequency Counting
    ↓
[collocation.py] Association Measure Computation
    ↓
[collocation.py] Ranking & Filtering
    ↓
[export.py] Export (CSV/JSON/TXT)
    ↓
[evaluation.py] Evaluation (optional)
    ↓
Results
```

### Module Dependencies

```
kcho_app.py
    ├── kcho_system.py
    │   ├── normalize.py
    │   ├── collocation.py
    │   │   └── normalize.py
    │   ├── export.py
    │   └── evaluation.py
    └── eng_kcho_parallel_extractor.py
```

---

## Core Modules

### normalize.py

**Purpose**: Text normalization and tokenization for K'Cho

**Key Classes**:
- `KChoNormalizer`: Main normalization class

**Features**:
- Unicode normalization (NFC)
- Tone mark preservation
- Vowel length handling
- Rule-based lemmatization
- Stem variant detection

**Example**:
```python
from kcho.normalize import KChoNormalizer

normalizer = KChoNormalizer(
    preserve_tones=True,    # Keep tone marks
    preserve_length=True,   # Keep vowel length
    lowercase=False         # Don't lowercase (preserves proper nouns)
)

# Normalize text
text = "Om noh Yóng am pàapai pe(k) ci."
normalized = normalizer.normalize_text(text)

# Tokenize
tokens = normalizer.tokenize(text)
# Result: ['Om', 'noh', 'Yóng', 'am', 'pàapai', 'pe', 'k', 'ci']

# Lemmatize
lemma = normalizer.lemmatize_simple('pe(k)')
# Result: 'pe' (removes tense marker)
```

### collocation.py

**Purpose**: Statistical collocation extraction

**Key Classes**:
- `CollocationExtractor`: Main extraction engine
- `CollocationResult`: Result data structure
- `AssociationMeasure`: Enum of available measures

**Association Measures**:
1. **PMI** (Pointwise Mutual Information): Classical measure
2. **NPMI** (Normalized PMI): Bounded [0,1] variant
3. **t-score**: Hypothesis testing
4. **Dice**: Symmetric coefficient
5. **Log-likelihood**: G² statistic

**Example**:
```python
from kcho.collocation import CollocationExtractor, AssociationMeasure

extractor = CollocationExtractor(
    window_size=5,      # Co-occurrence window
    min_freq=5,         # Minimum frequency threshold
    measures=[AssociationMeasure.PMI, AssociationMeasure.TSCORE]
)

# Extract collocations
results = extractor.extract(corpus)

# Access results by measure
pmi_collocations = results[AssociationMeasure.PMI]
for coll in pmi_collocations[:10]:
    print(f"{' '.join(coll.words)}: {coll.score:.3f} (freq={coll.frequency})")

# Detect multi-word expressions
mwe_candidates = extractor.detect_mwe(corpus, min_length=3, max_length=5)
for mwe, score in mwe_candidates[:5]:
    print(f"{' '.join(mwe)}: {score:.3f}")
```

### export.py

**Purpose**: Export collocation results to various formats

**Functions**:
- `to_csv()`: Export to CSV (for spreadsheets, analysis)
- `to_json()`: Export to JSON (for APIs, web apps)
- `to_text()`: Export to plain text (human-readable)

**Example**:
```python
from kcho import export

# Export to CSV
export.to_csv(results, 'collocations.csv', top_k=50)

# Export to JSON (pretty-printed)
export.to_json(results, 'collocations.json', indent=2)

# Export to text (for reading)
export.to_text(results, 'collocations.txt')
```

**CSV Format**:
```csv
word1,word2,measure,score,frequency
pe,ci,pmi,5.2341,25
noh,Yóng,pmi,4.8723,15
```

**JSON Format**:
```json
{
  "pmi": [
    {"words": ["pe", "ci"], "score": 5.2341, "frequency": 25},
    {"words": ["noh", "Yóng"], "score": 4.8723, "frequency": 15}
  ]
}
```

### evaluation.py

**Purpose**: Evaluate collocation extraction against gold standards

**Functions**:
- `compute_precision_recall()`: Basic P/R/F1
- `compute_mean_reciprocal_rank()`: MRR metric
- `compute_average_precision()`: AP metric
- `evaluate_ranking()`: Comprehensive evaluation
- `load_gold_standard()`: Load gold standard files

**Example**:
```python
from kcho.evaluation import evaluate_ranking, load_gold_standard

# Load gold standard
gold_set = load_gold_standard('gold_standard.txt')

# Evaluate predicted collocations
predicted = results[AssociationMeasure.PMI]
metrics = evaluate_ranking(predicted, gold_set, top_k_values=[10, 20, 50])

# Print metrics
print(f"MRR: {metrics['mrr']:.3f}")
print(f"MAP: {metrics['map']:.3f}")
print(f"P@10: {metrics['precision@10']:.3f}")
print(f"R@10: {metrics['recall@10']:.3f}")
print(f"F1@10: {metrics['f1@10']:.3f}")
```

### kcho_system.py

**Purpose**: High-level system wrapper integrating all components

**Key Class**:
- `KChoSystem`: Main system class

**Example**:
```python
from kcho import KChoSystem

system = KChoSystem()

# All-in-one extraction
results = system.extract_collocations(
    corpus,
    window_size=5,
    min_freq=5,
    measures=['pmi', 'tscore']
)

# All-in-one export
system.export_collocations(results, 'output.csv', format='csv', top_k=50)

# All-in-one evaluation
metrics = system.evaluate_collocations(results[AssociationMeasure.PMI], 'gold.txt')
```

---

## API Reference

### normalize Module

#### `normalize_text(text: str, **kwargs) -> str`
Normalize K'Cho text.

**Parameters**:
- `text` (str): Input text
- `preserve_tones` (bool): Keep tone marks (default: True)
- `preserve_length` (bool): Keep vowel length (default: True)
- `lowercase` (bool): Convert to lowercase (default: False)

**Returns**: Normalized text string

**Example**:
```python
normalized = normalize_text("Om noh Yóng am pàapai pe(k) ci.")
```

#### `tokenize(text: str, **kwargs) -> List[str]`
Tokenize K'Cho text.

**Parameters**:
- `text` (str): Input text
- Same kwargs as `normalize_text()`

**Returns**: List of tokens

**Example**:
```python
tokens = tokenize("Om noh Yóng am pàapai pe(k) ci.")
# Result: ['Om', 'noh', 'Yóng', 'am', 'pàapai', 'pe', 'k', 'ci']
```

#### `lemmatize(token: str, **kwargs) -> str`
Lemmatize a K'Cho token (rule-based approximation).

**Parameters**:
- `token` (str): Input token

**Returns**: Lemmatized token

**Example**:
```python
lemma = lemmatize('pe(k)')  # Result: 'pe'
lemma = lemmatize('ka-hngu')  # Result: 'hngu'
```

#### `KChoNormalizer` Class

**Constructor**:
```python
KChoNormalizer(preserve_tones=True, preserve_length=True, lowercase=False)
```

**Methods**:
- `normalize_text(text: str) -> str`
- `tokenize(text: str) -> List[str]`
- `lemmatize_simple(token: str) -> str`
- `get_stem_variants(verb: str) -> Dict[str, any]`

**Attributes**:
- `POSTPOSITIONS`: Set of K'Cho postpositions
- `PARTICLES`: Set of K'Cho particles
- `VERB_SUFFIXES`: Set of verb suffixes
- `AGREEMENT_PREFIXES`: Set of agreement prefixes

### collocation Module

#### `extract(corpus: List[str], **kwargs) -> Dict`
Extract collocations from corpus (module-level function).

**Parameters**:
- `corpus` (List[str]): List of sentences
- `window_size` (int): Co-occurrence window (default: 5)
- `min_freq` (int): Minimum frequency (default: 5)
- `measures` (List[str]): Measure names (default: ['pmi', 'tscore'])

**Returns**: Dictionary mapping AssociationMeasure to List[CollocationResult]

**Example**:
```python
from kcho import collocation
results = collocation.extract(corpus, measures=['pmi', 'dice'], min_freq=3)
```

#### `CollocationExtractor` Class

**Constructor**:
```python
CollocationExtractor(
    normalizer=None,
    window_size=5,
    min_freq=5,
    measures=None,
    pos_tagger=None
)
```

**Methods**:

##### `extract(corpus: List[str]) -> Dict[AssociationMeasure, List[CollocationResult]]`
Extract collocations with configured parameters.

##### `detect_mwe(corpus: List[str], min_length=3, max_length=5, min_score=5.0) -> List[Tuple]`
Detect multi-word expressions.

**Parameters**:
- `corpus`: List of sentences
- `min_length`: Minimum MWE length (default: 3)
- `max_length`: Maximum MWE length (default: 5)
- `min_score`: Minimum PMI score (default: 5.0)

**Returns**: List of (MWE tuple, score) pairs

##### `filter_by_pos(results: List, allowed_patterns: Set[Tuple]) -> List`
Filter by POS patterns (requires pos_tagger).

**Parameters**:
- `results`: List of CollocationResult
- `allowed_patterns`: Set of allowed POS tuples, e.g., {('NOUN', 'VERB')}

**Returns**: Filtered results

##### `extract_with_dependencies(corpus: List[str], dependency_parser=None) -> List`
Extract using dependency relations (requires parser).

#### `CollocationResult` DataClass

**Attributes**:
- `words`: Tuple[str, ...] - The collocation words
- `score`: float - Association score
- `measure`: AssociationMeasure - Measure used
- `frequency`: int - Corpus frequency
- `positions`: List[int] - Corpus positions (optional)

#### `AssociationMeasure` Enum

**Values**:
- `PMI`: Pointwise Mutual Information
- `NPMI`: Normalized PMI
- `TSCORE`: t-score
- `DICE`: Dice coefficient
- `LOG_LIKELIHOOD`: Log-likelihood ratio (G²)

### export Module

#### `to_csv(results, output_path, top_k=None)`
Export to CSV format.

**Parameters**:
- `results`: Dict or List of CollocationResult
- `output_path`: Output file path
- `top_k`: Limit to top K results (optional)

#### `to_json(results, output_path, top_k=None, indent=2)`
Export to JSON format.

**Parameters**:
- `results`: Dict or List of CollocationResult
- `output_path`: Output file path
- `top_k`: Limit to top K results (optional)
- `indent`: JSON indentation (default: 2)

#### `to_text(results, output_path, top_k=None)`
Export to plain text format.

**Parameters**:
- `results`: Dict or List of CollocationResult
- `output_path`: Output file path
- `top_k`: Limit to top K results (optional)

### evaluation Module

#### `compute_precision_recall(predicted, gold_standard, top_k=None) -> Dict`
Compute precision, recall, and F1.

**Parameters**:
- `predicted`: List[CollocationResult]
- `gold_standard`: Set[Tuple[str, ...]]
- `top_k`: Evaluate only top K (optional)

**Returns**: Dict with 'precision', 'recall', 'f1' keys

#### `evaluate_ranking(predicted, gold_standard, top_k_values=[10, 20, 50]) -> Dict`
Comprehensive ranking evaluation.

**Parameters**:
- `predicted`: List[CollocationResult]
- `gold_standard`: Set[Tuple[str, ...]]
- `top_k_values`: List of K values for P@K, R@K

**Returns**: Dict with MRR, MAP, P@K, R@K, F1@K metrics

#### `load_gold_standard(file_path: str) -> Set[Tuple[str, ...]]`
Load gold standard from file.

**Format**: One collocation per line, words space-separated

**Example file**:
```
pe ci
noh Yóng
luum-na ci
```

---

## CLI Commands

### Overview

The K'Cho CLI provides access to all major functions:

```bash
kcho --help
```

### normalize

Normalize K'Cho text.

**Usage**:
```bash
kcho normalize INPUT_FILE [OPTIONS]
```

**Options**:
- `-o, --output PATH`: Output file (default: stdout)

**Example**:
```bash
kcho normalize input.txt -o normalized.txt
```

### tokenize

Tokenize K'Cho text (renamed command to avoid conflict).

**Usage**:
```bash
kcho tokenize-cmd INPUT_FILE [OPTIONS]
```

**Options**:
- `-o, --output PATH`: Output file (default: stdout)

**Example**:
```bash
kcho tokenize-cmd input.txt -o tokens.txt
```

### extract-collocations

Extract collocations from corpus.

**Usage**:
```bash
kcho extract-collocations CORPUS_FILE [OPTIONS]
```

**Options**:
- `-o, --output PATH`: Output file (required)
- `-w, --window INT`: Co-occurrence window (default: 5)
- `-f, --min-freq INT`: Minimum frequency (default: 5)
- `-m, --measures TEXT`: Measures (can specify multiple)
  - Choices: `pmi`, `npmi`, `tscore`, `dice`, `log_likelihood`
- `-k, --top-k INT`: Limit to top K results

**Examples**:
```bash
# Basic extraction (PMI + t-score)
kcho extract-collocations corpus.txt -o collocations.csv

# Multiple measures
kcho extract-collocations corpus.txt -o output.csv \
    -m pmi -m npmi -m dice

# Custom parameters
kcho extract-collocations corpus.txt -o output.csv \
    --window 7 --min-freq 3 --top-k 100

# JSON output (detected from extension)
kcho extract-collocations corpus.txt -o collocations.json
```

### evaluate-collocations

Evaluate against gold standard.

**Usage**:
```bash
kcho evaluate-collocations PREDICTED_FILE GOLD_STANDARD_FILE [OPTIONS]
```

**Options**:
- `-m, --measure TEXT`: Measure to evaluate (default: pmi)
  - Choices: `pmi`, `npmi`, `tscore`, `dice`, `log_likelihood`

**Example**:
```bash
kcho evaluate-collocations predicted.csv gold_standard.txt --measure pmi
```

**Output**:
```
Evaluation Results:
==================================================
mrr: 0.8333
map: 0.7542
precision@10: 0.8000
recall@10: 0.4000
f1@10: 0.5333
precision@20: 0.7500
recall@20: 0.7500
f1@20: 0.7500
```

---

## Collocation Extraction

### Overview

Collocation extraction identifies word pairs (or n-grams) that co-occur more frequently than chance. The toolkit implements multiple statistical measures suitable for low-resource languages.

### Theory

#### Pointwise Mutual Information (PMI)

$$PMI(w_1, w_2) = \log_2 \frac{P(w_1, w_2)}{P(w_1) \cdot P(w_2)}$$

- **Range**: (-∞, +∞)
- **Interpretation**: High PMI = strong association
- **Pros**: Classical, interpretable
- **Cons**: Biased toward rare events

#### Normalized PMI (NPMI)

$$NPMI(w_1, w_2) = \frac{PMI(w_1, w_2)}{-\log_2 P(w_1, w_2)}$$

- **Range**: [0, 1]
- **Interpretation**: 0 = independent, 1 = perfect association
- **Pros**: Comparable across frequency ranges
- **Cons**: Less intuitive than PMI

#### t-score

$$t = \frac{O - E}{\sqrt{O}}$$

where $O$ = observed frequency, $E$ = expected frequency

- **Range**: (-∞, +∞)
- **Interpretation**: High t-score = statistically significant
- **Pros**: Less biased toward rare events than PMI
- **Cons**: Favors high-frequency pairs

#### Dice Coefficient

$$Dice(w_1, w_2) = \frac{2 \cdot f(w_1, w_2)}{f(w_1) + f(w_2)}$$

- **Range**: [0, 1]
- **Interpretation**: 1 = always co-occur
- **Pros**: Symmetric, bounded
- **Cons**: Ignores corpus size

#### Log-Likelihood Ratio (G²)

Based on contingency table, computes:

$$G^2 = 2 \sum O_i \log \frac{O_i}{E_i}$$

- **Range**: [0, +∞)
- **Interpretation**: Higher = more significant
- **Pros**: Robust for sparse data
- **Cons**: Computationally intensive

### Methodology

1. **Tokenization**: Split corpus into tokens
2. **Frequency Counting**: Count unigrams and bigrams
3. **Score Computation**: Calculate association measures
4. **Ranking**: Sort by score (descending)
5. **Filtering**: Apply frequency/score thresholds

### Best Practices

#### Corpus Size

- **Minimum**: 1,000 tokens (pilot studies)
- **Recommended**: 10,000+ tokens (reliable statistics)
- **Ideal**: 100,000+ tokens (comprehensive coverage)

For K'Cho (low-resource):
- Start with available texts (~5,000-20,000 tokens)
- Combine multiple sources (Bible, folktales, elicited data)
- Use multiple measures for robustness

#### Parameter Tuning

**Window Size**:
- `window_size=2`: Adjacent words only (strict collocations)
- `window_size=5`: Standard (recommended for most languages)
- `window_size=10`: Wider context (loose associations)

For K'Cho:
- Use `window_size=5` for general extraction
- Use `window_size=2` for grammatical collocations (verb+particle)

**Minimum Frequency**:
- `min_freq=2`: Include rare events (noisy)
- `min_freq=5`: Standard (balanced)
- `min_freq=10`: Conservative (high-frequency only)

For K'Cho:
- Use `min_freq=3` for small corpora (<10,000 tokens)
- Use `min_freq=5` for standard corpora
- Use `min_freq=10` for large corpora (>50,000 tokens)

**Measure Selection**:
- **PMI**: Best for identifying strong associations (idiomatic expressions)
- **t-score**: Best for frequent, stable collocations
- **Dice**: Best for balanced recall/precision
- **Multiple measures**: Recommended for consensus ranking

### K'Cho-Specific Considerations

#### Grammatical Collocations

K'Cho has predictable grammatical collocations:

1. **Verb + Particle** (VP): `pe ci`, `lo khai`
   - Use `window_size=2`, `min_freq=5`
   - Expect high PMI scores (>3.0)

2. **Postposition + Noun** (PP): `noh Yóng`, `am pàapai`
   - Use `window_size=2`, `min_freq=3`
   - Check for proper nouns (Yóng, Om)

3. **Applicative Construction** (APP): `luum-na ci`, `thah-na ci`
   - Look for `-na` or `-nák` suffix
   - Use lemmatization to normalize variants

4. **Agreement + Verb** (AGR): `a péit`, `ka hngu`
   - Agreement prefixes: ka-, a-, ani-, ami-
   - Check stem alternation (pe vs. péit)

#### Multi-Word Expressions (MWE)

K'Cho MWEs include:

1. **Compound Verbs**: `verb + auxiliary`
   - Example: `lo lo` (come begin = 'begin to come')

2. **Postpositional Phrases**: `P + NP + P`
   - Example: `noh Yóng am` (P Yong to)

3. **Fixed Expressions**: `clause + complementizer`
   - Example: `pe ci ah` (give NF COMP = 'that gave')

Use `detect_mwe()` method with `min_length=3`, `max_length=5`.

### Example Workflow

```python
from kcho.collocation import CollocationExtractor, AssociationMeasure

# Initialize extractor with conservative settings
extractor = CollocationExtractor(
    window_size=5,
    min_freq=5,
    measures=[AssociationMeasure.PMI, AssociationMeasure.TSCORE, AssociationMeasure.DICE]
)

# Extract collocations
results = extractor.extract(corpus)

# Analyze by measure
for measure, collocations in results.items():
    print(f"\n=== {measure.value} ===")
    
    # Top 10 collocations
    for i, coll in enumerate(collocations[:10], 1):
        print(f"{i}. {' '.join(coll.words):<20} "
              f"score={coll.score:>6.3f} freq={coll.frequency:>3}")

# Detect MWEs
mwe_candidates = extractor.detect_mwe(corpus, min_length=3, max_length=4)
print("\n=== Multi-Word Expressions ===")
for mwe, score in mwe_candidates[:10]:
    print(f"{' '.join(mwe):<30} score={score:>6.3f}")

# Export consensus ranking (average of all measures)
from kcho import export
export.to_csv(results, 'collocations_all_measures.csv')
```

---

## Gold Standard Creation

### Overview

A gold standard is a manually curated list of correct collocations used to evaluate extraction systems. High-quality gold standards are critical for reliable evaluation.

### Gold Standard Format

**File Format**: Plain text, one collocation per line

**Syntax**:
```
word1 word2 # category, freq=N, notes
```

**Example**:
```
# K'Cho Gold Standard Collocations

# === VP: Verb + Particle ===
pe ci          # VP, freq=25, give + non-future
lo ci          # VP, freq=20, come + non-future
that ci        # VP, freq=15, beat + non-future

# === PP: Postposition + Noun ===
noh Yóng       # PP, freq=15, postposition + proper_noun
am pàapai      # PP, freq=12, to + flower

# === APP: Applicative Constructions ===
luum-na ci     # APP, freq=8, play-APP + NF
thah-na ci     # APP, freq=6, beat-APP + NF
```

### Categories

| Code | Description | K'Cho Examples |
|------|-------------|----------------|
| VP | Verb + Particle | pe ci, lo khai |
| PP | Postposition + Noun | noh Yóng, am pàapai |
| APP | Applicative | luum-na ci, zèi-na ci |
| AGR | Agreement + Verb | a péit, ka hngu |
| AUX | Auxiliary | lo lo, pe lo |
| COMP | Complementizer | ci ah, khai ah |
| MWE | Multi-word (3+) | pe ci ah, noh Yóng am |
| COMPOUND | Compound Noun | k'am-k'zòi |
| LEX | Lexical | ui htui, khò na |
| DISC | Discourse Marker | cun ah, sin ah |

### Creation Workflow

#### Step 1: Automatic Extraction

```bash
# Extract candidates from corpus
python create_gold_standard.py \
    --corpus sample_corpus_kcho.txt \
    --output auto_candidates.txt \
    --auto \
    --top-k 100
```

This produces a file with automatically detected collocations:

```
# Auto-generated candidates

pe ci          # VP, freq=25, Auto-annotated (high confidence)
noh Yóng       # PP, freq=15, Auto-annotated (high confidence)
luum-na ci     # APP, freq=8, Auto-annotated (high confidence)
ka hngu        # AGR, freq=6, Auto-annotated (medium confidence)
...
```

#### Step 2: Manual Review

```bash
# Interactive annotation
python create_gold_standard.py \
    --corpus sample_corpus_kcho.txt \
    --output manual_gold.txt \
    --interactive \
    --top-k 50
```

Interactive interface:

```
[1/50] Candidate: pe ci
    Score: 8.2341, Frequency: 25
    Examples:
      - Om noh Yóng am pàapai pe(k) ci.
      - Ka teihpüi noh Yóng am pàapai pe(k) ci.
    Decision [y/n/s/q]: y
    Category [VP/PP/APP/AGR/AUX/COMP/MWE/LEX/OTHER]: VP
    Notes (optional): Very common pattern
    ✓ Added to gold standard
```

#### Step 3: Validation

```bash
# Validate gold standard
python validate_gold_standard.py \
    --gold manual_gold.txt \
    --corpus sample_corpus_kcho.txt
```

**Output**:
```
GOLD STANDARD VALIDATION
======================================================================
Total entries: 55

--- Format Validation ---
✓ Format is valid

--- Duplicate Check ---
✓ No duplicates found

--- Corpus Coverage ---
Coverage: 92.7%
Found in corpus: 51
Not found in corpus: 4

Not found (first 10):
  - ka-na khai
  - ng'mìng-na te
  ...

--- Category Distribution ---
  VP              10 ( 18.2%)
  PP               8 ( 14.5%)
  APP              7 ( 12.7%)
  AGR              6 ( 10.9%)
  COMP             4 (  7.3%)
  ...
```

#### Step 4: Inter-Annotator Agreement

For research-quality gold standards:

1. Have 2-3 annotators create independent gold standards
2. Compute agreement score
3. Discuss disagreements
4. Create consensus version

```bash
# Compute IAA
python validate_gold_standard.py \
    --gold annotator1_gold.txt \
    --compare annotator2_gold.txt annotator3_gold.txt \
    --corpus corpus.txt
```

**Target IAA**: >0.70 (good agreement)

### Best Practices

#### Size

- **Pilot Study**: 20-50 entries
- **Standard**: 50-200 entries
- **Comprehensive**: 200-500 entries

For K'Cho:
- Start with 50 high-frequency collocations
- Ensure category balance (not all VP)
- Include both grammatical and lexical collocations

#### Coverage

Ensure gold standard covers:
- Common patterns (high frequency)
- Rare patterns (low frequency, high PMI)
- All major categories
- Authentic corpus examples

#### Quality Control

- ✓ All entries appear in corpus (>90% coverage)
- ✓ Balanced category distribution
- ✓ No duplicates or near-duplicates
- ✓ Clear annotation guidelines
- ✓ Inter-annotator agreement >0.70

### Provided Gold Standard

The toolkit includes `gold_standard_kcho_collocations.txt` with 55 entries:

| Category | Count | Example |
|----------|-------|---------|
| VP | 10 | pe ci, lo ci, that ci |
| PP | 8 | noh Yóng, am pàapai |
| APP | 7 | luum-na ci, thah-na ci |
| AGR | 6 | a péit, ka hngu |
| AUX | 4 | lo lo, pe lo |
| COMP | 4 | ci ah, khai ah |
| MWE | 7 | pe ci ah, noh Yóng am |
| COMPOUND | 2 | k'am-k'zòi |
| LEX | 3 | ui htui, khò na |
| DISC | 2 | cun ah, sin ah |

**Coverage**: ~90% in `sample_corpus_kcho.txt`

---

## Evaluation

### Overview

Evaluation compares predicted collocations against a gold standard using precision, recall, and ranking metrics.

### Metrics

#### Precision, Recall, F1

$$Precision = \frac{TP}{TP + FP}$$

$$Recall = \frac{TP}{TP + FN}$$

$$F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$$

- **Precision**: Fraction of predicted collocations that are correct
- **Recall**: Fraction of gold standard collocations that are found
- **F1**: Harmonic mean of precision and recall

#### Precision@K, Recall@K

Evaluate only the top K predictions.

- **P@10**: Precision of top 10 predictions
- **R@10**: Recall of top 10 predictions
- **F1@10**: F1 score of top 10 predictions

Use for ranking evaluation (most important collocations first).

#### Mean Reciprocal Rank (MRR)

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$

where $rank_i$ is the rank of the first relevant item.

- **Range**: [0, 1]
- **Interpretation**: 1.0 = first prediction is correct
- **Use**: Evaluate if any correct collocation is ranked highly

#### Average Precision (AP)

$$AP = \frac{1}{|R|} \sum_{k=1}^{n} P(k) \cdot rel(k)$$

where $R$ = relevant items, $P(k)$ = precision at rank $k$, $rel(k)$ = 1 if item $k$ is relevant.

- **Range**: [0, 1]
- **Interpretation**: 1.0 = perfect ranking
- **Use**: Evaluate overall ranking quality

### Evaluation Workflow

#### Basic Evaluation

```python
from kcho.evaluation import compute_precision_recall, load_gold_standard

# Load gold standard
gold_set = load_gold_standard('gold_standard_kcho_collocations.txt')

# Load predicted collocations (from extraction)
predicted = results[AssociationMeasure.PMI]

# Compute P/R/F1 for top 10
metrics = compute_precision_recall(predicted, gold_set, top_k=10)
print(f"Precision@10: {metrics['precision']:.3f}")
print(f"Recall@10: {metrics['recall']:.3f}")
print(f"F1@10: {metrics['f1']:.3f}")
```

#### Comprehensive Evaluation

```python
from kcho.evaluation import evaluate_ranking

# Evaluate with multiple K values
metrics = evaluate_ranking(predicted, gold_set, top_k_values=[10, 20, 50])

# Print all metrics
for metric, value in metrics.items():
    print(f"{metric}: {value:.3f}")
```

**Output**:
```
mrr: 0.833
map: 0.754
precision@10: 0.800
recall@10: 0.400
f1@10: 0.533
precision@20: 0.750
recall@20: 0.750
f1@20: 0.750
precision@50: 0.640
recall@50: 1.000
f1@50: 0.781
```

#### Multi-Measure Comparison

```python
# Compare different association measures
for measure, collocations in results.items():
    metrics = evaluate_ranking(collocations, gold_set, top_k_values=[10, 20])
    print(f"\n=== {measure.value} ===")
    print(f"P@10: {metrics['precision@10']:.3f}")
    print(f"R@10: {metrics['recall@10']:.3f}")
    print(f"MRR: {metrics['mrr']:.3f}")
```

### CLI Evaluation

```bash
# Extract collocations
kcho extract-collocations corpus.txt -o predicted.csv

# Evaluate
kcho evaluate-collocations \
    predicted.csv \
    gold_standard_kcho_collocations.txt \
    --measure pmi
```

### Interpretation

#### Good Results

- **P@10 > 0.70**: Most top predictions are correct
- **R@10 > 0.40**: Captures substantial portion of gold standard
- **MRR > 0.80**: First prediction is usually correct
- **MAP > 0.70**: Overall ranking is good

#### Common Issues

**Low Precision**:
- Too many noise/spurious collocations
- Threshold too low (`min_freq` too small)
- **Fix**: Increase `min_freq`, use stricter measure (t-score)

**Low Recall**:
- Missing true collocations
- Threshold too high
- **Fix**: Decrease `min_freq`, use sensitive measure (PMI)

**Low MRR**:
- Correct collocations not ranked highly
- **Fix**: Try different measures, adjust `window_size`

### Baseline Comparison

Compare against frequency baseline:

```python
# Frequency baseline: rank by frequency only
from collections import Counter

bigram_freq = Counter()
for sent in corpus:
    tokens = tokenize(sent)
    for i in range(len(tokens) - 1):
        bigram_freq[(tokens[i], tokens[i+1])] += 1

# Convert to CollocationResult format
freq_baseline = [
    CollocationResult(words=bigram, score=freq, measure=None, frequency=freq)
    for bigram, freq in bigram_freq.most_common()
]

# Evaluate
baseline_metrics = evaluate_ranking(freq_baseline, gold_set)
pmi_metrics = evaluate_ranking(results[AssociationMeasure.PMI], gold_set)

print(f"Baseline P@10: {baseline_metrics['precision@10']:.3f}")
print(f"PMI P@10: {pmi_metrics['precision@10']:.3f}")
```

---

## Parallel Corpus Processing

### Overview

The parallel corpus module processes English-K'Cho sentence-aligned data for machine translation and other tasks.

### Module: eng_kcho_parallel_extractor.py

#### ParallelCorpusExtractor Class

**Purpose**: Extract and align parallel sentences

**Methods**:

##### `load_files(english_path: str, kcho_path: str)`
Load parallel corpus files.

**Format**: Plain text, one sentence per line

##### `align_sentences()`
Align sentences (simple 1:1 alignment).

##### `export_training_data(output_dir: str, force: bool = False)`
Export parallel data to CSV for training.

**Output**: `parallel_training.csv` with columns: `english`, `kcho`

### Usage

#### Python API

```python
from eng_kcho_parallel_extractor import ParallelCorpusExtractor

# Initialize extractor
extractor = ParallelCorpusExtractor()

# Load parallel files
extractor.load_files('english.txt', 'kcho.txt')

# Align sentences
extractor.align_sentences()

# Export to CSV
extractor.export_training_data('training_data/', force=True)
```

#### Standalone Script

```bash
python export_training_csv.py \
    --english-file data/parallel/english.txt \
    --kcho-file data/parallel/kcho.txt \
    --output-dir training_data/ \
    --force
```

**Output**: `training_data/parallel_training.csv`

```csv
english,kcho
"Om gave Yong a flower.","Om noh Yóng am pàapai pe(k) ci."
"The man came.","K'chàang lo(k) ci."
...
```

### Input Format

**english.txt**:
```
Om gave Yong a flower.
The man came.
I see the dog.
```

**kcho.txt**:
```
Om noh Yóng am pàapai pe(k) ci.
K'chàang lo(k) ci.
Ui ka hngu(k) ci.
```

**Requirements**:
- Equal number of lines
- 1:1 sentence alignment
- UTF-8 encoding

### Troubleshooting

#### Empty Output

**Problem**: `export_training_data()` produces empty CSV

**Fixes** (already implemented in fixed version):
```python
# Check 1: Verify files loaded
if not self.english_corpus or not self.kcho_corpus:
    logger.error("No corpus loaded")
    return

# Check 2: Verify alignment ran
if not self.aligned_pairs:
    logger.error("No aligned pairs - run align_sentences() first")
    return

# Check 3: Filter empty pairs
valid_pairs = [(en, kcho) for en, kcho in self.aligned_pairs 
               if en.strip() and kcho.strip()]
```

#### Alignment Errors

For advanced alignment (not implemented by default):
- Use external tools: `atools`, `fast_align`, `eflomal`
- Apply length-based filtering
- Use multilingual embeddings (LASER, LaBSE)

---

## Research Background

### K'Cho Linguistics

K'Cho is a Tibeto-Burman language of the Kuki-Chin branch, closely related to:
- Daai (Chin State, Myanmar)
- Lai/Haka (Chin State, Myanmar)
- Mizo (Mizoram, India)

#### Key Linguistic Features

**1. Verb Stem Alternation**

K'Cho verbs have two stems:
- **Stem I**: Used with tense particles (ci, khai)
- **Stem II**: Used in relative clauses, with applicative -na

| Meaning | Stem I | Stem II | Example Context |
|---------|--------|---------|-----------------|
| play | lùum | luum | lùum ci (plays) vs. luum-na (play-with) |
| beat | that | thah | that ci (beats) vs. thah-na (beat-with) |
| give | pe | péit | pe ci (gives) vs. a péit (he gives) |

**Alternation Patterns**:
- Tone change (low → high): lùum → luum
- Consonant change (t → h): that → thah
- Vowel length (short → long): pe → péit

**2. Agreement System**

K'Cho marks agreement on verbs:

| Person | Subject | Object | Genitive |
|--------|---------|--------|----------|
| 1S | ka- | na | ka |
| 2S | na- | mang | na |
| 3S | a- | - | a |
| 1PL | ka- ... gui | - | ka |
| 2PL | na- ... gui | - | na |
| 3PL | ami- | - | ami |
| 1DL | ka- ... goi | - | ka |
| 3DL | ani- | - | ani |

**Examples**:
```
Ka hngu(k) ci.        (1S see NF) "I see."
A hnguh.              (3S see.STEM2) "He/she sees."
Ami hnguh.            (3PL see.STEM2) "They see."
```

**3. Applicative Suffix -na/-nák**

The suffix -na adds an argument to verbs:

**Intransitive → Transitive**:
```
lùum ci               "play" (intransitive)
k'khìm luum-na ci     "play with knife" (transitive)
```

**Transitive → Ditransitive**:
```
ui that ci            "beat dog" (transitive)
htung ui thah-na ci   "beat dog with stick" (ditransitive)
```

**Stative → Active**:
```
Ka zèi ci.            "I am pleased." (stative)
Om ka zèi-na ci.      "I like Om." (active)
```

**Noun → Verb** (denominal derivation):
```
k'chú "wife"          → k'chú-na "marry, take as wife"
ng'mìng "name"        → ng'mìng-na "name, call"
khò "land"            → khò-na "own land"
```

**4. Postpositions**

K'Cho uses postpositions (not prepositions):

| Postposition | Function | Example |
|--------------|----------|---------|
| noh | Subject of transitive | Om noh (Om SUBJ) |
| am | Indirect object (to) | Yóng am (to Yong) |
| ah | Genitive, complementizer | Om ah (Om's, that Om...) |
| on | With, at | k'khìm on (with knife) |
| ung | At, when | a-ching ung (at that time) |
| tu | With (comitative) | tu bà (with again) |

**5. Relative Clauses**

K'Cho relative clauses are head-external, marked by complementizer `ah`:

```
[Yóng am pàapai pe(k) ci ah] k'chàang
 Yong to flower give NF C    man
"the man that gave Yong a flower"
```

**Key patterns**:
- Subject relativization: Stem I verb, no agreement
- Object relativization: Stem II verb, subject agreement
- Headless relatives: Just clause + ah (no overt head)

### Research Papers

The toolkit is based on the following research:

**1. Mang & Bedell (2006): "Relative Clauses in K'Cho"**
- SEALS XVI, Jakarta
- Describes relative clause syntax
- Documents verb stem alternation
- Provides authentic examples from corpus

**2. Bedell & Mang (2012): "The Applicative Suffix -na in K'Cho"**
- Language in India, Vol. 12:1
- Describes applicative constructions
- Documents -na attachment patterns
- Analyzes noun-to-verb derivation

**3. Nolan (2002): "Spelling and Alphabet in K'Cho"**
- ICU Asian Cultural Studies 28:127-138
- Documents orthography conventions
- Describes tone and vowel length marking
- Provides phonological analysis

**4. Nolan (2003): "Verb Alternation in K'Chò"**
- ICSTLL 36, Melbourne
- Detailed analysis of stem alternation
- Documents tone/consonant/vowel changes
- Provides morphophonological rules

**5. Mang (2006): MA Thesis**
- "An Explanation of Verb Stem Alternation in K'cho"
- Payap University
- Comprehensive morphological analysis
- Lexicon of verb stems

**6. Ng'thu-k'Thaì (2001)**
- K'Cho New Testament translation
- Largest available K'Cho corpus
- Authentic contemporary language
- Basis for gold standard examples

### Collocation Research

The collocation module implements methods from:

**1. Manning & Schütze (1999): "Foundations of Statistical NLP"**
- Classical association measures (PMI, t-score, χ²)
- Statistical significance testing
- Low-resource considerations

**2. Evert (2008): "Corpora and Collocations"**
- Comprehensive survey of measures
- Evaluation methodologies
- Computational implementation

**3. Pecina (2010): "Lexical Association Measures and Collocation Extraction"**
- Comparative evaluation of 82 measures
- Machine learning approaches
- Task-specific optimization

**4. Ramisch et al. (2013): "A Generic Framework for Multiword Expressions Treatment"**
- MWE detection algorithms
- Evaluation frameworks
- Cross-lingual approaches

---

## Advanced Usage

### Custom Normalization

Create a custom normalizer:

```python
from kcho.normalize import KChoNormalizer

class CustomKChoNormalizer(KChoNormalizer):
    """Custom normalizer with additional rules."""
    
    def lemmatize_simple(self, token: str) -> str:
        """Enhanced lemmatization with custom rules."""
        token = super().lemmatize_simple(token)
        
        # Custom rule: Handle specific verb forms
        if token.endswith('loo'):
            token = 'lo'  # Normalize auxiliary variant
        
        return token

# Use custom normalizer
normalizer = CustomKChoNormalizer()
extractor = CollocationExtractor(normalizer=normalizer)
```

### POS Tagging Integration

Integrate a POS tagger (if available):

```python
def kcho_pos_tagger(word: str) -> str:
    """Simple rule-based POS tagger (example)."""
    # Particles
    if word in {'ci', 'khai', 'ne', 'te'}:
        return 'PART'
    # Postpositions
    elif word in {'noh', 'ah', 'am', 'on'}:
        return 'ADP'
    # Agreement prefixes
    elif word in {'ka', 'a', 'ani', 'ami'}:
        return 'AGR'
    else:
        return 'WORD'  # Default

# Use with extractor
extractor = CollocationExtractor(pos_tagger=kcho_pos_tagger)

# Filter by POS patterns
allowed_patterns = {
    ('WORD', 'PART'),  # Verb + Particle
    ('ADP', 'WORD'),   # Postposition + Noun
}
filtered = extractor.filter_by_pos(results[AssociationMeasure.PMI], allowed_patterns)
```

### Batch Processing

Process multiple corpora:

```python
import os
from pathlib import Path

corpus_dir = Path('corpora/')
output_dir = Path('results/')
output_dir.mkdir(exist_ok=True)

for corpus_file in corpus_dir.glob('*.txt'):
    print(f"Processing {corpus_file.name}...")
    
    # Load corpus
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]
    
    # Extract collocations
    system = KChoSystem()
    results = system.extract_collocations(corpus, measures=['pmi', 'tscore'])
    
    # Export
    output_file = output_dir / f"{corpus_file.stem}_collocations.csv"
    system.export_collocations(results, output_file, format='csv', top_k=50)
    
    print(f"  → Saved to {output_file}")
```

### Cross-Validation

Evaluate with cross-validation:

```python
from sklearn.model_selection import KFold
import numpy as np

# Load gold standard
gold_data = load_gold_standard('gold_standard.txt')
gold_list = list(gold_data)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
precisions = []

for train_idx, test_idx in kf.split(gold_list):
    # Split gold standard
    train_gold = {gold_list[i] for i in train_idx}
    test_gold = {gold_list[i] for i in test_idx}
    
    # Train: Use train_gold to tune parameters (if applicable)
    # Test: Evaluate on test_gold
    
    # Extract collocations
    results = system.extract_collocations(corpus)
    predicted = results[AssociationMeasure.PMI]
    
    # Evaluate
    metrics = compute_precision_recall(predicted, test_gold, top_k=10)
    precisions.append(metrics['precision'])

print(f"Mean Precision@10: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
```

### Export for Web Application

Export to JSON for web API:

```python
import json

# Extract collocations
results = system.extract_collocations(corpus)

# Convert to web-friendly format
web_data = {
    'meta': {
        'corpus_size': len(corpus),
        'total_collocations': sum(len(colls) for colls in results.values()),
        'measures': [m.value for m in results.keys()]
    },
    'collocations': {}
}

for measure, collocations in results.items():
    web_data['collocations'][measure.value] = [
        {
            'words': list(coll.words),
            'score': round(coll.score, 4),
            'frequency': coll.frequency,
            'rank': i + 1
        }
        for i, coll in enumerate(collocations[:100])
    ]

# Save
with open('api_output.json', 'w', encoding='utf-8') as f:
    json.dump(web_data, f, ensure_ascii=False, indent=2)
```

### Integration with Stanza

Future integration with Stanza for POS/dependency parsing:

```python
# Requires: pip install stanza
# Note: No K'Cho model exists yet; this is for future use

import stanza

# Download related language model (e.g., Burmese)
# stanza.download('my')  # Myanmar/Burmese

# Load model
# nlp = stanza.Pipeline('my', processors='tokenize,pos,lemma,depparse')

# Process K'Cho (cross-lingual transfer - experimental)
# doc = nlp(kcho_text)

# Extract dependencies
# for sentence in doc.sentences:
#     for word in sentence.words:
#         print(f"{word.text}\t{word.upos}\t{word.head}\t{word.deprel}")
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**:
```
ModuleNotFoundError: No module named 'kcho'
```

**Solution**:
```bash
# Reinstall package
pip install -e .

# Or check Python path
python -c "import sys; print(sys.path)"
```

#### 2. Empty Collocation Results

**Problem**: `extract()` returns empty dictionary

**Causes**:
- Corpus too small
- `min_freq` threshold too high
- Empty/malformed input

**Solution**:
```python
# Check corpus size
print(f"Corpus size: {len(corpus)} sentences")
print(f"First sentence: {corpus[0]}")

# Lower threshold
extractor.min_freq = 2  # Instead of 5

# Check tokenization
tokens = normalizer.tokenize(corpus[0])
print(f"Tokens: {tokens}")
```

#### 3. Unicode/Encoding Errors

**Problem**:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte...
```

**Solution**:
```python
# Force UTF-8 encoding
with open('corpus.txt', 'r', encoding='utf-8', errors='replace') as f:
    corpus = f.readlines()

# Or detect encoding
import chardet
with open('corpus.txt', 'rb') as f:
    raw = f.read()
    detected = chardet.detect(raw)
    print(f"Detected encoding: {detected['encoding']}")
```

#### 4. Parallel Extractor Empty Output

**Problem**: `export_training_data()` produces empty CSV

**Solution** (already fixed in code):
```python
# Verify files loaded
if not extractor.english_corpus:
    print("English corpus not loaded!")
    extractor.load_files('english.txt', 'kcho.txt')

# Verify alignment
if not extractor.aligned_pairs:
    print("No aligned pairs!")
    extractor.align_sentences()

# Check output with force=True
extractor.export_training_data('output/', force=True)
```

#### 5. CLI Command Not Found

**Problem**:
```
bash: kcho: command not found
```

**Solution**:
```bash
# Check installation
pip list | grep kcho

# Reinstall with CLI
pip install -e .

# Or use module syntax
python -m kcho.kcho_app --help
```

### Performance Issues

#### Slow Extraction

For large corpora (>100,000 tokens):

```python
# Use subset for testing
test_corpus = corpus[:1000]  # First 1000 sentences

# Optimize parameters
extractor.window_size = 3     # Smaller window
extractor.min_freq = 10       # Higher threshold
extractor.measures = [AssociationMeasure.PMI]  # Single measure

# Process in batches
batch_size = 5000
for i in range(0, len(corpus), batch_size):
    batch = corpus[i:i+batch_size]
    results = extractor.extract(batch)
    # Process results...
```

#### Memory Issues

For very large corpora:

```python
# Use generator instead of list
def corpus_generator(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                yield line.strip()

# Process incrementally
from collections import Counter
unigram_freq = Counter()
bigram_freq = Counter()

for sentence in corpus_generator('large_corpus.txt'):
    tokens = normalizer.tokenize(sentence)
    unigram_freq.update(tokens)
    for i in range(len(tokens) - 1):
        bigram_freq[(tokens[i], tokens[i+1])] += 1
```

### Debugging

Enable detailed logging:

```python
import logging

# Set logging level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now run extraction
results = extractor.extract(corpus)
```

**Output**:
```
2025-01-15 10:30:45 - kcho.collocation - INFO - Extracting collocations from 95 sentences
2025-01-15 10:30:45 - kcho.collocation - DEBUG - Unigrams: 450, Bigrams: 385
2025-01-15 10:30:45 - kcho.collocation - DEBUG - Computing PMI for 385 bigrams
2025-01-15 10:30:46 - kcho.collocation - INFO - Extracted 156 collocations
```

---

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/kcho.git
cd kcho

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pre-commit install
```

### Code Style

Follow PEP 8 with these conventions:
- Line length: 100 characters (not 80)
- Use type hints for function signatures
- Docstrings: Google style
- Comments: Explain "why", not "what"

**Example**:
```python
def extract(corpus: List[str], 
            window_size: int = 5,
            min_freq: int = 5) -> Dict[AssociationMeasure, List[CollocationResult]]:
    """
    Extract collocations from K'Cho corpus.
    
    Args:
        corpus: List of sentences (one sentence per string)
        window_size: Co-occurrence window size (default: 5)
        min_freq: Minimum frequency threshold (default: 5)
    
    Returns:
        Dictionary mapping measures to ranked collocation lists
        
    Raises:
        ValueError: If corpus is empty or invalid
    
    Example:
        >>> results = extract(corpus, window_size=7, min_freq=3)
        >>> pmi_collocations = results[AssociationMeasure.PMI]
    """
    # Implementation...
```

### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=kcho tests/

# Run specific test
pytest tests/test_collocation.py::test_pmi
```

### Adding New Features

#### Adding a New Association Measure

1. Add to `AssociationMeasure` enum:
```python
class AssociationMeasure(Enum):
    # ... existing measures ...
    CHI_SQUARE = "chi_square"
```

2. Implement computation method:
```python
def _chi_square(self, freq_1: int, freq_2: int, freq_12: int) -> float:
    """
    Chi-square test.
    
    χ² = N * (O - E)² / E
    
    where N = corpus size, O = observed, E = expected
    """
    # Implementation...
```

3. Add to `_compute_measure()`:
```python
elif measure == AssociationMeasure.CHI_SQUARE:
    score = self._chi_square(freq_1, freq_2, freq_12)
```

4. Add tests:
```python
def test_chi_square():
    extractor = CollocationExtractor(measures=[AssociationMeasure.CHI_SQUARE])
    results = extractor.extract(sample_corpus)
    assert AssociationMeasure.CHI_SQUARE in results
```

#### Adding a New Export Format

1. Create export function in `export.py`:
```python
def to_xml(results, output_path, top_k=None):
    """Export to XML format."""
    # Implementation...
```

2. Update `KChoSystem.export_collocations()`:
```python
elif format == 'xml':
    to_xml(results, output_path, top_k=top_k)
```

3. Add CLI option:
```python
@click.option('--format', type=click.Choice(['csv', 'json', 'text', 'xml']))
```

### Documentation

Update documentation for:
- New features
- API changes
- Breaking changes
- Examples

**Docstring Template**:
```python
def new_function(param1: Type1, param2: Type2 = default) -> ReturnType:
    """
    Brief description (one line).
    
    Longer description explaining:
    - What the function does
    - When to use it
    - Important caveats
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: value)
    
    Returns:
        Description of return value
        
    Raises:
        ErrorType: When this error occurs
    
    Example:
        >>> result = new_function(value1, value2)
        >>> print(result)
        Expected output
        
    References:
        - Paper citation (if applicable)
        - Related functions
    """
```

### Pull Request Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Make changes with tests
4. Run tests: `pytest tests/`
5. Run linting: `flake8 kcho/`
6. Commit with clear message: `git commit -m "Add new feature: X"`
7. Push to fork: `git push origin feature/new-feature`
8. Create pull request with:
   - Description of changes
   - Motivation/use case
   - Test results
   - Breaking changes (if any)

---

## Citation

If you use this toolkit in your research, please cite:

### Toolkit

```bibtex
@software{kcho2025,
  title = {K'Cho Language Toolkit: A Comprehensive Package for Low-Resource NLP},
  author = {K'Cho Research Team},
  year = {2025},
  version = {0.2.0},
  url = {https://github.com/yourusername/kcho},
  note = {Python package for K'Cho language processing}
}
```

### Research Papers

```bibtex
@inproceedings{mang2006relative,
  title = {Relative Clauses in K'Cho},
  author = {Mang, Kee Shein and Bedell, George},
  booktitle = {Papers from the 16th Annual Meeting of the Southeast Asian Linguistics Society},
  pages = {21--34},
  year = {2006},
  publisher = {Pacific Linguistics},
  address = {Canberra}
}

@article{bedell2012applicative,
  title = {The Applicative Suffix -na in K'cho},
  author = {Bedell, George and Mang, Kee Shein},
  journal = {Language in India},
  volume = {12},
  number = {1},
  pages = {51--69},
  year = {2012}
}

@mastersthesis{mang2006thesis,
  title = {An Explanation of Verb Stem Alternation in K'cho, a Chin Language},
  author = {Mang, Kee Shein},
  school = {Payap University},
  year = {2006}
}
```

---

## Appendix

### A. K'Cho Orthography Reference

#### Tone Marks

| Tone | Mark | Example | Meaning |
|------|------|---------|---------|
| High | (none) | pe | give |
| Low | ` (grave) | lùum | play |
| Rising | ´ (acute) | Yóng | (name) |

#### Vowel Length

| Short | Long | Example |
|-------|------|---------|
| a | aa | that / thaat |
| e | ee | pe / pee |
| i | ii | ni / nii |
| o | oo | lo / loo |
| u | uu | lùum / luum |

#### Glottal Stops

| Position | Notation | Example |
|----------|----------|---------|
| Prefixed | k', m', ng' | k'am, m'hnii, ng'ai |
| Final | h | that (t) vs. thah (h) |

#### Common Words

| K'Cho | English | Category |
|-------|---------|----------|
| noh | P (subject) | Postposition |
| ah | P (genitive/COMP) | Postposition |
| am | to | Postposition |
| ci | non-future | Particle |
| khai | future | Particle |
| ka | 1S | Agreement |
| a | 3S | Agreement |
| pe | give | Verb |
| lo | come | Verb |
| that | beat | Verb |
| hngu | see | Verb |
| k'chàang | man | Noun |
| k'hngumí | woman | Noun |
| pàapai | flower | Noun |

### B. File Format Specifications

#### Corpus File Format

**Filename**: `*.txt` (UTF-8 encoding)

**Format**:
```
# Comments start with #
# Empty lines are ignored

Sentence 1 here.
Sentence 2 here.
...
```

**Requirements**:
- UTF-8 encoding
- One sentence per line
- Comments start with `#`
- Empty lines allowed (ignored)

#### Gold Standard Format

**Filename**: `gold_standard*.txt`

**Format**:
```
# Section header
word1 word2 # category, freq=N, notes
```

**Example**:
```
# === VP: Verb + Particle ===
pe ci          # VP, freq=25, give + non-future
lo ci          # VP, freq=20, come + non-future

# === PP: Postposition + Noun ===
noh Yóng       # PP, freq=15, postposition + name
```

#### CSV Export Format

**Columns**: `word1`, `word2`, `measure`, `score`, `frequency`

**Example**:
```csv
word1,word2,measure,score,frequency
pe,ci,pmi,5.2341,25
noh,Yóng,pmi,4.8723,15
luum-na,ci,pmi,6.1234,8
```

#### JSON Export Format

**Structure**:
```json
{
  "measure_name": [
    {
      "words": ["word1", "word2"],
      "score": 5.2341,
      "frequency": 25
    }
  ]
}
```

### C. Association Measure Reference

| Measure | Range | Best For | Bias |
|---------|-------|----------|------|
| PMI | (-∞, +∞) | Strong associations | Rare events |
| NPMI | [0, 1] | Normalized comparison | Rare events |
| t-score | (-∞, +∞) | Frequent collocations | High frequency |
| Dice | [0, 1] | Balanced P/R | Symmetric |
| Log-likelihood | [0, +∞) | Significance testing | Robust |

**Selection Guide**:
- **Idioms/Fixed expressions**: Use PMI (high scores)
- **Grammatical collocations**: Use t-score (stable, frequent)
- **Balanced evaluation**: Use Dice or NPMI
- **Statistical significance**: Use Log-likelihood

### D. Evaluation Metrics Reference

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| Precision | TP/(TP+FP) | [0, 1] | Accuracy of predictions |
| Recall | TP/(TP+FN) | [0, 1] | Coverage of gold standard |
| F1 | 2PR/(P+R) | [0, 1] | Harmonic mean of P & R |
| MRR | 1/rank | [0, 1] | First relevant rank |
| MAP | avg(P@k) | [0, 1] | Overall ranking quality |

**Good Scores**:
- Precision > 0.70
- Recall > 0.40 (for top-K)
- F1 > 0.50
- MRR > 0.80
- MAP > 0.70

### E. Sample Commands Reference

```bash
# Extract collocations (basic)
kcho extract-collocations corpus.txt -o output.csv

# Extract with custom parameters
kcho extract-collocations corpus.txt -o output.csv \
    --window 7 --min-freq 3 -m pmi -m tscore -m dice --top-k 50

# Normalize text
kcho normalize input.txt -o normalized.txt

# Tokenize text
kcho tokenize-cmd input.txt -o tokens.txt

# Evaluate extraction
kcho evaluate-collocations predicted.csv gold_standard.txt

# Export parallel corpus
python export_training_csv.py \
    --english-file english.txt \
    --kcho-file kcho.txt \
    --output-dir training_data/

# Create gold standard (auto)
python create_gold_standard.py \
    --corpus corpus.txt \
    --output gold.txt \
    --auto

# Create gold standard (interactive)
python create_gold_standard.py \
    --corpus corpus.txt \
    --output gold.txt \
    --interactive

# Validate gold standard
python validate_gold_standard.py \
    --gold gold.txt \
    --corpus corpus.txt
```

---

## License

MIT License

Copyright (c) 2025 K'Cho Language Research Team

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Contact

For questions, issues, or contributions:

- **GitHub**: https://github.com/yourusername/kcho
- **Issues**: https://github.com/yourusername/kcho/issues
- **Email**: kcho-toolkit@example.com
- **Documentation**: https://kcho.readthedocs.io

---

**End of K'Cho Language Toolkit System Guide v0.2.0**

*Last updated: January 2025*