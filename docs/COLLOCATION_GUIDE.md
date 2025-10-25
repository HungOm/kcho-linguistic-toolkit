# K'Cho Collocation Extraction Guide

This guide explains how to use the K'Cho Linguistic Toolkit's comprehensive collocation extraction features.

## Overview

The toolkit can extract collocations from **any K'Cho text file** (txt or json) using:
- **Statistical measures**: PMI, T-Score, Dice, Log-Likelihood, NPMI
- **Linguistic patterns**: VP, PP, APP, AGR, AUX, COMP, MWE, LEX, COMPOUND, DISC
- **Multiple output formats**: CSV, JSON, TXT
- **Configurable settings**: via config file and CLI arguments

## Supported File Formats

### Text Files (.txt)
Plain text files with one sentence per line:
```
Om noh Yóng am pàapai pe ci.
Ak'hmó lùum ci.
Yóng am pàapai a péit ah k'hngumí ka hngu ci.
```

### JSON Files (.json)
The toolkit automatically detects JSON structure:

**Bible JSON Structure** (example of nested JSON):
```json
{
  "GEN": {
    "GEN_1": {
      "1": "Akhtük bäih ah Khanpughi naw khankho la khomdek cu tüisho ci.",
      "2": "Khomdek cu a i ah kä ng'shing ne, kho k'häu ah kya ci."
    }
  }
}
```

**Simple JSON Structure**:
```json
[
  "Om noh Yóng am pàapai pe ci.",
  "Ak'hmó lùum ci."
]
```

## Statistical Measures

### 1. Pointwise Mutual Information (PMI)
- **Formula**: PMI(x,y) = log₂(P(x,y) / (P(x) × P(y)))
- **Range**: -∞ to +∞
- **Interpretation**: Higher values indicate stronger association
- **Use case**: General collocation strength

### 2. Normalized PMI (NPMI)
- **Formula**: NPMI(x,y) = PMI(x,y) / -log₂(P(x,y))
- **Range**: -1 to +1
- **Interpretation**: Normalized version of PMI, easier to compare
- **Use case**: Comparing collocations across different frequency ranges

### 3. T-Score
- **Formula**: t = (f(x,y) - f(x)f(y)/N) / √f(x,y)
- **Range**: -∞ to +∞
- **Interpretation**: Statistical significance test
- **Use case**: Filtering statistically significant collocations

### 4. Dice Coefficient
- **Formula**: Dice(x,y) = 2f(x,y) / (f(x) + f(y))
- **Range**: 0 to 1
- **Interpretation**: Symmetric measure, higher values = stronger association
- **Use case**: Balanced frequency consideration

### 5. Log-Likelihood Ratio (G²)
- **Formula**: 2 × Σ(O × log(O/E))
- **Range**: 0 to +∞
- **Interpretation**: Asymptotic significance test
- **Use case**: Large corpus analysis

## Linguistic Patterns

### 1. Verb + Particle (VP)
- **Pattern**: Verb followed by particle
- **Examples**: `pe ci`, `lo ci`, `that ci`
- **K'Cho Feature**: Core grammatical pattern

### 2. Postposition + Noun (PP)
- **Pattern**: Postposition followed by noun
- **Examples**: `noh Yóng`, `am pàapai`, `noh k'chàang`
- **K'Cho Feature**: Postpositional phrases

### 3. Applicative Construction (APP)
- **Pattern**: Verb with applicative suffix + particle
- **Examples**: `luum-na ci`, `thah-na ci`, `zèi-na ci`
- **K'Cho Feature**: Key morphological pattern

### 4. Agreement + Verb (AGR)
- **Pattern**: Agreement marker + verb
- **Examples**: `a péit`, `ka hngu`, `ami hnguh`
- **K'Cho Feature**: Subject/object agreement

### 5. Auxiliary Construction (AUX)
- **Pattern**: Main verb + auxiliary verb
- **Examples**: `lo lo`, `pe lo`, `that lo`
- **K'Cho Feature**: Aspectual marking

### 6. Complementizer Pattern (COMP)
- **Pattern**: Particle + complementizer
- **Examples**: `ci ah`, `khai ah`, `te ah`
- **K'Cho Feature**: Clause linking

### 7. Multi-Word Expression (MWE)
- **Pattern**: 3+ word fixed expressions
- **Examples**: `pe ci ah`, `noh Yóng am`, `Om noh Yóng`
- **K'Cho Feature**: Complex grammatical constructions

### 8. Lexical Collocation (LEX)
- **Pattern**: Semantically motivated word pairs
- **Examples**: `ui htui`, `khò na`, `àihli ng'dáng`
- **K'Cho Feature**: Idiomatic expressions

### 9. Compound Noun (COMPOUND)
- **Pattern**: Hyphenated compound nouns
- **Examples**: `k'am-k'zòi`, `k'chàang-k'ní`
- **K'Cho Feature**: Word formation

### 10. Discourse Marker (DISC)
- **Pattern**: Discourse linking elements
- **Examples**: `cun ah`, `sin ah`
- **K'Cho Feature**: Textual cohesion

## Configuration

### Configuration File (config.yaml)
```yaml
data_sources:
  custom_corpus: "path/to/your/corpus.txt"
  sample_corpus: "data/sample_corpus.txt"

collocation:
  window_size: 5
  min_freq: 5
  measures: [pmi, tscore, dice, log_likelihood, npmi]
  linguistic_patterns: [VP, PP, APP, AGR, AUX, COMP, MWE]

output:
  formats: [csv, json, txt]
  output_dir: "output"
```

### Environment Variables
```bash
export KCHO_CUSTOM_CORPUS="path/to/your/corpus.txt"
export KCHO_WINDOW_SIZE="3"
export KCHO_MIN_FREQ="10"
export KCHO_OUTPUT_DIR="results"
```

## CLI Usage

### Basic Extraction
```bash
python -m kcho.kcho_app extract-all-collocations \
  --input data/sample_corpus.txt \
  --output results/sample_collocations \
  --format csv json txt
```

### Custom Settings
```bash
python -m kcho.kcho_app extract-all-collocations \
  --input your_corpus.txt \
  --output results/my_collocations \
  --window-size 3 \
  --min-freq 10 \
  --measures pmi tscore dice \
  --patterns VP PP APP \
  --verbose
```

### With Configuration File
```bash
python -m kcho.kcho_app extract-all-collocations \
  --input your_corpus.json \
  --output results/collocations \
  --config my_config.yaml \
  --format csv json txt
```

### With Gold Standard Evaluation
```bash
python -m kcho.kcho_app extract-all-collocations \
  --input data/sample_corpus.txt \
  --output results/evaluated_collocations \
  --gold-standard data/gold_standard_collocations.txt \
  --format csv json txt
```

## Output Formats

### CSV Format
```csv
words,statistical_measure,score,frequency,linguistic_pattern,pattern_confidence,context_examples
pe ci,PMI,8.234,45,verb_particle,0.95,"Om noh Yóng am pàapai pe ci."
noh Yóng,TSCORE,12.456,32,postposition_phrase,0.88,"Om noh Yóng am pàapai pe ci."
```

### JSON Format
```json
{
  "metadata": {
    "input_file": "your_corpus.json",
    "total_sentences": 1000,
    "statistical_measures": ["pmi", "tscore", "dice", "log_likelihood", "npmi"],
    "linguistic_patterns": ["VP", "PP", "APP", "AGR", "AUX", "COMP", "MWE"]
  },
  "collocations": [
    {
      "words": ["pe", "ci"],
      "statistical_measures": {
        "pmi": 8.234,
        "tscore": 12.456,
        "dice": 0.789
      },
      "frequency": 45,
      "linguistic_pattern": "verb_particle",
      "pattern_confidence": 0.95,
      "context_examples": ["Om noh Yóng am pàapai pe ci."]
    }
  ]
}
```

### Text Format
```
=== K'Cho Collocation Analysis ===
Input: your_corpus.json
Total sentences: 1,000

=== VERB + PARTICLE (VP) ===
1. pe ci (PMI: 8.234, freq: 45)
   - "Om noh Yóng am pàapai pe ci."

=== POSTPOSITION + NOUN (PP) ===
1. noh Yóng (T-Score: 12.456, freq: 32)
   - "Om noh Yóng am pàapai pe ci."
```

## Python API Usage

### Basic Extraction
```python
from kcho.text_loader import TextLoader
from kcho.collocation import CollocationExtractor, AssociationMeasure

# Load text
sentences = TextLoader.load_from_file("your_corpus.json")

# Create extractor
extractor = CollocationExtractor(
    window_size=5,
    min_freq=5,
    measures=[AssociationMeasure.PMI, AssociationMeasure.TSCORE]
)

# Extract collocations
results = extractor.extract(sentences)
```

### With Linguistic Patterns
```python
# Extract with pattern classification
results = extractor.extract_with_patterns(sentences)

# Analyze results by pattern
for measure, collocations in results.items():
    for coll in collocations:
        if coll.linguistic_pattern:
            print(f"{' '.join(coll.words)}: {coll.linguistic_pattern.value} (confidence: {coll.pattern_confidence})")
```

### Configuration-Based Extraction
```python
from kcho.config import load_config

# Load configuration
config = load_config("config.yaml")

# Create extractor with config
measure_enums = [AssociationMeasure[m.upper()] for m in config.collocation.measures]
extractor = CollocationExtractor(
    window_size=config.collocation.window_size,
    min_freq=config.collocation.min_freq,
    measures=measure_enums
)

# Extract with patterns
results = extractor.extract_with_patterns(sentences, config.collocation.linguistic_patterns)
```

## Best Practices

### 1. Choose Appropriate Measures
- **PMI**: General collocation strength
- **T-Score**: Statistical significance
- **Dice**: Balanced frequency consideration
- **Log-Likelihood**: Large corpus analysis

### 2. Set Frequency Thresholds
- **Small corpus** (< 1000 sentences): min_freq = 2-3
- **Medium corpus** (1000-10000 sentences): min_freq = 5-10
- **Large corpus** (> 10000 sentences): min_freq = 10-20

### 3. Window Size Selection
- **Default**: window_size = 5
- **Tight collocations**: window_size = 3
- **Loose associations**: window_size = 7-10

### 4. Pattern Analysis
- Focus on **VP** and **PP** patterns for core grammar
- **APP** patterns for morphological analysis
- **MWE** patterns for complex constructions

### 5. Output Format Selection
- **CSV**: For spreadsheet analysis
- **JSON**: For programmatic processing
- **TXT**: For human reading

## Troubleshooting

### Common Issues

1. **Empty results**: Check min_freq threshold
2. **Too many results**: Increase min_freq or window_size
3. **JSON parsing errors**: Verify JSON structure
4. **Memory issues**: Process corpus in chunks

### Performance Tips

1. **Large files**: Use streaming processing
2. **Multiple measures**: Process in parallel
3. **Pattern classification**: Cache gold standard patterns
4. **Output**: Use compressed formats for large results

## Examples

See `examples/extract_all_collocations_example.py` for comprehensive usage examples.

## References

- Manning, C. D., & Schütze, H. (1999). Foundations of Statistical Natural Language Processing
- Evert, S. (2008). Corpora and collocations
- Pecina, P. (2010). Lexical association measures and collocation extraction
- Mang, K. S., & Bedell, G. (2006). K'Cho linguistic patterns
