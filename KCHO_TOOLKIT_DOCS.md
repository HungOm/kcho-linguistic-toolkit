# K'cho Language Toolkit - Complete Documentation

## Overview

A **unified, production-ready package** for K'cho language data creation, analysis, and preparation. This toolkit integrates all K'cho linguistic features based on research by Bedell & Mang (2012) into a single, well-designed system.

## 🎯 Features

### Core Capabilities

1. **Morphological Analysis**
   - Verb stem alternation (Stem I ↔ Stem II)
   - Applicative suffix detection (-na/-nák)
   - Agreement particle recognition
   - Postposition identification
   - Automatic glossing

2. **Text Validation**
   - K'cho text detection with confidence scoring
   - Character set validation
   - Morphological structure validation
   - Pattern matching

3. **Lexicon Management**
   - SQLite-based dictionary
   - Frequency tracking
   - Multi-lingual glosses (English, Myanmar)
   - Verb paradigm generation
   - Full-text search

4. **Corpus Building**
   - Automatic annotation
   - Quality control
   - Multiple export formats (JSON, CoNLL-U, CSV)
   - Train/dev/test splitting
   - Statistics and reporting

## 📦 Installation

### Requirements

```bash
pip install sqlite3  # Usually built-in with Python
```

### Quick Start

```python
from kcho_toolkit import KchoToolkit

# Initialize toolkit
toolkit = KchoToolkit(project_dir="./my_kcho_project")

# Analyze K'cho text
sentence = toolkit.analyze("Om noh Yong am paapai pe ci")
print(sentence.gloss)

# Add to corpus
toolkit.add_to_corpus(
    "Om noh Yong am paapai pe ci",
    translation="Om gave Yong flowers"
)

# Export everything
toolkit.export_all()
```

## 🚀 Usage Guide

### 1. Basic Text Analysis

```python
from kcho_toolkit import KchoToolkit

toolkit = KchoToolkit()

# Analyze a sentence
text = "Ak'hmó noh k'khìm luum-na ci"
sentence = toolkit.analyze(text)

print(f"Text:  {sentence.text}")
print(f"Gloss: {sentence.gloss}")

# Examine word structure
for word in sentence.words:
    print(f"\nWord: {word.surface}")
    print(f"  POS: {word.pos}")
    print(f"  Lemma: {word.lemma}")
    print(f"  Morphemes:")
    for m in word.morphemes:
        print(f"    {m.type}: {m.form} → {m.gloss}")
```

**Output:**
```
Text:  Ak'hmó noh k'khìm luum-na ci
Gloss: N P N V-APPL NF

Word: Ak'hmó
  POS: N
  Lemma: ak'hmó
  Morphemes:
    root: ak'hmó → ak'hmó

Word: noh
  POS: P
  Lemma: noh
  Morphemes:
    postposition: noh → P
```

### 2. Text Validation

```python
# Validate if text is K'cho
is_kcho, confidence, metrics = toolkit.validate("Om noh Yong am paapai pe ci")

print(f"Is K'cho: {is_kcho}")
print(f"Confidence: {confidence:.2%}")
print(f"Metrics: {metrics}")
```

**Output:**
```
Is K'cho: True
Confidence: 87.5%
Metrics: {
    'char_validity': 1.0,
    'marker_score': 0.75,
    'pattern_score': 0.5,
    'overall_confidence': 0.875
}
```

### 3. Building a Corpus

```python
# Add sentences with translations
examples = [
    ("Om noh Yong am paapai pe ci", "Om gave Yong flowers"),
    ("Ka teihpΫi noh Yong am paapai pe ci", "My friend gave Yong flowers"),
    ("Om ka zèi-na ci", "I like Om"),
]

for kcho, english in examples:
    toolkit.add_to_corpus(kcho, translation=english)

# Get statistics
stats = toolkit.corpus_stats()
print(f"Total sentences: {stats['total_sentences']}")
print(f"Vocabulary size: {stats['vocabulary_size']}")
print(f"Avg sentence length: {stats['avg_sentence_length']}")
```

### 4. Working with the Lexicon

```python
from kcho_toolkit import LexiconEntry

# Add a new word
entry = LexiconEntry(
    headword="paapai",
    pos="N",
    gloss_en="flower",
    gloss_my="ပန်း",
    definition="A flower or blossom",
    semantic_field="plants",
    examples=["Om noh Yong am paapai pe ci"]
)

toolkit.lexicon.add_entry(entry)

# Search lexicon
results = toolkit.search_lexicon("flower")
for entry in results:
    print(f"{entry.headword} ({entry.pos}): {entry.gloss_en}")

# Get frequency list
freq_list = toolkit.lexicon.get_frequency_list(10)
print("\nTop 10 words:")
for word, freq in freq_list:
    print(f"  {word}: {freq}")
```

### 5. Verb Paradigm Generation

```python
# Get complete verb paradigm
paradigm = toolkit.get_verb_forms('lùum')  # 'play'

print("Stem I forms:")
for form_name, form in paradigm['stem_I'].items():
    print(f"  {form_name:15s}: {form}")

print("\nStem II forms:")
for form_name, form in paradigm['stem_II'].items():
    print(f"  {form_name:15s}: {form}")
```

**Output:**
```
Stem I forms:
  base           : lùum
  1sg            : ka lùum
  2sg            : na lùum
  3sg            : lùum
  non_future     : lùum ci
  future         : lùum khai

Stem II forms:
  base           : luum
  3sg_agreement  : a luum
  applicative    : luum-na
  subordinate    : a luum
```

### 6. Data Export

```python
# Export to different formats

# JSON (for ML training)
toolkit.corpus.export_json('./output/corpus.json')

# CoNLL-U (for linguistic research)
toolkit.corpus.export_conllu('./output/corpus.conllu')

# CSV (for spreadsheet analysis)
toolkit.corpus.export_csv('./output/corpus.csv')

# Lexicon
toolkit.lexicon.export_json('./output/lexicon.json')

# Or export everything at once
report = toolkit.export_all()
```

### 7. Quality Control

```python
# Get quality report
quality = toolkit.corpus.quality_report()

print(f"Total sentences: {quality['total_sentences']}")
print(f"Validated: {quality['validated_sentences']}")
print(f"Average confidence: {quality['avg_confidence']:.2%}")

if quality['issues']:
    print("\nIssues found:")
    for issue in quality['issues']:
        print(f"  - {issue}")
```

### 8. Creating Train/Dev/Test Splits

```python
# Create splits for ML training
splits = toolkit.corpus.create_splits(
    train_ratio=0.8,
    dev_ratio=0.1
)

print(f"Train: {len(splits['train'])} sentences")
print(f"Dev:   {len(splits['dev'])} sentences")
print(f"Test:  {len(splits['test'])} sentences")

# Export splits separately
import json

for split_name, sentences in splits.items():
    data = [s.to_dict() for s in sentences]
    with open(f'{split_name}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
```

## 📊 Data Formats

### JSON Format

```json
{
  "text": "Om noh Yong am paapai pe ci",
  "words": [
    {
      "surface": "Om",
      "morphemes": [
        {
          "form": "Om",
          "lemma": "Om",
          "gloss": "Om",
          "type": "root",
          "features": {}
        }
      ],
      "pos": "N",
      "lemma": "Om",
      "features": {}
    }
  ],
  "gloss": "N P N to N V NF",
  "translation": "Om gave Yong flowers",
  "metadata": {
    "timestamp": "2025-01-23T10:30:00"
  }
}
```

### CoNLL-U Format

```
# sent_id = 1
# text = Om noh Yong am paapai pe ci
# translation = Om gave Yong flowers
# gloss = N P N to N V NF
1	Om	Om	N	_	root=Om	_	_	_	_
2	noh	noh	P	_	postposition=noh	_	_	_	_
3	Yong	Yong	N	_	root=Yong	_	_	_	_
4	am	am	P	_	postposition=am	_	_	_	_
5	paapai	paapai	N	_	root=paapai	_	_	_	_
6	pe	pe	V	_	root=pe	_	_	_	_
7	ci	ci	T	_	particle=ci	_	_	_	_

```

### CSV Format

```csv
text,gloss,translation,word_count,morphemes
"Om noh Yong am paapai pe ci","N P N to N V NF","Om gave Yong flowers",7,7
```

## 🎓 Advanced Usage

### Custom Morphological Rules

```python
from kcho_toolkit import KchoConfig

# Extend configuration
class MyKchoConfig(KchoConfig):
    # Add custom verb stems
    VERB_STEMS = {
        **KchoConfig.VERB_STEMS,
        'custom_verb': {'stem2': 'custom2', 'gloss': 'meaning', 'pattern': 'custom'}
    }

# Use custom config
toolkit = KchoToolkit()
toolkit.config = MyKchoConfig()
```

### Batch Processing

```python
# Process multiple files
import os

input_dir = './raw_texts/'
texts = []

for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
            content = f.read()
            # Split into sentences (simple approach)
            sentences = content.split('.')
            texts.extend([s.strip() for s in sentences if s.strip()])

# Add to corpus with validation
added = toolkit.corpus.add_batch(texts, validate=True)
print(f"Added {len(added)}/{len(texts)} sentences")
```

### Integration with ML Pipelines

```python
# Prepare data for transformer training
def prepare_for_training(toolkit, output_dir):
    # Create splits
    splits = toolkit.corpus.create_splits()
    
    # Export in format for Hugging Face
    for split_name, sentences in splits.items():
        data = []
        for sent in sentences:
            data.append({
                'source': sent.text,
                'target': sent.translation,
                'gloss': sent.gloss,
            })
        
        with open(f'{output_dir}/{split_name}.json', 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Training data ready in {output_dir}/")

prepare_for_training(toolkit, './ml_data')
```

## 🔧 Configuration

### Project Structure

When you initialize the toolkit, it creates this structure:

```
my_kcho_project/
├── corpus/
│   └── (corpus data files)
├── exports/
│   ├── corpus_20250123_103000.json
│   ├── corpus_20250123_103000.conllu
│   ├── corpus_20250123_103000.csv
│   └── lexicon_20250123_103000.json
├── reports/
│   └── report_20250123_103000.json
└── kcho_lexicon.db
```

### Logging

Configure logging level:

```python
import logging

logging.basicConfig(
    level=logging.INFO,  # or DEBUG, WARNING, ERROR
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

## 📈 Use Cases

### 1. Building a K'cho-English MT Dataset

```python
toolkit = KchoToolkit()

# Add parallel sentences
parallel_data = [
    ("Om noh Yong am paapai pe ci", "Om gave Yong flowers"),
    ("Ak'hmó lùum ci", "The child plays"),
    # ... add more
]

for kcho, english in parallel_data:
    toolkit.add_to_corpus(kcho, translation=english)

# Export for training
splits = toolkit.corpus.create_splits()

# Use with transformers
from datasets import Dataset

train_data = Dataset.from_dict({
    'kcho': [s.text for s in splits['train']],
    'english': [s.translation for s in splits['train']]
})
```

### 2. Linguistic Research

```python
toolkit = KchoToolkit()

# Collect data
# ... add corpus

# Analyze morphology patterns
stats = toolkit.corpus_stats()
print(f"POS distribution: {stats['pos_distribution']}")

# Study verb stem alternation
verbs_with_stems = []
for entry in toolkit.lexicon.search('', field='pos'):
    if entry.pos == 'V' and entry.stem2:
        verbs_with_stems.append((entry.stem1, entry.stem2))

print(f"Found {len(verbs_with_stems)} verbs with stem alternation")
```

### 3. Dictionary App Backend

```python
from flask import Flask, jsonify, request

app = Flask(__name__)
toolkit = KchoToolkit()

@app.route('/search')
def search():
    query = request.args.get('q', '')
    results = toolkit.search_lexicon(query)
    return jsonify([e.to_dict() for e in results])

@app.route('/analyze')
def analyze():
    text = request.args.get('text', '')
    sentence = toolkit.analyze(text)
    return jsonify(sentence.to_dict())

app.run()
```

## 🐛 Troubleshooting

### Common Issues

1. **"Text rejected" warnings**
   - Check if text is actually K'cho
   - Lower confidence threshold by setting `validate=False`

2. **SQLite database locked**
   - Close connections properly: `toolkit.close()`
   - Don't share database between processes

3. **Memory issues with large corpus**
   - Process in batches
   - Export periodically
   - Use generators for iteration

## 📚 API Reference

### KchoToolkit

Main class for the toolkit.

**Methods:**
- `analyze(text: str) -> Sentence`: Analyze K'cho text
- `validate(text: str) -> Tuple[bool, float, Dict]`: Validate if text is K'cho
- `add_to_corpus(text, translation, **kwargs) -> Sentence`: Add to corpus
- `search_lexicon(query: str) -> List[LexiconEntry]`: Search dictionary
- `get_verb_forms(verb: str) -> Dict`: Get verb paradigm
- `corpus_stats() -> Dict`: Get corpus statistics
- `export_all()`: Export all data
- `close()`: Clean up resources

### Sentence

Represents an annotated sentence.

**Attributes:**
- `text: str`: Original text
- `words: List[Word]`: Analyzed words
- `gloss: str`: Morphological gloss
- `translation: str`: Translation
- `metadata: Dict`: Additional metadata

### Word

Represents an analyzed word.

**Attributes:**
- `surface: str`: Surface form
- `morphemes: List[Morpheme]`: Component morphemes
- `pos: str`: Part of speech
- `lemma: str`: Citation form
- `features: Dict`: Grammatical features

### LexiconEntry

Dictionary entry.

**Attributes:**
- `headword: str`: Main form
- `pos: str`: Part of speech
- `stem1: str`: Stem I form
- `stem2: str`: Stem II form
- `gloss_en: str`: English gloss
- `gloss_my: str`: Myanmar gloss
- `definition: str`: Definition
- `examples: List[str]`: Example sentences
- `frequency: int`: Frequency count

## 🎯 Best Practices

1. **Always validate**: Use `validate=True` when adding to corpus
2. **Regular exports**: Export data frequently to avoid loss
3. **Clean input**: Preprocess text (remove extra whitespace, etc.)
4. **Batch operations**: Use `add_batch()` for multiple texts
5. **Check quality**: Run `quality_report()` periodically
6. **Close properly**: Call `toolkit.close()` when done

## 📖 References

- Bedell, G. & Mang, K. S. (2012). The Applicative Suffix -na in K'cho
- Jordan, M. (1969). Chin Dictionary and Grammar
- K'cho linguistic research papers

## 📝 License

This toolkit is provided for K'cho language research and preservation.

---

**Version**: 1.0.0  
**Last Updated**: January 2025
