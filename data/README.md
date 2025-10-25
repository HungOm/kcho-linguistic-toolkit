# Data Directory

**Author**: Hung Om  
**Research Foundation**: Based on foundational K'Cho linguistic research and Austroasiatic language studies  
**Version**: 0.2.0  
**Date**: 2024-10-25

## Abstract

This directory contains research data and corpora for the K'Cho Linguistic Processing Toolkit, including linguistic datasets, parallel corpora, and frequency analysis data used for computational linguistic research and pattern discovery applications.

## 📁 Data Structure

### Core Package Data
Located in `kcho/data/` (package data):
- **linguistic_data.json**: Core linguistic categories (verbs, pronouns, etc.)
- **gold_standard_collocations.txt**: Manually curated collocation patterns
- **word_frequency_top_1000.csv**: Statistical word frequency data
- **gold_standard_kcho_english.json**: Parallel sentence pairs
- **sample_corpus.txt**: Sample K'Cho texts for testing

### Research Data
Located in `data/` (research materials):
- **bible_versions/**: Multiple K'Cho Bible translations
- **parallel_corpora/**: K'Cho-English aligned texts
- **README.md**: This documentation file

## 📊 Data Formats

### JSON Files
- **Structure**: Hierarchical data with metadata
- **Encoding**: UTF-8 with proper Unicode handling
- **Validation**: Optional Pydantic schema validation

### Text Files
- **Encoding**: UTF-8
- **Line endings**: Unix-style (LF)
- **Format**: One sentence per line

### CSV Files
- **Delimiter**: Comma
- **Headers**: Descriptive column names
- **Encoding**: UTF-8

## 🔄 Data Management

### Migration System
The system includes automatic data migration:
```bash
# Check migration status
python -m kcho.kcho_app migrate check

# Run migration if needed
python -m kcho.kcho_app migrate run
```

### Data Validation
- **Checksums**: Automatic integrity checking
- **Versioning**: Track data changes over time
- **Backup**: Automatic backup before migrations

## 📈 Usage Examples

### Loading Core Data
```python
from kcho import KchoKnowledge

# Initialize with automatic data loading
knowledge = KchoKnowledge()

# Access linguistic data
verb_stems = knowledge.VERB_STEMS
collocations = knowledge.gold_standard_patterns
```

### Working with Research Data
```python
# Load custom corpus
with open('data/parallel_corpora/aligned_cho_english.csv', 'r') as f:
    # Process parallel data
    pass
```

## 🔒 Data Integrity

### Checksums
All data files include checksums for integrity verification:
- **SHA-256**: Primary integrity check
- **File size**: Secondary validation
- **Timestamp**: Change detection

### Version Control
- **Git LFS**: Large files stored efficiently
- **Compression**: Optimized storage for large datasets
- **Incremental updates**: Only changed data migrated

## 📝 Data Sources

### Academic Sources
- K'Cho linguistic research (Bedell & Mang 2012)
- Bible translation projects
- Community language documentation

### Generated Data
- Statistical analysis outputs
- Pattern discovery results
- Research findings and insights

## 🤝 Contributing Data

When adding new data:
1. Follow established naming conventions
2. Include proper metadata and documentation
3. Ensure data quality and accuracy
4. Update checksums and version information
5. Test with migration system