# K'Cho CLI Comprehensive Guide

**Author**: Hung Om  
**Research Foundation**: Based on foundational K'Cho linguistic research and Austroasiatic language studies  
**Version**: 0.2.0  
**Date**: 2024-10-25

## Abstract

This document provides comprehensive documentation for the K'Cho Linguistic Processing Toolkit command-line interface. The CLI provides access to computational linguistic analysis tools, research capabilities, and data management functions for K'Cho language processing.

## Overview

The K'Cho CLI (`kcho`) provides comprehensive access to all linguistic analysis tools, research capabilities, and data management functions. The CLI is organized into logical command groups for easy navigation.

## Installation and Setup

```bash
# Install the package
pip install -e .

# Verify installation
python -m kcho.cli --version
```

## Command Structure

```bash
python -m kcho.cli [GROUP] [COMMAND] [OPTIONS]
```

## Available Command Groups

### 1. Text Processing Commands

#### Normalize Text
```bash
python -m kcho.cli normalize input.txt --output normalized.txt
```
- Normalizes K'Cho text for analysis
- Handles Unicode normalization and text cleaning

#### Tokenize Text
```bash
python -m kcho.cli tokenize input.txt --output tokens.txt
```
- Tokenizes K'Cho text into individual tokens
- Outputs one token per line

### 2. Collocation Analysis Commands

#### Extract Collocations
```bash
python -m kcho.cli collocation --corpus corpus.txt --output results.csv --top-k 20 --min-freq 5
```
- Extracts collocations using statistical measures
- Supports PMI, t-score, dice, and other measures
- Configurable window size and frequency thresholds

#### Extract All Collocations
```bash
python -m kcho.cli extract-all-collocations --input corpus.txt --output results --format csv json txt
```
- Comprehensive collocation extraction with linguistic patterns
- Multiple output formats (CSV, JSON, TXT)
- Includes pattern classification and confidence scores

#### Evaluate Collocations
```bash
python -m kcho.cli evaluate-collocations predicted.csv gold_standard.txt --measure pmi
```
- Evaluates collocation extraction against gold standard
- Supports multiple association measures
- Provides precision, recall, and F1 metrics

### 3. N-gram Analysis Commands

#### Extract N-grams
```bash
python -m kcho.cli extract-ngrams --input corpus.txt --output ngrams --max-ngram-size 4 --min-freq 2
```
- Extracts n-grams (bigrams, trigrams, 4-grams, etc.)
- Configurable n-gram size and frequency thresholds
- Multiple statistical measures support

### 4. Research Commands

#### Comprehensive Research Analysis
```bash
python -m kcho.cli research analyze --corpus corpus.txt --output analysis.json --focus all --min-freq 3
```
- Conducts comprehensive linguistic research
- Focus areas: all, patterns, morphology, syntax
- Multiple output formats (JSON, CSV, TXT)

#### Pattern Discovery
```bash
python -m kcho.cli research discover-patterns --corpus corpus.txt --output patterns.json --min-freq 3 --max-patterns 100
```
- Discovers linguistic patterns from corpus
- Configurable frequency thresholds and pattern limits
- Pattern confidence scoring

#### Text Structure Analysis
```bash
python -m kcho.cli research analyze-text --corpus corpus.txt --output analysis.json --format json
```
- Analyzes text structure and linguistic features
- Token-level analysis with POS tagging
- Morphological and syntactic information

### 5. Data Generation Commands

#### Generate Research Data
```bash
python -m kcho.cli generate research-data --corpus corpus.txt --output research/ --include-research --include-collocations --include-ngrams
```
- Generates comprehensive research datasets
- Combines multiple analysis types
- Creates organized output directory structure

### 6. Data Migration Commands

#### Check Migration Status
```bash
python -m kcho.cli migrate check --verbose
```
- Checks if data migration is needed
- Shows file checksums and version information
- Displays migration status

#### Run Migration
```bash
python -m kcho.cli migrate run --verbose
```
- Runs data migration if needed
- Shows migration history
- Displays database statistics

#### View Migration History
```bash
python -m kcho.cli migrate history --limit 10
```
- Shows recent migration history
- Displays migration details and status
- Configurable history limit

#### Show Data File Status
```bash
python -m kcho.cli migrate status
```
- Shows status of tracked data files
- Displays file versions and checksums
- Shows loading timestamps

#### Force Fresh Migration
```bash
python -m kcho.cli migrate force --confirm
```
- Forces a fresh data migration
- Replaces all data in database
- Requires confirmation flag

#### Initialize Database
```bash
python -m kcho.cli migrate init
```
- Initializes database with fresh data
- Creates all necessary tables and indexes
- Loads initial data from files

### 7. Configuration Commands

#### Create Configuration Template
```bash
python -m kcho.cli create-config --output config.yaml
```
- Creates configuration template file
- Includes all available options
- Customizable output location

## Common Options

### Global Options
- `--verbose, -v`: Enable verbose output
- `--help`: Show help message
- `--version`: Show version information

### Output Options
- `--output, -o`: Specify output file/directory
- `--format, -f`: Choose output format (json, csv, txt)
- `--verbose`: Enable detailed logging

### Analysis Options
- `--min-freq`: Minimum frequency threshold
- `--max-patterns`: Maximum number of patterns to extract
- `--window-size`: Co-occurrence window size
- `--measures`: Statistical measures to use

## Examples

### Complete Research Workflow
```bash
# 1. Check data status
python -m kcho.cli migrate check

# 2. Analyze text structure
python -m kcho.cli research analyze-text --corpus corpus.txt --output text_analysis.json

# 3. Discover patterns
python -m kcho.cli research discover-patterns --corpus corpus.txt --output patterns.json --min-freq 3

# 4. Extract collocations
python -m kcho.cli collocation --corpus corpus.txt --output collocations.csv --top-k 50

# 5. Generate comprehensive research data
python -m kcho.cli generate research-data --corpus corpus.txt --output research/ --include-research --include-collocations --include-ngrams
```

### Data Management Workflow
```bash
# 1. Initialize database
python -m kcho.cli migrate init

# 2. Check status
python -m kcho.cli migrate status

# 3. Run migration if needed
python -m kcho.cli migrate run

# 4. View history
python -m kcho.cli migrate history
```

## Output Formats

### JSON Format
- Structured data with metadata
- Includes analysis parameters and results
- Machine-readable format

### CSV Format
- Tabular data for spreadsheet analysis
- Includes headers and statistical measures
- Easy to import into analysis tools

### TXT Format
- Human-readable format
- Includes summaries and examples
- Good for documentation and reports

## Error Handling

The CLI includes comprehensive error handling:
- Graceful handling of missing files
- Validation of input parameters
- Detailed error messages with suggestions
- Verbose mode for debugging

## Performance Considerations

- Database operations are optimized with indexes
- Large corpora are processed in batches
- Memory usage is optimized for large datasets
- Progress indicators for long-running operations

## Troubleshooting

### Common Issues
1. **Database locked**: Close other applications using the database
2. **Memory issues**: Process smaller corpus chunks
3. **File not found**: Check file paths and permissions
4. **Import errors**: Ensure all dependencies are installed

### Debug Mode
Use `--verbose` flag for detailed logging and debugging information.

## Integration

The CLI can be integrated into larger workflows:
- Script automation
- Pipeline processing
- Batch analysis
- Research data generation

## Support

For issues and questions:
- Check the main README.md
- Review the project documentation
- Create an issue on GitHub
- Check the examples directory
