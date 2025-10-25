# Examples

**Author**: Hung Om  
**Research Foundation**: Based on foundational K'Cho linguistic research and Austroasiatic language studies  
**Version**: 0.2.0  
**Date**: 2024-10-25

## Abstract

This directory contains practical examples demonstrating the K'Cho Linguistic Processing Toolkit capabilities for computational linguistic analysis, including pattern discovery, morphological analysis, and research data generation workflows.

## 🚀 Quick Start

### Basic System Demo
```bash
python enhanced_kcho_system_demo.py
```
Comprehensive demonstration of all system capabilities including pattern discovery, research analysis, and API integration.

### Practical Usage
```bash
python practical_demonstration.py
```
Real-world usage examples for common linguistic analysis tasks.

## 📚 Tutorials

### Collocation Analysis
```bash
python tutorials/collocation_extraction_tutorial.py
```
Learn how to extract and analyze collocation patterns from K'Cho texts.

### N-gram Analysis
```bash
python tutorials/ngram_collocation_test.py
```
Statistical analysis of n-gram patterns and word co-occurrences.

## 🔬 Research Tools

### Data Generation
```bash
python generate_research_data.py
```
Generate comprehensive research datasets for linguistic analysis.

### Research Analysis
```bash
python analyze_research_data.py
```
Analyze research data and generate insights.

## 📖 Example Code

### Basic Text Analysis
```python
from kcho import KchoSystem

# Initialize system
system = KchoSystem()

# Analyze text
sentence = system.analyze_text("Om noh Yóng am pàapai pe ci.")
print(f"Analyzed {len(sentence.tokens)} tokens")
```

### Pattern Discovery
```python
# Discover patterns from corpus
corpus = ["sentence1", "sentence2", ...]
patterns = system.discover_patterns(corpus, min_frequency=3)
print(f"Discovered {len(patterns)} patterns")
```

### API Integration
```python
# Get API response
response = system.get_api_response("What are common verb patterns?")
print(f"Response: {response['data']}")
```

## 🎯 Use Cases

- **Linguistic Research**: Pattern discovery and analysis
- **Language Documentation**: Systematic data collection
- **Machine Learning**: Training data preparation
- **API Integration**: External system connectivity
- **Educational**: Learning K'Cho language structure

## 📝 Notes

- All examples include error handling and logging
- Examples are designed to be self-contained and runnable
- Output formats include JSON, CSV, and human-readable text
- Examples demonstrate both basic and advanced usage patterns