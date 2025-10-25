#!/usr/bin/env python3
"""
Test Script for Trigram and Longer Collocation Extraction

This script demonstrates how to extract trigrams, 4-grams, and longer collocations
from K'Cho text using the enhanced NGramCollocationExtractor.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from kcho.ngram_collocation import NGramCollocationExtractor, NGramCollocationResult
from kcho.collocation import AssociationMeasure, LinguisticPattern
from kcho.text_loader import TextLoader
from kcho.config import load_config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_trigram_extraction():
    """Test trigram extraction on sample corpus."""
    print("🚀 Testing Trigram and Longer Collocation Extraction")
    print("=" * 60)
    
    # Load sample corpus
    sample_path = project_root / "data" / "sample_corpus.txt"
    print(f"Looking for sample corpus at: {sample_path}")
    print(f"Project root: {project_root}")
    print(f"Sample path exists: {sample_path.exists()}")
    print(f"Current working directory: {Path.cwd()}")
    
    if not sample_path.exists():
        print(f"❌ Sample corpus not found at {sample_path}")
        return
    
    print(f"📖 Loading corpus from {sample_path}")
    sentences = TextLoader.load_from_file(str(sample_path))
    print(f"✅ Loaded {len(sentences)} sentences")
    
    # Show sample sentences
    print(f"\n📝 Sample sentences:")
    for i, sentence in enumerate(sentences[:3]):
        print(f"  {i+1}. {sentence}")
    
    # Create n-gram extractor with support for up to 4-grams
    print(f"\n🔧 Creating N-gram extractor (max 4-grams)...")
    extractor = NGramCollocationExtractor(
        window_size=5,
        min_freq=2,  # Lower threshold for demo
        max_ngram_size=4,
        measures=[AssociationMeasure.PMI, AssociationMeasure.TSCORE, AssociationMeasure.DICE]
    )
    
    # Extract n-grams
    print(f"\n🔍 Extracting n-grams...")
    results = extractor.extract_ngrams(sentences)
    
    # Print summary
    extractor.print_ngram_summary(results)
    
    # Detailed analysis by n-gram size
    print(f"\n" + "=" * 60)
    print("📊 DETAILED ANALYSIS BY N-GRAM SIZE")
    print("=" * 60)
    
    for n, measure_results in results.items():
        print(f"\n🔸 {n}-GRAMS ANALYSIS")
        print("-" * 30)
        
        # Get top n-grams across all measures
        top_ngrams = extractor.get_top_ngrams_by_size({n: measure_results}, top_k=10)
        
        if n in top_ngrams and top_ngrams[n]:
            print(f"\nTop {n}-grams (across all measures):")
            for i, ngram in enumerate(top_ngrams[n], 1):
                pattern = ngram.linguistic_pattern.value if ngram.linguistic_pattern else 'UNK'
                print(f"  {i}. {' '.join(ngram.words)}")
                print(f"     Score: {ngram.score:.3f} ({ngram.measure.value})")
                print(f"     Frequency: {ngram.frequency}")
                print(f"     Pattern: {pattern} (confidence: {ngram.pattern_confidence:.2f})")
                if ngram.context_examples:
                    print(f"     Example: \"{ngram.context_examples[0]}\"")
                print()
        else:
            print(f"No {n}-grams found with current settings.")
    
    # Pattern analysis
    print(f"\n" + "=" * 60)
    print("🎯 LINGUISTIC PATTERN ANALYSIS")
    print("=" * 60)
    
    pattern_examples = {}
    for n, measure_results in results.items():
        for measure, ngrams in measure_results.items():
            for ngram in ngrams:
                if ngram.linguistic_pattern:
                    pattern = ngram.linguistic_pattern.value
                    if pattern not in pattern_examples:
                        pattern_examples[pattern] = []
                    pattern_examples[pattern].append(ngram)
    
    for pattern, ngrams in pattern_examples.items():
        print(f"\n📌 {pattern.upper()} Pattern:")
        # Sort by confidence and show top examples
        ngrams.sort(key=lambda x: x.pattern_confidence, reverse=True)
        for i, ngram in enumerate(ngrams[:5], 1):
            print(f"  {i}. {' '.join(ngram.words)} ({ngram.ngram_size}-gram, conf: {ngram.pattern_confidence:.2f})")

def test_with_larger_corpus():
    """Test with a larger corpus if available."""
    print(f"\n" + "=" * 60)
    print("🔬 TESTING WITH LARGER CORPUS")
    print("=" * 60)
    
    # Try to load a larger corpus
    bible_path = project_root / "data" / "bible_versions" / "2642_bible_data.json"
    if bible_path.exists():
        print(f"📖 Loading larger corpus from {bible_path}")
        sentences = TextLoader.load_from_file(str(bible_path))
        print(f"✅ Loaded {len(sentences)} sentences")
        
        # Use a subset for testing (first 1000 sentences)
        test_sentences = sentences[:1000]
        print(f"🔬 Testing with {len(test_sentences)} sentences")
        
        # Create extractor for larger corpus
        extractor = NGramCollocationExtractor(
            window_size=5,
            min_freq=5,  # Higher threshold for larger corpus
            max_ngram_size=3,  # Focus on trigrams for larger corpus
            measures=[AssociationMeasure.PMI, AssociationMeasure.TSCORE]
        )
        
        # Extract n-grams
        results = extractor.extract_ngrams(test_sentences)
        
        # Show trigram results
        if 3 in results:
            print(f"\n🔸 TRIGRAMS FROM LARGER CORPUS:")
            print("-" * 40)
            
            trigram_results = results[3]
            for measure, ngrams in trigram_results.items():
                if ngrams:
                    print(f"\n{measure.value.upper()} (top 10):")
                    for i, ngram in enumerate(ngrams[:10], 1):
                        pattern = ngram.linguistic_pattern.value if ngram.linguistic_pattern else 'UNK'
                        print(f"  {i}. {' '.join(ngram.words)} (score: {ngram.score:.3f}, freq: {ngram.frequency}, pattern: {pattern})")
    else:
        print(f"ℹ️  No larger corpus found at {bible_path}")
        print(f"   Place a larger K'Cho corpus there to test with more data")

def demonstrate_ngram_patterns():
    """Demonstrate different n-gram patterns found in K'Cho."""
    print(f"\n" + "=" * 60)
    print("📚 K'CHO N-GRAM PATTERN EXAMPLES")
    print("=" * 60)
    
    patterns = {
        "Bigrams (2-grams)": [
            ("pe", "ci"),  # Verb + Particle
            ("noh", "Yóng"),  # Postposition + Noun
            ("a", "péit"),  # Agreement + Verb
            ("ci", "ah"),  # Complementizer
        ],
        "Trigrams (3-grams)": [
            ("pe", "ci", "ah"),  # Verb + Particle + Complementizer
            ("noh", "Yóng", "am"),  # Postposition + Noun + Postposition
            ("luum", "-na", "ci"),  # Applicative construction
            ("a", "péit", "ah"),  # Agreement + Verb + Complementizer
        ],
        "4-grams": [
            ("Om", "noh", "Yóng", "am"),  # Subject + Postposition + Noun + Postposition
            ("pe", "ci", "ah", "k'chàang"),  # Verb + Particle + Complementizer + Noun
            ("noh", "Yóng", "am", "pàapai"),  # Postposition + Noun + Postposition + Noun
        ]
    }
    
    for pattern_type, examples in patterns.items():
        print(f"\n🔹 {pattern_type}:")
        for example in examples:
            print(f"  • {' '.join(example)}")

def main():
    """Main test function."""
    print("🧪 N-Gram Collocation Extraction Test Suite")
    print("=" * 60)
    
    # Test 1: Basic trigram extraction
    test_trigram_extraction()
    
    # Test 2: Larger corpus test
    test_with_larger_corpus()
    
    # Test 3: Pattern examples
    demonstrate_ngram_patterns()
    
    print(f"\n" + "=" * 60)
    print("✅ N-Gram Collocation Testing Complete!")
    print("=" * 60)
    
    print(f"\n🎯 Key Findings:")
    print(f"  • Trigrams capture more complex linguistic patterns")
    print(f"  • 4-grams reveal sentence-level constructions")
    print(f"  • Different association measures highlight different aspects")
    print(f"  • Linguistic pattern classification works across n-gram sizes")
    
    print(f"\n💡 Usage Tips:")
    print(f"  • Use lower min_freq for smaller corpora")
    print(f"  • Higher min_freq for larger corpora to reduce noise")
    print(f"  • PMI good for rare but meaningful patterns")
    print(f"  • T-Score good for frequent patterns")
    print(f"  • Dice coefficient good for balanced measures")

if __name__ == "__main__":
    main()
