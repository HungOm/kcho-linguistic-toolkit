#!/usr/bin/env python3
"""
Defaultdict Examples for K'Cho Toolkit

This script demonstrates the new defaultdict functionality added to the K'Cho toolkit.
It shows how defaultdict improves efficiency and code clarity for linguistic analysis.

Usage:
    python defaultdict_examples.py

Author: K'Cho Toolkit Team
Date: 2025
"""

import logging
from kcho.kcho_system import KchoSystem, KchoCorpus
from kcho.collocation import CollocationExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_defaultdict_functionality():
    """
    Demonstrate the new defaultdict functionality in the K'Cho toolkit.
    """
    print("=" * 60)
    print("K'CHO TOOLKIT - DEFAULTDICT FUNCTIONALITY DEMONSTRATION")
    print("=" * 60)
    
    # Sample K'Cho corpus for demonstration
    sample_corpus = [
        "Kcho language is beautiful",
        "We speak Kcho language daily", 
        "Kcho people love their culture",
        "Language learning is important",
        "Kcho culture has rich traditions",
        "We study Kcho language together",
        "Beautiful Kcho songs tell stories",
        "Kcho people are very friendly",
        "Language preservation is crucial",
        "Kcho traditions are passed down"
    ]
    
    print(f"\n📚 Sample Corpus ({len(sample_corpus)} sentences):")
    for i, sentence in enumerate(sample_corpus, 1):
        print(f"  {i}. {sentence}")
    
    # Initialize system components
    print(f"\n🔧 Initializing K'Cho System Components...")
    system = KchoSystem()
    corpus = KchoCorpus()
    collocation_extractor = CollocationExtractor()
    
    # Add sentences to corpus
    print(f"\n📝 Adding sentences to corpus...")
    for sentence in sample_corpus:
        corpus.add_sentence(sentence)
    
    print(f"✅ Corpus now contains {len(corpus.sentences)} sentences")
    
    # Demonstrate 1: POS Pattern Analysis using defaultdict
    print(f"\n🎯 DEMONSTRATION 1: POS Pattern Analysis")
    print("-" * 50)
    pos_patterns = corpus.analyze_pos_patterns()
    
    print("📊 POS Patterns by Length:")
    for length, patterns in pos_patterns.items():
        print(f"  Length {length}: {len(patterns)} unique patterns")
        # Show top 3 patterns for each length
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        for pattern, count in sorted_patterns:
            print(f"    {pattern}: {count} occurrences")
    
    # Demonstrate 2: Word Co-occurrence Matrix using nested defaultdict
    print(f"\n🎯 DEMONSTRATION 2: Word Co-occurrence Analysis")
    print("-" * 50)
    cooccurrence_matrix = corpus.build_word_cooccurrence_matrix(window_size=3)
    
    print("🔗 Top Word Co-occurrences:")
    # Find words with most co-occurrences
    word_cooccur_counts = {}
    for word, cooccur_dict in cooccurrence_matrix.items():
        word_cooccur_counts[word] = sum(cooccur_dict.values())
    
    top_words = sorted(word_cooccur_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for word, total_cooccur in top_words:
        print(f"  '{word}': {total_cooccur} total co-occurrences")
        # Show top co-occurring words
        top_cooccur = sorted(cooccurrence_matrix[word].items(), key=lambda x: x[1], reverse=True)[:3]
        for cooccur_word, count in top_cooccur:
            print(f"    with '{cooccur_word}': {count} times")
    
    # Demonstrate 3: Morphological Pattern Extraction using defaultdict
    print(f"\n🎯 DEMONSTRATION 3: Morphological Pattern Analysis")
    print("-" * 50)
    morph_patterns = corpus.extract_morphological_patterns()
    
    print("🔤 Morphological Patterns Found:")
    for pattern_type, patterns in morph_patterns.items():
        if patterns:  # Only show if patterns exist
            print(f"  {pattern_type.title()}:")
            sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5]
            for pattern, count in sorted_patterns:
                print(f"    {pattern}: {count} occurrences")
    
    # Demonstrate 4: Sentence Structure Analysis using defaultdict
    print(f"\n🎯 DEMONSTRATION 4: Sentence Structure Analysis")
    print("-" * 50)
    structure_patterns = corpus.analyze_sentence_structure_patterns()
    
    print("📐 Sentence Structure Patterns:")
    for structure_type, patterns in structure_patterns.items():
        print(f"  {structure_type.replace('_', ' ').title()}:")
        for pattern, count in patterns.items():
            print(f"    {pattern}: {count} sentences")
    
    # Demonstrate 5: Collocation Analysis with defaultdict grouping
    print(f"\n🎯 DEMONSTRATION 5: Collocation Analysis with POS Grouping")
    print("-" * 50)
    
    # Extract collocations
    collocations = collocation_extractor.extract(sample_corpus)
    print("🔍 Collocations Found:")
    for measure, results in collocations.items():
        print(f"  {measure.value}: {len(results)} collocations")
        if results:
            # Show top 3 collocations
            top_collocations = sorted(results, key=lambda x: x.score, reverse=True)[:3]
            for colloc in top_collocations:
                print(f"    {colloc.bigram[0]} {colloc.bigram[1]}: {colloc.score:.2f}")
    
    # Group collocations by POS patterns (if POS tagger available)
    try:
        pos_grouped = collocation_extractor.group_collocations_by_pos_pattern(sample_corpus)
        print(f"\n📋 Collocations Grouped by POS Patterns:")
        for pos_pattern, colloc_list in pos_grouped.items():
            print(f"  {pos_pattern}: {len(colloc_list)} collocations")
    except Exception as e:
        print(f"  Note: POS grouping requires POS tagger (not available in demo)")
    
    # Demonstrate 6: Word Context Analysis using nested defaultdict
    print(f"\n🎯 DEMONSTRATION 6: Word Context Analysis")
    print("-" * 50)
    
    try:
        word_contexts = collocation_extractor.analyze_word_contexts(sample_corpus, context_window=2)
        print("🌐 Word Context Analysis:")
        
        # Show contexts for most frequent words
        word_frequencies = {}
        for word, contexts in word_contexts.items():
            word_frequencies[word] = sum(contexts.values())
        
        top_context_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)[:3]
        for word, total_contexts in top_context_words:
            print(f"  '{word}' appears in {total_contexts} contexts:")
            top_contexts = sorted(word_contexts[word].items(), key=lambda x: x[1], reverse=True)[:3]
            for context_word, count in top_contexts:
                print(f"    with '{context_word}': {count} times")
    except Exception as e:
        print(f"  Note: Context analysis requires tokenization (demo limitation)")
    
    # Demonstrate 7: Linguistic Pattern Extraction
    print(f"\n🎯 DEMONSTRATION 7: Linguistic Pattern Extraction")
    print("-" * 50)
    
    try:
        linguistic_patterns = collocation_extractor.extract_linguistic_patterns(sample_corpus)
        print("🎭 Linguistic Patterns Found:")
        for pattern_type, patterns in linguistic_patterns.items():
            if patterns:  # Only show if patterns exist
                print(f"  {pattern_type.replace('_', ' ').title()}:")
                sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:3]
                for pattern, count in sorted_patterns:
                    print(f"    {pattern}: {count} occurrences")
    except Exception as e:
        print(f"  Note: Pattern extraction requires tokenization (demo limitation)")
    
    print(f"\n" + "=" * 60)
    print("✅ DEFAULTDICT DEMONSTRATION COMPLETE!")
    print("=" * 60)
    
    print(f"\n💡 KEY BENEFITS OF DEFAULTDICT IN K'CHO TOOLKIT:")
    print("  1. 🚀 Automatic default value creation - no KeyError exceptions")
    print("  2. 🧹 Cleaner code - no need for 'if key in dict' checks")
    print("  3. 📊 Efficient grouping - automatic list/dict creation")
    print("  4. 🔄 Nested structures - automatic nested defaultdict creation")
    print("  5. 📈 Better performance - fewer conditional checks")
    
    print(f"\n🎯 USE CASES DEMONSTRATED:")
    print("  • POS pattern grouping (defaultdict(list))")
    print("  • Word co-occurrence counting (nested defaultdict(int))")
    print("  • Morphological pattern extraction (defaultdict(lambda: defaultdict(int)))")
    print("  • Sentence structure analysis (defaultdict(lambda: defaultdict(int)))")
    print("  • Collocation grouping by linguistic features")
    print("  • Word context analysis with nested structures")
    print("  • Linguistic pattern categorization")

if __name__ == "__main__":
    demonstrate_defaultdict_functionality()
