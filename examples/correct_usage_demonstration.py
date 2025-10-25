#!/usr/bin/env python3
"""
K'Cho Enhanced System - Correct Usage Demonstration

This script demonstrates the CORRECT usage of the enhanced KchoSystem:
- kcho/data/ = Knowledge base for pattern recognition and classification
- User data (Bible, custom text) = Source for actual research data generation
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kcho import (
    EnhancedKchoSystem,
    ComprehensiveKchoKnowledge,
    TextLoader,
    CollocationExtractor,
    AssociationMeasure
)

def demonstrate_correct_usage():
    """Demonstrate correct usage: knowledge base guides analysis of user data"""
    
    print("🎯 K'CHO ENHANCED SYSTEM - CORRECT USAGE DEMONSTRATION")
    print("=" * 80)
    print("CORRECT APPROACH:")
    print("• kcho/data/ = Knowledge base for pattern recognition")
    print("• User data (Bible, custom text) = Source for research data")
    print("• Knowledge base guides analysis of user data")
    print("=" * 80)
    
    # Initialize enhanced system with knowledge base
    print("\n🧠 LOADING KNOWLEDGE BASE")
    print("-" * 50)
    
    system = EnhancedKchoSystem(use_comprehensive_knowledge=True)
    print("✅ Knowledge base loaded from kcho/data/")
    print(f"📊 Knowledge base contains:")
    print(f"  • {len(system.knowledge.all_words)} reference words")
    print(f"  • {len(system.knowledge.gold_standard_patterns)} known patterns")
    print(f"  • {len(system.knowledge.word_categories)} linguistic categories")
    
    # Load USER DATA (Bible text) - this is what gets analyzed
    print("\n📖 LOADING USER DATA (Bible Text)")
    print("-" * 50)
    
    bible_file = "data/bible_versions/2642_bible_data.json"
    
    if not Path(bible_file).exists():
        print(f"❌ Bible data not found: {bible_file}")
        print("Using sample sentences instead...")
        user_sentences = [
            "Om noh Yóng am pàapai pe ci.",
            "Yóng am pàapai pe ci ah k'chàang ka hngu ci.",
            "Ak'hmó noh k'khìm luum-na ci.",
            "Ui noh vok htui ci.",
            "Khò nàa ci."
        ]
    else:
        print(f"📚 Loading Bible data from: {bible_file}")
        user_sentences = TextLoader.load_from_file(bible_file)
        print(f"✅ Loaded {len(user_sentences)} sentences from Bible data")
        
        # Use subset for demonstration
        if len(user_sentences) > 1000:
            print(f"📊 Using first 1000 sentences for demonstration (total: {len(user_sentences)})")
            user_sentences = user_sentences[:1000]
    
    print(f"🎯 USER DATA TO ANALYZE: {len(user_sentences)} sentences")
    
    # Analyze USER DATA using knowledge base for guidance
    print("\n🔍 ANALYZING USER DATA WITH KNOWLEDGE BASE GUIDANCE")
    print("-" * 50)
    
    print("Using knowledge base to guide analysis of user data...")
    
    # Extract collocations from USER DATA
    print("\n📊 EXTRACTING COLLOCATIONS FROM USER DATA")
    print("-" * 50)
    
    extractor = CollocationExtractor(
        window_size=5,
        min_freq=5,  # Higher threshold for larger corpus
        measures=[
            AssociationMeasure.PMI,
            AssociationMeasure.TSCORE,
            AssociationMeasure.DICE,
            AssociationMeasure.LOG_LIKELIHOOD
        ]
    )
    
    print(f"Extracting collocations from {len(user_sentences)} user sentences...")
    collocation_results = extractor.extract(user_sentences)
    
    print("✅ Collocation extraction completed")
    
    # Show results for each measure
    total_collocations = 0
    for measure, results in collocation_results.items():
        print(f"\n📈 {measure.value.upper()} Results from USER DATA:")
        print(f"   Total collocations: {len(results)}")
        total_collocations += len(results)
        
        if results:
            print(f"   Top 5 collocations from USER DATA:")
            for i, result in enumerate(results[:5], 1):
                words = ' '.join(result.words)
                print(f"     {i}. {words}: {result.score:.3f} (freq: {result.frequency})")
    
    print(f"\n🎯 TOTAL COLLOCATIONS EXTRACTED FROM USER DATA: {total_collocations}")
    
    # Use knowledge base to classify patterns found in USER DATA
    print("\n🧠 USING KNOWLEDGE BASE TO CLASSIFY USER DATA PATTERNS")
    print("-" * 50)
    
    print("Classifying patterns found in user data using knowledge base...")
    
    # Get top collocations from user data
    top_collocations = []
    for measure, results in collocation_results.items():
        if results:
            top_collocations.extend(results[:10])  # Top 10 from each measure
    
    # Remove duplicates and sort by frequency
    unique_collocations = {}
    for result in top_collocations:
        words_key = ' '.join(result.words)
        if words_key not in unique_collocations or result.frequency > unique_collocations[words_key].frequency:
            unique_collocations[words_key] = result
    
    sorted_collocations = sorted(unique_collocations.values(), key=lambda x: x.frequency, reverse=True)
    
    print(f"🔍 Analyzing top {len(sorted_collocations)} patterns from user data:")
    
    known_patterns = 0
    novel_patterns = 0
    
    for i, result in enumerate(sorted_collocations[:20], 1):  # Top 20
        words = result.words
        words_key = ' '.join(words)
        
        # Use knowledge base to check if this is a known pattern
        if len(words) >= 2:
            is_known, pattern_type, confidence = system.is_gold_standard_pattern(words[0], words[1])
            
            if is_known:
                known_patterns += 1
                status = "✅ KNOWN"
                pattern_info = f"({pattern_type}, {confidence:.2f})"
            else:
                novel_patterns += 1
                status = "🆕 NOVEL"
                pattern_info = "(discovered)"
        else:
            status = "❓ UNKNOWN"
            pattern_info = "(single word)"
        
        print(f"  {i:2d}. {words_key}: {result.frequency} occurrences {status} {pattern_info}")
    
    print(f"\n📊 PATTERN CLASSIFICATION SUMMARY:")
    print(f"  ✅ Known patterns (from knowledge base): {known_patterns}")
    print(f"  🆕 Novel patterns (discovered in user data): {novel_patterns}")
    print(f"  📈 Total patterns analyzed: {known_patterns + novel_patterns}")
    
    # Generate research data from USER DATA
    print("\n📋 GENERATING RESEARCH DATA FROM USER DATA")
    print("-" * 50)
    
    print("Generating comprehensive research data from user data...")
    research_data = system.generate_research_data(user_sentences, output_format='json')
    
    print("✅ Research data generated from user data")
    
    # Show metadata
    if 'metadata' in research_data:
        metadata = research_data['metadata']
        print(f"\n📊 Research Data Metadata:")
        print(f"   Source: User data ({len(user_sentences)} sentences)")
        print(f"   Timestamp: {metadata['timestamp']}")
        print(f"   System version: {metadata['system_version']}")
        print(f"   Knowledge sources: {metadata['knowledge_sources']}")
    
    # Discover novel patterns in USER DATA
    print("\n🔍 DISCOVERING NOVEL PATTERNS IN USER DATA")
    print("-" * 50)
    
    print("Discovering novel patterns in user data...")
    discovered_patterns = system.discover_patterns(user_sentences, min_frequency=5)
    
    print(f"✅ Discovered {len(discovered_patterns)} novel patterns in user data")
    
    # Show top discovered patterns
    if discovered_patterns:
        print(f"\n🏆 Top 10 novel patterns discovered in user data:")
        sorted_patterns = sorted(discovered_patterns.items(), 
                               key=lambda x: x[1]['frequency'], reverse=True)
        
        for i, (pattern, data) in enumerate(sorted_patterns[:10], 1):
            print(f"  {i:2d}. {pattern}: {data['frequency']} occurrences")
            print(f"      Type: {data['pattern_type']}")
            print(f"      Confidence: {data['confidence']:.2f}")
    
    # Save research data from USER DATA
    print("\n💾 SAVING RESEARCH DATA FROM USER DATA")
    print("-" * 50)
    
    # Create results directory
    results_dir = Path("research/user_data_analysis")
    results_dir.mkdir(exist_ok=True)
    
    # Save comprehensive results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "data_source": "user_data",
        "user_sentences_count": len(user_sentences),
        "knowledge_base_stats": {
            "reference_words": len(system.knowledge.all_words),
            "known_patterns": len(system.knowledge.gold_standard_patterns),
            "linguistic_categories": len(system.knowledge.word_categories)
        },
        "user_data_analysis": {
            "total_collocations_extracted": total_collocations,
            "known_patterns_found": known_patterns,
            "novel_patterns_discovered": novel_patterns,
            "top_collocations": [
                {
                    "words": result.words,
                    "frequency": result.frequency,
                    "measures": {measure.value: getattr(result, 'score', 0) for measure in AssociationMeasure}
                }
                for result in sorted_collocations[:10]
            ]
        },
        "discovered_patterns": discovered_patterns,
        "research_data": research_data
    }
    
    results_file = results_dir / f"user_data_analysis_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Research data from user data saved to: {results_file}")
    
    # Final summary
    print("\n🎉 CORRECT USAGE DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print("✅ Knowledge base loaded from kcho/data/ (reference data)")
    print("✅ User data analyzed (Bible text - actual research source)")
    print("✅ Knowledge base used to guide pattern classification")
    print("✅ Research data generated from user data, not reference data")
    print("✅ Novel patterns discovered in user data")
    print("✅ Known patterns identified using knowledge base")
    print("✅ All analysis based on user data, informed by knowledge base")
    
    print(f"\n📊 SUMMARY:")
    print(f"  📚 Knowledge base: {len(system.knowledge.all_words)} reference words")
    print(f"  📖 User data analyzed: {len(user_sentences)} sentences")
    print(f"  📊 Collocations extracted: {total_collocations}")
    print(f"  ✅ Known patterns: {known_patterns}")
    print(f"  🆕 Novel patterns: {novel_patterns}")
    
    print(f"\n📁 Results saved to: {results_file}")
    print("🚀 System correctly uses knowledge base to guide analysis of user data!")

def main():
    """Main demonstration function"""
    try:
        demonstrate_correct_usage()
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

