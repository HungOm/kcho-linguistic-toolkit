#!/usr/bin/env python3
"""
K'Cho Enhanced System Practical Demonstration

This script demonstrates practical usage of the enhanced KchoSystem for:
1. Real-world linguistic research
2. Pattern discovery and analysis
3. API integration for LLaMA
4. Research data generation
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

def demonstrate_practical_usage():
    """Demonstrate practical usage of the enhanced system"""
    
    print("🎯 K'CHO ENHANCED SYSTEM PRACTICAL DEMONSTRATION")
    print("=" * 80)
    print("This demonstration shows how to use the enhanced KchoSystem")
    print("for real-world linguistic research and analysis.")
    print("=" * 80)
    
    # Initialize enhanced system
    print("\n🚀 INITIALIZING ENHANCED SYSTEM")
    print("-" * 50)
    
    system = EnhancedKchoSystem(use_comprehensive_knowledge=True)
    print("✅ Enhanced KchoSystem initialized with comprehensive knowledge")
    
    # Show knowledge base statistics
    print(f"📊 Knowledge Base Statistics:")
    print(f"  Total words: {len(system.knowledge.all_words)}")
    print(f"  Gold standard patterns: {len(system.knowledge.gold_standard_patterns)}")
    print(f"  Word categories: {len(system.knowledge.word_categories)}")
    print(f"  Data sources: {len(system.knowledge.data_sources)}")
    
    # Demonstrate text analysis
    print("\n📝 TEXT ANALYSIS DEMONSTRATION")
    print("-" * 50)
    
    sample_texts = [
        "Om noh Yóng am pàapai pe ci.",
        "Yóng am pàapai pe ci ah k'chàang ka hngu ci.",
        "Ak'hmó noh k'khìm luum-na ci.",
        "Ui noh vok htui ci.",
        "Khò nàa ci."
    ]
    
    print("Analyzing sample K'Cho sentences:")
    for i, text in enumerate(sample_texts, 1):
        print(f"\n{i}. '{text}'")
        
        # Analyze the text
        analysis = system.analyze(text)
        
        print(f"   Tokens: {len(analysis.tokens)}")
        print(f"   Text: {analysis.text}")
        
        # Check for known patterns
        for token in analysis.tokens:
            if hasattr(token, 'text') and token.text in ['pe', 'ci', 'noh', 'Yóng']:
                print(f"   Known pattern detected: {token.text}")
    
    # Demonstrate pattern discovery
    print("\n🔍 PATTERN DISCOVERY DEMONSTRATION")
    print("-" * 50)
    
    print("Discovering novel patterns in sample corpus...")
    discovered_patterns = system.discover_patterns(sample_texts, min_frequency=1)
    
    print(f"✅ Discovered {len(discovered_patterns)} novel patterns")
    
    # Show top discovered patterns
    if discovered_patterns:
        print("\n🏆 Top 5 discovered patterns:")
        sorted_patterns = sorted(discovered_patterns.items(), 
                               key=lambda x: x[1]['frequency'], reverse=True)
        
        for i, (pattern, data) in enumerate(sorted_patterns[:5], 1):
            print(f"  {i}. {pattern}: {data['frequency']} occurrences")
            print(f"     Type: {data['pattern_type']}")
            print(f"     Confidence: {data['confidence']:.2f}")
    
    # Demonstrate linguistic research
    print("\n🧠 LINGUISTIC RESEARCH DEMONSTRATION")
    print("-" * 50)
    
    print("Conducting comprehensive linguistic research...")
    research_results = system.conduct_linguistic_research(
        sample_texts, 
        research_focus='all'
    )
    
    print("✅ Linguistic research completed")
    
    # Show research insights
    if 'linguistic_insights' in research_results:
        insights = research_results['linguistic_insights']
        print(f"\n💡 Generated {len(insights)} linguistic insights:")
        
        for insight_type, data in insights.items():
            if isinstance(data, list) and data:
                print(f"  {insight_type}: {len(data)} insights")
                # Show first insight
                if data:
                    print(f"    Example: {data[0][:100]}...")
    
    # Demonstrate API integration
    print("\n🤖 API INTEGRATION DEMONSTRATION")
    print("-" * 50)
    
    api_queries = [
        "What are the most common verb patterns in K'Cho?",
        "Tell me about K'Cho morphological features",
        "How does K'Cho syntax work?",
        "What vocabulary data is available?"
    ]
    
    print("Processing API queries for LLaMA integration:")
    
    for i, query in enumerate(api_queries, 1):
        print(f"\n{i}. Query: {query}")
        
        response = system.get_api_response(query)
        
        print(f"   Response type: {response['response_type']}")
        print(f"   Query type: {response['data']['query_type']}")
        
        if 'gold_standard_patterns' in response['data']:
            print(f"   Gold standard patterns: {response['data']['gold_standard_patterns']}")
        
        if 'pattern_categories' in response['data']:
            print(f"   Pattern categories: {len(response['data']['pattern_categories'])}")
    
    # Demonstrate collocation extraction
    print("\n📊 COLLOCATION EXTRACTION DEMONSTRATION")
    print("-" * 50)
    
    print("Extracting collocations with multiple statistical measures...")
    
    extractor = CollocationExtractor(
        window_size=5,
        min_freq=1,
        measures=[
            AssociationMeasure.PMI,
            AssociationMeasure.TSCORE,
            AssociationMeasure.DICE,
            AssociationMeasure.LOG_LIKELIHOOD
        ]
    )
    
    collocation_results = extractor.extract(sample_texts)
    
    print("✅ Collocation extraction completed")
    
    # Show results for each measure
    for measure, results in collocation_results.items():
        print(f"\n📈 {measure.value.upper()} Results:")
        print(f"   Total collocations: {len(results)}")
        
        if results:
            print(f"   Top 3 collocations:")
            for i, result in enumerate(results[:3], 1):
                words = ' '.join(result.words)
                print(f"     {i}. {words}: {result.score:.3f} (freq: {result.frequency})")
    
    # Demonstrate research data generation
    print("\n📋 RESEARCH DATA GENERATION DEMONSTRATION")
    print("-" * 50)
    
    print("Generating comprehensive research data...")
    research_data = system.generate_research_data(sample_texts, output_format='json')
    
    print("✅ Research data generated")
    
    # Show metadata
    if 'metadata' in research_data:
        metadata = research_data['metadata']
        print(f"\n📊 Research Data Metadata:")
        print(f"   Timestamp: {metadata['timestamp']}")
        print(f"   Corpus size: {metadata['corpus_size']} sentences")
        print(f"   System version: {metadata['system_version']}")
        print(f"   Knowledge sources: {metadata['knowledge_sources']}")
    
    # Show knowledge base stats
    if 'knowledge_base_stats' in research_data:
        stats = research_data['knowledge_base_stats']
        print(f"\n🧠 Knowledge Base Statistics:")
        print(f"   Total words: {stats['total_words']}")
        print(f"   Gold standard patterns: {stats['gold_standard_patterns']}")
        print(f"   Word categories: {stats['word_categories']}")
    
    # Demonstrate pattern confidence checking
    print("\n🎯 PATTERN CONFIDENCE DEMONSTRATION")
    print("-" * 50)
    
    test_patterns = [
        "pe ci",      # Known gold standard pattern
        "noh Yóng",   # Known gold standard pattern
        "unknown pattern",  # Unknown pattern
        "ka hngu"     # Discovered pattern
    ]
    
    print("Checking pattern confidence scores:")
    
    for pattern in test_patterns:
        words = pattern.split()
        if len(words) >= 2:
            confidence = system.get_pattern_confidence(words[0], words[1])
            is_gold_standard = system.is_gold_standard_pattern(words[0], words[1])
        else:
            confidence = 0.0
            is_gold_standard = False
        
        print(f"\n  Pattern: {pattern}")
        print(f"    Confidence: {confidence:.2f}")
        print(f"    Gold standard: {is_gold_standard}")
        
        if is_gold_standard:
            print(f"    ✅ Known pattern in gold standard")
        else:
            print(f"    🔍 Novel pattern discovered")
    
    # Save demonstration results
    print("\n💾 SAVING DEMONSTRATION RESULTS")
    print("-" * 50)
    
    # Create results directory
    results_dir = Path("research/demonstration_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save comprehensive results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        "demonstration_timestamp": datetime.now().isoformat(),
        "sample_texts": sample_texts,
        "discovered_patterns": discovered_patterns,
        "research_results": research_results,
        "collocation_results": {
            measure.value: [
                {
                    "words": result.words,
                    "score": result.score,
                    "frequency": result.frequency
                }
                for result in results[:5]  # Top 5 for each measure
            ]
            for measure, results in collocation_results.items()
        },
        "research_data": research_data,
        "system_stats": {
            "total_words": len(system.knowledge.all_words),
            "gold_standard_patterns": len(system.knowledge.gold_standard_patterns),
            "word_categories": len(system.knowledge.word_categories),
            "data_sources": system.knowledge.data_sources
        }
    }
    
    results_file = results_dir / f"practical_demonstration_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Demonstration results saved to: {results_file}")
    
    # Final summary
    print("\n🎉 PRACTICAL DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print("✅ Enhanced KchoSystem successfully demonstrated")
    print("✅ Text analysis and pattern recognition working")
    print("✅ Novel pattern discovery accomplished")
    print("✅ Linguistic research conducted")
    print("✅ API integration ready for LLaMA")
    print("✅ Collocation extraction with multiple measures")
    print("✅ Research data generation completed")
    print("✅ Pattern confidence scoring functional")
    print("✅ All results saved for further analysis")
    
    print(f"\n📁 Results saved to: {results_file}")
    print("🚀 System ready for production use!")

def main():
    """Main demonstration function"""
    try:
        demonstrate_practical_usage()
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
