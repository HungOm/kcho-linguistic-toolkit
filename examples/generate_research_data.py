#!/usr/bin/env python3
"""
K'Cho Research Data Generator

This script generates comprehensive research data using the enhanced KchoSystem
with real K'Cho data from various sources.
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

def generate_research_data():
    """Generate comprehensive research data using enhanced KchoSystem"""
    
    print("🎯 K'CHO RESEARCH DATA GENERATION")
    print("=" * 60)
    
    # Initialize enhanced system
    print("🚀 Initializing Enhanced KchoSystem...")
    system = EnhancedKchoSystem(use_comprehensive_knowledge=True)
    
    # Load sample corpus
    print("📚 Loading sample corpus...")
    sample_corpus_path = "data/sample_corpus.txt"
    
    if Path(sample_corpus_path).exists():
        with open(sample_corpus_path, 'r', encoding='utf-8') as f:
            sample_sentences = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"  ✅ Loaded {len(sample_sentences)} sentences from sample corpus")
    else:
        print("  ⚠️  Sample corpus not found, using default sentences")
        sample_sentences = [
            "Om noh Yóng am pàapai pe ci.",
            "Yóng am pàapai pe ci ah k'chàang ka hngu ci.",
            "Om noh Yóng am a péit ah pàapai ka hngu ci.",
            "Ak'hmó noh k'khìm luum-na ci.",
            "Om lah Tam noh htung cuh ui thah-na ci goi.",
            "K'am zòi ci ah k'chàang lo ci.",
            "Ui noh vok htui ci.",
            "Khò nàa ci.",
            "Àihli ng'dáng lo ci.",
            "Cun ah k'chàang noh Yóng am pàapai pe ci."
        ]
    
    # Generate comprehensive research data
    print("\n🔬 Generating comprehensive research data...")
    research_data = system.generate_research_data(sample_sentences, output_format='json')
    
    # Save research data
    output_file = f"research/kcho_research_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path("research").mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(research_data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ Research data saved to: {output_file}")
    
    # Generate pattern discovery data
    print("\n🔍 Generating pattern discovery data...")
    discovered_patterns = system.discover_patterns(sample_sentences, min_frequency=2)
    
    pattern_file = f"research/discovered_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(pattern_file, 'w', encoding='utf-8') as f:
        json.dump(discovered_patterns, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ Discovered patterns saved to: {pattern_file}")
    
    # Generate collocation data
    print("\n📊 Generating collocation analysis...")
    extractor = CollocationExtractor(
        window_size=5,
        min_freq=2,
        measures=[AssociationMeasure.PMI, AssociationMeasure.TSCORE, AssociationMeasure.DICE]
    )
    
    collocation_results = extractor.extract(sample_sentences)
    
    # Convert results to serializable format
    collocation_data = {}
    for measure, results in collocation_results.items():
        collocation_data[measure.value] = [
            {
                'words': result.words,
                'score': result.score,
                'frequency': result.frequency,
                'context_examples': result.context_examples[:3] if result.context_examples else []
            }
            for result in results[:10]  # Top 10 for each measure
        ]
    
    collocation_file = f"research/collocation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(collocation_file, 'w', encoding='utf-8') as f:
        json.dump(collocation_data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ Collocation analysis saved to: {collocation_file}")
    
    # Generate API responses for LLaMA integration
    print("\n🤖 Generating API responses for LLaMA integration...")
    
    api_queries = [
        "What are the most common verb-particle patterns in K'Cho?",
        "Tell me about K'Cho morphological patterns and suffixes",
        "How does K'Cho syntax work with postpositions?",
        "What vocabulary data is available for K'Cho language learning?",
        "Give me a comprehensive overview of K'Cho linguistic features"
    ]
    
    api_responses = {}
    for i, query in enumerate(api_queries, 1):
        print(f"  Processing query {i}/5: {query[:50]}...")
        response = system.get_api_response(query)
        api_responses[f"query_{i}"] = response
    
    api_file = f"research/api_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(api_file, 'w', encoding='utf-8') as f:
        json.dump(api_responses, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ API responses saved to: {api_file}")
    
    # Generate comprehensive linguistic analysis
    print("\n🧠 Generating comprehensive linguistic analysis...")
    linguistic_analysis = system.conduct_linguistic_research(sample_sentences, research_focus='all')
    
    analysis_file = f"research/linguistic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(linguistic_analysis, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ Linguistic analysis saved to: {analysis_file}")
    
    # Generate summary report
    print("\n📋 Generating summary report...")
    
    summary_report = {
        "generation_timestamp": datetime.now().isoformat(),
        "corpus_info": {
            "total_sentences": len(sample_sentences),
            "sample_sentences": sample_sentences[:5]  # First 5 sentences
        },
        "knowledge_base_stats": {
            "total_words": len(system.knowledge.all_words),
            "gold_standard_patterns": len(system.knowledge.gold_standard_patterns),
            "word_categories": len(system.knowledge.word_categories),
            "data_sources": system.knowledge.data_sources
        },
        "research_outputs": {
            "research_data_file": output_file,
            "discovered_patterns_file": pattern_file,
            "collocation_analysis_file": collocation_file,
            "api_responses_file": api_file,
            "linguistic_analysis_file": analysis_file
        },
        "key_findings": {
            "total_discovered_patterns": len(discovered_patterns),
            "top_collocation_measures": list(collocation_data.keys()),
            "linguistic_insights_generated": len(linguistic_analysis.get('linguistic_insights', {})),
            "api_queries_processed": len(api_queries)
        }
    }
    
    summary_file = f"research/research_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ Summary report saved to: {summary_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎉 RESEARCH DATA GENERATION COMPLETE!")
    print("=" * 60)
    print(f"📊 Generated {len(sample_sentences)} sentences analyzed")
    print(f"🔍 Discovered {len(discovered_patterns)} novel patterns")
    print(f"📈 Analyzed collocations with {len(collocation_data)} measures")
    print(f"🤖 Processed {len(api_queries)} API queries")
    print(f"🧠 Generated comprehensive linguistic analysis")
    print(f"📁 All outputs saved to research/ directory")
    print("\n📋 Generated Files:")
    print(f"  • {output_file}")
    print(f"  • {pattern_file}")
    print(f"  • {collocation_file}")
    print(f"  • {api_file}")
    print(f"  • {analysis_file}")
    print(f"  • {summary_file}")
    
    return summary_report

def analyze_bible_data():
    """Analyze Bible data if available"""
    print("\n📖 ANALYZING BIBLE DATA")
    print("=" * 40)
    
    bible_file = "data/bible_versions/2642_bible_data.json"
    
    if not Path(bible_file).exists():
        print(f"  ⚠️  Bible data not found: {bible_file}")
        return None
    
    print(f"  📚 Loading Bible data from: {bible_file}")
    
    try:
        # Load Bible data using TextLoader
        sentences = TextLoader.load_from_file(bible_file)
        print(f"  ✅ Loaded {len(sentences)} sentences from Bible data")
        
        if len(sentences) > 1000:
            print(f"  📊 Using first 1000 sentences for analysis (total: {len(sentences)})")
            sentences = sentences[:1000]
        
        # Initialize system
        system = EnhancedKchoSystem(use_comprehensive_knowledge=True)
        
        # Generate research data
        print("  🔬 Generating Bible research data...")
        research_data = system.generate_research_data(sentences, output_format='json')
        
        # Save Bible analysis
        bible_output = f"research/bible_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(bible_output, 'w', encoding='utf-8') as f:
            json.dump(research_data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✅ Bible analysis saved to: {bible_output}")
        
        # Generate collocation analysis for Bible data
        print("  📊 Generating Bible collocation analysis...")
        extractor = CollocationExtractor(
            window_size=5,
            min_freq=10,  # Higher threshold for larger corpus
            measures=[AssociationMeasure.PMI, AssociationMeasure.TSCORE, AssociationMeasure.DICE]
        )
        
        collocation_results = extractor.extract(sentences)
        
        # Convert to serializable format
        bible_collocations = {}
        for measure, results in collocation_results.items():
            bible_collocations[measure.value] = [
                {
                    'words': result.words,
                    'score': result.score,
                    'frequency': result.frequency,
                    'context_examples': result.context_examples[:2] if result.context_examples else []
                }
                for result in results[:20]  # Top 20 for each measure
            ]
        
        bible_collocation_file = f"research/bible_collocations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(bible_collocation_file, 'w', encoding='utf-8') as f:
            json.dump(bible_collocations, f, ensure_ascii=False, indent=2)
        
        print(f"  ✅ Bible collocations saved to: {bible_collocation_file}")
        
        return {
            "bible_analysis": bible_output,
            "bible_collocations": bible_collocation_file,
            "total_sentences": len(sentences)
        }
        
    except Exception as e:
        print(f"  ❌ Error analyzing Bible data: {e}")
        return None

def main():
    """Main function to generate all research data"""
    print("🎯 K'CHO RESEARCH DATA GENERATOR")
    print("=" * 80)
    print("This script generates comprehensive research data using the enhanced KchoSystem")
    print("with real K'Cho data from various sources.")
    print("=" * 80)
    
    try:
        # Generate main research data
        summary_report = generate_research_data()
        
        # Analyze Bible data if available
        bible_analysis = analyze_bible_data()
        
        if bible_analysis:
            summary_report["bible_analysis"] = bible_analysis
        
        print("\n🎉 ALL RESEARCH DATA GENERATION COMPLETE!")
        print("=" * 80)
        print("✅ Enhanced KchoSystem successfully generated comprehensive research data")
        print("✅ All outputs saved to research/ directory")
        print("✅ Data ready for LLaMA integration and ML model training")
        print("✅ Unbiased linguistic analysis completed")
        print("✅ Novel pattern discovery accomplished")
        
    except Exception as e:
        print(f"❌ Error during research data generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

