#!/usr/bin/env python3
"""
K'Cho Comprehensive Data Generator

This script generates comprehensive research data using the correct approach:
- Uses kcho/data/ as knowledge base for pattern recognition
- Analyzes user data (Bible, custom text) to generate research findings
- Produces collocations, n-grams, and parallel training data
"""

import sys
import json
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

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

def generate_comprehensive_data():
    """Generate comprehensive research data from user data"""
    
    print("🎯 K'CHO COMPREHENSIVE DATA GENERATION")
    print("=" * 80)
    print("CORRECT APPROACH:")
    print("• kcho/data/ = Knowledge base for pattern recognition")
    print("• User data (Bible) = Source for research data generation")
    print("• Producing: Collocations, N-grams, Parallel training data")
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
    
    # Load user data (Bible text)
    print("\n📖 LOADING USER DATA (Bible Text)")
    print("-" * 50)
    
    bible_file = "data/bible_versions/2642_bible_data.json"
    
    if not Path(bible_file).exists():
        print(f"❌ Bible data not found: {bible_file}")
        return None
    
    print(f"📚 Loading Bible data from: {bible_file}")
    user_sentences = TextLoader.load_from_file(bible_file)
    print(f"✅ Loaded {len(user_sentences)} sentences from Bible data")
    
    # Use subset for comprehensive analysis
    if len(user_sentences) > 5000:
        print(f"📊 Using first 5000 sentences for comprehensive analysis (total: {len(user_sentences)})")
        user_sentences = user_sentences[:5000]
    
    print(f"🎯 USER DATA TO ANALYZE: {len(user_sentences)} sentences")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"research/comprehensive_data_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # 1. Generate Collocations
    print("\n📊 GENERATING COLLOCATIONS")
    print("-" * 50)
    
    extractor = CollocationExtractor(
        window_size=5,
        min_freq=10,  # Higher threshold for larger corpus
        measures=[
            AssociationMeasure.PMI,
            AssociationMeasure.TSCORE,
            AssociationMeasure.DICE,
            AssociationMeasure.LOG_LIKELIHOOD,
            AssociationMeasure.NPMI
        ]
    )
    
    print(f"Extracting collocations from {len(user_sentences)} user sentences...")
    collocation_results = extractor.extract(user_sentences)
    
    print("✅ Collocation extraction completed")
    
    # Process collocation results
    collocation_data = {}
    total_collocations = 0
    
    for measure, results in collocation_results.items():
        print(f"📈 {measure.value.upper()}: {len(results)} collocations")
        total_collocations += len(results)
        
        # Convert to serializable format
        collocation_data[measure.value] = [
            {
                'words': result.words,
                'score': result.score,
                'frequency': result.frequency,
                'context_examples': result.context_examples[:3] if result.context_examples else []
            }
            for result in results
        ]
    
    print(f"🎯 TOTAL COLLOCATIONS EXTRACTED: {total_collocations}")
    
    # Save collocations
    collocation_file = output_dir / "collocations.json"
    with open(collocation_file, 'w', encoding='utf-8') as f:
        json.dump(collocation_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Collocations saved to: {collocation_file}")
    
    # 2. Generate N-grams
    print("\n🔤 GENERATING N-GRAMS")
    print("-" * 50)
    
    print("Extracting n-grams (bigrams, trigrams, 4-grams)...")
    
    ngram_data = {
        'bigrams': defaultdict(int),
        'trigrams': defaultdict(int),
        'fourgrams': defaultdict(int),
        'fivegrams': defaultdict(int)
    }
    
    for sentence in user_sentences:
        tokens = sentence.split()
        
        # Bigrams
        for i in range(len(tokens) - 1):
            bigram = ' '.join(tokens[i:i+2])
            ngram_data['bigrams'][bigram] += 1
        
        # Trigrams
        for i in range(len(tokens) - 2):
            trigram = ' '.join(tokens[i:i+3])
            ngram_data['trigrams'][trigram] += 1
        
        # 4-grams
        for i in range(len(tokens) - 3):
            fourgram = ' '.join(tokens[i:i+4])
            ngram_data['fourgrams'][fourgram] += 1
        
        # 5-grams
        for i in range(len(tokens) - 4):
            fivegram = ' '.join(tokens[i:i+5])
            ngram_data['fivegrams'][fivegram] += 1
    
    # Convert to regular dicts and filter by frequency
    min_freq = 5
    filtered_ngrams = {}
    
    for ngram_type, ngrams in ngram_data.items():
        filtered = {ngram: freq for ngram, freq in ngrams.items() if freq >= min_freq}
        filtered_ngrams[ngram_type] = dict(sorted(filtered.items(), key=lambda x: x[1], reverse=True))
        print(f"📈 {ngram_type}: {len(filtered)} n-grams (freq >= {min_freq})")
    
    # Save n-grams
    ngram_file = output_dir / "ngrams.json"
    with open(ngram_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_ngrams, f, ensure_ascii=False, indent=2)
    
    print(f"✅ N-grams saved to: {ngram_file}")
    
    # 3. Generate Parallel Training Data
    print("\n📚 GENERATING PARALLEL TRAINING DATA")
    print("-" * 50)
    
    print("Generating parallel training data from user sentences...")
    
    # Create parallel training data structure
    parallel_data = []
    
    for i, sentence in enumerate(user_sentences):
        # Tokenize sentence
        tokens = sentence.split()
        
        # Create parallel training entry
        parallel_entry = {
            'id': i + 1,
            'kcho_text': sentence,
            'kcho_tokens': tokens,
            'token_count': len(tokens),
            'sentence_length': len(sentence),
            'has_punctuation': any(char in sentence for char in '.!?'),
            'word_types': list(set(tokens)),  # Unique words
            'word_count': len(set(tokens))
        }
        
        parallel_data.append(parallel_entry)
    
    print(f"✅ Generated {len(parallel_data)} parallel training entries")
    
    # Save parallel training data
    parallel_file = output_dir / "parallel_training_data.json"
    with open(parallel_file, 'w', encoding='utf-8') as f:
        json.dump(parallel_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Parallel training data saved to: {parallel_file}")
    
    # 4. Generate Pattern Discovery
    print("\n🔍 GENERATING PATTERN DISCOVERY")
    print("-" * 50)
    
    print("Discovering novel patterns in user data...")
    discovered_patterns = system.discover_patterns(user_sentences, min_frequency=10)
    
    print(f"✅ Discovered {len(discovered_patterns)} novel patterns")
    
    # Save discovered patterns
    patterns_file = output_dir / "discovered_patterns.json"
    with open(patterns_file, 'w', encoding='utf-8') as f:
        json.dump(discovered_patterns, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Discovered patterns saved to: {patterns_file}")
    
    # 5. Generate Statistical Analysis
    print("\n📈 GENERATING STATISTICAL ANALYSIS")
    print("-" * 50)
    
    print("Generating comprehensive statistical analysis...")
    
    # Token statistics
    all_tokens = []
    for sentence in user_sentences:
        all_tokens.extend(sentence.split())
    
    token_stats = {
        'total_tokens': len(all_tokens),
        'unique_tokens': len(set(all_tokens)),
        'total_sentences': len(user_sentences),
        'average_sentence_length': sum(len(s.split()) for s in user_sentences) / len(user_sentences),
        'token_frequency': dict(Counter(all_tokens).most_common(100)),
        'sentence_length_distribution': {
            'short': len([s for s in user_sentences if len(s.split()) <= 5]),
            'medium': len([s for s in user_sentences if 5 < len(s.split()) <= 15]),
            'long': len([s for s in user_sentences if len(s.split()) > 15])
        }
    }
    
    # Save statistical analysis
    stats_file = output_dir / "statistical_analysis.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(token_stats, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Statistical analysis saved to: {stats_file}")
    
    # 6. Generate CSV Exports
    print("\n📊 GENERATING CSV EXPORTS")
    print("-" * 50)
    
    # Collocations CSV
    collocation_csv = output_dir / "collocations.csv"
    with open(collocation_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['words', 'measure', 'score', 'frequency', 'context_examples'])
        
        for measure, results in collocation_data.items():
            for result in results:
                writer.writerow([
                    ' '.join(result['words']),
                    measure,
                    result['score'],
                    result['frequency'],
                    '; '.join(result['context_examples'])
                ])
    
    print(f"✅ Collocations CSV saved to: {collocation_csv}")
    
    # N-grams CSV
    ngram_csv = output_dir / "ngrams.csv"
    with open(ngram_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ngram', 'type', 'frequency'])
        
        for ngram_type, ngrams in filtered_ngrams.items():
            for ngram, freq in ngrams.items():
                writer.writerow([ngram, ngram_type, freq])
    
    print(f"✅ N-grams CSV saved to: {ngram_csv}")
    
    # Parallel training CSV
    parallel_csv = output_dir / "parallel_training_data.csv"
    with open(parallel_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'kcho_text', 'token_count', 'sentence_length', 'word_count'])
        
        for entry in parallel_data:
            writer.writerow([
                entry['id'],
                entry['kcho_text'],
                entry['token_count'],
                entry['sentence_length'],
                entry['word_count']
            ])
    
    print(f"✅ Parallel training CSV saved to: {parallel_csv}")
    
    # 7. Generate Summary Report
    print("\n📋 GENERATING SUMMARY REPORT")
    print("-" * 50)
    
    summary_report = {
        "generation_timestamp": datetime.now().isoformat(),
        "data_source": "user_data_bible",
        "analysis_scope": {
            "total_sentences": len(user_sentences),
            "knowledge_base_words": len(system.knowledge.all_words),
            "knowledge_base_patterns": len(system.knowledge.gold_standard_patterns)
        },
        "generated_data": {
            "collocations": {
                "total_extracted": total_collocations,
                "measures": list(collocation_data.keys()),
                "file": str(collocation_file)
            },
            "ngrams": {
                "types": list(filtered_ngrams.keys()),
                "total_ngrams": sum(len(ngrams) for ngrams in filtered_ngrams.values()),
                "file": str(ngram_file)
            },
            "parallel_training": {
                "total_entries": len(parallel_data),
                "file": str(parallel_file)
            },
            "discovered_patterns": {
                "total_patterns": len(discovered_patterns),
                "file": str(patterns_file)
            },
            "statistical_analysis": {
                "file": str(stats_file)
            }
        },
        "output_files": {
            "json_files": [
                str(collocation_file),
                str(ngram_file),
                str(parallel_file),
                str(patterns_file),
                str(stats_file)
            ],
            "csv_files": [
                str(collocation_csv),
                str(ngram_csv),
                str(parallel_csv)
            ]
        }
    }
    
    # Save summary report
    summary_file = output_dir / "summary_report.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Summary report saved to: {summary_file}")
    
    # Final summary
    print("\n🎉 COMPREHENSIVE DATA GENERATION COMPLETE!")
    print("=" * 80)
    print("✅ Knowledge base used for pattern recognition")
    print("✅ User data analyzed for research findings")
    print("✅ Collocations extracted with multiple measures")
    print("✅ N-grams generated (bigrams to 5-grams)")
    print("✅ Parallel training data created")
    print("✅ Novel patterns discovered")
    print("✅ Statistical analysis completed")
    print("✅ CSV exports generated")
    print("✅ All data saved to research directory")
    
    print(f"\n📊 SUMMARY:")
    print(f"  📚 Knowledge base: {len(system.knowledge.all_words)} reference words")
    print(f"  📖 User data analyzed: {len(user_sentences)} sentences")
    print(f"  📊 Collocations extracted: {total_collocations}")
    print(f"  🔤 N-grams generated: {sum(len(ngrams) for ngrams in filtered_ngrams.values())}")
    print(f"  📚 Parallel training entries: {len(parallel_data)}")
    print(f"  🔍 Novel patterns discovered: {len(discovered_patterns)}")
    
    print(f"\n📁 All data saved to: {output_dir}")
    print("🚀 Data ready for ML training and linguistic research!")
    
    return summary_report

def main():
    """Main data generation function"""
    try:
        summary_report = generate_comprehensive_data()
        
        if summary_report:
            print(f"\n✅ Data generation completed successfully!")
            print(f"📁 Output directory: {summary_report['generated_data']['collocations']['file'].split('/')[0]}")
        else:
            print("❌ Data generation failed")
            
    except Exception as e:
        print(f"❌ Error during data generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

