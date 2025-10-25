#!/usr/bin/env python3
"""
K'Cho Collocation Extraction Tutorial

This tutorial demonstrates how to use the K'Cho Linguistic Toolkit to extract
collocations from any K'Cho text file with comprehensive linguistic analysis.

Features covered:
- Universal text loading (txt/json files)
- All statistical measures (PMI, T-Score, Dice, etc.)
- Linguistic pattern classification (VP, PP, APP, etc.)
- Configuration management
- Multiple output formats
- CLI and Python API usage
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from kcho.config import load_config
from kcho.text_loader import TextLoader
from kcho.collocation import CollocationExtractor, AssociationMeasure, LinguisticPattern


def main():
    """Main tutorial function."""
    print("🚀 K'Cho Collocation Extraction Tutorial")
    print("=" * 50)
    
    # Load configuration
    config_path = project_root / "config.yaml"
    if config_path.exists():
        config = load_config(str(config_path))
        print(f"✅ Loaded configuration from {config_path}")
    else:
        print("⚠️  No config file found, using defaults")
        config = load_config()
    
    # Tutorial 1: Process any JSON data
    print("\n📖 Tutorial 1: Processing JSON Data")
    print("-" * 40)
    
    # Check if user has their own JSON file
    json_path = project_root / "data" / "your_corpus.json"
    if json_path.exists():
        print(f"Loading JSON data from {json_path}")
        
        # Load text using universal loader
        sentences = TextLoader.load_from_file(str(json_path))
        print(f"✅ Loaded {len(sentences)} sentences")
        
        # Show sample sentences
        print("\nSample sentences:")
        for i, sentence in enumerate(sentences[:3]):
            print(f"  {i+1}. {sentence[:80]}...")
        
        # Extract collocations with all measures
        print(f"\nExtracting collocations...")
        print(f"  Window size: {config.collocation.window_size}")
        print(f"  Min frequency: {config.collocation.min_freq}")
        print(f"  Measures: {config.collocation.measures}")
        print(f"  Patterns: {config.collocation.linguistic_patterns}")
        
        # Create extractor
        measure_enums = []
        for measure_name in config.collocation.measures:
            try:
                measure_enums.append(AssociationMeasure[measure_name.upper()])
            except KeyError:
                print(f"⚠️  Unknown measure: {measure_name}")
        
        extractor = CollocationExtractor(
            window_size=config.collocation.window_size,
            min_freq=config.collocation.min_freq,
            measures=measure_enums
        )
        
        # Extract with pattern classification
        results = extractor.extract_with_patterns(sentences, config.collocation.linguistic_patterns)
        
        # Analyze results
        print(f"\n📊 Results Analysis:")
        total_collocations = sum(len(collocations) for collocations in results.values())
        print(f"  Total collocations found: {total_collocations}")
        
        # Group by linguistic pattern
        pattern_counts = {}
        for measure, collocations in results.items():
            for coll in collocations:
                pattern = coll.linguistic_pattern.value if coll.linguistic_pattern else 'UNKNOWN'
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        print(f"\nLinguistic Pattern Distribution:")
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern}: {count}")
        
        # Show top collocations by pattern
        print(f"\nTop Collocations by Pattern:")
        pattern_groups = {}
        for measure, collocations in results.items():
            for coll in collocations:
                pattern = coll.linguistic_pattern.value if coll.linguistic_pattern else 'UNKNOWN'
                if pattern not in pattern_groups:
                    pattern_groups[pattern] = []
                pattern_groups[pattern].append(coll)
        
        for pattern, collocations in pattern_groups.items():
            print(f"\n=== {pattern.upper()} ===")
            # Sort by frequency and show top 5
            top_collocations = sorted(collocations, key=lambda x: x.frequency, reverse=True)[:5]
            for i, coll in enumerate(top_collocations, 1):
                print(f"  {i}. {' '.join(coll.words)} (freq: {coll.frequency}, score: {coll.score:.3f})")
                if coll.context_examples:
                    print(f"     Example: \"{coll.context_examples[0][:60]}...\"")
    
    else:
        print(f"ℹ️  No JSON corpus found at {json_path}")
        print(f"   Place your K'Cho JSON file there to test JSON processing")
    
    # Tutorial 2: Process sample corpus
    print(f"\n\n📝 Tutorial 2: Processing Sample Corpus")
    print("-" * 40)
    
    sample_path = project_root / "data" / "sample_corpus.txt"
    if sample_path.exists():
        print(f"Loading sample corpus from {sample_path}")
        
        sentences = TextLoader.load_from_file(str(sample_path))
        print(f"✅ Loaded {len(sentences)} sentences")
        
        # Quick extraction with fewer measures for demo
        extractor = CollocationExtractor(
            window_size=5,
            min_freq=2,
            measures=[AssociationMeasure.PMI, AssociationMeasure.TSCORE]
        )
        
        results = extractor.extract_with_patterns(sentences)
        
        print(f"\nSample Corpus Results:")
        total_collocations = sum(len(collocations) for collocations in results.values())
        print(f"  Total collocations: {total_collocations}")
        
        # Show top PMI collocations
        if AssociationMeasure.PMI in results:
            print(f"\nTop PMI Collocations:")
            pmi_collocations = sorted(results[AssociationMeasure.PMI], 
                                    key=lambda x: x.score, reverse=True)[:10]
            for i, coll in enumerate(pmi_collocations, 1):
                pattern = coll.linguistic_pattern.value if coll.linguistic_pattern else 'UNK'
                print(f"  {i}. {' '.join(coll.words)} (PMI: {coll.score:.3f}, freq: {coll.frequency}, pattern: {pattern})")
    
    else:
        print(f"❌ Sample corpus not found at {sample_path}")
    
    # Tutorial 3: Configuration and CLI usage
    print(f"\n\n⚙️  Tutorial 3: Configuration and CLI Usage")
    print("-" * 40)
    
    print("You can customize the extraction by:")
    print("1. Editing config.yaml file")
    print("2. Using CLI arguments")
    print("3. Setting environment variables")
    
    print(f"\nCLI Usage Examples:")
    print(f"  # Extract from any K'Cho text file with all measures")
    print(f"  python -m kcho.kcho_app extract-all-collocations \\")
    print(f"    --input your_corpus.txt \\")
    print(f"    --output results/collocations \\")
    print(f"    --format csv json txt")
    
    print(f"\n  # Extract with custom settings")
    print(f"  python -m kcho.kcho_app extract-all-collocations \\")
    print(f"    --input your_corpus.json \\")
    print(f"    --output results/my_collocations \\")
    print(f"    --window-size 3 \\")
    print(f"    --min-freq 10 \\")
    print(f"    --measures pmi tscore dice")
    
    print(f"\n  # Create configuration template")
    print(f"  python -m kcho.kcho_app create-config --output my_config.yaml")
    
    print(f"\n✅ Tutorial complete!")
    print(f"\nKey Features Demonstrated:")
    print(f"  ✓ Universal text loading (txt, json)")
    print(f"  ✓ All statistical measures (PMI, T-Score, Dice, etc.)")
    print(f"  ✓ Linguistic pattern classification (VP, PP, APP, etc.)")
    print(f"  ✓ Configurable via config file and CLI")
    print(f"  ✓ Multiple output formats (CSV, JSON, TXT)")
    print(f"  ✓ Context examples for each collocation")


if __name__ == "__main__":
    main()