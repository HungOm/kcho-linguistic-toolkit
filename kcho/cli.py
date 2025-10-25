"""
K'Cho CLI Application

Author: Hung Om
Research Foundation: Based on foundational K'Cho linguistic research and Austroasiatic language studies
Version: 0.2.0
Date: 2024-10-25

Abstract:
Comprehensive command-line interface for K'Cho language processing toolkit.
Provides access to all analysis tools, research capabilities, and data management
functions for computational linguistic analysis of K'Cho language structure.
"""

import click
import logging
from pathlib import Path
from typing import List

from .kcho_system import KchoSystem, PatternDiscoveryEngine, LinguisticResearchEngine
from .normalize import normalize_text, tokenize
from .collocation import AssociationMeasure, LinguisticPattern
from .ngram_collocation import NGramCollocationExtractor
from .export import to_csv, to_json, to_text
from .evaluation import load_gold_standard, evaluate_ranking
from .config import load_config, save_config_template
from .text_loader import TextLoader
from .knowledge_base import KchoKnowledgeBase, init_database
from .data_migration import DataMigrationManager, FreshDataLoader, check_and_migrate_data
from .llama_integration import load_env_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
@click.version_option(version='0.2.0', prog_name='K\'Cho Linguistic Toolkit')
def cli():
    """K'Cho Linguistic Toolkit - Low-resource language processing."""
    pass

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file (default: stdout)')
def normalize(input_file, output):
    """Normalize K'Cho text."""
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    normalized = normalize_text(text)
    
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(normalized)
        logger.info(f"Normalized text written to {output}")
    else:
        print(normalized)

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file (default: stdout)')
def tokenize_cmd(input_file, output):
    """Tokenize K'Cho text."""
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokens = tokenize(text)
    result = '\n'.join(tokens)
    
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(result)
        logger.info(f"Tokens written to {output}")
    else:
        print(result)

@cli.command()
@click.option('--corpus', '-c', required=True, type=click.Path(exists=True), help='Input corpus file')
@click.option('--output', '-o', required=True, type=click.Path(), help='Output file')
@click.option('--top-k', default=10, type=click.IntRange(min=1), help='Number of top collocations to extract')
@click.option('--min-freq', default=5, type=click.IntRange(min=1), help='Minimum frequency threshold')
@click.option('--measures', multiple=True, default=['pmi', 'tscore'], help='Association measures to use')
@click.option('--window-size', default=5, type=click.IntRange(min=1), help='Co-occurrence window size')
@click.option('--verbose', is_flag=True, help='Verbose output')
def collocation(corpus, output, top_k, min_freq, measures, window_size, verbose):
    """Extract collocations from K'Cho corpus."""
    # Load corpus
    with open(corpus, 'r', encoding='utf-8') as f:
        corpus_lines = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(corpus_lines)} sentences from {corpus}")
    
    # Set default measures if none specified
    if not measures:
        measures = ['pmi', 'tscore']
    
    # Extract collocations
    system = KchoSystem()
    results = system.extract_collocations(
        corpus_lines,
        window_size=window_size,
        min_freq=min_freq,
        measures=list(measures)
    )
    
    # Export results
    output_format = Path(output).suffix[1:]  # Remove leading dot
    if output_format not in ['csv', 'json']:
        output_format = 'csv'
    
    system.export_collocations(results, output, format=output_format, top_k=top_k)
    logger.info(f"Collocations exported to {output}")

@cli.command()
@click.argument('predicted_file', type=click.Path(exists=True))
@click.argument('gold_standard_file', type=click.Path(exists=True))
@click.option('--measure', '-m', default='pmi', 
              type=click.Choice(['pmi', 'npmi', 'tscore', 'dice', 'log_likelihood']),
              help='Association measure to evaluate')
def evaluate_collocations(predicted_file, gold_standard_file, measure):
    """Evaluate collocation extraction against gold standard."""
    import json
    
    # Load predicted results
    with open(predicted_file, 'r', encoding='utf-8') as f:
        if predicted_file.endswith('.json'):
            data = json.load(f)
            predicted = data.get(measure, [])
        else:
            # Parse CSV
            import csv
            reader = csv.DictReader(f)
            predicted = [row for row in reader if row['measure'] == measure]
    
    # Convert to CollocationResult objects (simplified)
    from .collocation import CollocationResult
    predicted_collocations = [
        CollocationResult(
            words=tuple([p['word1'], p['word2']]),
            score=float(p['score']),
            measure=AssociationMeasure(measure),
            frequency=int(p['frequency'])
        ) for p in predicted
    ]
    
    # Load gold standard
    gold_set = load_gold_standard(gold_standard_file)
    
    # Evaluate
    metrics = evaluate_ranking(predicted_collocations, gold_set)
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


@cli.command(name='extract-all-collocations')
@click.option('--input', '-i', required=True, type=click.Path(exists=True), 
              help='Input file (txt or json)')
@click.option('--output', '-o', required=True, type=click.Path(), 
              help='Output file prefix')
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Configuration file (yaml/json)')
@click.option('--window-size', type=int, 
              help='Co-occurrence window size (overrides config)')
@click.option('--min-freq', type=int, 
              help='Minimum frequency (overrides config)')
@click.option('--measures', multiple=True, 
              help='Statistical measures to compute (overrides config)')
@click.option('--patterns', multiple=True, 
              help='Linguistic patterns to detect (overrides config)')
@click.option('--format', '-f', multiple=True, 
              type=click.Choice(['csv', 'json', 'txt']), 
              help='Output formats')
@click.option('--gold-standard', type=click.Path(exists=True), 
              help='Gold standard file for evaluation')
@click.option('--json-structure', 
              type=click.Choice(['nested', 'simple', 'auto']), 
              default='auto',
              help='JSON structure type')
@click.option('--verbose', is_flag=True, help='Verbose output')
def extract_all_collocations(input, output, config, window_size, min_freq, measures, 
                           patterns, format, gold_standard, json_structure, verbose):
    """Extract all types of collocations with full analysis."""
    from .collocation import CollocationExtractor
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if config:
        kcho_config = load_config(config)
    else:
        kcho_config = load_config()
    
    # Override config with CLI arguments
    if window_size:
        kcho_config.collocation.window_size = window_size
    if min_freq:
        kcho_config.collocation.min_freq = min_freq
    if measures:
        kcho_config.collocation.measures = list(measures)
    if patterns:
        kcho_config.collocation.linguistic_patterns = list(patterns)
    if format:
        kcho_config.output.formats = list(format)
    
    logger.info(f"Loading text from {input}")
    
    # Load text using universal loader
    try:
        sentences = TextLoader.load_from_file(input)
        logger.info(f"Loaded {len(sentences)} sentences")
    except Exception as e:
        logger.error(f"Error loading text: {e}")
        return
    
    # Create extractor with all measures
    measure_enums = []
    for measure_name in kcho_config.collocation.measures:
        try:
            measure_enums.append(AssociationMeasure[measure_name.upper()])
        except KeyError:
            logger.warning(f"Unknown measure: {measure_name}")
    
    if not measure_enums:
        measure_enums = [AssociationMeasure.PMI, AssociationMeasure.TSCORE]
    
    extractor = CollocationExtractor(
        window_size=kcho_config.collocation.window_size,
        min_freq=kcho_config.collocation.min_freq,
        measures=measure_enums
    )
    
    logger.info("Extracting collocations with linguistic patterns...")
    
    # Extract with pattern classification
    results = extractor.extract_with_patterns(sentences, kcho_config.collocation.linguistic_patterns)
    
    # Prepare output directory
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export in requested formats
    for fmt in kcho_config.output.formats:
        output_file = f"{output}.{fmt}"
        
        if fmt == 'csv':
            export_collocations_csv(results, output_file, include_patterns=True)
        elif fmt == 'json':
            export_collocations_json(results, output_file, input_file=input, 
                                  total_sentences=len(sentences))
        elif fmt == 'txt':
            export_collocations_txt(results, output_file, input_file=input, 
                                 total_sentences=len(sentences))
        
        logger.info(f"Exported {fmt} format to {output_file}")
    
    # Evaluate against gold standard if provided
    if gold_standard:
        logger.info("Evaluating against gold standard...")
        try:
            gold_set = load_gold_standard(gold_standard)
            
            # Convert results to evaluation format
            predicted = []
            for measure, collocations in results.items():
                predicted.extend(collocations)
            
            metrics = evaluate_ranking(predicted, gold_set)
            
            print("\nEvaluation Results:")
            print("=" * 50)
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
                
        except Exception as e:
            logger.error(f"Error evaluating against gold standard: {e}")
    
    # Print summary
    total_collocations = sum(len(collocations) for collocations in results.values())
    print(f"\nExtraction Complete:")
    print(f"  Input: {input}")
    print(f"  Sentences: {len(sentences)}")
    print(f"  Total collocations: {total_collocations}")
    print(f"  Measures: {[m.value for m in measure_enums]}")
    print(f"  Output formats: {kcho_config.output.formats}")


def export_collocations_csv(results, output_file, include_patterns=True):
    """Export collocations to CSV format."""
    import csv
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        if include_patterns:
            writer = csv.writer(f)
            writer.writerow(['words', 'statistical_measure', 'score', 'frequency', 
                           'linguistic_pattern', 'pattern_confidence', 'context_examples'])
            
            for measure, collocations in results.items():
                for coll in collocations:
                    pattern = coll.linguistic_pattern.value if coll.linguistic_pattern else 'UNK'
                    examples = '; '.join(coll.context_examples[:2])  # Limit examples
                    writer.writerow([
                        ' '.join(coll.words),
                        measure.value,
                        coll.score,
                        coll.frequency,
                        pattern,
                        coll.pattern_confidence,
                        examples
                    ])
        else:
            writer = csv.writer(f)
            writer.writerow(['words', 'statistical_measure', 'score', 'frequency'])
            
            for measure, collocations in results.items():
                for coll in collocations:
                    writer.writerow([
                        ' '.join(coll.words),
                        measure.value,
                        coll.score,
                        coll.frequency
                    ])


def export_collocations_json(results, output_file, input_file=None, total_sentences=None):
    """Export collocations to JSON format."""
    import json
    
    metadata = {
        'input_file': input_file,
        'total_sentences': total_sentences,
        'statistical_measures': [m.value for m in results.keys()],
        'total_collocations': sum(len(collocations) for collocations in results.values())
    }
    
    collocations = []
    for measure, collocations_list in results.items():
        for coll in collocations_list:
            collocation_data = {
                'words': list(coll.words),
                'statistical_measures': {measure.value: coll.score},
                'frequency': coll.frequency,
                'linguistic_pattern': coll.linguistic_pattern.value if coll.linguistic_pattern else None,
                'pattern_confidence': coll.pattern_confidence,
                'context_examples': coll.context_examples
            }
            collocations.append(collocation_data)
    
    output_data = {
        'metadata': metadata,
        'collocations': collocations
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


def export_collocations_txt(results, output_file, input_file=None, total_sentences=None):
    """Export collocations to text format."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== K'Cho Collocation Analysis ===\n")
        if input_file:
            f.write(f"Input: {input_file}\n")
        if total_sentences:
            f.write(f"Total sentences: {total_sentences:,}\n")
        f.write("\n")
        
        # Group by linguistic pattern
        pattern_groups = {}
        for measure, collocations in results.items():
            for coll in collocations:
                pattern = coll.linguistic_pattern.value if coll.linguistic_pattern else 'UNKNOWN'
                if pattern not in pattern_groups:
                    pattern_groups[pattern] = []
                pattern_groups[pattern].append(coll)
        
        for pattern, collocations in pattern_groups.items():
            f.write(f"=== {pattern.upper()} ===\n")
            for i, coll in enumerate(collocations[:10], 1):  # Top 10 per pattern
                f.write(f"{i}. {' '.join(coll.words)} ({coll.measure.value}: {coll.score:.3f}, freq: {coll.frequency})\n")
                for example in coll.context_examples[:2]:
                    f.write(f"   - \"{example}\"\n")
                f.write("\n")


def export_ngrams_csv(results: dict, output_file: str):
    """Export n-gram results to CSV format."""
    import csv
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ngram_size', 'words', 'measure', 'score', 'frequency', 'pattern', 'confidence', 'examples'])
        
        for n, measure_results in results.items():
            for measure, ngrams in measure_results.items():
                for ngram in ngrams:
                    pattern = ngram.linguistic_pattern.value if ngram.linguistic_pattern else 'UNK'
                    examples = '; '.join(ngram.context_examples[:2]) if ngram.context_examples else ''
                    writer.writerow([
                        ngram.ngram_size,
                        ' '.join(ngram.words),
                        measure.value,
                        ngram.score,
                        ngram.frequency,
                        pattern,
                        ngram.pattern_confidence,
                        examples
                    ])


def export_ngrams_json(results: dict, output_file: str):
    """Export n-gram results to JSON format."""
    import json
    output_data = {
        'metadata': {
            'total_ngram_sizes': len(results),
            'ngram_sizes': list(results.keys())
        },
        'results': {}
    }
    
    for n, measure_results in results.items():
        output_data['results'][f'{n}-grams'] = {}
        for measure, ngrams in measure_results.items():
            output_data['results'][f'{n}-grams'][measure.value] = []
            for ngram in ngrams:
                pattern = ngram.linguistic_pattern.value if ngram.linguistic_pattern else 'UNK'
                output_data['results'][f'{n}-grams'][measure.value].append({
                    'words': list(ngram.words),
                    'score': ngram.score,
                    'frequency': ngram.frequency,
                    'pattern': pattern,
                    'confidence': ngram.pattern_confidence,
                    'examples': ngram.context_examples[:3]
                })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def export_ngrams_txt(results: dict, output_file: str):
    """Export n-gram results to text format."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("K'Cho N-Gram Collocation Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        for n, measure_results in results.items():
            f.write(f"{n}-GRAMS ANALYSIS\n")
            f.write("-" * 20 + "\n\n")
            
            for measure, ngrams in measure_results.items():
                f.write(f"{measure.value.upper()} Results:\n")
                for i, ngram in enumerate(ngrams[:10], 1):  # Top 10
                    pattern = ngram.linguistic_pattern.value if ngram.linguistic_pattern else 'UNK'
                    f.write(f"  {i}. {' '.join(ngram.words)}\n")
                    f.write(f"     Score: {ngram.score:.3f}, Freq: {ngram.frequency}, Pattern: {pattern}\n")
                    if ngram.context_examples:
                        f.write(f"     Example: \"{ngram.context_examples[0]}\"\n")
                    f.write("\n")
                f.write("\n")


@cli.command()
@click.option('--output', '-o', default='config.yaml', type=click.Path(), 
              help='Output file for configuration template')
def create_config(output):
    """Create a configuration template file."""
    save_config_template(output)
    logger.info(f"Configuration template saved to {output}")


@cli.command(name='extract-ngrams')
@click.option('--input', '-i', required=True, type=click.Path(exists=True), 
              help='Input file (txt or json)')
@click.option('--output', '-o', required=True, type=click.Path(), 
              help='Output file prefix')
@click.option('--max-ngram-size', type=int, default=3, 
              help='Maximum n-gram size to extract (default: 3)')
@click.option('--min-freq', type=int, default=2, 
              help='Minimum frequency threshold (default: 2)')
@click.option('--measures', multiple=True, 
              help='Statistical measures to compute')
@click.option('--format', '-f', multiple=True, 
              type=click.Choice(['csv', 'json', 'txt']), 
              help='Output formats')
@click.option('--verbose', is_flag=True, help='Verbose output')
def extract_ngrams(input, output, max_ngram_size, min_freq, measures, format, verbose):
    """Extract n-grams (bigrams, trigrams, 4-grams, etc.) from K'Cho text."""
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    try:
        # Load text
        sentences = TextLoader.load_from_file(input)
        if not sentences:
            click.echo("❌ No sentences found in input file", err=True)
            return
        
        click.echo(f"✅ Loaded {len(sentences)} sentences")
        
        # Set default measures if none specified
        if not measures:
            measures = ['pmi', 'tscore']
        
        # Convert measure names to enums
        measure_enums = []
        for measure_name in measures:
            try:
                measure_enums.append(AssociationMeasure[measure_name.upper()])
            except KeyError:
                click.echo(f"⚠️  Unknown measure: {measure_name}", err=True)
                continue
        
        if not measure_enums:
            click.echo("❌ No valid measures specified", err=True)
            return
        
        # Create n-gram extractor
        extractor = NGramCollocationExtractor(
            window_size=5,
            min_freq=min_freq,
            max_ngram_size=max_ngram_size,
            measures=measure_enums
        )
        
        # Extract n-grams
        click.echo(f"🔍 Extracting n-grams (2-{max_ngram_size})...")
        results = extractor.extract_ngrams(sentences)
        
        # Print summary
        extractor.print_ngram_summary(results)
        
        # Export results
        if format:
            for fmt in format:
                if fmt == 'csv':
                    export_ngrams_csv(results, f"{output}.csv")
                elif fmt == 'json':
                    export_ngrams_json(results, f"{output}.json")
                elif fmt == 'txt':
                    export_ngrams_txt(results, f"{output}.txt")
        
        click.echo(f"✅ N-gram extraction complete!")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()


# ============================================================================
# RESEARCH AND ANALYSIS COMMANDS
# ============================================================================

@cli.group()
def research():
    """Research and analysis commands for linguistic research."""
    pass


@research.command()
@click.option('--corpus', '-c', required=True, type=click.Path(exists=True), 
              help='Input corpus file')
@click.option('--output', '-o', required=True, type=click.Path(), 
              help='Output file for research results')
@click.option('--focus', type=click.Choice(['all', 'patterns', 'morphology', 'syntax']), 
              default='all', help='Research focus area')
@click.option('--min-freq', default=3, type=int, 
              help='Minimum frequency for pattern discovery')
@click.option('--format', '-f', type=click.Choice(['json', 'csv', 'txt']), 
              default='json', help='Output format')
@click.option('--verbose', is_flag=True, help='Verbose output')
def analyze(corpus, output, focus, min_freq, format, verbose):
    """Conduct comprehensive linguistic research analysis."""
    try:
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        click.echo("🔬 Starting comprehensive linguistic research...")
        
        # Load corpus
        sentences = TextLoader.load_from_file(corpus)
        if not sentences:
            click.echo("❌ No sentences found in corpus", err=True)
            return
        
        click.echo(f"✅ Loaded {len(sentences)} sentences")
        
        # Initialize system and research engine
        system = KchoSystem()
        research_engine = LinguisticResearchEngine(system.knowledge)
        
        # Conduct research
        click.echo(f"🔍 Conducting {focus} research analysis...")
        results = research_engine.conduct_comprehensive_research(sentences, research_focus=focus)
        
        # Export results
        if format == 'json':
            import json
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        elif format == 'csv':
            # Convert to CSV format
            import csv
            with open(output, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['analysis_type', 'metric', 'value'])
                
                for analysis_type, data in results.items():
                    if isinstance(data, dict):
                        for metric, value in data.items():
                            writer.writerow([analysis_type, metric, value])
                    else:
                        writer.writerow([analysis_type, 'result', str(data)])
        elif format == 'txt':
            with open(output, 'w', encoding='utf-8') as f:
                f.write("K'Cho Linguistic Research Analysis\n")
                f.write("=" * 50 + "\n\n")
                
                for analysis_type, data in results.items():
                    f.write(f"{analysis_type.upper()} ANALYSIS\n")
                    f.write("-" * 30 + "\n")
                    
                    if isinstance(data, dict):
                        for metric, value in data.items():
                            f.write(f"{metric}: {value}\n")
                    else:
                        f.write(f"Result: {data}\n")
                    f.write("\n")
        
        click.echo(f"✅ Research analysis complete! Results saved to {output}")
        
        # Print summary
        click.echo("\n📊 Research Summary:")
        if 'known_patterns' in results:
            click.echo(f"  Known patterns analyzed: {results['known_patterns'].get('total_patterns_analyzed', 0)}")
        if 'novel_patterns' in results:
            click.echo(f"  Novel patterns discovered: {results['novel_patterns'].get('total_discovered', 0)}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()


@research.command()
@click.option('--corpus', '-c', required=True, type=click.Path(exists=True), 
              help='Input corpus file')
@click.option('--output', '-o', required=True, type=click.Path(), 
              help='Output file for pattern discovery results')
@click.option('--min-freq', default=3, type=int, 
              help='Minimum frequency threshold')
@click.option('--max-patterns', default=100, type=int, 
              help='Maximum number of patterns to discover')
@click.option('--format', '-f', type=click.Choice(['json', 'csv', 'txt']), 
              default='json', help='Output format')
@click.option('--verbose', is_flag=True, help='Verbose output')
def discover_patterns(corpus, output, min_freq, max_patterns, format, verbose):
    """Discover linguistic patterns from corpus."""
    try:
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        click.echo("🔍 Starting pattern discovery...")
        
        # Load corpus
        sentences = TextLoader.load_from_file(corpus)
        if not sentences:
            click.echo("❌ No sentences found in corpus", err=True)
            return
        
        click.echo(f"✅ Loaded {len(sentences)} sentences")
        
        # Initialize system and pattern discovery engine
        system = KchoSystem()
        pattern_engine = PatternDiscoveryEngine(system.knowledge)
        
        # Discover patterns
        click.echo(f"🔍 Discovering patterns (min_freq={min_freq})...")
        results = pattern_engine.discover_patterns(sentences, min_frequency=min_freq)
        
        # Export results
        if format == 'json':
            import json
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        elif format == 'csv':
            import csv
            with open(output, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['pattern_type', 'words', 'frequency', 'confidence'])
                
                for pattern_type, patterns in results.items():
                    for pattern in patterns[:max_patterns]:
                        writer.writerow([
                            pattern_type,
                            ' '.join(pattern.get('words', [])),
                            pattern.get('frequency', 0),
                            pattern.get('confidence', 0.0)
                        ])
        elif format == 'txt':
            with open(output, 'w', encoding='utf-8') as f:
                f.write("K'Cho Pattern Discovery Results\n")
                f.write("=" * 50 + "\n\n")
                
                for pattern_type, patterns in results.items():
                    f.write(f"{pattern_type.upper()} PATTERNS\n")
                    f.write("-" * 30 + "\n")
                    
                    for i, pattern in enumerate(patterns[:max_patterns], 1):
                        f.write(f"{i}. {' '.join(pattern.get('words', []))}\n")
                        f.write(f"   Frequency: {pattern.get('frequency', 0)}\n")
                        f.write(f"   Confidence: {pattern.get('confidence', 0.0):.3f}\n\n")
        
        click.echo(f"✅ Pattern discovery complete! Results saved to {output}")
        
        # Print summary
        total_patterns = sum(len(patterns) for patterns in results.values())
        click.echo(f"\n📊 Discovery Summary:")
        click.echo(f"  Total patterns discovered: {total_patterns}")
        for pattern_type, patterns in results.items():
            click.echo(f"  {pattern_type}: {len(patterns)} patterns")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()


@research.command()
@click.option('--corpus', '-c', required=True, type=click.Path(exists=True), 
              help='Input corpus file')
@click.option('--output', '-o', required=True, type=click.Path(), 
              help='Output file for analysis results')
@click.option('--format', '-f', type=click.Choice(['json', 'csv', 'txt']), 
              default='json', help='Output format')
@click.option('--verbose', is_flag=True, help='Verbose output')
def analyze_text(corpus, output, format, verbose):
    """Analyze text structure and linguistic features."""
    try:
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        click.echo("📝 Starting text analysis...")
        
        # Load corpus
        sentences = TextLoader.load_from_file(corpus)
        if not sentences:
            click.echo("❌ No sentences found in corpus", err=True)
            return
        
        click.echo(f"✅ Loaded {len(sentences)} sentences")
        
        # Initialize system
        system = KchoSystem()
        
        # Analyze sentences
        analysis_results = []
        for i, sentence_text in enumerate(sentences[:100]):  # Limit to first 100 for performance
            try:
                sentence = system.analyze_text(sentence_text)
                analysis_results.append({
                    'sentence_id': i,
                    'text': sentence_text,
                    'tokens': len(sentence.tokens),
                    'token_details': [
                        {
                            'surface': token.surface,
                            'pos': token.pos.value,
                            'stem': getattr(token, 'stem', None),
                            'gloss': getattr(token, 'gloss', None)
                        } for token in sentence.tokens
                    ]
                })
            except Exception as e:
                if verbose:
                    click.echo(f"⚠️  Error analyzing sentence {i}: {e}")
                continue
        
        # Export results
        if format == 'json':
            import json
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        elif format == 'csv':
            import csv
            with open(output, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['sentence_id', 'text', 'token_count', 'tokens'])
                
                for result in analysis_results:
                    tokens_str = '; '.join([f"{t['surface']}({t['pos']})" for t in result['token_details']])
                    writer.writerow([
                        result['sentence_id'],
                        result['text'],
                        result['tokens'],
                        tokens_str
                    ])
        elif format == 'txt':
            with open(output, 'w', encoding='utf-8') as f:
                f.write("K'Cho Text Analysis Results\n")
                f.write("=" * 50 + "\n\n")
                
                for result in analysis_results:
                    f.write(f"Sentence {result['sentence_id']}: {result['text']}\n")
                    f.write(f"Tokens ({result['tokens']}):\n")
                    for token in result['token_details']:
                        f.write(f"  {token['surface']} ({token['pos']})")
                        if token['stem']:
                            f.write(f" [stem: {token['stem']}]")
                        if token['gloss']:
                            f.write(f" [gloss: {token['gloss']}]")
                        f.write("\n")
                    f.write("\n")
        
        click.echo(f"✅ Text analysis complete! Results saved to {output}")
        click.echo(f"\n📊 Analysis Summary:")
        click.echo(f"  Sentences analyzed: {len(analysis_results)}")
        click.echo(f"  Total tokens: {sum(r['tokens'] for r in analysis_results)}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()


@cli.group()
def generate():
    """Data generation commands for research and analysis."""
    pass


@generate.command()
@click.option('--corpus', '-c', required=True, type=click.Path(exists=True), 
              help='Input corpus file')
@click.option('--output', '-o', required=True, type=click.Path(), 
              help='Output directory for generated data')
@click.option('--include-ngrams', is_flag=True, help='Include n-gram analysis')
@click.option('--include-research', is_flag=True, help='Include research analysis')
@click.option('--include-collocations', is_flag=True, help='Include collocation extraction')
@click.option('--verbose', is_flag=True, help='Verbose output')
def research_data(corpus, output, include_ngrams, include_research, include_collocations, verbose):
    """Generate comprehensive research data from corpus."""
    try:
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        click.echo("📊 Generating comprehensive research data...")
        
        # Create output directory
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load corpus
        sentences = TextLoader.load_from_file(corpus)
        if not sentences:
            click.echo("❌ No sentences found in corpus", err=True)
            return
        
        click.echo(f"✅ Loaded {len(sentences)} sentences")
        
        # Initialize system
        system = KchoSystem()
        
        # Generate collocation data
        if include_collocations:
            click.echo("🔍 Extracting collocations...")
            collocation_results = system.extract_collocations(sentences)
            
            # Export collocations
            collocation_file = output_path / "collocations.json"
            import json
            with open(collocation_file, 'w', encoding='utf-8') as f:
                json.dump(collocation_results, f, ensure_ascii=False, indent=2)
            click.echo(f"✅ Collocations saved to {collocation_file}")
        
        # Generate n-gram data
        if include_ngrams:
            click.echo("🔍 Extracting n-grams...")
            ngram_extractor = NGramCollocationExtractor()
            ngram_results = ngram_extractor.extract_ngrams(sentences)
            
            # Export n-grams
            ngram_file = output_path / "ngrams.json"
            import json
            with open(ngram_file, 'w', encoding='utf-8') as f:
                json.dump(ngram_results, f, ensure_ascii=False, indent=2)
            click.echo(f"✅ N-grams saved to {ngram_file}")
        
        # Generate research analysis
        if include_research:
            click.echo("🔬 Conducting research analysis...")
            research_engine = LinguisticResearchEngine(system.knowledge)
            research_results = research_engine.conduct_comprehensive_research(sentences)
            
            # Export research results
            research_file = output_path / "research_analysis.json"
            import json
            with open(research_file, 'w', encoding='utf-8') as f:
                json.dump(research_results, f, ensure_ascii=False, indent=2)
            click.echo(f"✅ Research analysis saved to {research_file}")
        
        # Generate summary
        summary_file = output_path / "generation_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("K'Cho Research Data Generation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input corpus: {corpus}\n")
            f.write(f"Sentences processed: {len(sentences)}\n")
            f.write(f"Generated files:\n")
            
            if include_collocations:
                f.write(f"  - collocations.json\n")
            if include_ngrams:
                f.write(f"  - ngrams.json\n")
            if include_research:
                f.write(f"  - research_analysis.json\n")
            
            f.write(f"  - generation_summary.txt\n")
        
        click.echo(f"✅ Research data generation complete!")
        click.echo(f"📁 Output directory: {output_path}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()


# ============================================================================
# DATA MIGRATION COMMANDS
# ============================================================================

@cli.group()
def migrate():
    """Data migration commands for managing K'Cho knowledge base."""
    pass


@migrate.command()
@click.option('--db-path', type=click.Path(), help='Path to SQLite database file')
@click.option('--data-dir', type=click.Path(), help='Path to data directory')
@click.option('--force', is_flag=True, help='Force migration even if no changes detected')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def check(db_path, data_dir, force, verbose):
    """Check if data migration is needed."""
    try:
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Initialize knowledge base
        kb = KchoKnowledgeBase(db_path=db_path, data_dir=data_dir)
        
        # Check migration status
        migration_manager = DataMigrationManager(kb.db_path, str(kb.data_dir))
        
        try:
            has_changed = migration_manager.has_data_changed()
            current_version = migration_manager.calculate_current_data_version()
            stored_version = migration_manager.get_current_data_version()
            
            click.echo("📊 Migration Status Check")
            click.echo("=" * 40)
            
            if stored_version:
                click.echo(f"📋 Stored version: {stored_version.version}")
                click.echo(f"📅 Last updated: {stored_version.timestamp}")
            else:
                click.echo("📋 No stored version found")
            
            click.echo(f"🔍 Current version: {current_version.version}")
            click.echo(f"📅 Calculated: {current_version.timestamp}")
            
            if has_changed or force:
                click.echo("🔄 Status: Migration needed")
                if force:
                    click.echo("⚡ Force flag enabled - migration will be performed")
                else:
                    click.echo("💡 Run 'kcho migrate run' to perform migration")
            else:
                click.echo("✅ Status: No migration needed")
            
            # Show file checksums
            click.echo("\n📁 Data File Checksums:")
            for filename, checksum in current_version.checksums.items():
                click.echo(f"  {filename}: {checksum[:16]}...")
            
        finally:
            migration_manager.close()
            kb.close()
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()


@migrate.command()
@click.option('--db-path', type=click.Path(), help='Path to SQLite database file')
@click.option('--data-dir', type=click.Path(), help='Path to data directory')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def run(db_path, data_dir, verbose):
    """Run data migration if needed."""
    try:
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        click.echo("🔄 Starting data migration...")
        
        # Initialize knowledge base (this will trigger migration if needed)
        kb = KchoKnowledgeBase(db_path=db_path, data_dir=data_dir)
        
        # Get migration history
        history = kb.get_migration_history(limit=5)
        
        click.echo("\n📋 Recent Migration History:")
        click.echo("=" * 50)
        
        if history:
            for migration in history:
                status_emoji = "✅" if migration['status'] == 'completed' else "❌" if migration['status'] == 'failed' else "🔄"
                click.echo(f"{status_emoji} {migration['version_to']} ({migration['migration_type']}) - {migration['status']}")
                if migration['records_affected']:
                    click.echo(f"   📊 Records: {migration['records_affected']}")
                if migration['error_message']:
                    click.echo(f"   ⚠️  Error: {migration['error_message']}")
        else:
            click.echo("No migration history found")
        
        # Get database statistics
        stats = kb.get_statistics()
        click.echo(f"\n📊 Database Statistics:")
        click.echo(f"  Verb stems: {stats['verb_stems']}")
        click.echo(f"  Collocations: {stats['collocations']}")
        click.echo(f"  Word frequencies: {stats['word_frequencies']}")
        click.echo(f"  Parallel sentences: {stats['parallel_sentences']}")
        click.echo(f"  Raw texts: {stats['raw_texts']}")
        
        kb.close()
        click.echo("✅ Migration check completed")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()


@migrate.command()
@click.option('--db-path', type=click.Path(), help='Path to SQLite database file')
@click.option('--data-dir', type=click.Path(), help='Path to data directory')
@click.option('--limit', default=10, help='Number of migrations to show')
def history(db_path, data_dir, limit):
    """Show migration history."""
    try:
        kb = KchoKnowledgeBase(db_path=db_path, data_dir=data_dir)
        
        history = kb.get_migration_history(limit=limit)
        
        click.echo("📋 Migration History")
        click.echo("=" * 60)
        
        if history:
            for migration in history:
                status_emoji = "✅" if migration['status'] == 'completed' else "❌" if migration['status'] == 'failed' else "🔄"
                click.echo(f"{status_emoji} Migration {migration['id']}")
                click.echo(f"   From: {migration['version_from'] or 'None'}")
                click.echo(f"   To: {migration['version_to']}")
                click.echo(f"   Type: {migration['migration_type']}")
                click.echo(f"   Status: {migration['status']}")
                click.echo(f"   Started: {migration['started_at']}")
                if migration['completed_at']:
                    click.echo(f"   Completed: {migration['completed_at']}")
                if migration['records_affected']:
                    click.echo(f"   Records: {migration['records_affected']}")
                if migration['error_message']:
                    click.echo(f"   Error: {migration['error_message']}")
                click.echo()
        else:
            click.echo("No migration history found")
        
        kb.close()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


@migrate.command()
@click.option('--db-path', type=click.Path(), help='Path to SQLite database file')
@click.option('--data-dir', type=click.Path(), help='Path to data directory')
def status(db_path, data_dir):
    """Show data file status."""
    try:
        kb = KchoKnowledgeBase(db_path=db_path, data_dir=data_dir)
        
        file_status = kb.get_data_file_status()
        
        click.echo("📁 Data File Status")
        click.echo("=" * 60)
        
        if file_status:
            for file_info in file_status:
                click.echo(f"📄 {file_info['filename']}")
                click.echo(f"   Version: {file_info['version']}")
                click.echo(f"   Checksum: {file_info['checksum'][:16]}...")
                click.echo(f"   Size: {file_info['file_size']} bytes")
                click.echo(f"   Modified: {file_info['last_modified']}")
                click.echo(f"   Loaded: {file_info['loaded_at']}")
                click.echo()
        else:
            click.echo("No data files tracked")
        
        kb.close()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


@migrate.command()
@click.option('--db-path', type=click.Path(), help='Path to SQLite database file')
@click.option('--data-dir', type=click.Path(), help='Path to data directory')
@click.option('--confirm', is_flag=True, help='Confirm the operation')
def force(db_path, data_dir, confirm):
    """Force a fresh data migration."""
    if not confirm:
        click.echo("⚠️  This will replace ALL data in the database!")
        click.echo("Use --confirm flag to proceed")
        return
    
    try:
        click.echo("🔄 Forcing fresh data migration...")
        
        kb = KchoKnowledgeBase(db_path=db_path, data_dir=data_dir)
        
        success = kb.force_fresh_migration()
        
        if success:
            click.echo("✅ Fresh migration completed successfully")
            
            # Show updated statistics
            stats = kb.get_statistics()
            click.echo(f"\n📊 Updated Database Statistics:")
            click.echo(f"  Verb stems: {stats['verb_stems']}")
            click.echo(f"  Collocations: {stats['collocations']}")
            click.echo(f"  Word frequencies: {stats['word_frequencies']}")
            click.echo(f"  Parallel sentences: {stats['parallel_sentences']}")
        else:
            click.echo("❌ Fresh migration failed")
        
        kb.close()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


@migrate.command()
@click.option('--db-path', type=click.Path(), help='Path to SQLite database file')
@click.option('--data-dir', type=click.Path(), help='Path to data directory')
def init(db_path, data_dir):
    """Initialize database with fresh data."""
    try:
        click.echo("🚀 Initializing K'Cho knowledge base...")
        
        # Initialize database
        kb = init_database(db_path=db_path, data_dir=data_dir)
        
        # Get statistics
        stats = kb.get_statistics()
        
        click.echo("✅ Database initialized successfully!")
        click.echo(f"\n📊 Database Statistics:")
        click.echo(f"  Verb stems: {stats['verb_stems']}")
        click.echo(f"  Collocations: {stats['collocations']}")
        click.echo(f"  Word frequencies: {stats['word_frequencies']}")
        click.echo(f"  Parallel sentences: {stats['parallel_sentences']}")
        click.echo(f"  Raw texts: {stats['raw_texts']}")
        
        kb.close()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


@cli.group()
def env():
    """Environment configuration management."""
    pass


@env.command()
@click.option('--file', '-f', default='.env', help='Path to .env file')
def load(file: str):
    """Load environment variables from .env file."""
    try:
        success = load_env_file(file)
        if success:
            click.echo(f"✅ Environment variables loaded from {file}")
        else:
            click.echo(f"⚠️  No .env file found at {file}")
            click.echo("💡 Create a .env file using: kcho env create")
    except Exception as e:
        click.echo(f"❌ Error loading .env file: {e}")


@env.command()
@click.option('--file', '-f', default='.env', help='Path to .env file')
def create(file: str):
    """Create a sample .env file."""
    try:
        env_content = """# K'Cho Linguistic Processing Toolkit - Environment Configuration
# Fill in your actual values below

# =============================================================================
# LLaMA API Configuration
# =============================================================================

# LLaMA Provider: ollama, openai, anthropic
LLAMA_PROVIDER=ollama

# =============================================================================
# Ollama Configuration (Local LLaMA)
# =============================================================================

# Ollama base URL (default: http://localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434

# Ollama model to use (default: llama3.1:8b)
OLLAMA_MODEL=llama3.1:8b

# =============================================================================
# OpenAI Configuration
# =============================================================================

# OpenAI API Key (get from https://platform.openai.com/api-keys)
# OPENAI_API_KEY=sk-your-openai-api-key-here

# OpenAI model to use (default: gpt-3.5-turbo)
# OPENAI_MODEL=gpt-3.5-turbo

# =============================================================================
# Anthropic Configuration
# =============================================================================

# Anthropic API Key (get from https://console.anthropic.com/)
# ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here

# Anthropic model to use (default: claude-3-sonnet-20240229)
# ANTHROPIC_MODEL=claude-3-sonnet-20240229

# =============================================================================
# General LLaMA Settings
# =============================================================================

# Request timeout in seconds (default: 30)
LLAMA_TIMEOUT=30

# Maximum tokens to generate (default: 1000)
LLAMA_MAX_TOKENS=1000

# Temperature for generation (default: 0.7)
LLAMA_TEMPERATURE=0.7

# =============================================================================
# Database Configuration
# =============================================================================

# SQLite database path (default: auto-generated)
# KCHO_DB_PATH=kcho_lexicon.db

# =============================================================================
# Logging Configuration
# =============================================================================

# Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)
# LOG_LEVEL=INFO

# =============================================================================
# Development Settings
# =============================================================================

# Enable debug mode (default: false)
# DEBUG=false

# Enable Pydantic validation (default: false)
# ENABLE_PYDANTIC_VALIDATION=false
"""
        
        with open(file, 'w') as f:
            f.write(env_content)
        
        click.echo(f"✅ Sample .env file created at {file}")
        click.echo("💡 Edit the file to add your API keys and configuration")
        
    except Exception as e:
        click.echo(f"❌ Error creating .env file: {e}")


@env.command()
@click.option('--file', '-f', default='.env', help='Path to .env file')
def check(file: str):
    """Check current environment configuration."""
    try:
        import os
        
        # Load .env file if it exists
        load_env_file(file)
        
        click.echo("🔍 Environment Configuration Check:")
        click.echo("=" * 40)
        
        # Check LLaMA provider
        provider = os.getenv("LLAMA_PROVIDER", "ollama")
        click.echo(f"🤖 LLaMA Provider: {provider}")
        
        if provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
            click.echo(f"🌐 Ollama URL: {base_url}")
            click.echo(f"🧠 Model: {model}")
            
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            click.echo(f"🔑 OpenAI API Key: {'✅ Set' if api_key else '❌ Not set'}")
            click.echo(f"🧠 Model: {model}")
            
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            model = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
            click.echo(f"🔑 Anthropic API Key: {'✅ Set' if api_key else '❌ Not set'}")
            click.echo(f"🧠 Model: {model}")
        
        # Check general settings
        timeout = os.getenv("LLAMA_TIMEOUT", "30")
        max_tokens = os.getenv("LLAMA_MAX_TOKENS", "1000")
        temperature = os.getenv("LLAMA_TEMPERATURE", "0.7")
        
        click.echo(f"⏱️  Timeout: {timeout}s")
        click.echo(f"📝 Max Tokens: {max_tokens}")
        click.echo(f"🌡️  Temperature: {temperature}")
        
        # Check database
        db_path = os.getenv("KCHO_DB_PATH", "auto-generated")
        click.echo(f"🗄️  Database: {db_path}")
        
    except Exception as e:
        click.echo(f"❌ Error checking environment: {e}")


@cli.command()
@click.option('--query', '-q', required=True, help='Query about K\'Cho language')
@click.option('--deep-research', is_flag=True, help='Use deep research mode (3000 tokens)')
@click.option('--no-llama', is_flag=True, help='Use structured data only (no LLaMA)')
@click.option('--log', is_flag=True, help='Log response to file')
@click.option('--log-format', type=click.Choice(['json', 'csv']), default='json', help='Log format')
@click.option('--context', help='Additional context for the query')
def query(query: str, deep_research: bool, no_llama: bool, log: bool, log_format: str, context: str):
    """Query the K'Cho linguistic system with LLaMA integration."""
    try:
        # Initialize system
        system = KchoSystem()
        
        # Check for deep research mode warning
        if deep_research:
            click.echo("⚠️  Deep research mode requested (3000 tokens)")
            click.echo("💰 This may incur significant costs with external APIs")
            if not click.confirm("Do you want to proceed?"):
                click.echo("❌ Query cancelled by user")
                return
        
        # Get API response
        response = system.get_api_response(
            query=query,
            use_llama=not no_llama,
            context=context,
            deep_research=deep_research,
            log_response=log,
            log_format=log_format
        )
        
        # Display results
        click.echo(f"\n📊 Query: {query}")
        click.echo(f"🕒 Timestamp: {response['timestamp']}")
        click.echo(f"🔧 Response Type: {response['response_type']}")
        click.echo(f"🤖 LLaMA Enabled: {response['llama_enabled']}")
        click.echo(f"🔬 Deep Research: {response['deep_research']}")
        click.echo(f"🎯 Tokens Used: {response['tokens_used']}")
        
        if response['llama_enabled'] and response.get('llama_response', {}).get('success'):
            click.echo(f"\n🤖 LLaMA Response:")
            click.echo("-" * 40)
            click.echo(response['llama_response']['response'])
            
            click.echo(f"\n📊 Structured Data:")
            click.echo("-" * 40)
            structured = response['structured_data']
            click.echo(f"Query Type: {structured.get('query_type', 'unknown')}")
            if 'total_patterns' in structured:
                click.echo(f"Total Patterns: {structured['total_patterns']}")
            if 'pattern_categories' in structured:
                click.echo(f"Categories: {', '.join(structured['pattern_categories'])}")
            if 'context' in structured:
                click.echo(f"Context: {structured['context']}")
        else:
            click.echo(f"\n📊 Structured Data:")
            click.echo("-" * 40)
            data = response.get('data', response.get('structured_data', {}))
            click.echo(f"Query Type: {data.get('query_type', 'unknown')}")
            if 'total_patterns' in data:
                click.echo(f"Total Patterns: {data['total_patterns']}")
            if 'pattern_categories' in data:
                click.echo(f"Categories: {', '.join(data['pattern_categories'])}")
            if 'context' in data:
                click.echo(f"Context: {data['context']}")
            
            if 'llama_error' in response:
                click.echo(f"\n⚠️  LLaMA Error: {response['llama_error']}")
        
        if log:
            click.echo(f"\n📝 Response logged to file")
        
        system.close()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


if __name__ == '__main__':
    cli()