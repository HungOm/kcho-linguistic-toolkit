# FILE: kcho_app.py (ENHANCED CLI)
# CHANGES:
#   - Line ~50: Added collocation extraction command
#   - Line ~80: Added collocation export command
#   - Line ~110: Added collocation evaluation command
#   - Added imports for new modules

"""
K'Cho CLI Application.

Command-line interface for K'Cho language toolkit.
"""

import click
import logging
from pathlib import Path
from typing import List

from .kcho_system import KChoSystem
from .normalize import normalize_text, tokenize
from .collocation import AssociationMeasure
from .export import to_csv, to_json, to_text
from .evaluation import load_gold_standard, evaluate_ranking

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """K'Cho Language Toolkit - Low-resource language processing."""
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
@click.argument('corpus_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), required=True, help='Output CSV file')
@click.option('--window', '-w', default=5, help='Co-occurrence window size')
@click.option('--min-freq', '-f', default=5, help='Minimum frequency threshold')
@click.option('--measures', '-m', multiple=True, 
              type=click.Choice(['pmi', 'npmi', 'tscore', 'dice', 'log_likelihood']),
              help='Association measures (can specify multiple)')
@click.option('--top-k', '-k', type=int, help='Limit to top K results per measure')
def extract_collocations(corpus_file, output, window, min_freq, measures, top_k):
    """Extract collocations from K'Cho corpus."""
    # Load corpus
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(corpus)} sentences from {corpus_file}")
    
    # Set default measures if none specified
    if not measures:
        measures = ['pmi', 'tscore']
    
    # Extract collocations
    system = KChoSystem()
    results = system.extract_collocations(
        corpus,
        window_size=window,
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

if __name__ == '__main__':
    cli()