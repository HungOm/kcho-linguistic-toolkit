# FILE: export.py (NEW)
# PURPOSE: Export collocation results to CSV, JSON, and other formats

"""
Export utilities for collocation results.

Supports:
- CSV export (for lexicon, spreadsheet analysis)
- JSON export (for API, programmatic use)
- Plain text export (human-readable)
"""

import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Union
from .collocation import CollocationResult, AssociationMeasure

logger = logging.getLogger(__name__)

def to_csv(results: Union[List[CollocationResult], Dict[AssociationMeasure, List[CollocationResult]]],
           output_path: Union[str, Path],
           top_k: int = None) -> None:
    """
    Export collocation results to CSV.
    
    Args:
        results: Either a list of CollocationResult or dict mapping measures to results
        output_path: Path to output CSV file
        top_k: Optional limit on number of results per measure
    
    CSV columns: word1, word2, measure, score, frequency
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Normalize input to dict format
    if isinstance(results, list):
        results = {'default': results}
    
    rows = []
    for measure, collocations in results.items():
        measure_name = measure.value if isinstance(measure, AssociationMeasure) else str(measure)
        for i, coll in enumerate(collocations):
            if top_k and i >= top_k:
                break
            rows.append({
                'word1': coll.words[0],
                'word2': coll.words[1],
                'measure': measure_name,
                'score': round(coll.score, 4),
                'frequency': coll.frequency
            })
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=['word1', 'word2', 'measure', 'score', 'frequency'])
            writer.writeheader()
            writer.writerows(rows)
    
    logger.info(f"Exported {len(rows)} collocations to {output_path}")

def to_json(results: Union[List[CollocationResult], Dict[AssociationMeasure, List[CollocationResult]]],
            output_path: Union[str, Path],
            top_k: int = None,
            indent: int = 2) -> None:
    """
    Export collocation results to JSON.
    
    Args:
        results: Either a list of CollocationResult or dict mapping measures to results
        output_path: Path to output JSON file
        top_k: Optional limit on number of results per measure
        indent: JSON indentation (default 2)
    
    JSON structure:
    {
        "measure_name": [
            {"words": ["w1", "w2"], "score": 5.2, "frequency": 10},
            ...
        ]
    }
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Normalize input to dict format
    if isinstance(results, list):
        results = {'default': results}
    
    output_data = {}
    for measure, collocations in results.items():
        measure_name = measure.value if isinstance(measure, AssociationMeasure) else str(measure)
        output_data[measure_name] = []
        for i, coll in enumerate(collocations):
            if top_k and i >= top_k:
                break
            output_data[measure_name].append({
                'words': list(coll.words),
                'score': round(coll.score, 4),
                'frequency': coll.frequency
            })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=indent)
    
    total = sum(len(v) for v in output_data.values())
    logger.info(f"Exported {total} collocations to {output_path}")

def to_text(results: Union[List[CollocationResult], Dict[AssociationMeasure, List[CollocationResult]]],
            output_path: Union[str, Path],
            top_k: int = None) -> None:
    """
    Export collocation results to plain text (human-readable).
    
    Args:
        results: Either a list of CollocationResult or dict mapping measures to results
        output_path: Path to output text file
        top_k: Optional limit on number of results per measure
    
    Format:
    === Measure: pmi ===
    w1 w2 (score=5.2, freq=10)
    ...
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Normalize input to dict format
    if isinstance(results, list):
        results = {'default': results}
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for measure, collocations in results.items():
            measure_name = measure.value if isinstance(measure, AssociationMeasure) else str(measure)
            f.write(f"\n=== Measure: {measure_name} ===\n\n")
            for i, coll in enumerate(collocations):
                if top_k and i >= top_k:
                    break
                words_str = ' '.join(coll.words)
                f.write(f"{words_str} (score={coll.score:.4f}, freq={coll.frequency})\n")
    
    logger.info(f"Exported collocations to {output_path}")