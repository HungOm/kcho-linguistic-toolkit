# FILE: export_training_csv.py (NEW)
# PURPOSE: Reproducible script to export parallel corpus training data

"""
Reproducible script for exporting parallel corpus training data.

Usage:
    python export_training_csv.py --english-file path/to/english.txt --kcho-file path/to/kcho.txt --output-dir output/
"""

import argparse
import logging
from pathlib import Path
from .eng_kcho_parallel_extractor import ParallelCorpusExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Export parallel corpus training data')
    parser.add_argument('--english-file', required=True, help='Path to English corpus file')
    parser.add_argument('--kcho-file', required=True, help='Path to K\'Cho corpus file')
    parser.add_argument('--output-dir', default='training_data', help='Output directory for CSV')
    parser.add_argument('--force', action='store_true', help='Overwrite existing files')
    
    args = parser.parse_args()
    
    # Validate input files
    english_path = Path(args.english_file)
    kcho_path = Path(args.kcho_file)
    
    if not english_path.exists():
        logger.error(f"English file not found: {english_path}")
        return
    
    if not kcho_path.exists():
        logger.error(f"K'Cho file not found: {kcho_path}")
        return
    
    # Initialize extractor
    logger.info("Initializing parallel corpus extractor")
    extractor = ParallelCorpusExtractor()
    
    # Load files
    logger.info(f"Loading files: {english_path}, {kcho_path}")
    extractor.load_files(str(english_path), str(kcho_path))
    
    # Align sentences
    logger.info("Aligning sentences")
    extractor.align_sentences()
    
    # Export training data
    logger.info(f"Exporting training data to {args.output_dir}")
    extractor.export_training_data(args.output_dir, force=args.force)
    
    logger.info("Export complete!")

if __name__ == '__main__':
    main()