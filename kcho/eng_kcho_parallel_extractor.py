"""
English-K'Cho Parallel Corpus Extractor
Extracts and exports parallel training data for machine translation models.
"""
import csv
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict

@dataclass
class ParallelSentence:
    """Represents a parallel sentence pair."""
    english: str
    kcho: str
    source: str = ""
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        return asdict(self)

class ParallelCorpusExtractor:
    """Extract and process parallel English-K'Cho corpus."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / 'output'
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.corpus: List[ParallelSentence] = []
    
    def add_sentence_pair(self, english: str, kcho: str, source: str = "", 
                          metadata: Dict = None):
        """Add a parallel sentence pair to corpus."""
        # Validate inputs
        if not english or not english.strip():
            print(f"Warning: Empty English sentence from {source}")
            return
        
        if not kcho or not kcho.strip():
            print(f"Warning: Empty K'Cho sentence from {source}")
            return
        
        pair = ParallelSentence(
            english=english.strip(),
            kcho=kcho.strip(),
            source=source,
            metadata=metadata or {}
        )
        self.corpus.append(pair)
    
    def load_from_file(self, filepath: Path):
        """Load parallel corpus from various formats."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Corpus file not found: {filepath}")
        
        try:
            if filepath.suffix == '.json':
                self._load_json(filepath)
            elif filepath.suffix == '.csv':
                self._load_csv(filepath)
            elif filepath.suffix == '.txt':
                self._load_txt(filepath)
            else:
                print(f"Warning: Unknown file format {filepath.suffix}, trying as text")
                self._load_txt(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            raise
    
    def _load_json(self, filepath: Path):
        """Load from JSON format."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'english' in item and 'kcho' in item:
                        self.add_sentence_pair(
                            item['english'],
                            item['kcho'],
                            source=str(filepath),
                            metadata=item.get('metadata', {})
                        )
    
    def _load_csv(self, filepath: Path):
        """Load from CSV format."""
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                english = row.get('english', row.get('English', row.get('en', '')))
                kcho = row.get('kcho', row.get('K\'Cho', row.get('kc', '')))
                
                if english and kcho:
                    metadata = {k: v for k, v in row.items() if k not in ['english', 'kcho', 'English', "K'Cho", 'en', 'kc']}
                    self.add_sentence_pair(
                        english,
                        kcho,
                        source=str(filepath),
                        metadata=metadata
                    )
    
    def _load_txt(self, filepath: Path):
        """Load from tab-separated text format."""
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        self.add_sentence_pair(
                            parts[0],
                            parts[1],
                            source=f"{filepath}:line{line_num}"
                        )
    
    def export_training_data(self, output_dir: str, force: bool = False) -> None:
        """
        Export parallel data to CSV for training.
        
        FIXED: Added checks for empty data and proper force parameter handling
        
        Args:
            output_dir: Directory to save training files
            force: If True, overwrite existing files; if False, skip if exists
        """
        import os
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        csv_file = output_path / 'parallel_training.csv'
        
        # Check if file exists and force=False
        if csv_file.exists() and not force:
            logger.info(f"File {csv_file} already exists, skipping (use force=True to overwrite)")
            return
        
        # Extract parallel pairs
        pairs = self.extract_parallel_pairs()
        
        # CRITICAL FIX: Check if pairs is empty
        if not pairs:
            logger.error("No parallel pairs extracted - check alignment and input files")
            return
        
        # Validate pairs structure
        valid_pairs = [(en, kcho) for en, kcho in pairs if en.strip() and kcho.strip()]
        if not valid_pairs:
            logger.error("All extracted pairs are empty - check sentence alignment")
            return
        
        # Export to CSV
        import csv as csv_module
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv_module.writer(f)
            writer.writerow(['english', 'kcho'])
            writer.writerows(valid_pairs)
        
        logger.info(f"Exported {len(valid_pairs)} parallel pairs to {csv_file}")
    
    def export_json(self, filepath: Optional[Path] = None) -> Path:
        """Export corpus to JSON format."""
        if not self.corpus:
            raise ValueError("Cannot export empty corpus.")
        
        filepath = Path(filepath) if filepath else self.output_dir / 'parallel_corpus.json'
        
        data = [pair.to_dict() for pair in self.corpus]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Exported {len(self.corpus)} sentence pairs to {filepath}")
        return filepath
    
    def get_statistics(self) -> Dict:
        """Get corpus statistics."""
        if not self.corpus:
            return {
                'total_pairs': 0,
                'avg_length_en': 0,
                'avg_length_kc': 0,
                'sources': []
            }
        
        en_lengths = [len(pair.english.split()) for pair in self.corpus]
        kc_lengths = [len(pair.kcho.split()) for pair in self.corpus]
        sources = list(set(pair.source for pair in self.corpus))
        
        return {
            'total_pairs': len(self.corpus),
            'avg_length_en': sum(en_lengths) / len(en_lengths),
            'avg_length_kc': sum(kc_lengths) / len(kc_lengths),
            'min_length_en': min(en_lengths),
            'max_length_en': max(en_lengths),
            'min_length_kc': min(kc_lengths),
            'max_length_kc': max(kc_lengths),
            'sources': sources
        }