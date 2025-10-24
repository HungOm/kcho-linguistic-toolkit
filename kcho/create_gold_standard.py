# FILE: create_gold_standard.py (NEW)
# PURPOSE: Helper script for creating gold standard files

"""
Gold Standard Creation Helper

This script assists in creating gold standard collocation files by:
1. Extracting candidate collocations from corpus
2. Ranking by frequency and association measures
3. Providing interactive annotation interface
4. Validating against linguistic categories

Usage:
    python create_gold_standard.py --corpus sample_corpus_kcho.txt --output gold_standard.txt
"""

import argparse
import logging
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Set

from .collocation import CollocationExtractor, AssociationMeasure
from .normalize import KChoNormalizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# K'Cho linguistic categories for annotation
CATEGORIES = {
    'VP': 'Verb + Particle',
    'PP': 'Postposition + Noun',
    'APP': 'Applicative construction',
    'AGR': 'Agreement + Verb',
    'AUX': 'Auxiliary construction',
    'COMP': 'Complementizer pattern',
    'MWE': 'Multi-word expression (3+)',
    'COMPOUND': 'Compound noun',
    'LEX': 'Lexical collocation',
    'DISC': 'Discourse marker',
    'OTHER': 'Other/unclassified'
}

class GoldStandardCreator:
    """Interactive tool for creating gold standard files."""
    
    def __init__(self, corpus_path: str):
        self.corpus_path = Path(corpus_path)
        self.normalizer = KChoNormalizer()
        self.extractor = CollocationExtractor(
            normalizer=self.normalizer,
            window_size=5,
            min_freq=3,
            measures=[AssociationMeasure.PMI, AssociationMeasure.TSCORE]
        )
        self.corpus: List[str] = []
        self.candidates: List[Tuple[Tuple[str, ...], float, int]] = []
        self.gold_standard: Dict[Tuple[str, ...], Dict] = {}
    
    def load_corpus(self):
        """Load corpus from file."""
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            self.corpus = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        logger.info(f"Loaded {len(self.corpus)} sentences from {self.corpus_path}")
    
    def extract_candidates(self, top_k: int = 100):
        """Extract candidate collocations."""
        results = self.extractor.extract(self.corpus)
        
        # Combine results from all measures
        candidate_dict = {}
        for measure, collocations in results.items():
            for coll in collocations:
                if coll.words not in candidate_dict:
                    candidate_dict[coll.words] = {
                        'max_score': coll.score,
                        'frequency': coll.frequency
                    }
                else:
                    candidate_dict[coll.words]['max_score'] = max(
                        candidate_dict[coll.words]['max_score'],
                        coll.score
                    )
        
        # Sort by combined score (max_score * log(frequency))
        import math
        self.candidates = sorted(
            [(words, data['max_score'] * math.log(data['frequency'] + 1), data['frequency'])
             for words, data in candidate_dict.items()],
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        logger.info(f"Extracted {len(self.candidates)} candidate collocations")
    
    def interactive_annotation(self):
        """Interactive annotation interface."""
        print("\n" + "=" * 70)
        print("GOLD STANDARD ANNOTATION")
        print("=" * 70)
        print("\nCategories:")
        for code, desc in CATEGORIES.items():
            print(f"  {code}: {desc}")
        print("\nCommands:")
        print("  y = Accept (will prompt for category)")
        print("  n = Reject")
        print("  s = Skip (decide later)")
        print("  q = Quit and save")
        print("=" * 70 + "\n")
        
        for i, (words, score, freq) in enumerate(self.candidates, 1):
            print(f"\n[{i}/{len(self.candidates)}] Candidate: {' '.join(words)}")
            print(f"    Score: {score:.4f}, Frequency: {freq}")
            
            # Show example sentences
            examples = self._find_examples(words, max_examples=3)
            if examples:
                print(f"    Examples:")
                for ex in examples:
                    print(f"      - {ex}")
            
            decision = input(f"    Decision [y/n/s/q]: ").strip().lower()
            
            if decision == 'q':
                logger.info("Quitting annotation session")
                break
            elif decision == 'y':
                category = input(f"    Category [{'/'.join(CATEGORIES.keys())}]: ").strip().upper()
                if category not in CATEGORIES:
                    category = 'OTHER'
                notes = input(f"    Notes (optional): ").strip()
                
                self.gold_standard[words] = {
                    'category': category,
                    'frequency': freq,
                    'score': score,
                    'notes': notes
                }
                print(f"    ✓ Added to gold standard")
            elif decision == 'n':
                print(f"    ✗ Rejected")
            else:
                print(f"    ~ Skipped")
    
    def _find_examples(self, words: Tuple[str, ...], max_examples: int = 3) -> List[str]:
        """Find example sentences containing the collocation."""
        examples = []
        words_list = list(words)
        
        for sent in self.corpus:
            tokens = self.normalizer.tokenize(sent)
            # Check if words appear consecutively
            for i in range(len(tokens) - len(words_list) + 1):
                if tokens[i:i+len(words_list)] == words_list:
                    examples.append(sent)
                    break
            if len(examples) >= max_examples:
                break
        
        return examples
    
    def auto_annotate(self):
        """Automatic annotation based on linguistic patterns."""
        logger.info("Starting automatic annotation based on K'Cho patterns...")
        
        # K'Cho particles and postpositions
        particles = {'ci', 'khai', 'ne', 'te', 'gui', 'goi', 'ni'}
        postpositions = {'noh', 'ah', 'am', 'on', 'ung', 'tu', 'bà'}
        agreement = {'ka', 'a', 'ani', 'ami', 'an'}
        
        for words, score, freq in self.candidates:
            if words in self.gold_standard:
                continue  # Already annotated
            
            w1, w2 = words[0], words[1] if len(words) > 1 else ''
            
            # Rule-based classification
            category = None
            confidence = 'low'
            
            # VP pattern: verb + particle
            if w2 in particles:
                category = 'VP'
                confidence = 'high'
            
            # PP pattern: postposition + noun
            elif w1 in postpositions:
                category = 'PP'
                confidence = 'high'
            
            # APP pattern: contains -na or -nák
            elif '-na' in w1 or '-nák' in w1:
                category = 'APP'
                confidence = 'high'
            
            # AGR pattern: agreement prefix + verb
            elif w1 in agreement:
                category = 'AGR'
                confidence = 'medium'
            
            # COMP pattern: particle + ah
            elif w2 == 'ah' and w1 in particles:
                category = 'COMP'
                confidence = 'high'
            
            if category and confidence == 'high':
                self.gold_standard[words] = {
                    'category': category,
                    'frequency': freq,
                    'score': score,
                    'notes': f'Auto-annotated ({confidence} confidence)'
                }
        
        logger.info(f"Auto-annotated {len(self.gold_standard)} collocations")
    
    def save_gold_standard(self, output_path: str):
        """Save gold standard to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sort by category, then by score
        sorted_items = sorted(
            self.gold_standard.items(),
            key=lambda x: (x[1]['category'], -x[1]['score'])
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("# K'Cho Gold Standard Collocations\n")
            f.write(f"# Generated from: {self.corpus_path}\n")
            f.write(f"# Total entries: {len(self.gold_standard)}\n")
            f.write("# FORMAT: word1 word2 [category] [frequency] [notes]\n\n")
            
            # Category sections
            current_category = None
            for words, data in sorted_items:
                category = data['category']
                
                # Category header
                if category != current_category:
                    current_category = category
                    f.write(f"\n# === {category}: {CATEGORIES.get(category, 'Unknown')} ===\n")
                
                # Write collocation
                words_str = ' '.join(words)
                f.write(f"{words_str:<20} # {category}, freq={data['frequency']}")
                if data.get('notes'):
                    f.write(f", {data['notes']}")
                f.write("\n")
        
        logger.info(f"Saved {len(self.gold_standard)} collocations to {output_path}")
        
        # Print statistics
        print("\n" + "=" * 70)
        print("GOLD STANDARD STATISTICS")
        print("=" * 70)
        category_counts = Counter(data['category'] for data in self.gold_standard.values())
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count}")
        print(f"\nTotal: {len(self.gold_standard)} collocations")
        print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description='Create gold standard collocation file')
    parser.add_argument('--corpus', required=True, help='Path to corpus file')
    parser.add_argument('--output', default='gold_standard_kcho.txt', help='Output gold standard file')
    parser.add_argument('--top-k', type=int, default=100, help='Number of candidates to extract')
    parser.add_argument('--auto', action='store_true', help='Use automatic annotation only')
    parser.add_argument('--interactive', action='store_true', help='Use interactive annotation')
    
    args = parser.parse_args()
    
    # Create gold standard
    creator = GoldStandardCreator(args.corpus)
    creator.load_corpus()
    creator.extract_candidates(top_k=args.top_k)
    
    if args.auto or not args.interactive:
        creator.auto_annotate()
    
    if args.interactive:
        creator.interactive_annotation()
    
    creator.save_gold_standard(args.output)

if __name__ == '__main__':
    main()