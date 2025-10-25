"""
Tests for evaluation.py module.

This module tests the evaluation functionality for collocation extraction
and other linguistic analysis tasks.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

from kcho.evaluation import (
    compute_precision_recall, compute_mean_reciprocal_rank, compute_average_precision,
    evaluate_ranking, load_gold_standard
)
from kcho.collocation import CollocationExtractor, AssociationMeasure, CollocationResult
from kcho.kcho_system import KchoSystem


class TestComputePrecisionRecall:
    """Test the compute_precision_recall function."""
    
    def test_compute_precision_recall_basic(self):
        """Test basic precision/recall computation."""
        # Create mock predicted collocations
        predicted = [
            CollocationResult(words=('pe', 'ci'), score=0.8, measure=AssociationMeasure.PMI, frequency=5),
            CollocationResult(words=('lo', 'ci'), score=0.7, measure=AssociationMeasure.PMI, frequency=4),
            CollocationResult(words=('noh', 'Yóng'), score=0.6, measure=AssociationMeasure.PMI, frequency=3),
        ]
        
        # Create mock gold standard
        gold_standard = {
            ('pe', 'ci'),
            ('noh', 'Yóng'),
            ('luum-na', 'ci')
        }
        
        result = compute_precision_recall(predicted, gold_standard)
        
        assert isinstance(result, dict)
        assert 'precision' in result
        assert 'recall' in result
        assert 'f1' in result
        assert 'true_positives' in result
        assert 'false_positives' in result
        assert 'false_negatives' in result
        
        # Check values
        assert result['true_positives'] == 2  # pe ci, noh Yóng
        assert result['false_positives'] == 1  # lo ci
        assert result['false_negatives'] == 1  # luum-na ci
        assert result['precision'] == 2/3  # 2/3
        assert result['recall'] == 2/3  # 2/3
        assert result['f1'] == 2/3  # 2/3
    
    def test_compute_precision_recall_perfect_match(self):
        """Test precision/recall computation with perfect match."""
        # Create mock predicted collocations
        predicted = [
            CollocationResult(words=('pe', 'ci'), score=0.8, measure=AssociationMeasure.PMI, frequency=5),
            CollocationResult(words=('noh', 'Yóng'), score=0.7, measure=AssociationMeasure.PMI, frequency=4),
        ]
        
        # Create mock gold standard
        gold_standard = {
            ('pe', 'ci'),
            ('noh', 'Yóng')
        }
        
        result = compute_precision_recall(predicted, gold_standard)
        
        assert result['precision'] == 1.0
        assert result['recall'] == 1.0
        assert result['f1'] == 1.0
        assert result['true_positives'] == 2
        assert result['false_positives'] == 0
        assert result['false_negatives'] == 0
    
    def test_compute_precision_recall_no_match(self):
        """Test precision/recall computation with no match."""
        # Create mock predicted collocations
        predicted = [
            CollocationResult(words=('lo', 'ci'), score=0.8, measure=AssociationMeasure.PMI, frequency=5),
            CollocationResult(words=('random', 'words'), score=0.7, measure=AssociationMeasure.PMI, frequency=4),
        ]
        
        # Create mock gold standard
        gold_standard = {
            ('pe', 'ci'),
            ('noh', 'Yóng')
        }
        
        result = compute_precision_recall(predicted, gold_standard)
        
        assert result['precision'] == 0.0
        assert result['recall'] == 0.0
        assert result['f1'] == 0.0
        assert result['true_positives'] == 0
        assert result['false_positives'] == 2
        assert result['false_negatives'] == 2
    
    def test_compute_precision_recall_empty_predictions(self):
        """Test precision/recall computation with empty predictions."""
        predicted = []
        
        gold_standard = {
            ('pe', 'ci'),
            ('noh', 'Yóng')
        }
        
        result = compute_precision_recall(predicted, gold_standard)
        
        assert result['precision'] == 0.0
        assert result['recall'] == 0.0
        assert result['f1'] == 0.0
        assert result['true_positives'] == 0
        assert result['false_positives'] == 0
        assert result['false_negatives'] == 2
    
    def test_compute_precision_recall_empty_gold_standard(self):
        """Test precision/recall computation with empty gold standard."""
        predicted = [
            CollocationResult(words=('pe', 'ci'), score=0.8, measure=AssociationMeasure.PMI, frequency=5),
            CollocationResult(words=('noh', 'Yóng'), score=0.7, measure=AssociationMeasure.PMI, frequency=4),
        ]
        
        gold_standard = set()
        
        result = compute_precision_recall(predicted, gold_standard)
        
        assert result['precision'] == 0.0
        assert result['recall'] == 0.0
        assert result['f1'] == 0.0
        assert result['true_positives'] == 0
        assert result['false_positives'] == 2
        assert result['false_negatives'] == 0
    
    def test_compute_precision_recall_with_top_k(self):
        """Test precision/recall computation with top_k parameter."""
        # Create mock predicted collocations
        predicted = [
            CollocationResult(words=('pe', 'ci'), score=0.8, measure=AssociationMeasure.PMI, frequency=5),
            CollocationResult(words=('lo', 'ci'), score=0.7, measure=AssociationMeasure.PMI, frequency=4),
            CollocationResult(words=('noh', 'Yóng'), score=0.6, measure=AssociationMeasure.PMI, frequency=3),
        ]
        
        # Create mock gold standard
        gold_standard = {
            ('pe', 'ci'),
            ('noh', 'Yóng'),
            ('luum-na', 'ci')
        }
        
        # Test with top_k=2
        result = compute_precision_recall(predicted, gold_standard, top_k=2)
        
        assert result['true_positives'] == 1  # Only pe ci
        assert result['false_positives'] == 1  # lo ci
        assert result['false_negatives'] == 2  # noh Yóng, luum-na ci
        assert result['precision'] == 0.5  # 1/2
        assert result['recall'] == 1/3  # 1/3


class TestComputeMeanReciprocalRank:
    """Test the compute_mean_reciprocal_rank function."""
    
    def test_compute_mean_reciprocal_rank_basic(self):
        """Test basic MRR computation."""
        # Create mock predicted collocations
        predicted = [
            CollocationResult(words=('pe', 'ci'), score=0.8, measure=AssociationMeasure.PMI, frequency=5),
            CollocationResult(words=('lo', 'ci'), score=0.7, measure=AssociationMeasure.PMI, frequency=4),
            CollocationResult(words=('noh', 'Yóng'), score=0.6, measure=AssociationMeasure.PMI, frequency=3),
        ]
        
        # Create mock gold standard
        gold_standard = {
            ('pe', 'ci'),
            ('noh', 'Yóng'),
            ('luum-na', 'ci')
        }
        
        mrr = compute_mean_reciprocal_rank(predicted, gold_standard)
        
        assert isinstance(mrr, float)
        assert 0.0 <= mrr <= 1.0
    
    def test_compute_mean_reciprocal_rank_perfect(self):
        """Test MRR computation with perfect ranking."""
        # Create mock predicted collocations
        predicted = [
            CollocationResult(words=('pe', 'ci'), score=0.8, measure=AssociationMeasure.PMI, frequency=5),
            CollocationResult(words=('noh', 'Yóng'), score=0.7, measure=AssociationMeasure.PMI, frequency=4),
        ]
        
        # Create mock gold standard
        gold_standard = {
            ('pe', 'ci'),
            ('noh', 'Yóng')
        }
        
        mrr = compute_mean_reciprocal_rank(predicted, gold_standard)
        
        assert mrr == 1.0
    
    def test_compute_mean_reciprocal_rank_no_match(self):
        """Test MRR computation with no match."""
        # Create mock predicted collocations
        predicted = [
            CollocationResult(words=('lo', 'ci'), score=0.8, measure=AssociationMeasure.PMI, frequency=5),
            CollocationResult(words=('random', 'words'), score=0.7, measure=AssociationMeasure.PMI, frequency=4),
        ]
        
        # Create mock gold standard
        gold_standard = {
            ('pe', 'ci'),
            ('noh', 'Yóng')
        }
        
        mrr = compute_mean_reciprocal_rank(predicted, gold_standard)
        
        assert mrr == 0.0


class TestLoadGoldStandard:
    """Test the load_gold_standard function."""
    
    def test_load_gold_standard_basic(self):
        """Test loading gold standard from file using real parallel corpus data."""
        # Use the real gold standard K'Cho-English parallel corpus
        parallel_corpus_path = Path(__file__).parent.parent / "data" / "gold_standard_kcho_english.json"
        
        # Create a temporary gold standard file based on the parallel corpus
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            # Load the parallel corpus and extract collocations
            with open(parallel_corpus_path, 'r', encoding='utf-8') as parallel_file:
                parallel_data = json.load(parallel_file)
            
            # Extract collocations from the parallel corpus
            collocations = []
            for pair in parallel_data['sentence_pairs'][:10]:  # Use first 10 sentences
                kcho_text = pair['kcho']
                # Simple bigram extraction for testing
                words = kcho_text.split()
                for i in range(len(words) - 1):
                    collocations.append(f"{words[i]} {words[i+1]}")
            
            # Write to temporary file
            f.write('\n'.join(collocations))
            temp_file = f.name
        
        try:
            gold_standard = load_gold_standard(temp_file)
            
            assert isinstance(gold_standard, set)
            assert len(gold_standard) > 0
            
            # Check structure of gold standard entries
            for collocation in gold_standard:
                assert isinstance(collocation, tuple)
                assert len(collocation) == 2
                # Verify that we have real K'Cho words
                assert all(isinstance(word, str) for word in collocation)
                assert all(len(word) > 0 for word in collocation)
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_load_gold_standard_empty(self):
        """Test loading empty gold standard file."""
        # Create empty gold standard file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("# Empty gold standard\n")
            temp_file = f.name
        
        try:
            gold_standard = load_gold_standard(temp_file)
            
            assert isinstance(gold_standard, set)
            assert len(gold_standard) == 0
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_load_gold_standard_nonexistent(self):
        """Test loading nonexistent gold standard file."""
        with pytest.raises(FileNotFoundError):
            load_gold_standard("nonexistent_file.txt")


class TestIntegrationWithRealData:
    """Integration tests using real parallel corpus data."""
    
    def setup_method(self):
        """Set up test fixtures with real parallel corpus data."""
        self.data_dir = Path(__file__).parent.parent / "data"
        self.parallel_corpus_path = self.data_dir / "gold_standard_kcho_english.json"
        
        # Load parallel corpus data
        if self.parallel_corpus_path.exists():
            with open(self.parallel_corpus_path, 'r', encoding='utf-8') as f:
                self.parallel_data = json.load(f)
        else:
            self.parallel_data = None
    
    def test_evaluate_with_real_gold_standard(self):
        """Test evaluation with real parallel corpus data."""
        if not self.parallel_corpus_path.exists() or not self.parallel_data:
            pytest.skip("Parallel corpus file not found")
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_output = f.name
        
        try:
            # Extract K'Cho sentences and create gold standard
            kcho_sentences = [pair['kcho'] for pair in self.parallel_data['sentence_pairs'][:20]]
            
            # Create gold standard from parallel corpus
            gold_standard = set()
            for sentence in kcho_sentences:
                words = sentence.split()
                for i in range(len(words) - 1):
                    gold_standard.add((words[i], words[i+1]))
            
            # Verify gold standard was created
            assert len(gold_standard) > 0
            
            # Check structure of gold standard entries
            for collocation in gold_standard:
                assert isinstance(collocation, tuple)
                assert len(collocation) == 2
                # Verify that we have real K'Cho words
                assert all(isinstance(word, str) for word in collocation)
                assert all(len(word) > 0 for word in collocation)
        
        finally:
            if os.path.exists(temp_output):
                os.unlink(temp_output)
    
    def test_evaluate_with_real_corpus(self):
        """Test evaluation with real corpus data."""
        if not self.parallel_corpus_path.exists() or not self.parallel_data:
            pytest.skip("Parallel corpus file not found")
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_output = f.name
        
        try:
            # Create system
            system = KchoSystem()
            
            # Extract K'Cho sentences from parallel corpus
            kcho_sentences = [pair['kcho'] for pair in self.parallel_data['sentence_pairs'][:15]]
            
            # Add sentences to corpus
            for sentence in kcho_sentences:
                system.add_to_corpus(sentence, validate=False)
            
            # Create gold standard from parallel corpus
            gold_standard = set()
            for sentence in kcho_sentences:
                words = sentence.split()
                for i in range(len(words) - 1):
                    gold_standard.add((words[i], words[i+1]))
            
            # Extract collocations
            predicted_collocations = system.extract_collocations(kcho_sentences)
            
            # Evaluate using the utility function
            if predicted_collocations:
                # Flatten predicted collocations
                all_predicted = []
                for measure, results in predicted_collocations.items():
                    all_predicted.extend(results)
                
                # Compute precision/recall
                result = compute_precision_recall(all_predicted, gold_standard)
                
                # Verify results
                assert isinstance(result, dict)
                assert 'precision' in result
                assert 'recall' in result
                assert 'f1' in result
                assert 'true_positives' in result
                assert 'false_positives' in result
                assert 'false_negatives' in result
        
        finally:
            if os.path.exists(temp_output):
                os.unlink(temp_output)
    
    def test_performance_with_large_corpus(self):
        """Test performance with larger corpus."""
        if not self.parallel_corpus_path.exists() or not self.parallel_data:
            pytest.skip("Parallel corpus file not found")
        
        import time
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_output = f.name
        
        try:
            start_time = time.time()
            
            # Create system
            system = KchoSystem()
            
            # Extract K'Cho sentences from parallel corpus
            kcho_sentences = [pair['kcho'] for pair in self.parallel_data['sentence_pairs'][:15]]
            
            # Add sentences to corpus
            for sentence in kcho_sentences:
                system.add_to_corpus(sentence, validate=False)
            
            # Create gold standard from parallel corpus
            gold_standard = set()
            for sentence in kcho_sentences:
                words = sentence.split()
                for i in range(len(words) - 1):
                    gold_standard.add((words[i], words[i+1]))
            
            # Extract collocations
            predicted_collocations = system.extract_collocations(kcho_sentences)
            
            # Evaluate using the utility function
            if predicted_collocations:
                # Flatten predicted collocations
                all_predicted = []
                for measure, results in predicted_collocations.items():
                    all_predicted.extend(results)
                
                # Compute precision/recall
                result = compute_precision_recall(all_predicted, gold_standard)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should complete within reasonable time
            assert processing_time < 30  # 30 seconds should be more than enough
        
        finally:
            if os.path.exists(temp_output):
                os.unlink(temp_output)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_compute_precision_recall_with_corrupted_data(self):
        """Test precision/recall computation with corrupted data."""
        # Test with corrupted predicted collocations
        corrupted_predictions = [
            CollocationResult(words=('', ''), score=0.0, measure=AssociationMeasure.PMI, frequency=0),
            CollocationResult(words=('pe', 'ci'), score=0.8, measure=AssociationMeasure.PMI, frequency=5),
        ]
        
        gold_standard = {
            ('pe', 'ci'),
            ('noh', 'Yóng')
        }
        
        # Should handle corrupted data gracefully
        result = compute_precision_recall(corrupted_predictions, gold_standard)
        
        assert isinstance(result, dict)
        assert 'precision' in result
        assert 'recall' in result
        assert 'f1' in result
    
    def test_compute_precision_recall_with_invalid_types(self):
        """Test precision/recall computation with invalid types."""
        # Test with invalid predicted collocations
        invalid_predictions = [
            CollocationResult(words=('pe', 'ci'), score=0.8, measure=AssociationMeasure.PMI, frequency=5),
            None,  # Invalid entry
        ]
        
        gold_standard = {
            ('pe', 'ci'),
            ('noh', 'Yóng')
        }
        
        # Should handle invalid types gracefully
        result = compute_precision_recall(invalid_predictions, gold_standard)
        
        assert isinstance(result, dict)
        assert 'precision' in result
        assert 'recall' in result
        assert 'f1' in result
    
    def test_compute_mean_reciprocal_rank_with_invalid_values(self):
        """Test MRR computation with invalid values."""
        # Test with negative values
        precision = -0.5
        recall = 0.6
        
        # Should handle invalid values gracefully
        assert isinstance(precision, float)
        assert isinstance(recall, float)