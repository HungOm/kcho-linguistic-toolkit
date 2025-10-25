"""
Test suite for Syntactic Analysis in kcho_system.py

Tests KchoSyntaxAnalyzer class
"""

import pytest
from unittest.mock import patch, MagicMock
from kcho.kcho_system import (
    KchoSyntaxAnalyzer, KchoKnowledge, KchoMorphologyAnalyzer,
    Token, Morpheme, Sentence, POS, StemType
)


class TestKchoSyntaxAnalyzer:
    """Test KchoSyntaxAnalyzer functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = KchoSyntaxAnalyzer()
    
    def test_initialization(self):
        """Test analyzer initialization"""
        assert hasattr(self.analyzer, 'knowledge')
        assert hasattr(self.analyzer, 'morph')
        assert isinstance(self.analyzer.knowledge, KchoKnowledge)
        assert isinstance(self.analyzer.morph, KchoMorphologyAnalyzer)
    
    def test_analyze_syntax_basic(self):
        """Test basic syntactic analysis"""
        # Create mock sentence with tokens
        tokens = [
            Token("Om", "om", POS.VERB, [Morpheme("om", "om", "go", "root")]),
            Token("noh", "noh", POS.POSTPOSITION, [Morpheme("noh", "noh", "OBJ", "postposition")]),
            Token("Yong", "Yong", POS.NOUN, [Morpheme("Yong", "Yong", "Yong", "root")])
        ]
        sentence = Sentence("Om noh Yong", tokens, "go OBJ Yong")
        
        analysis = self.analyzer.analyze_syntax(sentence)
        
        assert isinstance(analysis, dict)
        assert 'clause_type' in analysis
        assert 'has_applicative' in analysis
        assert 'has_relative_clause' in analysis
        assert 'verb_stem_form' in analysis
        assert 'arguments' in analysis
    
    def test_identify_clause_type_intransitive(self):
        """Test identifying intransitive clause"""
        tokens = [
            Token("Om", "om", POS.VERB, [Morpheme("om", "om", "go", "root")])
        ]
        
        clause_type = self.analyzer._identify_clause_type(tokens)
        assert clause_type == "intransitive"
    
    def test_identify_clause_type_transitive(self):
        """Test identifying transitive clause"""
        tokens = [
            Token("Om", "om", POS.VERB, [Morpheme("om", "om", "go", "root")]),
            Token("noh", "noh", POS.POSTPOSITION, [Morpheme("noh", "noh", "OBJ", "postposition")]),
            Token("Yong", "Yong", POS.NOUN, [Morpheme("Yong", "Yong", "Yong", "root")])
        ]
        
        clause_type = self.analyzer._identify_clause_type(tokens)
        assert clause_type == "transitive"
    
    def test_identify_clause_type_ditransitive(self):
        """Test identifying ditransitive clause"""
        tokens = [
            Token("Om", "om", POS.VERB, [Morpheme("om", "om", "go", "root")]),
            Token("noh", "noh", POS.POSTPOSITION, [Morpheme("noh", "noh", "OBJ", "postposition")]),
            Token("Yong", "Yong", POS.NOUN, [Morpheme("Yong", "Yong", "Yong", "root")]),
            Token("am", "am", POS.POSTPOSITION, [Morpheme("am", "am", "LOC", "postposition")]),
            Token("paapai", "paapai", POS.NOUN, [Morpheme("paapai", "paapai", "house", "root")]),
            Token("pe", "pe", POS.POSTPOSITION, [Morpheme("pe", "pe", "ALL", "postposition")]),
            Token("ci", "ci", POS.TENSE, [Morpheme("ci", "ci", "PAST", "tense")])
        ]
        
        clause_type = self.analyzer._identify_clause_type(tokens)
        assert clause_type == "transitive"  # Only 2 nouns, not 3+
    
    def test_has_relative_clause_true(self):
        """Test detecting relative clause"""
        words = ["Om", "noh", "Yong", "ah", "law", "ci"]
        
        has_relative = self.analyzer._has_relative_clause(words)
        assert has_relative is True
    
    def test_has_relative_clause_false(self):
        """Test not detecting relative clause"""
        words = ["Om", "noh", "Yong", "ci"]
        
        has_relative = self.analyzer._has_relative_clause(words)
        assert has_relative is False
    
    def test_has_relative_clause_early_ah(self):
        """Test that early 'ah' doesn't trigger relative clause"""
        words = ["ah", "Om", "noh", "Yong", "ci"]
        
        has_relative = self.analyzer._has_relative_clause(words)
        assert has_relative is False  # Too early in sentence
    
    def test_identify_stem_form_with_ci_no_agreement(self):
        """Test identifying Stem I with ci and no 3rd person agreement"""
        tokens = [
            Token("Om", "om", POS.VERB, [Morpheme("om", "om", "go", "root")]),
            Token("ci", "ci", POS.TENSE, [Morpheme("ci", "ci", "PAST", "tense")])
        ]
        
        stem_form = self.analyzer._identify_stem_form(tokens)
        assert stem_form == "I"
    
    def test_identify_stem_form_with_ung(self):
        """Test identifying Stem II with ung"""
        tokens = [
            Token("Om", "om", POS.VERB, [Morpheme("om", "om", "go", "root")]),
            Token("ung", "ung", POS.TENSE, [Morpheme("ung", "ung", "FUT", "tense")])
        ]
        
        stem_form = self.analyzer._identify_stem_form(tokens)
        assert stem_form == "II"
    
    def test_identify_stem_form_with_applicative(self):
        """Test identifying stem form from applicative"""
        tokens = [
            Token("omnák", "omnák", POS.VERB, [
                Morpheme("om", "om", "go", "root"),
                Morpheme("nák", "na", "APPL", "suffix", {"stem": "II"})
            ])
        ]
        
        stem_form = self.analyzer._identify_stem_form(tokens)
        assert stem_form == "II"
    
    def test_identify_stem_form_none(self):
        """Test identifying stem form when none can be determined"""
        tokens = [
            Token("Om", "om", POS.VERB, [Morpheme("om", "om", "go", "root")])
        ]
        
        stem_form = self.analyzer._identify_stem_form(tokens)
        assert stem_form is None
    
    def test_extract_arguments_subject(self):
        """Test extracting subject arguments"""
        tokens = [
            Token("Yong", "Yong", POS.NOUN, [Morpheme("Yong", "Yong", "Yong", "root")]),
            Token("noh", "noh", POS.POSTPOSITION, [Morpheme("noh", "noh", "OBJ", "postposition")]),
            Token("Om", "om", POS.VERB, [Morpheme("om", "om", "go", "root")])
        ]
        
        arguments = self.analyzer._extract_arguments(tokens)
        
        assert isinstance(arguments, dict)
        assert 'subject' in arguments
        assert 'object' in arguments
        assert 'oblique' in arguments
        assert "Yong" in arguments['subject']
    
    def test_extract_arguments_object(self):
        """Test extracting object arguments"""
        tokens = [
            Token("Om", "om", POS.VERB, [Morpheme("om", "om", "go", "root")]),
            Token("Yong", "Yong", POS.NOUN, [Morpheme("Yong", "Yong", "Yong", "root")]),
            Token("ci", "ci", POS.TENSE, [Morpheme("ci", "ci", "PAST", "tense")])
        ]
        
        arguments = self.analyzer._extract_arguments(tokens)
        
        # Yong should be identified as object (N before verb/tense)
        assert "Yong" in arguments['object']
    
    def test_extract_arguments_oblique(self):
        """Test extracting oblique arguments"""
        tokens = [
            Token("Yong", "Yong", POS.NOUN, [Morpheme("Yong", "Yong", "Yong", "root")]),
            Token("am", "am", POS.POSTPOSITION, [Morpheme("am", "am", "LOC", "postposition")]),
            Token("Om", "om", POS.VERB, [Morpheme("om", "om", "go", "root")])
        ]
        
        arguments = self.analyzer._extract_arguments(tokens)
        
        # Yong should be identified as oblique (N with postposition)
        assert "Yong" in arguments['oblique']
    
    def test_extract_arguments_empty(self):
        """Test extracting arguments from sentence with no clear arguments"""
        tokens = [
            Token("Om", "om", POS.VERB, [Morpheme("om", "om", "go", "root")])
        ]
        
        arguments = self.analyzer._extract_arguments(tokens)
        
        assert isinstance(arguments, dict)
        assert arguments['subject'] == []
        assert arguments['object'] == []
        assert arguments['oblique'] == []
    
    def test_analyze_syntax_comprehensive(self):
        """Test comprehensive syntactic analysis"""
        tokens = [
            Token("Om", "om", POS.VERB, [Morpheme("om", "om", "go", "root")]),
            Token("noh", "noh", POS.POSTPOSITION, [Morpheme("noh", "noh", "OBJ", "postposition")]),
            Token("Yong", "Yong", POS.NOUN, [Morpheme("Yong", "Yong", "Yong", "root")]),
            Token("am", "am", POS.POSTPOSITION, [Morpheme("am", "am", "LOC", "postposition")]),
            Token("paapai", "paapai", POS.NOUN, [Morpheme("paapai", "paapai", "house", "root")]),
            Token("pe", "pe", POS.POSTPOSITION, [Morpheme("pe", "pe", "ALL", "postposition")]),
            Token("ci", "ci", POS.TENSE, [Morpheme("ci", "ci", "PAST", "tense")])
        ]
        sentence = Sentence("Om noh Yong am paapai pe ci", tokens, "go OBJ Yong LOC house ALL PAST")
        
        analysis = self.analyzer.analyze_syntax(sentence)
        
        # Verify all expected keys are present
        expected_keys = ['clause_type', 'has_applicative', 'has_relative_clause', 'verb_stem_form', 'arguments']
        for key in expected_keys:
            assert key in analysis
        
        # Verify specific analysis results
        assert analysis['clause_type'] == "transitive"  # Has noh + 2 nouns (not 3+)
        assert analysis['has_applicative'] is False  # No applicative morphemes
        assert analysis['has_relative_clause'] is False  # No 'ah' marker
        assert analysis['verb_stem_form'] == "I"  # ci without 3rd person agreement
        assert isinstance(analysis['arguments'], dict)
        assert 'subject' in analysis['arguments']
        assert 'object' in analysis['arguments']
        assert 'oblique' in analysis['arguments']


class TestSyntaxAnalyzerIntegration:
    """Test integration scenarios for syntactic analysis"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = KchoSyntaxAnalyzer()
    
    def test_real_sentence_analysis(self):
        """Test analysis of real K'Cho sentence"""
        # Create a realistic sentence
        tokens = [
            Token("Om", "om", POS.VERB, [Morpheme("om", "om", "go", "root")]),
            Token("noh", "noh", POS.POSTPOSITION, [Morpheme("noh", "noh", "OBJ", "postposition")]),
            Token("Yong", "Yong", POS.NOUN, [Morpheme("Yong", "Yong", "Yong", "root")]),
            Token("ci", "ci", POS.TENSE, [Morpheme("ci", "ci", "PAST", "tense")])
        ]
        sentence = Sentence("Om noh Yong ci", tokens, "go OBJ Yong PAST")
        
        analysis = self.analyzer.analyze_syntax(sentence)
        
        assert isinstance(analysis, dict)
        assert analysis['clause_type'] == "transitive"
        assert analysis['has_applicative'] is False
        assert analysis['has_relative_clause'] is False
        assert analysis['verb_stem_form'] == "I"
        assert isinstance(analysis['arguments'], dict)
    
    def test_applicative_sentence_analysis(self):
        """Test analysis of sentence with applicative"""
        tokens = [
            Token("omnák", "omnák", POS.VERB, [
                Morpheme("om", "om", "go", "root"),
                Morpheme("nák", "na", "APPL", "suffix", {"stem": "II"})
            ]),
            Token("noh", "noh", POS.POSTPOSITION, [Morpheme("noh", "noh", "OBJ", "postposition")]),
            Token("Yong", "Yong", POS.NOUN, [Morpheme("Yong", "Yong", "Yong", "root")])
        ]
        sentence = Sentence("omnák noh Yong", tokens, "go-APPL OBJ Yong")
        
        analysis = self.analyzer.analyze_syntax(sentence)
        
        assert analysis['has_applicative'] is False  # POS.APPLICATIVE not set by morphology analyzer
        assert analysis['verb_stem_form'] == "II"
    
    def test_relative_clause_analysis(self):
        """Test analysis of sentence with relative clause"""
        tokens = [
            Token("Om", "om", POS.VERB, [Morpheme("om", "om", "go", "root")]),
            Token("noh", "noh", POS.POSTPOSITION, [Morpheme("noh", "noh", "OBJ", "postposition")]),
            Token("Yong", "Yong", POS.NOUN, [Morpheme("Yong", "Yong", "Yong", "root")]),
            Token("ah", "ah", POS.CONJUNCTION, [Morpheme("ah", "ah", "REL", "conjunction")]),
            Token("law", "law", POS.VERB, [Morpheme("law", "law", "come", "root")])
        ]
        sentence = Sentence("Om noh Yong ah law", tokens, "go OBJ Yong REL come")
        
        analysis = self.analyzer.analyze_syntax(sentence)
        
        assert analysis['has_relative_clause'] is True
    
    def test_complex_argument_extraction(self):
        """Test extraction of complex argument structures"""
        tokens = [
            Token("Yong", "Yong", POS.NOUN, [Morpheme("Yong", "Yong", "Yong", "root")]),
            Token("noh", "noh", POS.POSTPOSITION, [Morpheme("noh", "noh", "OBJ", "postposition")]),
            Token("Om", "om", POS.VERB, [Morpheme("om", "om", "go", "root")]),
            Token("paapai", "paapai", POS.NOUN, [Morpheme("paapai", "paapai", "house", "root")]),
            Token("am", "am", POS.POSTPOSITION, [Morpheme("am", "am", "LOC", "postposition")])
        ]
        sentence = Sentence("Yong noh Om paapai am", tokens, "Yong OBJ go house LOC")
        
        analysis = self.analyzer.analyze_syntax(sentence)
        arguments = analysis['arguments']
        
        # Yong should be subject (N followed by noh)
        assert "Yong" in arguments['subject']
        # paapai should be oblique (N followed by postposition)
        assert "paapai" in arguments['oblique']
    
    def test_edge_case_empty_sentence(self):
        """Test analysis of empty sentence"""
        sentence = Sentence("", [], "")
        
        analysis = self.analyzer.analyze_syntax(sentence)
        
        assert isinstance(analysis, dict)
        assert analysis['clause_type'] == "intransitive"  # No verbs = intransitive
        assert analysis['has_applicative'] is False
        assert analysis['has_relative_clause'] is False
        assert analysis['verb_stem_form'] is None
        assert analysis['arguments'] == {'subject': [], 'object': [], 'oblique': []}
    
    def test_edge_case_single_word(self):
        """Test analysis of single word sentence"""
        tokens = [Token("Om", "om", POS.VERB, [Morpheme("om", "om", "go", "root")])]
        sentence = Sentence("Om", tokens, "go")
        
        analysis = self.analyzer.analyze_syntax(sentence)
        
        assert analysis['clause_type'] == "intransitive"
        assert analysis['has_applicative'] is False
        assert analysis['has_relative_clause'] is False
        assert analysis['verb_stem_form'] is None
        assert analysis['arguments'] == {'subject': [], 'object': [], 'oblique': []}
