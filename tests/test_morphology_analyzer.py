"""
Test suite for Morphological Analysis in kcho_system.py

Tests KchoMorphologyAnalyzer class
"""

import pytest
from unittest.mock import patch, MagicMock
from kcho.kcho_system import (
    KchoMorphologyAnalyzer, KchoKnowledge, 
    Token, Morpheme, Sentence, POS, StemType
)


class TestKchoMorphologyAnalyzer:
    """Test KchoMorphologyAnalyzer functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = KchoMorphologyAnalyzer()
    
    def test_initialization(self):
        """Test analyzer initialization"""
        assert hasattr(self.analyzer, 'knowledge')
        assert isinstance(self.analyzer.knowledge, KchoKnowledge)
    
    def test_analyze_token_empty_word(self):
        """Test analyzing empty word"""
        token = self.analyzer.analyze_token("")
        
        assert isinstance(token, Token)
        assert token.surface == ""
        assert token.lemma == ""
        assert token.pos == POS.UNKNOWN
        assert len(token.morphemes) == 0
        assert token.stem_type is None
        assert token.features == {}
    
    def test_analyze_token_whitespace_only(self):
        """Test analyzing whitespace-only word"""
        token = self.analyzer.analyze_token("   ")
        
        assert isinstance(token, Token)
        assert token.surface == ""
        assert token.pos == POS.UNKNOWN
    
    def test_analyze_token_postposition(self):
        """Test analyzing postposition"""
        # Mock knowledge base to return postposition info
        with patch.object(self.analyzer.knowledge, 'is_postposition', return_value=True), \
             patch.object(self.analyzer.knowledge, 'POSTPOSITIONS', {'noh': {'gloss': 'OBJ', 'function': 'accusative'}}):
            
            token = self.analyzer.analyze_token("noh")
            
            assert isinstance(token, Token)
            assert token.surface == "noh"
            assert token.lemma == "noh"
            assert token.pos == POS.POSTPOSITION
            assert len(token.morphemes) == 1
            assert token.morphemes[0].form == "noh"
            assert token.morphemes[0].gloss == "OBJ"
            assert token.morphemes[0].type == "postposition"
    
    def test_analyze_token_tense_marker(self):
        """Test analyzing tense marker"""
        with patch.object(self.analyzer.knowledge, 'is_tense_marker', return_value=True), \
             patch.object(self.analyzer.knowledge, 'TENSE_ASPECT', {'ci': {'gloss': 'PAST', 'tense': 'past'}}):
            
            token = self.analyzer.analyze_token("ci")
            
            assert isinstance(token, Token)
            assert token.surface == "ci"
            assert token.lemma == "ci"
            assert token.pos == POS.TENSE
            assert len(token.morphemes) == 1
            assert token.morphemes[0].gloss == "PAST"
            assert token.morphemes[0].type == "particle"
    
    def test_analyze_token_agreement(self):
        """Test analyzing agreement particle"""
        with patch.object(self.analyzer.knowledge, 'is_agreement', return_value=True), \
             patch.object(self.analyzer.knowledge, 'AGREEMENT', {'a': {'person': '3', 'number': 'sg'}}):
            
            token = self.analyzer.analyze_token("a")
            
            assert isinstance(token, Token)
            assert token.surface == "a"
            assert token.lemma == "a"
            assert token.pos == POS.AGREEMENT
            assert len(token.morphemes) == 1
            assert "3" in token.morphemes[0].gloss
            assert "sg" in token.morphemes[0].gloss
    
    def test_analyze_token_verb_with_applicative_nak(self):
        """Test analyzing verb with -nák applicative"""
        token = self.analyzer.analyze_token("omnák")
        
        assert isinstance(token, Token)
        assert token.surface == "omnák"
        assert token.pos == POS.VERB
        assert token.stem_type == StemType.STEM_II
        assert len(token.morphemes) == 2
        
        # Check root morpheme
        root_morph = token.morphemes[0]
        assert root_morph.form == "om"
        assert root_morph.gloss == "V"
        assert root_morph.type == "root"
        
        # Check applicative morpheme
        app_morph = token.morphemes[1]
        assert app_morph.form == "nák"
        assert app_morph.gloss == "APPL"
        assert app_morph.type == "suffix"
    
    def test_analyze_token_verb_with_applicative_na(self):
        """Test analyzing verb with -na applicative"""
        token = self.analyzer.analyze_token("omna")
        
        assert isinstance(token, Token)
        assert token.surface == "omna"
        assert token.pos == POS.VERB
        assert token.stem_type == StemType.STEM_I
        assert len(token.morphemes) == 2
        
        # Check root morpheme
        root_morph = token.morphemes[0]
        assert root_morph.form == "om"
        assert root_morph.gloss == "V"
        assert root_morph.type == "root"
        
        # Check applicative morpheme
        app_morph = token.morphemes[1]
        assert app_morph.form == "na"
        assert app_morph.gloss == "APPL"
        assert app_morph.type == "suffix"
    
    def test_analyze_token_known_verb(self):
        """Test analyzing known verb"""
        with patch.object(self.analyzer.knowledge, 'VERB_STEMS', {'om': {'gloss': 'go', 'stem2': 'law'}}):
            token = self.analyzer.analyze_token("om")
            
            assert isinstance(token, Token)
            assert token.surface == "om"
            assert token.lemma == "om"
            assert token.pos == POS.VERB
            assert token.stem_type == StemType.STEM_I
            assert len(token.morphemes) == 1
            assert token.morphemes[0].gloss == "go"
    
    def test_analyze_token_known_noun(self):
        """Test analyzing known noun"""
        with patch.object(self.analyzer.knowledge, 'COMMON_NOUNS', {'yong': {'gloss': 'house'}}):
            token = self.analyzer.analyze_token("yong")
            
            assert isinstance(token, Token)
            assert token.surface == "yong"
            assert token.lemma == "yong"
            assert token.pos == POS.NOUN
            assert len(token.morphemes) == 1
            assert token.morphemes[0].gloss == "yong"  # Uses word as gloss when not in verb stems
    
    def test_analyze_token_proper_noun(self):
        """Test analyzing proper noun (capitalized)"""
        token = self.analyzer.analyze_token("Yong")
        
        assert isinstance(token, Token)
        assert token.surface == "Yong"
        assert token.pos == POS.NOUN  # Should be identified as noun due to capitalization
    
    def test_analyze_token_kcho_prefix(self):
        """Test analyzing word with K'Cho prefix"""
        token = self.analyzer.analyze_token("K'Cho")
        
        assert isinstance(token, Token)
        assert token.surface == "K'Cho"
        assert token.pos == POS.NOUN  # Should be identified as noun due to K' prefix
    
    def test_analyze_token_unknown_word(self):
        """Test analyzing unknown word"""
        token = self.analyzer.analyze_token("unknownword")
        
        assert isinstance(token, Token)
        assert token.surface == "unknownword"
        assert token.lemma == "unknownword"
        assert token.pos == POS.UNKNOWN
        assert len(token.morphemes) == 1
        assert token.morphemes[0].form == "unknownword"
    
    def test_get_lemma_stem_ii_to_stem_i(self):
        """Test getting Stem I from Stem II"""
        with patch.object(self.analyzer.knowledge, 'get_stem_i', return_value='om'):
            lemma = self.analyzer._get_lemma("law")
            assert lemma == "om"
    
    def test_get_lemma_no_stem_ii(self):
        """Test getting lemma when not Stem II"""
        with patch.object(self.analyzer.knowledge, 'get_stem_i', return_value=None):
            lemma = self.analyzer._get_lemma("om")
            assert lemma == "om"
    
    def test_analyze_sentence_basic(self):
        """Test analyzing complete sentence"""
        with patch.object(self.analyzer, 'analyze_token') as mock_analyze:
            # Mock token analysis
            mock_token1 = Token("Om", "om", POS.VERB, [Morpheme("om", "om", "go", "root")])
            mock_token2 = Token("noh", "noh", POS.POSTPOSITION, [Morpheme("noh", "noh", "OBJ", "postposition")])
            mock_token3 = Token("Yong", "Yong", POS.NOUN, [Morpheme("Yong", "Yong", "Yong", "root")])
            
            mock_analyze.side_effect = [mock_token1, mock_token2, mock_token3]
            
            sentence = self.analyzer.analyze_sentence("Om noh Yong")
            
            assert isinstance(sentence, Sentence)  # Should return Sentence object
            assert sentence.text == "Om noh Yong"
            assert len(sentence.tokens) == 3
            assert sentence.tokens[0] == mock_token1
            assert sentence.tokens[1] == mock_token2
            assert sentence.tokens[2] == mock_token3
            assert sentence.gloss == "go OBJ Yong"
            assert 'timestamp' in sentence.metadata
    
    def test_generate_gloss_single_morpheme(self):
        """Test generating gloss for tokens with single morphemes"""
        token1 = Token("om", "om", POS.VERB, [Morpheme("om", "om", "go", "root")])
        token2 = Token("noh", "noh", POS.POSTPOSITION, [Morpheme("noh", "noh", "OBJ", "postposition")])
        
        gloss = self.analyzer._generate_gloss([token1, token2])
        assert gloss == "go OBJ"
    
    def test_generate_gloss_multiple_morphemes(self):
        """Test generating gloss for tokens with multiple morphemes"""
        token = Token("omnák", "omnák", POS.VERB, [
            Morpheme("om", "om", "go", "root"),
            Morpheme("nák", "na", "APPL", "suffix")
        ])
        
        gloss = self.analyzer._generate_gloss([token])
        assert gloss == "go-APPL"
    
    def test_generate_gloss_mixed_tokens(self):
        """Test generating gloss for mixed single and multiple morpheme tokens"""
        token1 = Token("om", "om", POS.VERB, [Morpheme("om", "om", "go", "root")])
        token2 = Token("omnák", "omnák", POS.VERB, [
            Morpheme("om", "om", "go", "root"),
            Morpheme("nák", "na", "APPL", "suffix")
        ])
        
        gloss = self.analyzer._generate_gloss([token1, token2])
        assert gloss == "go go-APPL"


class TestMorphologyAnalyzerIntegration:
    """Test integration scenarios for morphological analysis"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = KchoMorphologyAnalyzer()
    
    def test_complex_sentence_analysis(self):
        """Test analysis of complex sentence with multiple word types"""
        # This test uses real knowledge base data
        sentence = self.analyzer.analyze_sentence("Om noh Yong am paapai pe ci")
        
        assert isinstance(sentence, Sentence)  # Should be Sentence
        assert sentence.text == "Om noh Yong am paapai pe ci"
        assert len(sentence.tokens) > 0
        assert sentence.gloss is not None
        assert len(sentence.gloss) > 0
    
    def test_applicative_verb_analysis(self):
        """Test analysis of verb with applicative"""
        token = self.analyzer.analyze_token("omnák")
        
        assert token.pos == POS.VERB
        assert token.stem_type == StemType.STEM_II
        assert len(token.morphemes) == 2
        assert any(morph.gloss == "APPL" for morph in token.morphemes)
    
    def test_postposition_analysis(self):
        """Test analysis of postposition"""
        token = self.analyzer.analyze_token("noh")
        
        # Should be identified as postposition if in knowledge base
        assert isinstance(token, Token)
        assert token.surface == "noh"
    
    def test_tense_marker_analysis(self):
        """Test analysis of tense marker"""
        token = self.analyzer.analyze_token("ci")
        
        # Should be identified as tense marker if in knowledge base
        assert isinstance(token, Token)
        assert token.surface == "ci"
    
    def test_unknown_word_handling(self):
        """Test handling of completely unknown words"""
        token = self.analyzer.analyze_token("xyzunknown")
        
        assert isinstance(token, Token)
        assert token.surface == "xyzunknown"
        assert token.pos == POS.UNKNOWN
        assert len(token.morphemes) == 1
        assert token.morphemes[0].form == "xyzunknown"
    
    def test_punctuation_stripping(self):
        """Test that punctuation is properly stripped"""
        token = self.analyzer.analyze_token("om.")
        
        assert isinstance(token, Token)
        assert token.surface == "om"  # Should strip punctuation
        assert "." not in token.surface
    
    def test_case_preservation(self):
        """Test that case is preserved in analysis"""
        token = self.analyzer.analyze_token("Yong")
        
        assert isinstance(token, Token)
        assert token.surface == "Yong"  # Should preserve case
        assert token.surface[0].isupper()
