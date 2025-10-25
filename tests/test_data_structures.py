"""
Test suite for Core Data Structures in kcho_system.py

Tests the fundamental data structures: POS, StemType, Morpheme, Token, Sentence
"""

import pytest
from kcho.kcho_system import POS, StemType, Morpheme, Token, Sentence


class TestPOS:
    """Test POS enum functionality"""
    
    def test_pos_values(self):
        """Test that POS enum has correct values"""
        assert POS.NOUN.value == 'N'
        assert POS.VERB.value == 'V'
        assert POS.POSTPOSITION.value == 'P'
        assert POS.AGREEMENT.value == 'AGR'
        assert POS.TENSE.value == 'T'
        assert POS.DEICTIC.value == 'D'
        assert POS.CONJUNCTION.value == 'CONJ'
        assert POS.ADJECTIVE.value == 'ADJ'
        assert POS.ADVERB.value == 'ADV'
        assert POS.DETERMINER.value == 'DET'
        assert POS.APPLICATIVE.value == 'APPL'
        assert POS.TENSE_ASPECT.value == 'T/A'
        assert POS.UNKNOWN.value == 'UNK'
    
    def test_pos_iteration(self):
        """Test POS enum iteration"""
        pos_values = [pos.value for pos in POS]
        expected_values = ['N', 'V', 'P', 'AGR', 'T', 'D', 'CONJ', 'ADJ', 'ADV', 'DET', 'APPL', 'T/A', 'UNK']
        assert set(pos_values) == set(expected_values)
    
    def test_pos_membership(self):
        """Test POS enum membership"""
        assert 'N' in POS
        assert 'V' in POS
        assert 'UNKNOWN' not in POS  # Should be 'UNK'
        assert 'UNK' in POS


class TestStemType:
    """Test StemType enum functionality"""
    
    def test_stem_type_values(self):
        """Test that StemType enum has correct values"""
        assert StemType.STEM_I.value == 'I'
        assert StemType.STEM_II.value == 'II'
    
    def test_stem_type_iteration(self):
        """Test StemType enum iteration"""
        stem_values = [stem.value for stem in StemType]
        assert set(stem_values) == {'I', 'II'}
    
    def test_stem_type_membership(self):
        """Test StemType enum membership"""
        assert 'I' in StemType
        assert 'II' in StemType
        assert 'STEM_I' not in StemType  # Should be 'I'


class TestMorpheme:
    """Test Morpheme dataclass functionality"""
    
    def test_morpheme_creation(self):
        """Test basic morpheme creation"""
        morpheme = Morpheme(
            form="om",
            lemma="om",
            gloss="go",
            type="root"
        )
        assert morpheme.form == "om"
        assert morpheme.lemma == "om"
        assert morpheme.gloss == "go"
        assert morpheme.type == "root"
        assert morpheme.features == {}
    
    def test_morpheme_with_features(self):
        """Test morpheme creation with features"""
        features = {"person": "3", "number": "sg"}
        morpheme = Morpheme(
            form="a",
            lemma="a",
            gloss="3SG",
            type="agreement",
            features=features
        )
        assert morpheme.features == features
        assert morpheme.features["person"] == "3"
        assert morpheme.features["number"] == "sg"
    
    def test_morpheme_to_dict(self):
        """Test morpheme to_dict conversion"""
        morpheme = Morpheme(
            form="noh",
            lemma="noh",
            gloss="OBJ",
            type="postposition",
            features={"case": "accusative"}
        )
        morpheme_dict = morpheme.to_dict()
        
        assert isinstance(morpheme_dict, dict)
        assert morpheme_dict["form"] == "noh"
        assert morpheme_dict["lemma"] == "noh"
        assert morpheme_dict["gloss"] == "OBJ"
        assert morpheme_dict["type"] == "postposition"
        assert morpheme_dict["features"]["case"] == "accusative"
    
    def test_morpheme_equality(self):
        """Test morpheme equality"""
        morpheme1 = Morpheme("om", "om", "go", "root")
        morpheme2 = Morpheme("om", "om", "go", "root")
        morpheme3 = Morpheme("law", "law", "come", "root")
        
        assert morpheme1 == morpheme2
        assert morpheme1 != morpheme3


class TestToken:
    """Test Token dataclass functionality"""
    
    def test_token_creation(self):
        """Test basic token creation"""
        morpheme = Morpheme("om", "om", "go", "root")
        token = Token(
            surface="om",
            lemma="om",
            pos=POS.VERB,
            morphemes=[morpheme]
        )
        
        assert token.surface == "om"
        assert token.lemma == "om"
        assert token.pos == POS.VERB
        assert len(token.morphemes) == 1
        assert token.morphemes[0] == morpheme
        assert token.stem_type is None
        assert token.features == {}
    
    def test_token_with_stem_type(self):
        """Test token creation with stem type"""
        morpheme = Morpheme("law", "law", "come", "root")
        token = Token(
            surface="law",
            lemma="law",
            pos=POS.VERB,
            morphemes=[morpheme],
            stem_type=StemType.STEM_I
        )
        
        assert token.stem_type == StemType.STEM_I
    
    def test_token_with_features(self):
        """Test token creation with features"""
        morpheme = Morpheme("ci", "ci", "PAST", "tense")
        features = {"tense": "past", "aspect": "perfective"}
        token = Token(
            surface="ci",
            lemma="ci",
            pos=POS.TENSE,
            morphemes=[morpheme],
            features=features
        )
        
        assert token.features == features
        assert token.features["tense"] == "past"
    
    def test_token_to_dict(self):
        """Test token to_dict conversion"""
        morpheme = Morpheme("noh", "noh", "OBJ", "postposition")
        token = Token(
            surface="noh",
            lemma="noh",
            pos=POS.POSTPOSITION,
            morphemes=[morpheme],
            stem_type=None,
            features={"case": "accusative"}
        )
        
        token_dict = token.to_dict()
        
        assert isinstance(token_dict, dict)
        assert token_dict["surface"] == "noh"
        assert token_dict["lemma"] == "noh"
        assert token_dict["pos"] == "P"
        assert token_dict["stem_type"] is None
        assert len(token_dict["morphemes"]) == 1
        assert token_dict["morphemes"][0]["form"] == "noh"
        assert token_dict["features"]["case"] == "accusative"
    
    def test_token_to_dict_with_stem_type(self):
        """Test token to_dict with stem type"""
        morpheme = Morpheme("om", "om", "go", "root")
        token = Token(
            surface="om",
            lemma="om",
            pos=POS.VERB,
            morphemes=[morpheme],
            stem_type=StemType.STEM_I
        )
        
        token_dict = token.to_dict()
        assert token_dict["stem_type"] == "I"
    
    def test_token_equality(self):
        """Test token equality"""
        morpheme1 = Morpheme("om", "om", "go", "root")
        morpheme2 = Morpheme("om", "om", "go", "root")
        
        token1 = Token("om", "om", POS.VERB, [morpheme1])
        token2 = Token("om", "om", POS.VERB, [morpheme2])
        token3 = Token("law", "law", POS.VERB, [morpheme1])
        
        assert token1 == token2
        assert token1 != token3


class TestSentence:
    """Test Sentence dataclass functionality"""
    
    def test_sentence_creation(self):
        """Test basic sentence creation"""
        morpheme = Morpheme("om", "om", "go", "root")
        token = Token("om", "om", POS.VERB, [morpheme])
        
        sentence = Sentence(
            text="Om noh Yong am paapai pe ci",
            tokens=[token],
            gloss="go OBJ Yong am paapai pe PAST"
        )
        
        assert sentence.text == "Om noh Yong am paapai pe ci"
        assert len(sentence.tokens) == 1
        assert sentence.tokens[0] == token
        assert sentence.gloss == "go OBJ Yong am paapai pe PAST"
        assert sentence.translation is None
        assert sentence.syntax is None
        assert sentence.metadata == {}
    
    def test_sentence_with_translation(self):
        """Test sentence creation with translation"""
        morpheme = Morpheme("om", "om", "go", "root")
        token = Token("om", "om", POS.VERB, [morpheme])
        
        sentence = Sentence(
            text="Om noh Yong am paapai pe ci",
            tokens=[token],
            gloss="go OBJ Yong am paapai pe PAST",
            translation="I went to Yong's house yesterday"
        )
        
        assert sentence.translation == "I went to Yong's house yesterday"
    
    def test_sentence_with_syntax(self):
        """Test sentence creation with syntax analysis"""
        morpheme = Morpheme("om", "om", "go", "root")
        token = Token("om", "om", POS.VERB, [morpheme])
        
        syntax = {"clause_type": "transitive", "arguments": {"subject": ["I"], "object": ["house"]}}
        sentence = Sentence(
            text="Om noh Yong am paapai pe ci",
            tokens=[token],
            gloss="go OBJ Yong am paapai pe PAST",
            syntax=syntax
        )
        
        assert sentence.syntax == syntax
        assert sentence.syntax["clause_type"] == "transitive"
    
    def test_sentence_with_metadata(self):
        """Test sentence creation with metadata"""
        morpheme = Morpheme("om", "om", "go", "root")
        token = Token("om", "om", POS.VERB, [morpheme])
        
        metadata = {"source": "fieldwork", "speaker": "informant_1", "date": "2024-01-01"}
        sentence = Sentence(
            text="Om noh Yong am paapai pe ci",
            tokens=[token],
            gloss="go OBJ Yong am paapai pe PAST",
            metadata=metadata
        )
        
        assert sentence.metadata == metadata
        assert sentence.metadata["source"] == "fieldwork"
    
    def test_sentence_to_dict(self):
        """Test sentence to_dict conversion"""
        morpheme = Morpheme("om", "om", "go", "root")
        token = Token("om", "om", POS.VERB, [morpheme])
        
        sentence = Sentence(
            text="Om noh Yong am paapai pe ci",
            tokens=[token],
            gloss="go OBJ Yong am paapai pe PAST",
            translation="I went to Yong's house yesterday",
            syntax={"clause_type": "transitive"},
            metadata={"source": "fieldwork"}
        )
        
        sentence_dict = sentence.to_dict()
        
        assert isinstance(sentence_dict, dict)
        assert sentence_dict["text"] == "Om noh Yong am paapai pe ci"
        assert sentence_dict["gloss"] == "go OBJ Yong am paapai pe PAST"
        assert sentence_dict["translation"] == "I went to Yong's house yesterday"
        assert sentence_dict["syntax"]["clause_type"] == "transitive"
        assert sentence_dict["metadata"]["source"] == "fieldwork"
        assert len(sentence_dict["tokens"]) == 1
        assert sentence_dict["tokens"][0]["surface"] == "om"
    
    def test_sentence_equality(self):
        """Test sentence equality"""
        morpheme1 = Morpheme("om", "om", "go", "root")
        morpheme2 = Morpheme("om", "om", "go", "root")
        token1 = Token("om", "om", POS.VERB, [morpheme1])
        token2 = Token("om", "om", POS.VERB, [morpheme2])
        
        sentence1 = Sentence("Om", [token1], "go")
        sentence2 = Sentence("Om", [token2], "go")
        sentence3 = Sentence("Law", [token1], "come")
        
        assert sentence1 == sentence2
        assert sentence1 != sentence3


class TestDataStructureIntegration:
    """Test integration between data structures"""
    
    def test_complex_sentence_structure(self):
        """Test creating a complex sentence with multiple tokens and morphemes"""
        # Create morphemes
        om_morph = Morpheme("om", "om", "go", "root")
        noh_morph = Morpheme("noh", "noh", "OBJ", "postposition")
        yong_morph = Morpheme("Yong", "Yong", "Yong", "root")
        ci_morph = Morpheme("ci", "ci", "PAST", "tense")
        
        # Create tokens
        om_token = Token("om", "om", POS.VERB, [om_morph], StemType.STEM_I)
        noh_token = Token("noh", "noh", POS.POSTPOSITION, [noh_morph])
        yong_token = Token("Yong", "Yong", POS.NOUN, [yong_morph])
        ci_token = Token("ci", "ci", POS.TENSE, [ci_morph])
        
        # Create sentence
        sentence = Sentence(
            text="Om noh Yong ci",
            tokens=[om_token, noh_token, yong_token, ci_token],
            gloss="go OBJ Yong PAST",
            translation="I went to Yong",
            syntax={"clause_type": "transitive", "arguments": {"subject": ["I"], "object": ["Yong"]}},
            metadata={"source": "test", "complexity": "simple"}
        )
        
        # Verify structure
        assert len(sentence.tokens) == 4
        assert sentence.tokens[0].pos == POS.VERB
        assert sentence.tokens[0].stem_type == StemType.STEM_I
        assert sentence.tokens[1].pos == POS.POSTPOSITION
        assert sentence.tokens[2].pos == POS.NOUN
        assert sentence.tokens[3].pos == POS.TENSE
        
        # Verify to_dict works
        sentence_dict = sentence.to_dict()
        assert len(sentence_dict["tokens"]) == 4
        assert sentence_dict["tokens"][0]["pos"] == "V"
        assert sentence_dict["tokens"][0]["stem_type"] == "I"
    
    def test_morpheme_features_propagation(self):
        """Test that morpheme features are properly handled"""
        # Create morpheme with complex features
        features = {
            "person": "3",
            "number": "sg",
            "gender": "m",
            "case": "nom"
        }
        morpheme = Morpheme("a", "a", "3SG.M.NOM", "agreement", features)
        
        # Create token with features
        token_features = {"agreement": "3sg", "tense": "past"}
        token = Token("a", "a", POS.AGREEMENT, [morpheme], features=token_features)
        
        # Verify features are preserved
        assert morpheme.features["person"] == "3"
        assert token.features["agreement"] == "3sg"
        
        # Verify to_dict preserves features
        token_dict = token.to_dict()
        assert token_dict["features"]["agreement"] == "3sg"
        assert token_dict["morphemes"][0]["features"]["person"] == "3"
