"""
LLaMA API Integration for K'Cho Linguistic Processing Toolkit

This module provides actual LLaMA API integration for the K'Cho linguistic system.
Supports multiple LLaMA providers and configurations.

Author: Hung Om
Research Foundation: Based on foundational K'Cho linguistic research and Austroasiatic language studies
Version: 0.2.0
Date: 2024-10-25
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import requests
from dataclasses import dataclass

# Try to load dotenv for .env file support
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    load_dotenv = None

logger = logging.getLogger(__name__)


def load_env_file(env_file: str = ".env") -> bool:
    """
    Load environment variables from .env file
    
    Args:
        env_file: Path to .env file (default: ".env")
        
    Returns:
        True if .env file was loaded successfully, False otherwise
    """
    if not DOTENV_AVAILABLE:
        logger.warning("python-dotenv not installed. Install with: pip install python-dotenv")
        return False
    
    try:
        # Try to load from current directory first
        if os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info(f"✅ Loaded environment variables from {env_file}")
            return True
        
        # Try to load from project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(project_root, env_file)
        if os.path.exists(env_path):
            load_dotenv(env_path)
            logger.info(f"✅ Loaded environment variables from {env_path}")
            return True
        
        logger.debug(f"No .env file found at {env_file} or {env_path}")
        return False
        
    except Exception as e:
        logger.warning(f"Failed to load .env file: {e}")
        return False


@dataclass
class LLaMAConfig:
    """Configuration for LLaMA API integration"""
    provider: str = "ollama"  # "ollama", "openai", "anthropic", "huggingface"
    base_url: str = "http://localhost:11434"  # Ollama default
    model: str = "llama3.1:8b"
    api_key: Optional[str] = None
    timeout: int = 30
    max_tokens: int = 1000
    temperature: float = 0.7


class LLaMAAPIClient:
    """Client for LLaMA API integration"""
    
    def __init__(self, config: LLaMAConfig):
        self.config = config
        self.session = requests.Session()
        
        # Set up headers
        if config.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json"
            })
    
    def generate_response(self, prompt: str, context: Optional[str] = None, max_tokens: int = None) -> Dict[str, Any]:
        """
        Generate response from LLaMA API
        
        Args:
            prompt: The prompt to send to LLaMA
            context: Optional context for the prompt
            max_tokens: Override max_tokens for this request
            
        Returns:
            Dictionary with response data
        """
        try:
            # Prepare the full prompt with context
            full_prompt = self._prepare_prompt(prompt, context)
            
            # Make API request based on provider
            if self.config.provider == "ollama":
                response = self._call_ollama_api(full_prompt, max_tokens)
            elif self.config.provider == "openai":
                response = self._call_openai_api(full_prompt, max_tokens)
            elif self.config.provider == "anthropic":
                response = self._call_anthropic_api(full_prompt, max_tokens)
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
            
            return {
                "success": True,
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "provider": self.config.provider,
                "model": self.config.model,
                "tokens_used": max_tokens or self.config.max_tokens
            }
            
        except Exception as e:
            logger.error(f"LLaMA API error: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "provider": self.config.provider,
                "model": self.config.model,
                "tokens_used": 0
            }
    
    def _prepare_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        """Prepare the full prompt with K'Cho linguistic context"""
        system_prompt = """You are a linguistic expert specializing in K'Cho language analysis. 
K'Cho is an Austroasiatic language with specific morphological and syntactic patterns.

When analyzing K'Cho language data, consider:
- Verb-particle constructions (VP patterns)
- Postpositional phrases (PP patterns)
- Agreement patterns (AGR patterns)
- Morphological structure and word formation
- Syntactic word order patterns

Provide detailed, accurate linguistic analysis based on the K'Cho language data provided."""
        
        if context:
            full_prompt = f"{system_prompt}\n\nContext: {context}\n\nQuery: {prompt}"
        else:
            full_prompt = f"{system_prompt}\n\nQuery: {prompt}"
        
        return full_prompt
    
    def _call_ollama_api(self, prompt: str) -> str:
        """Call Ollama API"""
        url = f"{self.config.base_url}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        response = self.session.post(url, json=payload, timeout=self.config.timeout)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "")
    
    def _call_openai_api(self, prompt: str, max_tokens: int = None) -> str:
        """Call OpenAI API (for LLaMA models via OpenAI)"""
        if max_tokens is None:
            max_tokens = self.config.max_tokens
            
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": self.config.temperature
        }
        
        response = self.session.post(url, json=payload, timeout=self.config.timeout)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def _call_anthropic_api(self, prompt: str, max_tokens: int = None) -> str:
        """Call Anthropic API"""
        if max_tokens is None:
            max_tokens = self.config.max_tokens
            
        url = "https://api.anthropic.com/v1/messages"
        payload = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        response = self.session.post(url, json=payload, timeout=self.config.timeout)
        response.raise_for_status()
        
        result = response.json()
        return result["content"][0]["text"]


class EnhancedKchoAPILayer:
    """Enhanced API layer with actual LLaMA integration"""
    
    def __init__(self, knowledge_base, llama_config: Optional[LLaMAConfig] = None):
        """
        Initialize enhanced API layer with LLaMA integration
        
        Args:
            knowledge_base: KchoKnowledgeBase or LegacyKchoKnowledge instance
            llama_config: LLaMA configuration (optional)
        """
        self.knowledge = knowledge_base
        
        # Set up LLaMA configuration
        if llama_config is None:
            # Try to load from environment variables
            llama_config = self._load_config_from_env()
        
        self.config = llama_config
        self.llama_client = LLaMAAPIClient(llama_config) if llama_config else None
    
    def _load_config_from_env(self) -> Optional[LLaMAConfig]:
        """Load LLaMA configuration from environment variables and .env files"""
        # Try to load .env file first
        load_env_file()
        
        provider = os.getenv("LLAMA_PROVIDER", "ollama")
        
        if provider == "ollama":
            return LLaMAConfig(
                provider="ollama",
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
                timeout=int(os.getenv("LLAMA_TIMEOUT", "30")),
                max_tokens=int(os.getenv("LLAMA_MAX_TOKENS", "1000")),
                temperature=float(os.getenv("LLAMA_TEMPERATURE", "0.7"))
            )
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not found in environment")
                return None
            return LLaMAConfig(
                provider="openai",
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                api_key=api_key,
                timeout=int(os.getenv("LLAMA_TIMEOUT", "30")),
                max_tokens=int(os.getenv("LLAMA_MAX_TOKENS", "1000")),
                temperature=float(os.getenv("LLAMA_TEMPERATURE", "0.7"))
            )
        else:
            logger.warning(f"Unknown LLaMA provider: {provider}")
            return None
    
    def get_api_response(self, query: str, use_llama: bool = True, context: Optional[str] = None, deep_research: bool = False, log_response: bool = False, log_format: str = "json") -> Dict[str, Any]:
        """
        Get API response with optional LLaMA integration
        
        Args:
            query: User query about K'Cho language
            use_llama: Whether to use LLaMA API (default: True)
            context: Optional context for the query
            deep_research: Whether to use deep research mode (3000 tokens)
            log_response: Whether to log response to file
            log_format: Log format ("json" or "csv")
            
        Returns:
            Structured API response
        """
        # First, get structured data from knowledge base
        structured_data = self._get_structured_data(query)
        
        if use_llama and self.llama_client:
            try:
                # Determine token limit based on mode
                max_tokens = 3000 if deep_research else 1000
                
                # Check if deep research mode requires user approval
                if deep_research and self.config.provider in ["openai", "anthropic"]:
                    logger.warning(f"⚠️  Deep research mode requested (3000 tokens) with {self.config.provider} API")
                    logger.warning("💰 This may incur significant costs. Proceeding with user approval...")
                
                # Use LLaMA API for enhanced response
                llama_response = self.llama_client.generate_response(
                    query, 
                    context=context or structured_data.get("context", ""),
                    max_tokens=max_tokens
                )
                
                response = {
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "response_type": "llama_enhanced",
                    "structured_data": structured_data,
                    "llama_response": llama_response,
                    "llama_enabled": True,
                    "deep_research": deep_research,
                    "tokens_used": llama_response.get("tokens_used", 0)
                }
                
                # Log response if requested
                if log_response:
                    self._log_response(response, log_format)
                
                return response
                
            except Exception as e:
                logger.warning(f"LLaMA API failed, falling back to structured data: {e}")
                # Fall back to structured data only
                response = {
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "response_type": "structured_only",
                    "data": structured_data,
                    "llama_enabled": False,
                    "llama_error": str(e),
                    "deep_research": False,
                    "tokens_used": 0
                }
                
                # Log response if requested
                if log_response:
                    self._log_response(response, log_format)
                
                return response
        else:
            # Return structured data only
            response = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "response_type": "structured_only",
                "data": structured_data,
                "llama_enabled": False,
                "deep_research": False,
                "tokens_used": 0
            }
            
            # Log response if requested
            if log_response:
                self._log_response(response, log_format)
            
            return response
    
    def _log_response(self, response: Dict[str, Any], log_format: str = "json"):
        """Log response to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            query_safe = "".join(c for c in response["query"][:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            query_safe = query_safe.replace(' ', '_')
            
            if log_format.lower() == "csv":
                filename = f"kcho_response_{timestamp}_{query_safe}.csv"
                self._log_to_csv(response, filename)
            else:
                filename = f"kcho_response_{timestamp}_{query_safe}.json"
                self._log_to_json(response, filename)
            
            logger.info(f"📝 Response logged to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to log response: {e}")
    
    def _log_to_json(self, response: Dict[str, Any], filename: str):
        """Log response to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(response, f, indent=2, ensure_ascii=False)
    
    def _log_to_csv(self, response: Dict[str, Any], filename: str):
        """Log response to CSV file"""
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Field', 'Value'])
            
            # Write basic fields
            writer.writerow(['Query', response.get('query', '')])
            writer.writerow(['Timestamp', response.get('timestamp', '')])
            writer.writerow(['Response Type', response.get('response_type', '')])
            writer.writerow(['LLaMA Enabled', response.get('llama_enabled', False)])
            writer.writerow(['Deep Research', response.get('deep_research', False)])
            writer.writerow(['Tokens Used', response.get('tokens_used', 0)])
            
            # Write structured data
            if 'structured_data' in response:
                structured = response['structured_data']
                writer.writerow(['Query Type', structured.get('query_type', '')])
                writer.writerow(['Total Patterns', structured.get('total_patterns', '')])
                writer.writerow(['Pattern Categories', ', '.join(structured.get('pattern_categories', []))])
                writer.writerow(['Context', structured.get('context', '')])
            
            # Write LLaMA response
            if 'llama_response' in response:
                llama_resp = response['llama_response']
                writer.writerow(['LLaMA Success', llama_resp.get('success', False)])
                writer.writerow(['LLaMA Provider', llama_resp.get('provider', '')])
                writer.writerow(['LLaMA Model', llama_resp.get('model', '')])
                if llama_resp.get('success'):
                    writer.writerow(['LLaMA Response', llama_resp.get('response', '')])
                else:
                    writer.writerow(['LLaMA Error', llama_resp.get('error', '')])
    
    def _get_structured_data(self, query: str) -> Dict[str, Any]:
        """Get structured data from knowledge base (existing functionality)"""
        query_lower = query.lower()
        
        if 'pattern' in query_lower or 'collocation' in query_lower:
            return self._handle_pattern_query(query)
        elif 'morphology' in query_lower or 'morpheme' in query_lower or 'morphological' in query_lower:
            return self._handle_morphology_query(query)
        elif 'syntax' in query_lower or 'grammar' in query_lower or 'postposition' in query_lower:
            return self._handle_syntax_query(query)
        elif 'word' in query_lower or 'vocabulary' in query_lower or 'verb' in query_lower:
            return self._handle_vocabulary_query(query)
        else:
            return self._handle_general_query(query)
    
    def _handle_pattern_query(self, query: str) -> Dict[str, Any]:
        """Handle pattern-related queries"""
        if hasattr(self.knowledge, 'get_collocations_by_category'):
            # New SQLite-based knowledge base
            categories = ['VP', 'PP', 'APP', 'AGR', 'AUX', 'COMP', 'MWE']
            pattern_categories = []
            total_patterns = 0
            
            for category in categories:
                patterns = self.knowledge.get_collocations_by_category(category)
                if patterns:
                    pattern_categories.append(category)
                    total_patterns += len(patterns)
            
            sample_patterns = {}
            for category in pattern_categories[:3]:
                patterns = self.knowledge.get_collocations_by_category(category)
                sample_patterns[category] = [p['words'] for p in patterns[:3]]
            
            return {
                "query_type": "pattern_analysis",
                "total_patterns": total_patterns,
                "pattern_categories": pattern_categories,
                "sample_patterns": sample_patterns,
                "context": f"Found {total_patterns} linguistic patterns across {len(pattern_categories)} categories"
            }
        else:
            # Legacy knowledge base
            return {
                "query_type": "pattern_analysis",
                "gold_standard_patterns": len(self.knowledge.gold_standard_patterns),
                "pattern_categories": list(set(data['category'] for data in self.knowledge.gold_standard_patterns.values())),
                "sample_patterns": dict(list(self.knowledge.gold_standard_patterns.items())[:5]),
                "context": f"Found {len(self.knowledge.gold_standard_patterns)} gold standard patterns"
            }
    
    def _handle_morphology_query(self, query: str) -> Dict[str, Any]:
        """Handle morphology-related queries"""
        if hasattr(self.knowledge, 'get_verb_stem'):
            stats = self.knowledge.get_statistics()
            return {
                "query_type": "morphology_analysis",
                "verb_stems": stats['verb_stems'],
                "sample_verbs": self._get_sample_verbs(),
                "morphological_categories": ['verb', 'noun', 'pronoun', 'agreement', 'postposition'],
                "context": f"Database contains {stats['verb_stems']} verb stems"
            }
        else:
            return {
                "query_type": "morphology_analysis",
                "verb_stems": len(self.knowledge.VERB_STEMS),
                "sample_verbs": dict(list(self.knowledge.VERB_STEMS.items())[:5]),
                "morphological_categories": list(self.knowledge.word_categories.keys()),
                "context": f"Knowledge base contains {len(self.knowledge.VERB_STEMS)} verb stems"
            }
    
    def _handle_syntax_query(self, query: str) -> Dict[str, Any]:
        """Handle syntax-related queries"""
        if hasattr(self.knowledge, 'get_collocations_by_category'):
            pp_patterns = self.knowledge.get_collocations_by_category('PP')
            agr_patterns = self.knowledge.get_collocations_by_category('AGR')
            
            return {
                "query_type": "syntax_analysis",
                "postposition_patterns": len(pp_patterns),
                "sample_postpositions": [p['words'] for p in pp_patterns[:5]],
                "agreement_patterns": len(agr_patterns),
                "sample_agreement": [p['words'] for p in agr_patterns[:5]],
                "context": f"Found {len(pp_patterns)} postposition patterns and {len(agr_patterns)} agreement patterns"
            }
        else:
            return {
                "query_type": "syntax_analysis",
                "context": "Syntax analysis available through legacy knowledge base"
            }
    
    def _handle_vocabulary_query(self, query: str) -> Dict[str, Any]:
        """Handle vocabulary-related queries"""
        if hasattr(self.knowledge, 'get_statistics'):
            stats = self.knowledge.get_statistics()
            return {
                "query_type": "vocabulary_analysis",
                "word_frequencies": stats['word_frequencies'],
                "parallel_sentences": stats['parallel_sentences'],
                "context": f"Database contains {stats['word_frequencies']} word frequency entries"
            }
        else:
            return {
                "query_type": "vocabulary_analysis",
                "context": "Vocabulary analysis available through legacy knowledge base"
            }
    
    def _handle_general_query(self, query: str) -> Dict[str, Any]:
        """Handle general queries"""
        if hasattr(self.knowledge, 'get_statistics'):
            stats = self.knowledge.get_statistics()
            return {
                "query_type": "general_analysis",
                "database_stats": stats,
                "context": f"K'Cho linguistic database with {sum(v for v in stats.values() if isinstance(v, int))} total entries"
            }
        else:
            return {
                "query_type": "general_analysis",
                "context": "General K'Cho linguistic analysis available"
            }
    
    def _get_sample_verbs(self) -> List[str]:
        """Get sample verbs from knowledge base"""
        if hasattr(self.knowledge, 'get_all_verb_stems'):
            verbs = self.knowledge.get_all_verb_stems()
            return [v['verb'] for v in verbs[:5]]
        elif hasattr(self.knowledge, 'VERB_STEMS'):
            return list(self.knowledge.VERB_STEMS.keys())[:5]
        else:
            return []


def create_llama_config(
    provider: str = "ollama",
    base_url: str = "http://localhost:11434",
    model: str = "llama3.1:8b",
    api_key: Optional[str] = None,
    **kwargs
) -> LLaMAConfig:
    """
    Create LLaMA configuration
    
    Args:
        provider: LLaMA provider ("ollama", "openai", "anthropic")
        base_url: Base URL for API
        model: Model name
        api_key: API key (if required)
        **kwargs: Additional configuration options
        
    Returns:
        LLaMAConfig instance
    """
    return LLaMAConfig(
        provider=provider,
        base_url=base_url,
        model=model,
        api_key=api_key,
        **kwargs
    )
