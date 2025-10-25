#!/usr/bin/env python3
"""
Enhanced LLaMA Integration Demo

This script demonstrates the enhanced LLaMA integration features including:
- Token-limited responses (1000 max, 3000 for deep research)
- Cost warnings for external APIs
- Automatic fallback to Ollama or structured data
- Response logging in JSON/CSV formats
- SQLite data as context for LLaMA responses

Author: Hung Om
Research Foundation: Based on foundational K'Cho linguistic research and Austroasiatic language studies
Version: 0.2.0
Date: 2024-10-25
"""

import os
import json
from kcho import KchoSystem, load_env_file


def demo_enhanced_llama_features():
    """Demonstrate enhanced LLaMA integration features"""
    
    print("🤖 Enhanced LLaMA Integration Demo")
    print("=" * 50)
    
    # Load environment configuration
    load_env_file()
    
    # Initialize system
    print("\n1. Initializing K'Cho system...")
    system = KchoSystem()
    print("✅ K'Cho system initialized")
    
    # Check current configuration
    provider = os.getenv("LLAMA_PROVIDER", "ollama")
    print(f"🤖 LLaMA Provider: {provider}")
    
    # Test queries with different modes
    queries = [
        {
            "query": "What are common verb patterns in K'Cho?",
            "description": "Standard query (1000 tokens max)",
            "deep_research": False,
            "log": True
        },
        {
            "query": "Analyze the morphological structure of K'Cho verbs",
            "description": "Deep research query (3000 tokens max)",
            "deep_research": True,
            "log": True
        },
        {
            "query": "How does K'Cho handle postpositional phrases?",
            "description": "Structured data only (no LLaMA)",
            "deep_research": False,
            "log": True,
            "use_llama": False
        }
    ]
    
    print(f"\n2. Testing {len(queries)} query modes...")
    
    for i, test_case in enumerate(queries, 1):
        print(f"\n--- Test {i}: {test_case['description']} ---")
        print(f"Query: {test_case['query']}")
        
        try:
            # Get API response
            response = system.get_api_response(
                query=test_case['query'],
                use_llama=test_case.get('use_llama', True),
                deep_research=test_case['deep_research'],
                log_response=test_case['log'],
                log_format='json'
            )
            
            # Display results
            print(f"✅ Response Type: {response['response_type']}")
            print(f"🤖 LLaMA Enabled: {response['llama_enabled']}")
            print(f"🔬 Deep Research: {response['deep_research']}")
            print(f"🎯 Tokens Used: {response['tokens_used']}")
            
            if response['llama_enabled'] and response.get('llama_response', {}).get('success'):
                print(f"🤖 LLaMA Response Preview:")
                llama_resp = response['llama_response']['response']
                preview = llama_resp[:200] + "..." if len(llama_resp) > 200 else llama_resp
                print(f"   {preview}")
                
                print(f"📊 Structured Data:")
                structured = response['structured_data']
                print(f"   Query Type: {structured.get('query_type', 'unknown')}")
                if 'total_patterns' in structured:
                    print(f"   Total Patterns: {structured['total_patterns']}")
                if 'context' in structured:
                    print(f"   Context: {structured['context']}")
            else:
                print(f"📊 Structured Data:")
                data = response.get('data', response.get('structured_data', {}))
                print(f"   Query Type: {data.get('query_type', 'unknown')}")
                if 'total_patterns' in data:
                    print(f"   Total Patterns: {data['total_patterns']}")
                if 'context' in data:
                    print(f"   Context: {data['context']}")
                
                if 'llama_error' in response:
                    print(f"⚠️  LLaMA Error: {response['llama_error']}")
            
            if test_case['log']:
                print(f"📝 Response logged to file")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Demonstrate cost warning for external APIs
    print(f"\n3. Cost Warning Demonstration:")
    print("-" * 35)
    
    if provider in ["openai", "anthropic"]:
        print(f"⚠️  External API detected: {provider}")
        print(f"💰 Deep research mode (3000 tokens) may incur significant costs")
        print(f"💡 Consider using Ollama for local processing")
    else:
        print(f"✅ Using local Ollama - no external API costs")
    
    # Show SQLite data usage
    print(f"\n4. SQLite Data as Context:")
    print("-" * 30)
    
    try:
        # Get some sample data from the knowledge base
        stats = system.knowledge.get_statistics()
        print(f"📊 Database Statistics:")
        print(f"   Verb Stems: {stats['verb_stems']}")
        print(f"   Collocations: {stats['collocations']}")
        print(f"   Word Frequencies: {stats['word_frequencies']}")
        print(f"   Parallel Sentences: {stats['parallel_sentences']}")
        
        print(f"\n💡 This data is automatically used as context for LLaMA responses")
        print(f"   - Provides linguistic background")
        print(f"   - Enhances response accuracy")
        print(f"   - Reduces hallucination")
        
    except Exception as e:
        print(f"❌ Error getting database stats: {e}")
    
    # Show logging capabilities
    print(f"\n5. Response Logging:")
    print("-" * 20)
    
    print(f"📝 Logging Features:")
    print(f"   - JSON format: Complete structured data")
    print(f"   - CSV format: Tabular data for analysis")
    print(f"   - Automatic filename generation")
    print(f"   - Timestamp and query-based naming")
    
    # Clean up
    system.close()
    
    print(f"\n✅ Enhanced LLaMA integration demo completed!")


def demo_error_handling():
    """Demonstrate error handling and fallback mechanisms"""
    
    print("\n" + "=" * 50)
    print("🛡️  Error Handling and Fallback Demo")
    print("=" * 50)
    
    system = KchoSystem()
    
    # Test scenarios
    scenarios = [
        {
            "name": "No LLaMA API available",
            "query": "What are K'Cho verb patterns?",
            "use_llama": False
        },
        {
            "name": "LLaMA API failure (simulated)",
            "query": "Analyze K'Cho syntax",
            "use_llama": True,
            "context": "This is a test query for error handling"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i}: {scenario['name']} ---")
        
        try:
            response = system.get_api_response(
                query=scenario['query'],
                use_llama=scenario['use_llama'],
                context=scenario.get('context'),
                log_response=True,
                log_format='json'
            )
            
            print(f"✅ Response received")
            print(f"   Type: {response['response_type']}")
            print(f"   LLaMA Enabled: {response['llama_enabled']}")
            print(f"   Tokens Used: {response['tokens_used']}")
            
            if not response['llama_enabled'] and 'llama_error' in response:
                print(f"⚠️  LLaMA Error: {response['llama_error']}")
                print(f"✅ Graceful fallback to structured data")
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    system.close()


def demo_cli_usage():
    """Demonstrate CLI usage examples"""
    
    print("\n" + "=" * 50)
    print("💻 CLI Usage Examples")
    print("=" * 50)
    
    examples = [
        {
            "command": "python -m kcho.cli query -q \"What are K'Cho patterns?\"",
            "description": "Basic query with LLaMA"
        },
        {
            "command": "python -m kcho.cli query -q \"Analyze K'Cho\" --deep-research",
            "description": "Deep research mode (3000 tokens)"
        },
        {
            "command": "python -m kcho.cli query -q \"K'Cho syntax\" --no-llama",
            "description": "Structured data only"
        },
        {
            "command": "python -m kcho.cli query -q \"K'Cho patterns\" --log --log-format csv",
            "description": "Query with CSV logging"
        },
        {
            "command": "python -m kcho.cli env check",
            "description": "Check environment configuration"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}:")
        print(f"   {example['command']}")
    
    print(f"\n💡 All commands support:")
    print(f"   - Automatic .env file loading")
    print(f"   - Cost warnings for external APIs")
    print(f"   - Graceful fallback on errors")
    print(f"   - Response logging")
    print(f"   - Token usage tracking")


if __name__ == "__main__":
    try:
        demo_enhanced_llama_features()
        demo_error_handling()
        demo_cli_usage()
        
        print(f"\n🎉 All demos completed successfully!")
        print(f"🔧 The K'Cho system now supports:")
        print(f"   ✅ Token-limited responses (1000/3000)")
        print(f"   ✅ Cost warnings for external APIs")
        print(f"   ✅ Automatic fallback mechanisms")
        print(f"   ✅ Response logging (JSON/CSV)")
        print(f"   ✅ SQLite data as context")
        print(f"   ✅ Comprehensive error handling")
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
