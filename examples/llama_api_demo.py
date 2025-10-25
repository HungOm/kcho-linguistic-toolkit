#!/usr/bin/env python3
"""
LLaMA API Integration Demo

This script demonstrates how to use the K'Cho Linguistic Processing Toolkit
with actual LLaMA API integration for enhanced linguistic analysis.

Author: Hung Om
Research Foundation: Based on foundational K'Cho linguistic research and Austroasiatic language studies
Version: 0.2.0
Date: 2024-10-25
"""

import os
import json
from kcho import KchoSystem, create_llama_config


def demo_llama_integration():
    """Demonstrate LLaMA API integration with K'Cho linguistic analysis"""
    
    print("🤖 K'Cho LLaMA API Integration Demo")
    print("=" * 50)
    
    # Initialize the K'Cho system
    print("\n1. Initializing K'Cho system...")
    system = KchoSystem()
    print("✅ K'Cho system initialized")
    
    # Configure LLaMA API (optional - will fall back to structured data if not available)
    print("\n2. Configuring LLaMA API...")
    
    # Try to configure Ollama (local LLaMA)
    try:
        system.configure_llama_api(
            provider="ollama",
            base_url="http://localhost:11434",
            model="llama3.1:8b",
            temperature=0.7,
            max_tokens=1000
        )
        print("✅ LLaMA API configured (Ollama)")
        llama_available = True
    except Exception as e:
        print(f"⚠️  LLaMA API not available: {e}")
        print("   Continuing with structured data only...")
        llama_available = False
    
    # Test queries
    queries = [
        "What are common verb patterns in K'Cho?",
        "How does K'Cho handle postpositional phrases?",
        "What morphological features are important in K'Cho?",
        "Analyze the pattern: pàapai pe ci"
    ]
    
    print(f"\n3. Testing {len(queries)} linguistic queries...")
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        
        try:
            # Get API response
            response = system.get_api_response(
                query, 
                use_llama=llama_available,
                context="This is a computational linguistic analysis of K'Cho language patterns."
            )
            
            print(f"Response Type: {response['response_type']}")
            print(f"LLaMA Enabled: {response['llama_enabled']}")
            
            if response['llama_enabled'] and response.get('llama_response', {}).get('success'):
                print("\n🤖 LLaMA Response:")
                print(response['llama_response']['response'])
                
                print("\n📊 Structured Data:")
                structured = response['structured_data']
                print(f"  Query Type: {structured.get('query_type', 'unknown')}")
                if 'total_patterns' in structured:
                    print(f"  Total Patterns: {structured['total_patterns']}")
                if 'pattern_categories' in structured:
                    print(f"  Categories: {structured['pattern_categories']}")
                if 'context' in structured:
                    print(f"  Context: {structured['context']}")
            else:
                print("\n📊 Structured Data Only:")
                data = response.get('data', response.get('structured_data', {}))
                print(f"  Query Type: {data.get('query_type', 'unknown')}")
                if 'total_patterns' in data:
                    print(f"  Total Patterns: {data['total_patterns']}")
                if 'pattern_categories' in data:
                    print(f"  Categories: {data['pattern_categories']}")
                if 'context' in data:
                    print(f"  Context: {data['context']}")
                if 'llama_error' in response:
                    print(f"  LLaMA Error: {response['llama_error']}")
                    
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    # Demonstrate configuration options
    print(f"\n4. Configuration Options:")
    print("   - Provider: ollama (local), openai, anthropic")
    print("   - Models: llama3.1:8b, gpt-3.5-turbo, claude-3-sonnet")
    print("   - Parameters: temperature, max_tokens, timeout")
    
    # Show environment variable setup
    print(f"\n5. Environment Variables:")
    print("   export LLAMA_PROVIDER=ollama")
    print("   export OLLAMA_BASE_URL=http://localhost:11434")
    print("   export OLLAMA_MODEL=llama3.1:8b")
    print("   export LLAMA_TEMPERATURE=0.7")
    print("   export LLAMA_MAX_TOKENS=1000")
    
    print(f"\n✅ Demo completed successfully!")
    
    # Clean up
    system.close()


def demo_without_llama():
    """Demonstrate system without LLaMA API (structured data only)"""
    
    print("\n" + "=" * 50)
    print("📊 Structured Data Only Demo")
    print("=" * 50)
    
    system = KchoSystem()
    
    query = "What are common verb patterns in K'Cho?"
    response = system.get_api_response(query, use_llama=False)
    
    print(f"Query: {query}")
    print(f"Response Type: {response['response_type']}")
    print(f"LLaMA Enabled: {response['llama_enabled']}")
    print(f"Data: {json.dumps(response['data'], indent=2)}")
    
    system.close()


if __name__ == "__main__":
    try:
        demo_llama_integration()
        demo_without_llama()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
