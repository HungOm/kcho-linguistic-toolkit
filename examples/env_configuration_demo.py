#!/usr/bin/env python3
"""
Environment Configuration Demo

This script demonstrates how to use .env files for configuration
with the K'Cho Linguistic Processing Toolkit.

Author: Hung Om
Research Foundation: Based on foundational K'Cho linguistic research and Austroasiatic language studies
Version: 0.2.0
Date: 2024-10-25
"""

import os
from kcho import KchoSystem, load_env_file


def demo_env_configuration():
    """Demonstrate .env file configuration"""
    
    print("🔧 Environment Configuration Demo")
    print("=" * 50)
    
    # 1. Show current environment variables
    print("\n1. Current Environment Variables:")
    print("-" * 30)
    
    env_vars = [
        "LLAMA_PROVIDER",
        "OLLAMA_BASE_URL", 
        "OLLAMA_MODEL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "LLAMA_TIMEOUT",
        "LLAMA_MAX_TOKENS",
        "LLAMA_TEMPERATURE"
    ]
    
    for var in env_vars:
        value = os.getenv(var, "Not set")
        if "API_KEY" in var and value != "Not set":
            value = "***hidden***"
        print(f"  {var}: {value}")
    
    # 2. Load .env file
    print("\n2. Loading .env file:")
    print("-" * 20)
    
    success = load_env_file()
    if success:
        print("✅ .env file loaded successfully")
    else:
        print("⚠️  No .env file found or failed to load")
        print("💡 Create one with: python -m kcho.cli env create")
    
    # 3. Show updated environment variables
    print("\n3. Environment Variables After Loading .env:")
    print("-" * 45)
    
    for var in env_vars:
        value = os.getenv(var, "Not set")
        if "API_KEY" in var and value != "Not set":
            value = "***hidden***"
        print(f"  {var}: {value}")
    
    # 4. Initialize system with .env configuration
    print("\n4. Initializing K'Cho System:")
    print("-" * 30)
    
    try:
        system = KchoSystem()
        print("✅ K'Cho system initialized")
        
        # The system automatically loads .env configuration
        print("✅ LLaMA API configured from .env file")
        
        # 5. Test API response
        print("\n5. Testing API Response:")
        print("-" * 25)
        
        response = system.get_api_response(
            "What are common verb patterns in K'Cho?",
            use_llama=False  # Use structured data only for demo
        )
        
        print(f"✅ Query processed successfully")
        print(f"📊 Query Type: {response['data']['query_type']}")
        print(f"📈 Total Patterns: {response['data']['total_patterns']}")
        print(f"🏷️  Categories: {len(response['data']['pattern_categories'])}")
        
        # 6. Show configuration summary
        print("\n6. Configuration Summary:")
        print("-" * 25)
        
        provider = os.getenv("LLAMA_PROVIDER", "ollama")
        print(f"🤖 LLaMA Provider: {provider}")
        
        if provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
            print(f"🌐 Ollama URL: {base_url}")
            print(f"🧠 Model: {model}")
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            print(f"🔑 OpenAI API Key: {'✅ Set' if api_key else '❌ Not set'}")
            print(f"🧠 Model: {model}")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            model = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
            print(f"🔑 Anthropic API Key: {'✅ Set' if api_key else '❌ Not set'}")
            print(f"🧠 Model: {model}")
        
        timeout = os.getenv("LLAMA_TIMEOUT", "30")
        max_tokens = os.getenv("LLAMA_MAX_TOKENS", "1000")
        temperature = os.getenv("LLAMA_TEMPERATURE", "0.7")
        
        print(f"⏱️  Timeout: {timeout}s")
        print(f"📝 Max Tokens: {max_tokens}")
        print(f"🌡️  Temperature: {temperature}")
        
        system.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 7. CLI Commands
    print("\n7. Available CLI Commands:")
    print("-" * 30)
    print("📝 Create .env file:    python -m kcho.cli env create")
    print("📂 Load .env file:      python -m kcho.cli env load")
    print("🔍 Check configuration: python -m kcho.cli env check")
    print("📋 Show all commands:   python -m kcho.cli --help")


def demo_env_file_creation():
    """Demonstrate creating and using .env files"""
    
    print("\n" + "=" * 50)
    print("📝 .env File Creation Demo")
    print("=" * 50)
    
    # Show sample .env content
    sample_env = """# K'Cho Linguistic Processing Toolkit - Environment Configuration
# Fill in your actual values below

# LLaMA Provider: ollama, openai, anthropic
LLAMA_PROVIDER=ollama

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# OpenAI Configuration (uncomment and fill)
# OPENAI_API_KEY=sk-your-openai-api-key-here
# OPENAI_MODEL=gpt-3.5-turbo

# Anthropic Configuration (uncomment and fill)
# ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
# ANTHROPIC_MODEL=claude-3-sonnet-20240229

# General Settings
LLAMA_TIMEOUT=30
LLAMA_MAX_TOKENS=1000
LLAMA_TEMPERATURE=0.7"""
    
    print("Sample .env file content:")
    print("-" * 25)
    print(sample_env)
    
    print("\n💡 To create your own .env file:")
    print("   1. Run: python -m kcho.cli env create")
    print("   2. Edit the .env file with your API keys")
    print("   3. Run: python -m kcho.cli env check")
    print("   4. Use the system normally - it will auto-load .env")


if __name__ == "__main__":
    try:
        demo_env_configuration()
        demo_env_file_creation()
        
        print("\n✅ Environment configuration demo completed!")
        print("🔧 The K'Cho system now supports .env files for easy configuration management")
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
