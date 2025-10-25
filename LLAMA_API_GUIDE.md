# LLaMA API Integration Guide

**Author**: Hung Om  
**Research Foundation**: Based on foundational K'Cho linguistic research and Austroasiatic language studies  
**Version**: 0.2.0  
**Date**: 2024-10-25

## Abstract

This guide explains how to configure and use LLaMA API integration with the K'Cho Linguistic Processing Toolkit. The system now supports actual LLaMA API calls for enhanced linguistic analysis and natural language responses.

## Overview

The K'Cho system now includes **real LLaMA API integration** that can:
- Connect to various LLaMA providers (Ollama, OpenAI, Anthropic)
- Generate enhanced responses using LLaMA models
- Provide both structured data and natural language responses
- Fall back to structured-only responses when LLaMA is unavailable

## Configuration Options

### 1. Environment Variables (.env file) - Recommended

**Create a .env file**:
```bash
# Create sample .env file
python -m kcho.cli env create

# Or manually create .env file
```

**Sample .env file**:
```bash
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
LLAMA_TEMPERATURE=0.7
```

**Check configuration**:
```bash
# Check current environment settings
python -m kcho.cli env check
```

### 2. Ollama (Local LLaMA)

**Setup**:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a LLaMA model
ollama pull llama3.1:8b
```

**Configuration**:
```python
from kcho import KchoSystem, create_llama_config

# Initialize system
system = KchoSystem()

# Configure Ollama
system.configure_llama_api(
    provider="ollama",
    base_url="http://localhost:11434",
    model="llama3.1:8b",
    temperature=0.7,
    max_tokens=1000
)
```

**Environment Variables**:
```bash
export LLAMA_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.1:8b
export LLAMA_TEMPERATURE=0.7
export LLAMA_MAX_TOKENS=1000
```

### 3. OpenAI API

**Configuration**:
```python
system.configure_llama_api(
    provider="openai",
    model="gpt-3.5-turbo",
    api_key="your-openai-api-key"
)
```

**Environment Variables**:
```bash
export LLAMA_PROVIDER=openai
export OPENAI_API_KEY=your-openai-api-key
export OPENAI_MODEL=gpt-3.5-turbo
```

### 4. Anthropic API

**Configuration**:
```python
system.configure_llama_api(
    provider="anthropic",
    model="claude-3-sonnet-20240229",
    api_key="your-anthropic-api-key"
)
```

**Environment Variables**:
```bash
export LLAMA_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your-anthropic-api-key
```

## Usage Examples

### Basic Usage

```python
from kcho import KchoSystem

# Initialize system
system = KchoSystem()

# Configure LLaMA API (optional)
system.configure_llama_api(provider="ollama", model="llama3.1:8b")

# Get API response with LLaMA
response = system.get_api_response("What are common verb patterns?")

print("Response Type:", response['response_type'])
print("LLaMA Enabled:", response['llama_enabled'])

if response['llama_enabled']:
    print("LLaMA Response:", response['llama_response']['response'])
    print("Structured Data:", response['structured_data'])
else:
    print("Structured Data:", response['data'])
```

### Advanced Usage

```python
# Get response without LLaMA (structured only)
response = system.get_api_response(
    "What are common verb patterns?", 
    use_llama=False
)

# Get response with custom context
response = system.get_api_response(
    "Analyze this pattern: pàapai pe ci",
    context="This is a verb-particle construction in K'Cho"
)
```

### Error Handling

```python
try:
    response = system.get_api_response("What are common verb patterns?")
    
    if response['llama_enabled']:
        if response['llama_response']['success']:
            print("LLaMA Response:", response['llama_response']['response'])
        else:
            print("LLaMA Error:", response['llama_response']['error'])
            print("Falling back to structured data:", response['structured_data'])
    else:
        print("LLaMA not configured, using structured data:", response['data'])
        
except Exception as e:
    print(f"Error: {e}")
```

## Response Format

### With LLaMA Enabled

```json
{
    "query": "What are common verb patterns?",
    "timestamp": "2024-10-25T16:30:00",
    "response_type": "llama_enhanced",
    "structured_data": {
        "query_type": "pattern_analysis",
        "total_patterns": 53,
        "pattern_categories": ["VP", "PP", "AGR"],
        "sample_patterns": {
            "VP": ["pàapai pe ci", "k'chàang noh"],
            "PP": ["am pàapai", "noh Yóng"]
        },
        "context": "Found 53 linguistic patterns across 3 categories"
    },
    "llama_response": {
        "success": true,
        "response": "Based on the K'Cho linguistic data, common verb patterns include...",
        "timestamp": "2024-10-25T16:30:00",
        "provider": "ollama",
        "model": "llama3.1:8b"
    },
    "llama_enabled": true
}
```

### Without LLaMA (Structured Only)

```json
{
    "query": "What are common verb patterns?",
    "timestamp": "2024-10-25T16:30:00",
    "response_type": "structured_only",
    "data": {
        "query_type": "pattern_analysis",
        "total_patterns": 53,
        "pattern_categories": ["VP", "PP", "AGR"],
        "sample_patterns": {
            "VP": ["pàapai pe ci", "k'chàang noh"],
            "PP": ["am pàapai", "noh Yóng"]
        },
        "context": "Found 53 linguistic patterns across 3 categories"
    },
    "llama_enabled": false
}
```

## CLI Integration

The CLI now supports LLaMA API configuration:

```bash
# Set environment variables
export LLAMA_PROVIDER=ollama
export OLLAMA_MODEL=llama3.1:8b

# Use CLI with LLaMA integration
python -m kcho.cli research analyze --corpus corpus.txt --output analysis.json
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**:
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Start Ollama if not running
   ollama serve
   ```

2. **Model Not Found**:
   ```bash
   # List available models
   ollama list
   
   # Pull the model
   ollama pull llama3.1:8b
   ```

3. **API Key Issues**:
   ```bash
   # Check environment variables
   echo $OPENAI_API_KEY
   echo $ANTHROPIC_API_KEY
   ```

4. **Timeout Errors**:
   ```python
   # Increase timeout
   system.configure_llama_api(
       provider="ollama",
       timeout=60,  # Increase from default 30
       max_tokens=2000
   )
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed LLaMA API logs
response = system.get_api_response("What are common verb patterns?")
```

## Performance Considerations

- **Local Ollama**: Fastest for development, requires local GPU/CPU resources
- **OpenAI API**: Good balance of performance and ease of use
- **Anthropic API**: High quality responses, may be slower
- **Fallback**: System gracefully falls back to structured data if LLaMA fails

## Security Notes

- **API Keys**: Store securely in environment variables
- **Local Models**: Ollama runs locally, no data sent externally
- **Rate Limits**: Be aware of API rate limits for external providers
- **Data Privacy**: Consider data privacy implications for external API calls

## Examples

See `examples/enhanced_kcho_system_demo.py` for complete working examples with LLaMA integration.
