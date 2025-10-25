# CHANGELOG.md

All notable changes to the K'Cho Linguistic Processing Toolkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **SQLite Knowledge Base**: Complete refactor to use SQLite backend for efficient storage and querying
- **LLaMA API Integration**: Full integration with Ollama, OpenAI, and Anthropic APIs
- **Environment Configuration**: Support for `.env` files for easy configuration management
- **Enhanced CLI**: Comprehensive command-line interface with new query commands
- **Response Logging**: JSON and CSV logging capabilities for all responses
- **Token Management**: Configurable token limits (1000 standard, 3000 deep research)
- **Cost Warnings**: Automatic warnings for expensive external API operations
- **Data Migration**: CLI tools for database migration and management
- **Professional Documentation**: Academic citation format and comprehensive guides
- **Error Handling**: Robust fallback mechanisms and graceful error recovery

### Changed
- **Version**: Updated from 0.1.0 to 0.2.0 (development stage)
- **Architecture**: Migrated from in-memory dictionaries to SQLite database
- **API Layer**: Separated into dedicated module for better modularity
- **Class Naming**: Consistent naming convention (removed "Enhanced", "Comprehensive" prefixes)
- **Documentation**: Consolidated multiple markdown files into comprehensive guides

### Deprecated
- **Legacy Knowledge System**: `LegacyKchoKnowledge` class marked for deprecation
- **Old CLI**: `kcho_app.py` replaced with `cli.py`

### Removed
- **Duplicate Files**: Cleaned up redundant documentation and temporary files
- **Old Classes**: Removed duplicate and deprecated class implementations

### Fixed
- **Indentation Errors**: Fixed multiple syntax and indentation issues
- **Import Errors**: Resolved module import and dependency issues
- **API Integration**: Fixed LLaMA API parameter passing and error handling

### Security
- **API Key Management**: Secure storage of API keys in `.env` files
- **Environment Variables**: Proper handling of sensitive configuration data

## [0.1.0] - 2025-01-24

### Added
- Initial release of K'Cho Linguistic Processing Toolkit
- Basic linguistic analysis capabilities
- Pattern discovery and collocation extraction
- Evaluation metrics and gold standard comparison
- Basic CLI interface

---

## Feature Branches Overview

### 1. `feature/sqlite-backend` 
- SQLite knowledge base implementation
- Database schema and migration tools
- Performance optimizations with indexing

### 2. `feature/llama-integration`
- LLaMA API integration (Ollama, OpenAI, Anthropic)
- Token management and cost warnings
- Enhanced response generation

### 3. `feature/env-configuration`
- `.env` file support
- Environment variable management
- CLI configuration tools

### 4. `feature/cli-enhancement`
- Enhanced CLI with new commands
- Response logging capabilities
- Query interface improvements

### 5. `feature/documentation`
- Professional documentation updates
- Academic citation format
- Comprehensive guides and examples

### 6. `feature/error-handling`
- Robust error handling and fallback mechanisms
- Graceful degradation
- Comprehensive logging

---

## Breaking Changes

### Version 0.2.0
- **Database Backend**: Migration from in-memory to SQLite requires data migration
- **Class Names**: Some class names changed for consistency
- **CLI Commands**: New command structure with additional options
- **Configuration**: New environment variable requirements

### Migration Guide
1. **Database Migration**: Run `python -m kcho.cli migrate fresh` to migrate data
2. **Environment Setup**: Create `.env` file using `python -m kcho.cli env create`
3. **API Configuration**: Configure LLaMA API using `python -m kcho.cli env check`
4. **Code Updates**: Update imports if using deprecated classes directly

---

## Contributors

- **Hung Om** - Primary developer and maintainer
- **Research Foundation** - Based on foundational K'Cho linguistic research and Austroasiatic language studies

---

## Acknowledgments

- Bedell & Mang (2012) - Foundational K'Cho linguistic research
- Austroasiatic linguistics community
- Python open source ecosystem
- SQLite development team
- LLaMA model developers
