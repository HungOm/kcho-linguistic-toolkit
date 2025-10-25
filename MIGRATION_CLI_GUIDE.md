# K'Cho Data Migration CLI Guide

**Author**: Hung Om  
**Research Foundation**: Based on foundational K'Cho linguistic research and Austroasiatic language studies  
**Version**: 0.2.0  
**Date**: 2024-10-25

## Abstract

This document provides comprehensive documentation for the K'Cho Linguistic Processing Toolkit data migration system. The migration commands follow database best practices with automatic change detection, fresh data loading, and complete audit trails for managing the SQLite knowledge base.

## Migration Commands

### 1. Check Migration Status

```bash
# Check if migration is needed
kcho migrate check

# Check with verbose output
kcho migrate check --verbose

# Force check (show status even if no changes)
kcho migrate check --force
```

**What it does:**
- Calculates checksums for all data files in `kcho/data/`
- Compares with stored version in database
- Shows current vs stored version
- Displays file checksums
- Indicates if migration is needed

**Example Output:**
```
📊 Migration Status Check
========================================
📋 Stored version: ec6e07ecaae3629b
📅 Last updated: 2025-10-25T16:11:07
🔍 Current version: ec6e07ecaae3629b
📅 Calculated: 2025-10-25T16:15:03
✅ Status: No migration needed

📁 Data File Checksums:
  linguistic_data.json: 1b05c7370bf7c0d5...
  gold_standard_collocations.txt: b47e3ee0ddb137a6...
  word_frequency_top_1000.csv: 2967871f70a46afb...
  gold_standard_kcho_english.json: c5656217f9a9861c...
```

### 2. Run Migration

```bash
# Run migration if needed
kcho migrate run

# Run with verbose output
kcho migrate run --verbose
```

**What it does:**
- Automatically detects if data files have changed
- Performs fresh data migration if needed
- Shows migration history
- Displays database statistics

**Example Output:**
```
🔄 Starting data migration...

📋 Recent Migration History:
==================================================
✅ ec6e07ecaae3629b (fresh_load) - completed
   📊 Records: 1340

📊 Database Statistics:
  Verb stems: 453
  Collocations: 53
  Word frequencies: 0
  Parallel sentences: 88
  Raw texts: 0
✅ Migration check completed
```

### 3. View Migration History

```bash
# Show recent migration history
kcho migrate history

# Show more migrations
kcho migrate history --limit 20
```

**What it does:**
- Shows complete audit trail of all migrations
- Displays migration status (completed/failed)
- Shows record counts and error messages
- Provides timestamps for all operations

**Example Output:**
```
📋 Migration History
============================================================
✅ Migration 1
   From: None
   To: ec6e07ecaae3629b
   Type: fresh_load
   Status: completed
   Started: 2025-10-25 08:11:07
   Completed: 2025-10-25 08:11:07
   Records: 1340
```

### 4. Check Data File Status

```bash
# Show status of all tracked data files
kcho migrate status
```

**What it does:**
- Shows all data files being tracked
- Displays file checksums and sizes
- Shows last modification times
- Indicates when files were loaded

**Example Output:**
```
📁 Data File Status
============================================================
📄 linguistic_data.json
   Version: ec6e07ecaae3629b
   Checksum: 1b05c7370bf7c0d5...
   Size: 81471 bytes
   Modified: 2025-10-25T14:02:25
   Loaded: 2025-10-25 08:11:07

📄 gold_standard_collocations.txt
   Version: ec6e07ecaae3629b
   Checksum: b47e3ee0ddb137a6...
   Size: 4071 bytes
   Modified: 2025-10-25T11:46:18
   Loaded: 2025-10-25 08:11:07
```

### 5. Force Fresh Migration

```bash
# Force fresh migration (requires confirmation)
kcho migrate force --confirm

# This will replace ALL data in the database
```

**What it does:**
- Clears all existing data from database
- Loads fresh data from all files
- Records migration in history
- Shows updated statistics

**⚠️ Warning:** This replaces ALL data in the database!

### 6. Initialize Database

```bash
# Initialize a new database with fresh data
kcho migrate init

# Initialize with custom paths
kcho migrate init --db-path /path/to/db.db --data-dir /path/to/data
```

**What it does:**
- Creates new database with fresh data
- Loads all data files
- Shows initial statistics
- Sets up migration tracking

## Command Options

### Global Options

- `--db-path PATH`: Specify custom database file path
- `--data-dir PATH`: Specify custom data directory path
- `--verbose, -v`: Enable verbose output with debug information

### Examples

```bash
# Use custom database location
kcho migrate check --db-path /custom/path/kcho.db

# Use custom data directory
kcho migrate run --data-dir /custom/data/path

# Verbose output for debugging
kcho migrate check --verbose

# Force migration with custom paths
kcho migrate force --confirm --db-path /tmp/test.db --data-dir ./data
```

## Migration Process

### Automatic Migration

The system automatically performs migrations when:

1. **First Run**: No stored version exists
2. **File Changes**: Any data file has been modified
3. **Manual Trigger**: Using `kcho migrate run` or `kcho migrate force`

### Migration Types

- **fresh_load**: Complete replacement of all data
- **incremental**: Future support for partial updates
- **schema_change**: Future support for database schema changes

### Data Files Tracked

The migration system tracks these files in `kcho/data/`:

- `linguistic_data.json` - Verb stems, collocations, word categories
- `gold_standard_collocations.txt` - Gold standard patterns
- `word_frequency_top_1000.csv` - Word frequency data
- `gold_standard_kcho_english.json` - Parallel sentences
- `sample_corpus.txt` - Sample corpus data

## Best Practices

### 1. Regular Checks

```bash
# Check migration status regularly
kcho migrate check
```

### 2. Before Data Changes

```bash
# Check current status before modifying data files
kcho migrate status
```

### 3. After Data Changes

```bash
# Run migration after updating data files
kcho migrate run
```

### 4. Backup Before Force Migration

```bash
# Always backup before force migration
cp kcho/kcho_lexicon.db kcho/kcho_lexicon.db.backup
kcho migrate force --confirm
```

## Troubleshooting

### Common Issues

1. **Migration Failed**
   ```bash
   # Check migration history for errors
   kcho migrate history
   
   # Check data file status
   kcho migrate status
   ```

2. **Database Locked**
   ```bash
   # Ensure no other processes are using the database
   # Check for running Python processes
   ps aux | grep python
   ```

3. **Data File Not Found**
   ```bash
   # Check data directory
   ls -la kcho/data/
   
   # Verify file paths
   kcho migrate check --verbose
   ```

### Debug Mode

```bash
# Enable verbose logging for debugging
kcho migrate check --verbose
kcho migrate run --verbose
```

## Integration with Development

### Pre-commit Hook

```bash
#!/bin/bash
# Check if data files have changed
kcho migrate check
if [ $? -ne 0 ]; then
    echo "Data files have changed, running migration..."
    kcho migrate run
fi
```

### CI/CD Pipeline

```bash
# In your CI pipeline
kcho migrate check
if [ $? -eq 0 ]; then
    echo "No migration needed"
else
    echo "Migration required"
    kcho migrate run
fi
```

## Summary

The K'Cho CLI migration system provides:

- ✅ **Automatic change detection** using SHA-256 checksums
- ✅ **Fresh data loading** following database best practices
- ✅ **Complete audit trail** with migration history
- ✅ **Error handling** with detailed status reporting
- ✅ **Flexible configuration** with custom paths
- ✅ **Safe operations** with confirmation prompts

This ensures that any changes to files in `kcho/data/` are automatically detected and the entire SQLite database is refreshed with the new data, maintaining data integrity and consistency.
