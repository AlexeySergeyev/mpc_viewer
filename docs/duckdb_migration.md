# DuckDB Migration Summary

## Overview
The MPC Viewer application has been successfully migrated from CSV/JSON file storage to DuckDB databases. This provides better data management, query performance, and metadata tracking.

## Changes Made

### 1. Database Structure

Four separate DuckDB databases have been created:

#### a. **mpc.duckdb** - MPC Observation Data
- Each asteroid gets its own table named `mpc_{safe_designation}`
- Stores observation data from the Minor Planet Center
- Table structure matches the original CSV columns from ADES_DF format

#### b. **miriade.duckdb** - Miriade Ephemeris Data
- Each asteroid gets its own table named `miriade_{safe_designation}`
- Stores ephemeris calculations from IMCCE Miriade service
- Contains phase angles, distances, and other computed values

#### c. **ztf.duckdb** - ZTF Observation Data
- Each asteroid gets its own table named `ztf_{safe_designation}`
- Stores Zwicky Transient Facility observations from Fink Portal
- Includes photometry and ephemeris data

#### d. **metadata.duckdb** - Metadata and Download Tracking
- **asteroids** table: Tracks all asteroids with their designations and names
  - `asteroid_number`: Input identifier
  - `iau_designation`: Official IAU designation
  - `safe_designation`: Filesystem-safe designation (used for table names)
  - `iau_name`: Optional asteroid name
  - `first_seen`, `last_updated`: Timestamps
  
- **data_downloads** table: Records all data download events
  - Tracks when data was downloaded
  - Records success/error status
  - Stores error messages for debugging
  - Links to the asteroids table

### 2. New Files

#### **db_utils.py** - Database Utility Module
A comprehensive module providing:

**Initialization Functions:**
- `init_all_databases()`: Initialize all four databases
- `init_metadata_db()`: Create metadata tables
- `init_mpc_db()`, `init_miriade_db()`, `init_ztf_db()`: Initialize data databases

**Data Management Functions:**
- `save_mpc_data()`, `save_miriade_data()`, `save_ztf_data()`: Save DataFrames to database
- `load_mpc_data()`, `load_miriade_data()`, `load_ztf_data()`: Load data from database
- `mpc_data_exists()`, `miriade_data_exists()`, `ztf_data_exists()`: Check data availability

**Metadata Functions:**
- `register_asteroid()`: Register new asteroids in metadata
- `get_asteroid_info()`: Retrieve asteroid information
- `record_data_download()`: Log download events
- `get_download_history()`: View download history for an asteroid
- `list_all_asteroids()`: List all tracked asteroids

### 3. Modified Files

#### **requirements.txt**
Added: `duckdb`

#### **app.py**
Major refactoring to use DuckDB instead of CSV files:

**Imports:**
- Added `import db_utils`

**Initialization:**
- Calls `db_utils.init_all_databases()` on startup

**Modified Functions:**

1. **`get_id()`**
   - Now checks metadata database first before making API calls
   - Automatically registers asteroids in metadata database
   - Returns tuple: (iau_designation, safe_designation, iau_name)

2. **`fetch_mpc_data()`**
   - Checks database before fetching from API
   - Saves to DuckDB instead of CSV files
   - Records download events in metadata
   - Better error handling with metadata logging

3. **`fetch_ztf_data()`**
   - Uses database for caching
   - Records download status in metadata
   - Handles empty results properly

4. **`fetch_miriade()`**
   - Loads MPC data from database
   - Checks for existing Miriade data in database
   - Saves computed ephemeris to database
   - Tracks all download attempts

5. **`plot_observations()`**
   - Loads data from DuckDB databases
   - Uses `db_utils.mpc_data_exists()` and `db_utils.ztf_data_exists()`
   - No changes to plotting logic

6. **`plot_phase()`**
   - Loads MPC and Miriade data from databases
   - Validates data availability before processing
   - Maintains original plotting functionality

7. **`make_folders()`**
   - Simplified to only create `db/` and `db/designation/` directories
   - No longer creates separate subdirectories for mpc, miriade, ztf

## Benefits

1. **Better Data Management:**
   - Single source of truth per data type
   - Structured schema with proper types
   - No file naming issues

2. **Metadata Tracking:**
   - Complete download history
   - Error tracking for debugging
   - Asteroid registry with multiple identifiers

3. **Performance:**
   - Faster data access through SQL queries
   - Efficient storage (DuckDB compression)
   - Ability to query across asteroids

4. **Maintainability:**
   - Centralized database operations in `db_utils.py`
   - Cleaner code separation
   - Easier to add new features

5. **Data Integrity:**
   - Referential integrity with foreign keys
   - Automatic timestamp tracking
   - Transaction support

## Migration Path

The old CSV files in `db/mpc/`, `db/miriade/`, and `db/ztf/` directories are no longer used. However:

1. The application still saves IAU designation JSON files in `db/designation/`
2. Existing CSV files are not automatically migrated
3. When an asteroid is queried, data will be re-fetched if not in the database

To migrate existing data, you could create a migration script that:
- Reads CSV files from old directories
- Loads them as DataFrames
- Saves to DuckDB using `db_utils.save_*_data()` functions

## Usage

### Starting Fresh
The databases are automatically created when the application starts.

### Querying Metadata
```python
import db_utils

# List all asteroids
asteroids = db_utils.list_all_asteroids()

# Get asteroid info
info = db_utils.get_asteroid_info("2000 LE29")

# Check download history
history = db_utils.get_download_history("2000 LE29")
```

### Direct Database Access
You can also query the databases directly:
```python
import duckdb

# Connect to metadata database
conn = duckdb.connect('./db/metadata.duckdb')

# Query asteroids
result = conn.execute("SELECT * FROM asteroids").df()

# Query download history
downloads = conn.execute("""
    SELECT a.iau_designation, d.data_source, d.record_count, d.status
    FROM asteroids a
    JOIN data_downloads d ON a.asteroid_number = d.asteroid_number
    ORDER BY d.download_timestamp DESC
""").df()

conn.close()
```

## Files Location

All DuckDB database files are stored in `./db/`:
- `./db/mpc.duckdb`
- `./db/miriade.duckdb`
- `./db/ztf.duckdb`
- `./db/metadata.duckdb`

## Testing Recommendations

1. Test asteroid data fetching with a known asteroid
2. Verify data persists across application restarts
3. Check metadata tables are being populated
4. Verify plotting functions work with database data
5. Test error handling when data sources are unavailable

## Future Enhancements

With DuckDB in place, you can now easily add:
- Query interface to search across all asteroids
- Bulk data analysis features
- Export capabilities
- Data validation and quality checks
- Caching strategies with expiration
- Multi-user support with proper locking
