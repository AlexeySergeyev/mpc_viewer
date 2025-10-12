# DuckDB Database Migration

## Overview

The MPC Viewer application has been migrated from CSV/JSON file storage to DuckDB databases for improved performance, data management, and metadata tracking.

## What Changed?

### Database Structure

The application now uses **4 separate DuckDB databases**:

1. **mpc.duckdb** - MPC observation data
2. **miriade.duckdb** - Miriade ephemeris data  
3. **ztf.duckdb** - ZTF observation data
4. **metadata.duckdb** - Asteroid registry and download tracking

Each asteroid gets its own table within the respective database, named by its safe designation (e.g., `mpc_2000_LE29`, `ztf_A807_FA`).

### Benefits

- ✅ Better data organization and query performance
- ✅ Automatic metadata tracking of all downloads
- ✅ Centralized data management
- ✅ Efficient storage with compression
- ✅ Easier to add new features and analytics

## Installation

Update your dependencies:

```bash
pip install -r requirements.txt
```

The new dependency added is:
- `duckdb` - For database management

## Migration from CSV Files

If you have existing CSV files in `db/mpc/`, `db/miriade/`, or `db/ztf/` directories, you can migrate them to DuckDB:

```bash
python migrate_csv_to_duckdb.py
```

This script will:
1. Find all CSV files in the old directories
2. Read each file and import it into the appropriate DuckDB database
3. Skip files that are already in the database
4. Log progress and any errors

**Note:** The migration script does NOT populate the metadata database with asteroid information. This will be populated automatically when asteroids are queried through the web application.

## Starting the Application

No changes to how you run the application:

```bash
python app.py
```

The databases will be automatically initialized on first run.

## Database Files

All database files are stored in `./db/`:

```
db/
├── mpc.duckdb          # MPC observation data
├── miriade.duckdb      # Miriade ephemeris data
├── ztf.duckdb          # ZTF observation data
├── metadata.duckdb     # Asteroid registry & download tracking
└── designation/        # IAU designation JSON files (still used)
```

## Querying the Databases

You can query the databases directly using Python:

```python
import duckdb
import pandas as pd

# Connect to a database
conn = duckdb.connect('./db/metadata.duckdb')

# Query asteroids
asteroids = conn.execute("SELECT * FROM asteroids").df()
print(asteroids)

# Query download history
history = conn.execute("""
    SELECT a.iau_designation, d.data_source, d.record_count, 
           d.download_timestamp, d.status
    FROM asteroids a
    JOIN data_downloads d ON a.asteroid_number = d.asteroid_number
    ORDER BY d.download_timestamp DESC
    LIMIT 10
""").df()
print(history)

conn.close()
```

### Useful Database Queries

**List all asteroids:**
```sql
SELECT asteroid_number, iau_designation, iau_name, last_updated 
FROM asteroids 
ORDER BY last_updated DESC;
```

**Check download status:**
```sql
SELECT 
    a.iau_designation,
    d.data_source,
    d.record_count,
    d.status,
    d.download_timestamp
FROM asteroids a
JOIN data_downloads d ON a.asteroid_number = d.asteroid_number
WHERE d.status = 'error';
```

**Get list of tables (asteroids) in MPC database:**
```sql
SELECT table_name 
FROM information_schema.tables 
WHERE table_name LIKE 'mpc_%';
```

## Using the Database Utilities

The `db_utils.py` module provides convenient functions:

```python
import db_utils

# Initialize all databases
db_utils.init_all_databases()

# Check if data exists
if db_utils.mpc_data_exists('2000_LE29'):
    # Load data
    df = db_utils.load_mpc_data('2000_LE29')
    print(f"Loaded {len(df)} observations")

# Get asteroid information
info = db_utils.get_asteroid_info('2000 LE29')
print(info)

# List all asteroids
asteroids = db_utils.list_all_asteroids()
print(asteroids)

# Get download history
history = db_utils.get_download_history('2000 LE29')
print(history)
```

## Backup and Maintenance

### Backing Up Databases

Simply copy the `.duckdb` files:

```bash
# Create backup directory
mkdir -p backups/$(date +%Y%m%d)

# Copy all databases
cp db/*.duckdb backups/$(date +%Y%m%d)/
```

### Database Size

Check database file sizes:

```bash
ls -lh db/*.duckdb
```

### Rebuilding Databases

To start fresh, simply delete the database files:

```bash
rm db/*.duckdb
```

They will be recreated automatically when the application starts.

## Troubleshooting

### Data Type Issues

DuckDB may store numeric columns differently than pandas expects. The application automatically converts columns to the correct types when loading data:
- `mag` columns are converted to numeric with `pd.to_numeric()`
- `obstime` columns are converted to datetime
- Distance and phase columns (`Dhelio`, `Dobs`, `Phase`) are converted to numeric

If you encounter type-related errors, the application will handle them gracefully.

### Database Locked Error

If you get a "database is locked" error:
- Make sure only one instance of the application is running
- Close any DuckDB connections you may have opened in Python

### Missing Data

If data seems to be missing:
- Check the `data_downloads` table in `metadata.duckdb` for download status
- Look at application logs in `./logs/` for errors
- Verify the safe designation matches the table name (check with `information_schema.tables`)

### Migration Issues

If migration fails:
- Check CSV file formats match expected columns
- Verify file permissions
- Look at migration logs for specific errors

## Performance

DuckDB provides excellent performance for analytical queries:
- Fast scans of observation data
- Efficient joins between MPC and Miriade data
- Built-in compression reduces storage size
- Support for parallel query execution

## Future Enhancements

With DuckDB in place, potential new features include:
- Web interface to browse all asteroids
- Bulk analysis across multiple asteroids
- Advanced filtering and search
- Data quality validation
- Time-series analysis tools

## Export Feature

The application includes a data export feature that allows you to download asteroid observation data in CSV format:

### Using the Web Interface

1. Load asteroid data using the "Load MPC" button
2. Once data is loaded, the "Export Data" dropdown button becomes active
3. Click the dropdown to select export options:
   - **All Data (ZIP)**: Downloads all available data (MPC, Miriade, ZTF) in a single ZIP file
   - **MPC Data (CSV)**: Downloads only MPC observation data
   - **Miriade Data (CSV)**: Downloads only Miriade ephemeris data
   - **ZTF Data (CSV)**: Downloads only ZTF observation data

### API Endpoint

You can also export data programmatically using the `/export_data` endpoint:

```bash
# Export all data as ZIP
curl -X POST http://localhost:5000/export_data \
  -H "Content-Type: application/json" \
  -d '{"asteroid_id": "2000 LE29", "data_source": "all"}' \
  --output asteroid_data.zip

# Export MPC data only
curl -X POST http://localhost:5000/export_data \
  -H "Content-Type: application/json" \
  -d '{"asteroid_id": "2000 LE29", "data_source": "mpc"}' \
  --output mpc_data.csv
```

**Supported data sources:** `all`, `mpc`, `miriade`, `ztf`

### Export File Format

- **CSV files**: Standard comma-separated values with headers
- **ZIP files**: Contains multiple CSV files (one for each data source available)
- **Filenames**: Include the safe designation for easy identification

Example filenames:
- `2000_LE29_mpc.csv`
- `2000_LE29_miriade.csv`
- `2000_LE29_ztf.csv`
- `2000_LE29_all_data.zip` (contains all of the above)

## Documentation

For more details, see:
- `docs/duckdb_migration.md` - Complete migration documentation
- `db_utils.py` - Database utility functions with docstrings
- DuckDB documentation: https://duckdb.org/docs/

## Support

For issues or questions:
1. Check the logs in `./logs/`
2. Review error messages in `data_downloads` table
3. Verify database integrity with DuckDB CLI
4. Check that all dependencies are installed correctly
