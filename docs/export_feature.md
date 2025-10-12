# Data Export Feature

## Overview

The MPC Viewer application now supports exporting asteroid observation data to CSV files. You can export data from any of the three sources (MPC, Miriade, ZTF) individually or all at once as a ZIP archive.

## Features

- ✅ Export individual data sources (MPC, Miriade, or ZTF)
- ✅ Export all available data at once (ZIP file)
- ✅ Web interface with dropdown menu
- ✅ REST API endpoint for programmatic access
- ✅ Automatic filename generation based on asteroid designation
- ✅ Proper CSV formatting with headers

## Using the Web Interface

### Step-by-Step Guide

1. **Load Asteroid Data**
   - Enter an asteroid name or number in the input field
   - Click "Load MPC" to fetch observation data
   - Wait for the data to load successfully

2. **Access Export Menu**
   - Once data is loaded, the "Export Data" dropdown button becomes active
   - Click the dropdown to see available export options

3. **Choose Export Type**
   - **All Data (ZIP)**: Downloads all available data sources in one ZIP file
   - **MPC Data (CSV)**: Downloads Minor Planet Center observations
   - **Miriade Data (CSV)**: Downloads IMCCE Miriade ephemeris calculations
   - **ZTF Data (CSV)**: Downloads Zwicky Transient Facility photometry

4. **Download**
   - Click on your choice
   - The file will automatically download to your default downloads folder
   - A success message will appear when the export is complete

### Export Menu Location

The export dropdown is located in the "Show Observations" section, below the plot buttons.

## Using the API

### Endpoint

```
POST /export_data
```

### Request Format

**Headers:**
```
Content-Type: application/json
```

**Body:**
```json
{
  "asteroid_id": "2000 LE29",
  "data_source": "all"
}
```

**Parameters:**
- `asteroid_id` (required): The asteroid designation (e.g., "2000 LE29", "Ceres", "1")
- `data_source` (required): One of `"all"`, `"mpc"`, `"miriade"`, or `"ztf"`

### Response Format

**Success:**
- Returns file data as attachment
- Content-Type: `text/csv` for CSV files, `application/zip` for ZIP files
- Content-Disposition header includes filename

**Error:**
```json
{
  "status": "error",
  "message": "Error description"
}
```

### cURL Examples

**Export all data:**
```bash
curl -X POST http://localhost:5000/export_data \
  -H "Content-Type: application/json" \
  -d '{"asteroid_id": "2000 LE29", "data_source": "all"}' \
  --output 2000_LE29_all_data.zip
```

**Export MPC data only:**
```bash
curl -X POST http://localhost:5000/export_data \
  -H "Content-Type: application/json" \
  -d '{"asteroid_id": "2000 LE29", "data_source": "mpc"}' \
  --output 2000_LE29_mpc.csv
```

**Export Miriade data:**
```bash
curl -X POST http://localhost:5000/export_data \
  -H "Content-Type: application/json" \
  -d '{"asteroid_id": "2000 LE29", "data_source": "miriade"}' \
  --output 2000_LE29_miriade.csv
```

**Export ZTF data:**
```bash
curl -X POST http://localhost:5000/export_data \
  -H "Content-Type: application/json" \
  -d '{"asteroid_id": "2000 LE29", "data_source": "ztf"}' \
  --output 2000_LE29_ztf.csv
```

### Python Example

```python
import requests

# Export all data
response = requests.post(
    'http://localhost:5000/export_data',
    json={
        'asteroid_id': '2000 LE29',
        'data_source': 'all'
    }
)

if response.status_code == 200:
    with open('asteroid_data.zip', 'wb') as f:
        f.write(response.content)
    print("Export successful!")
else:
    print(f"Error: {response.json()['message']}")
```

## File Formats

### CSV Files

All CSV exports include:
- **Header row** with column names
- **UTF-8 encoding**
- **Standard comma delimiters**
- **No index column**

### MPC Data Columns

The MPC CSV export includes all columns from the ADES_DF format:
- `obstime`: Observation timestamp (ISO format)
- `ra`, `dec`: Right ascension and declination
- `mag`: Magnitude measurement
- `band`: Photometric band
- `stn`: Observatory code
- And other ADES_DF fields

### Miriade Data Columns

The Miriade CSV export includes ephemeris calculations:
- `Dhelio`: Heliocentric distance (AU)
- `Dobs`: Observer distance (AU)
- `Phase`: Phase angle (degrees)
- `Elong`: Elongation
- And other Miriade ephemeris fields

### ZTF Data Columns

The ZTF CSV export includes photometry and ephemeris:
- `Date`: Julian Date
- `i:magpsf`: PSF magnitude
- `i:fid`: Filter ID (1=g, 2=r)
- `Dhelio`, `Dobs`, `Phase`: From ephemeris
- And other ZTF/Fink fields

### ZIP Archive Structure

When exporting "All Data", the ZIP file contains:
```
asteroid_designation_all_data.zip
├── asteroid_designation_mpc.csv
├── asteroid_designation_miriade.csv
└── asteroid_designation_ztf.csv
```

Only files with available data are included. If an asteroid has no ZTF observations, the ZIP will only contain MPC and Miriade files.

## Filename Convention

All exported files follow a consistent naming pattern:

**Single source exports:**
- Format: `{safe_designation}_{source}.csv`
- Example: `2000_LE29_mpc.csv`

**All data export:**
- Format: `{safe_designation}_all_data.zip`
- Example: `2000_LE29_all_data.zip`

**Safe designation:**
- Slashes replaced with underscores: `2000/LE29` → `2000_LE29`
- Spaces replaced with underscores: `A807 FA` → `A807_FA`

## Error Handling

### Common Errors

**400 Bad Request:**
```json
{
  "status": "error",
  "message": "Asteroid ID is required"
}
```
- Cause: Missing `asteroid_id` in request
- Solution: Include asteroid_id in request body

**404 Not Found:**
```json
{
  "status": "error",
  "message": "No MPC data found for 2000 LE29"
}
```
- Cause: Requested data source not available for asteroid
- Solution: Load the data first or choose a different source

**500 Internal Server Error:**
```json
{
  "status": "error",
  "message": "Error exporting data: [details]"
}
```
- Cause: Server-side processing error
- Solution: Check server logs for details

## Data Availability

Before exporting, ensure the data has been loaded:

1. **MPC data**: Must be loaded via "Load MPC" button
2. **Miriade data**: Requires MPC data first, then "Load Miriade"
3. **ZTF data**: Can be loaded independently via "Load ZTF"

You can export individual sources even if other sources aren't available. For example, you can export MPC data without having Miriade or ZTF data loaded.

## Use Cases

### Scientific Analysis
Export data for analysis in your preferred tools (Python, R, Excel, etc.)

### Data Backup
Download local copies of asteroid observations for archival purposes

### Sharing Data
Export specific datasets to share with collaborators

### Offline Work
Download data to work with when internet connection is unavailable

### Custom Processing
Export raw data for custom analysis pipelines

## Performance Notes

- Small asteroids (few hundred observations): Export is nearly instant
- Large asteroids (thousands of observations): May take a few seconds
- ZIP archives: Slightly slower due to compression
- All exports are done in-memory for efficiency

## Limitations

- Maximum file size limited by available RAM
- Very large asteroids (10,000+ observations) may take longer to export
- ZIP archives require all requested data sources to be loaded first
- Export operation blocks until complete (no streaming)

## Security

- Only data that has been previously loaded can be exported
- No direct database access through export endpoint
- Filenames are sanitized to prevent path traversal
- All exports go through the same validation as data loading

## Troubleshooting

### Export button is disabled
- **Solution**: Load asteroid data first using "Load MPC" button

### "No data found" error
- **Solution**: Make sure you've loaded the specific data source you're trying to export

### Download not starting
- **Solution**: Check browser's download settings and popup blocker

### Corrupted ZIP file
- **Solution**: Make sure all data sources were loaded before exporting "All Data"

### Empty CSV file
- **Solution**: Data might not have been saved properly; try reloading the data

## Future Enhancements

Potential improvements to the export feature:
- [ ] Custom column selection
- [ ] Date range filtering
- [ ] Export to other formats (JSON, FITS, etc.)
- [ ] Batch export for multiple asteroids
- [ ] Scheduled/automated exports
- [ ] Export presets for common use cases
- [ ] Compression level options for ZIP files
- [ ] Direct integration with analysis tools

## Feedback

If you encounter issues with the export feature or have suggestions for improvements, please file an issue on the project repository.
