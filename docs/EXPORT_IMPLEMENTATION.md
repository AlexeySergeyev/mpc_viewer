# Export Feature Implementation Summary

## Overview
Added comprehensive data export functionality to the MPC Viewer application, allowing users to download asteroid observation data in CSV format.

## Changes Made

### 1. Backend (app.py)
Added `/export_data` endpoint that:
- Accepts POST requests with asteroid_id and data_source
- Supports exporting individual sources: `mpc`, `miriade`, `ztf`
- Supports exporting all data at once: `all` (creates ZIP archive)
- Returns CSV files for single sources
- Returns ZIP archive containing all available data sources
- Includes proper error handling and logging
- Uses in-memory processing for efficiency

### 2. Frontend HTML (templates/index.html)
Added export UI elements:
- Export dropdown button in the "Show Observations" section
- Menu items for each export option:
  - All Data (ZIP)
  - MPC Data (CSV)
  - Miriade Data (CSV)
  - ZTF Data (CSV)
- Export button is disabled until data is loaded
- Uses Bootstrap dropdown component for clean UI

### 3. Frontend JavaScript (static/js/app.js)
Added export functionality:
- Event listeners for all export menu items
- `handleExport()` function to manage export requests
- Automatic file download using Blob API
- Progress messages and error handling
- Enable/disable export button based on data availability
- Proper filename extraction from response headers

### 4. Documentation
Created comprehensive documentation:
- **docs/export_feature.md**: Complete export feature documentation
  - Web interface usage guide
  - API endpoint documentation
  - cURL and Python examples
  - File format specifications
  - Error handling guide
  - Troubleshooting section

- **docs/DUCKDB_README.md**: Added export section
  - Quick reference for web and API usage
  - File format examples

## Features

✅ **Multiple Export Options**
- Individual data sources (MPC, Miriade, ZTF)
- All available data in single ZIP file

✅ **User-Friendly Interface**
- Bootstrap dropdown menu
- Clear status messages
- Automatic file downloads

✅ **Robust API**
- RESTful endpoint
- JSON request/response
- Proper HTTP status codes
- Detailed error messages

✅ **Smart File Handling**
- Automatic filename generation
- Safe filename sanitization
- Proper MIME types
- In-memory processing

✅ **Data Validation**
- Checks data availability before export
- Returns 404 if data not found
- Validates data source parameter
- Requires asteroid_id

## Usage Examples

### Web Interface
1. Load asteroid data (e.g., "2000 LE29")
2. Click "Export Data" dropdown
3. Select desired export option
4. File downloads automatically

### API Call
```bash
curl -X POST http://localhost:5000/export_data \
  -H "Content-Type: application/json" \
  -d '{"asteroid_id": "2000 LE29", "data_source": "all"}' \
  --output asteroid_data.zip
```

### Python Script
```python
import requests

response = requests.post(
    'http://localhost:5000/export_data',
    json={'asteroid_id': '2000 LE29', 'data_source': 'mpc'}
)

with open('mpc_data.csv', 'wb') as f:
    f.write(response.content)
```

## File Formats

### Single Source Exports
- Format: CSV with headers
- Filename: `{safe_designation}_{source}.csv`
- Example: `2000_LE29_mpc.csv`

### All Data Export
- Format: ZIP archive
- Filename: `{safe_designation}_all_data.zip`
- Contains: One CSV file per available source
- Example: `2000_LE29_all_data.zip`
  - Contains: `2000_LE29_mpc.csv`, `2000_LE29_miriade.csv`, `2000_LE29_ztf.csv`

## Technical Details

### Backend Implementation
- Uses pandas `to_csv()` for CSV generation
- BytesIO for in-memory file handling
- zipfile module for ZIP archives
- Flask send_file() for file downloads
- Proper content-disposition headers

### Frontend Implementation
- Fetch API for HTTP requests
- Blob API for file handling
- Dynamic download link creation
- Automatic cleanup after download

### Data Flow
1. User clicks export option
2. JavaScript sends POST to /export_data
3. Backend loads data from DuckDB
4. Data converted to CSV (or multiple CSVs in ZIP)
5. File sent as download
6. Browser saves file automatically

## Testing Recommendations

1. ✅ Export with single data source loaded
2. ✅ Export with multiple data sources loaded
3. ✅ Export all data as ZIP
4. ✅ Try exporting before loading data (should fail gracefully)
5. ✅ Test with different asteroid designations
6. ✅ Verify CSV file structure and content
7. ✅ Test API endpoint with curl
8. ✅ Check error messages for missing data

## Error Handling

The implementation includes comprehensive error handling:
- Missing asteroid_id: 400 Bad Request
- Data not found: 404 Not Found
- Invalid data source: 400 Bad Request
- Processing errors: 500 Internal Server Error
- All errors logged for debugging

## Benefits

1. **Data Portability**: Users can work with data offline
2. **Integration**: Easy to use with other analysis tools
3. **Backup**: Create local copies of important data
4. **Sharing**: Simple way to share data with collaborators
5. **Flexibility**: Choose exactly what data to export

## Future Enhancements

Potential improvements:
- Custom column selection
- Date range filtering
- Batch export for multiple asteroids
- Additional export formats (JSON, FITS)
- Direct cloud storage integration
- Export scheduling/automation

## Dependencies

No new dependencies required! Uses existing packages:
- Flask (already installed)
- pandas (already installed)
- zipfile (Python standard library)
- io.BytesIO (Python standard library)

## Compatibility

- Works with all modern browsers
- Compatible with current DuckDB implementation
- No breaking changes to existing functionality
- Backward compatible with existing API

## Files Modified

1. `/app.py` - Added export endpoint
2. `/templates/index.html` - Added export UI
3. `/static/js/app.js` - Added export JavaScript
4. `/docs/DUCKDB_README.md` - Added export documentation
5. `/docs/export_feature.md` - New comprehensive guide

## Notes

- Export button automatically enables/disables based on data availability
- ZIP files only include data sources that are actually available
- All exports use safe filenames (slashes/spaces replaced with underscores)
- Exports are done in-memory for better performance
- No temporary files created on disk
