from flask import Flask, render_template, request, jsonify, send_from_directory # type: ignore
import requests
import json
import pandas as pd
import os
import plotly # type: ignore
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
import numpy as np
from astropy.time import Time
from astroquery.mpc import MPC
import logging
from logging.handlers import RotatingFileHandler
import time
import urllib.parse
import db_utils


app = Flask(__name__)

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_name = os.path.join(log_dir, f'{time.strftime("%Y-%m-%d")}.log')
log_file = os.path.join(log_dir, log_name)

# Set up rotating file handler to avoid log files getting too large
handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
handler.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)s] %(module)s:%(lineno)d - %(message)s'
))
logger = logging.getLogger('mpc_viewer')
logger.propagate = False  # Prevent log messages from being propagated to the root logger
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

# Add handler to Flask's logger as well
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)
app.logger.info('MPC Viewer application starting up')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Create db directory if it doesn't exist
os.makedirs('./db/', exist_ok=True)
logger.info('Ensuring database directories exist')

# Initialize all DuckDB databases
db_utils.init_all_databases()
logger.info('All DuckDB databases initialized')

# Constants for API URLs
MPC_API_IDENTIFIER_URL = "https://data.minorplanetcenter.net/api/query-identifier"
MPC_API_OBSERVATIONS_URL = "https://data.minorplanetcenter.net/api/get-obs"
# IMCCE_MIRIADE_URL = "https://ssp.imcce.fr/webservices/miriade/api/ephemcc.php"
IMCCE_MIRIADE_URL = "http://vo.imcce.fr/webservices/miriade/ephemcc_query.php"

init_miriade_params = {
        '-rplane': 1,
        '-tcoor': 5,
        "-oscelem": "astorb",
        "-mime": "json",
        "-output": "--jd",
        }

def make_folders():
    """
    Create necessary folders for storing data.
    This is now mainly for backward compatibility and temporary files.
    """
    folders = [
        './db/',
        './db/designation/'
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        logger.info(f"Ensured folder exists: {folder}")


def load_obsevatory_codes():
    """
    Downloads the list of observatory codes from the Minor Planet Center (MPC) 
    
    Returns:
        dict: A dictionary mapping observatory codes to their names
    """
    logger.info("Loading observatory codes from MPC")
    obs = MPC.get_observatory_codes() # type: ignore
    d = dict(zip(obs['Code'].tolist(), 
             obs['Name'].tolist()))
    logger.debug(f"Loaded {len(d)} observatory codes")
    return d


def get_band_color_map():
    """
    Get a color mapping for photometric bands that matches real filter colors.
    
    Returns:
        dict: Mapping of band names to hex color codes
    """
    return {
        # Standard photometric bands
        'U': '#5E3C99',    # Ultraviolet - purple
        'B': '#56B4E9',    # Blue
        'V': '#32CD32',    # Visual/Green
        'R': '#CC0000',    # Red
        'I': '#D95F02',    # Infrared - dark red
        
        # Lowercase variants
        'u': '#5E3C99',
        'b': '#56B4E9',
        'v': '#32CD32',
        'g': '#1B9E77',
        'r': '#CC0000',
        'i': '#D95F02',
        'z': '#333333',

        # SDSS bands
        # 'g': '#00CED1',    # g' - cyan/turquoise
        # 'r': '#D55E00',    # r' - red-orange
        # 'i': '#8B0000',    # i' - dark red
        # 'z': '#4B0082',    # z' - indigo
        
        # SDSS with prime notation
        # "g'": '#00CED1',
        # "r'": '#FF4500',
        # "i'": '#8B0000',
        # "z'": '#4B0082',
        
        # Other common bands
        'C': '#808080',    # Clear/unfiltered - gray
        'o': '#FFA500',    # Orange
        'w': '#D3D3D3',    # White/unfiltered - light gray
        'G': '#32CD32',    # Gaia G band - green
        
        # Special cases
        'Empty': '#777777',  # Dark gray for missing band data
        'unknown': '#BBBBBB'  # Gray for unknown bands
    }


def create_band_legend_name(band, stn=None, obs_name=None):
    """
    Create a legend name combining band and observatory information.
    Station (stn) is shown first, then band.
    
    Args:
        band: Filter/band name
        stn: Observatory code
        obs_name: Observatory name
        
    Returns:
        str: Formatted legend name with station first
    """
    if band and band != 'unknown':
        if stn:
            return f"{stn}: {band}-band"
        else:
            return f"{band}-band"
    else:
        if obs_name and obs_name != stn:
            return f"{stn} - {obs_name}"
        else:
            return stn if stn else "Unknown"


def get_marker_shapes():
    """
    Get a list of Plotly marker symbols for different observatory stations.
    
    Returns:
        list: List of marker symbol names that work well together
    """
    return [
        'circle',           # 0
        'square',           # 1
        'diamond',          # 2
        'cross',            # 3
        'x',                # 4
        'triangle-up',      # 5
        'triangle-down',    # 6
        'triangle-left',    # 7
        'triangle-right',   # 8
        'pentagon',         # 9
        'hexagon',          # 10
        'star',             # 11
        'hexagram',         # 12
        # 'circle-open',      # 13
        # 'square-open',      # 14
        # 'diamond-open',     # 15
    ]


def assign_marker_shapes(stations):
    """
    Assign marker shapes to observatory stations.
    
    Args:
        stations: List or Series of observatory station codes
        
    Returns:
        dict: Mapping of station codes to marker symbols
    """
    unique_stations = sorted(list(set(stations)))
    marker_shapes = get_marker_shapes()
    
    # Create mapping, cycling through marker shapes if needed
    station_marker_map = {}
    for i, station in enumerate(unique_stations):
        station_marker_map[station] = marker_shapes[i % len(marker_shapes)]
    
    return station_marker_map


def get_id(asteroid_number: str) -> tuple[str, str, str] | None:
    """
    Generate a unique ID for the asteroid based on its number.
    
    Args
        asteroid_number (str): The asteroid number to generate an ID for
        
    Returns:
        tuple | None: (iau_designation, safe_designation, iau_name) or None if not found

    """
    logger.info(f"Getting IAU designation for asteroid {asteroid_number}")
    
    # Check if we already have this asteroid in the database
    asteroid_info = db_utils.get_asteroid_info(asteroid_number)
    if asteroid_info:
        logger.info(f"Found asteroid {asteroid_number} in database")
        return (asteroid_info['iau_designation'], 
                asteroid_info['safe_designation'], 
                asteroid_info['iau_name'])
    
    # Use the asteroid number as the ID
    response = requests.get(MPC_API_IDENTIFIER_URL, data=str(asteroid_number))
    
    iau_designation = None
    iau_name = None
    safe_designation = None

    if response.ok:
        mpc_identification = response.json()
        iau_designation = mpc_identification.get('unpacked_primary_provisional_designation', None)
        iau_name = mpc_identification.get('name', None)
        logger.debug(f"Response data: {json.dumps(mpc_identification, indent=4)}")

        safe_designation = iau_designation.replace('/', '_').replace(' ', '_')
        filename = f"./db/designation/{safe_designation}.json"
        if not os.path.exists(filename):
            logger.debug(f"Saving IAU designation to {filename}")
            # Save the response to a file
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                json.dump(response.json(), f, indent=4)
            logger.info(f"Asteroid {asteroid_number} has IAU designation {iau_designation}")
        else:
            logger.info(f"IAU designation for {asteroid_number} already exists in {filename}")
        
        # Register asteroid in metadata database
        db_utils.register_asteroid(asteroid_number, iau_designation, safe_designation, iau_name)
    else:
        logger.error(f"Error getting IAU designation: {response.status_code} {response.content}")
        return None
    
    return iau_designation, safe_designation, iau_name


def fetch_mpc_data(asteroid_name: str):
    """
    Fetch observation data for a specific asteroid from the Minor Planet Center API.
    Checks online source for number of observations and updates local database if different.
    
    Args:
        asteroid_name (str): The asteroid number to fetch data for
        
    Returns:
        dict: Parsed JSON response from the MPC API
        iau_designation (str): The asteroid ID
    """
    
    # Check asteroid id
    iau_designation = None
    safe_designation = None
    
    id = get_id(asteroid_name)
    if id is not None:
        iau_designation, safe_designation, _ = id

    # MPC sometimes requires specific formatting (e.g., '00001' for Ceres)
    if iau_designation is None or safe_designation is None:
        logger.error(f"Asteroid {asteroid_name} not found in MPC database")
        # If the asteroid number is not found, return None or raise an error
        raise Exception(f"Asteroid {asteroid_name} not found in MPC database.")
    
    # Check if data already exists in database
    needs_update = False
    if db_utils.mpc_data_exists(safe_designation):
        # Load existing data from cache
        mpc_df_local = db_utils.load_mpc_data(safe_designation)
        
        if mpc_df_local is not None:
            local_count = len(mpc_df_local)
            
            # MPC API has a default limit of 2000 observations and doesn't support high limits reliably
            # For now, skip automatic count checking to avoid API errors
            # Users can manually refresh if they suspect new data is available
            logger.info(f"Data for {iau_designation} loaded from database (count={local_count}). Skipping online check.")
            return mpc_df_local.to_json(), iau_designation
    
    logger.info(f"Fetching MPC observations for {iau_designation}")
    # Note: MPC API may have observation limits. For asteroids with >2000 observations,
    # the API might not return all data in a single request.
    response = requests.get(MPC_API_OBSERVATIONS_URL, 
                            json={"desigs": [iau_designation], 
                                  "output_format":["ADES_DF"]})
    if response.ok:
        mpc_data = response.json()[0]['ADES_DF']
        mpc_df = pd.DataFrame(mpc_data)
        action = "Updating" if needs_update else "Saving"
        logger.info(f"{action} {mpc_df.shape[0]} observations for {iau_designation} to database")
        
        # Save to DuckDB database
        db_utils.save_mpc_data(safe_designation, mpc_df)
        
        # Record download in metadata
        db_utils.record_data_download(asteroid_name, 'mpc', len(mpc_df), 'success')
        
        # Always return JSON to keep the response consistent with cached data
        return mpc_df.to_json(), iau_designation
    else:
        logger.error(f"Error fetching MPC data: {response.status_code} {response.content}")
        db_utils.record_data_download(asteroid_name, 'mpc', 0, 'error', 
                                      f"HTTP {response.status_code}")
        return None, iau_designation


def fetch_miriade_data(iau_name: str, jd=None):
    """
    Fetches data from the IMCCE Miriade web service.
    
    Args:
        iau_designation (str | tuple[str, str]): The IAU designation of the asteroid or tuple of (name, iau_designation)
        jd (list, optional): List of Julian Dates to fetch data for.
        
    Returns:
        dict: Parsed JSON response from the Miriade web service.
    """
    # Handle case where iau_designation is a tuple (name, iau_designation)
    
    # "-name": urllib.parse.quote(iau_designation),
    if str(iau_name).startswith('C') or str(iau_name).startswith('P'):
        params = {
            "-name": f"c:{iau_name.replace(' ', '_')}",
            **init_miriade_params,
        }
    else:
        params = {
            "-name": iau_name,
            **init_miriade_params,
        }
    # Reduced chunk size for faster processing and to avoid worker timeouts
    CHUNK_SIZE = 2000  # Reduced from 4000 to process faster
    epoch_len = len(jd) if jd is not None else 0
    logger.debug(f"Miriade parameters: {json.dumps(params, indent=4)}")
    try:
        logger.info(f"Sending request to Miriade for {iau_name} with {len(jd) if jd is not None else 0} epochs")
        miriade_data = None
        total_chunks = (epoch_len // CHUNK_SIZE) + (1 if epoch_len % CHUNK_SIZE > 0 else 0)
        
        for chunk_idx, chunk_jd in enumerate(range(0, epoch_len, CHUNK_SIZE)):
            chunk_num = chunk_idx + 1
            logger.info(f"Processing chunk {chunk_num}/{total_chunks} (epochs {chunk_jd} to {min(chunk_jd + CHUNK_SIZE, epoch_len)})")
            chunk_epochs = {
                'epochs': (
                    'epochs', 
                    '\n'.join(['%.6f' % epoch for epoch in jd[chunk_jd:chunk_jd + CHUNK_SIZE]])
                )
            } if jd is not None else {}
            logger.debug(f"Chunk epochs: {chunk_epochs['epochs'][1][:20]}... (total {len(chunk_epochs['epochs'])} epochs)")
            
            # Increased timeout to handle slow IMCCE API responses
            response = requests.post(IMCCE_MIRIADE_URL, params=params, 
            files=chunk_epochs, timeout=180)  # Increased timeout from 120 to 180 seconds
            logger.debug(f"Request URL: {response.url}")
            response.raise_for_status()  # Raise an error for bad responses
            logger.debug(f"Response status code: {response.status_code}")
            
            response_data = response.json()
            if miriade_data is None:
                # First chunk - initialize miriade_data
                if 'data' not in response_data:
                    logger.error(f"No data found for {iau_name} in Miriade response")
                    return None
                miriade_data = response_data
            else:
                # Subsequent chunks - merge the new data with the existing data
                if 'data' in response_data:
                    miriade_data['data'].extend(response_data['data'])
            logger.info(f"Received Miriade data with {len(miriade_data.get('data', [])) if miriade_data else 0} records")
        return miriade_data
    
    except requests.RequestException as e:
        logger.error(f"Request error fetching Miriade data: {e}")
        return None
    except ValueError as e:
        logger.error(f"JSON parsing error with Miriade response: {e}")
        return None


def fetch_ztf_data(asteroid_name: str):
    """
    Fetches ZTF data for a specific asteroid.
    Checks online source for number of observations and updates local database if different.
    
    Args:
        asteroid_name (str): The name of the asteroid to fetch data for.
        
    Returns:
        dict: Parsed JSON response from the ZTF API.
        iau_designation (str): The asteroid ID
    """
    # Check asteroid id
    id = get_id(asteroid_name)
    iau_designation = None
    safe_designation = None

    if id is not None:
        iau_designation, safe_designation, _ = id
    
    # MPC sometimes requires specific formatting (e.g., '00001' for Ceres)
    if iau_designation is None or safe_designation is None:
        logger.error(f"Asteroid {asteroid_name} not found in ZTF database")
        # If the asteroid number is not found, return None or raise an error
        raise Exception(f"Asteroid {asteroid_name} not found in ZTF database.")
    
    # Check if data already exists in database
    needs_update = False
    if db_utils.ztf_data_exists(safe_designation):
        # Load existing data to check count
        ztf_df_local = db_utils.load_ztf_data(safe_designation)
        
        if ztf_df_local is not None:
            local_count = len(ztf_df_local)
            
            # Check online source for observation count
            logger.info(f"Checking online ZTF for observation count of {iau_designation}")
            try:
                query = {
                    'n_or_d': iau_designation,
                    'withEphem': True,
                    'withResiduals': True,
                    'output-format': 'json'
                }
                response_check = requests.post(
                    'https://api.fink-portal.org/api/v1/sso',
                    json=query,
                    timeout=10
                )
                if response_check.ok:
                    online_data = response_check.json()
                    online_count = len(online_data) if online_data else 0
                    
                    if online_count != local_count:
                        logger.info(f"Observation count mismatch for {iau_designation}: local={local_count}, online={online_count}. Updating database.")
                        needs_update = True
                    else:
                        logger.info(f"ZTF data for {iau_designation} is up to date (count={local_count}). Loading from database.")
                        return ztf_df_local.to_json(), iau_designation
                else:
                    logger.warning(f"Could not check online ZTF count (HTTP {response_check.status_code}). Using cached data.")
                    return ztf_df_local.to_json(), iau_designation
            except Exception as e:
                logger.warning(f"Error checking online ZTF count: {str(e)}. Using cached data.")
                return ztf_df_local.to_json(), iau_designation
    
    logger.info(f"Fetching ZTF data for {iau_designation}")
    query = {
        'n_or_d': iau_designation,
        'withEphem': True,
        'withResiduals': True,
        'output-format': 'json'
    }
    response = requests.post(
        'https://api.fink-portal.org/api/v1/sso',
        json=query
    )
    logger.debug(f"Request URL: {response.url}")
    if response.ok:
        ztf_data = response.json()
        ztf_df = pd.DataFrame(ztf_data)
        if ztf_df.empty:
            logger.warning(f"No ZTF data found for {iau_designation}")
            db_utils.record_data_download(asteroid_name, 'ztf', 0, 'success', 
                                         "No data available")
            return None
        action = "Updating" if needs_update else "Saving"
        logger.info(f"{action} {ztf_df.shape[0]} ZTF observations for {iau_designation} to database")
        
        # Save to DuckDB database
        db_utils.save_ztf_data(safe_designation, ztf_df)
        
        # Record download in metadata
        db_utils.record_data_download(asteroid_name, 'ztf', len(ztf_df), 'success')
        
        return ztf_df.to_json(), iau_designation
    else:
        logger.error(f"Error fetching ZTF data: {response.status_code} {response.content}")
        db_utils.record_data_download(asteroid_name, 'ztf', 0, 'error', 
                                      f"HTTP {response.status_code}")
        return jsonify({
            "status": "error",
            "message": f"Error fetching data for {iau_designation}: {response.status_code} {response.content}"
        })


@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the main index page."""
    logger.info("Rendering index page")
    # Ensure necessary folders exist
    make_folders()
    
    return render_template('index.html')


@app.route('/fetch_mpc', methods=['POST'])
def fetch_asteroid():
    data = request.get_json()
    asteroid_name = data.get('userInput', '')
    logger.info(f"Fetching MPC data for asteroid {asteroid_name}")
    try:
        data, iau_designation = fetch_mpc_data(asteroid_name)
        logger.info(f"Successfully retrieved MPC data for {asteroid_name} (ID: {iau_designation})")
        return jsonify({
            "status": "success", 
            "message": f"Data for asteroid {asteroid_name} (ID: {iau_designation}) loaded successfully from MPC",
            "data": data,
            "id": iau_designation
        })
    except Exception as e:
        logger.error(f"Error fetching MPC data for {asteroid_name}: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/fetch_ztf', methods=['POST'])
def fetch_ztf():
    data = request.get_json()
    asteroid_name = data.get('userInput', '')
    logger.info(f"Fetching ZTF data for asteroid {asteroid_name}")
    try:
        data = fetch_ztf_data(asteroid_name)
        if data is None:
            logger.warning(f"No ZTF data found for asteroid {asteroid_name}")
            return jsonify({
                "status": "error", 
                "message": f"No ZTF data found for asteroid {asteroid_name}"
            })
        elif isinstance(data, tuple) and len(data) == 2:
            data_obs, iau_designation = data
        else:
            # Handle the case when fetch_ztf_data returns a Response object
            logger.warning(f"Unexpected return type from fetch_ztf_data: {type(data)}")
            return data  # Return the response object directly
        logger.info(f"Successfully retrieved ZTF data for {asteroid_name} (ID: {iau_designation})")
        return jsonify({
            "status": "success", 
            "message": f"Data for asteroid {asteroid_name} (ID: {iau_designation}) retrieved successfully from ZTF",
            "data": data_obs,
            "id": iau_designation
        })
    except Exception as e:
        logger.error(f"Error fetching ZTF data for {asteroid_name}: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/fetch_miriade', methods=['POST'])
def fetch_miriade():
    """
    Fetches Miriade data for a specific asteroid.
    """
    data = request.get_json()
    input_name = data.get('userInput', '')
    logger.info(f"Fetching Miriade data for asteroid {input_name}")
    print("Input name:", input_name)
    # if "C/" in input_name:
    #     input_name = f"c:/{input_name}"

    # Check asteroid id
    id = get_id(input_name)
    print("ID:", id)
    iau_designation = None
    safe_designation = None
    iau_name = None
    if id is not None:
        iau_designation, safe_designation, iau_name = id
    
    # Handle case where get_id returns None
    if iau_designation is None or safe_designation is None:
        logger.error(f"Asteroid {input_name} not found in MPC database")
        return jsonify({
            "status": "error", 
            "message": f"Asteroid {input_name} not found in MPC database."
        })

    logger.debug(f"Safe designation for {iau_designation}: {safe_designation}")
    
    # Check if MPC data exists in database
    if not db_utils.mpc_data_exists(safe_designation):
        logger.warning(f"No MPC data found for asteroid {iau_designation}")
        return jsonify({
            "status": "error", 
            "message": f"No MPC data found for asteroid {iau_designation}"
        })
    
    # Load MPC data from database
    df_mpc = db_utils.load_mpc_data(safe_designation)
    if df_mpc is None:
        logger.error(f"Failed to load MPC data for {iau_designation}")
        return jsonify({
            "status": "error", 
            "message": f"Failed to load MPC data for asteroid {iau_designation}"
        })
    logger.info(f"MPC data for {iau_designation} loaded successfully from database, shape: {df_mpc.shape}")
    
    # Check if Miriade data already exists in database
    needs_update = False
    if db_utils.miriade_data_exists(safe_designation):
        miriade_df = db_utils.load_miriade_data(safe_designation)
        if miriade_df is not None:
            local_count = len(miriade_df)
            mpc_count = len(df_mpc)
            
            # Check if Miriade data count matches MPC data count
            if local_count != mpc_count:
                logger.info(f"Miriade observation count mismatch for {iau_designation}: miriade={local_count}, mpc={mpc_count}. Updating database.")
                needs_update = True
            else:
                logger.info(f"Miriade data for {iau_designation} is up to date (count={local_count}). Loading from database.")
                return jsonify({
                    "status": "success", 
                    "message": f"Data for asteroid {input_name} (ID: {iau_designation}) loaded successfully from Miriade",
                    "data": miriade_df.to_json(),
                    "id": iau_designation
                })
    
    # Get the epochs from the DataFrame
    epochs = df_mpc.loc[:, 'obstime'].to_list()
    logger.debug(f"Epochs for Miriade request: {epochs[:5]}... (total {len(epochs)} epochs)")
    # Convert to Julian Date
    try:
        
        nchunks = len(epochs) // 500 + 1
        logger.debug(f"Total epochs: {len(epochs)}, splitting into {nchunks} chunks of 500 epochs each")
        epochs_jd = np.array([])  # Initialize as empty array to avoid unbound error
        # Convert epochs to Julian Date using astropy Time
        for i in range(nchunks):
            start = i * 500
            end = min((i + 1) * 500, len(epochs))
            logger.debug(f"Processing chunk {i + 1}/{nchunks}: epochs {start} to {end}")
            epochs_time = Time(epochs[start:end], format='isot', scale='utc')
            if i == 0:
                epochs_jd = np.array(epochs_time.jd)
            else:
                epochs_jd = np.concatenate((epochs_jd, np.array(epochs_time.jd)))
        # Limit to the first 500 epochs for Miriade request
        # epochs = Time(epochs[:500], format='isot', scale='utc')
        logger.debug(f"Converted epochs to Julian Date: {epochs_jd[:5]}... (total {len(epochs)} epochs)")
    except Exception as e:
        logger.error(f"Error converting epochs to Julian Date: {str(e)}")
        return jsonify({"status": "error", "message": f"Error converting epochs to Julian Date: {str(e)}"})

    # Convert to Julian Date
    # epochs_jd = np.array(epochs.jd)
    # logger.debug(f"Epochs for Miriade request: {epochs_jd[:5]}... (total {len(epochs_jd)} epochs)")
    
    # # Send the request to Miriade
    logger.info(f"Fetching Miriade data for asteroid {input_name} with {len(epochs_jd) } epochs")
    try:
        if iau_name is not None:
            miriade_data = fetch_miriade_data(iau_name, epochs_jd)
        else:
            miriade_data = fetch_miriade_data(iau_designation, epochs_jd)
        
        if miriade_data and "data" in miriade_data:
            miriade_df = pd.DataFrame(miriade_data["data"])
            # logger.info(f"Columns in Miriade data: {miriade_df.columns.tolist()}")
            # Rename columns to set Upper CamelCase
            if 'dobs' in miriade_df.columns:
                miriade_df = miriade_df.rename(columns={'dobs': 'Dobs', 'dhelio': 'Dhelio', 
                                                        'vmag': 'VMag', 'epoch': 'Date',
                                                        'phase': 'Phase'})
            # miriade_df.columns = [col[0].upper() + col[1:] for col in miriade_df.columns]
             # Ensure that 'Jd' column is present
            action = "Updating" if needs_update else "Saving"
            logger.info(f"{action} {miriade_df.shape[0]} Miriade records for {iau_designation} to database")
            
            # Save to DuckDB database
            db_utils.save_miriade_data(safe_designation, miriade_df)
            
            # Record download in metadata
            db_utils.record_data_download(input_name, 'miriade', len(miriade_df), 'success')
            
            return jsonify({
                "status": "success", 
                "message": f"Data for asteroid {input_name} (ID: {iau_designation}) fetched successfully from Miriade",
                "data": miriade_df.to_json(),  # Convert DataFrame to JSON, not raw miriade_data
                "id": iau_designation
            })
        else:
            logger.error(f"Failed to get valid Miriade data for {input_name}")
            db_utils.record_data_download(input_name, 'miriade', 0, 'error', 
                                         "No valid data returned")
            return jsonify({"status": "error", "message": "Failed to get valid Miriade data"})
    except Exception as e:
        logger.error(f"Error fetching Miriade data for {input_name}: {str(e)}")
        db_utils.record_data_download(input_name, 'miriade', 0, 'error', str(e))
        return jsonify({"status": "error", "message": str(e)})


@app.route('/plot_observations', methods=['POST'])
def plot_observations():
    asteroid_id = request.form.get('asteroid_id')
    logger.info(f"Generating observations plot for asteroid {asteroid_id}")
    
    # Replace slashes with underscores in the filename to avoid directory issues
    safe_designation = asteroid_id.replace('/', '_').replace(' ', '_') # type: ignore

    # Load ZTF data from database
    df_ztf = None
    if db_utils.ztf_data_exists(safe_designation):
        df_ztf = db_utils.load_ztf_data(safe_designation)
        if df_ztf is not None:
            logger.info(f"ZTF data for {asteroid_id} loaded successfully, shape: {df_ztf.shape}")
            # Convert numeric columns to proper types
            df_ztf['Date'] = pd.to_numeric(df_ztf['Date'], errors='coerce')
            df_ztf['i:fid'] = pd.to_numeric(df_ztf['i:fid'], errors='coerce')
            df_ztf['i:magpsf'] = pd.to_numeric(df_ztf['i:magpsf'], errors='coerce')
            df_ztf['obstime'] = pd.to_datetime(df_ztf['Date'], origin='julian', unit='D')
            show_ztf = True
        else:
            show_ztf = False
    else:
        show_ztf = False
        logger.info(f"ZTF data for {asteroid_id} not found, skipping ZTF plot")
    
    # Load MPC data from database
    df_mpc = None
    if db_utils.mpc_data_exists(safe_designation):
        df_mpc = db_utils.load_mpc_data(safe_designation)
        if df_mpc is not None:
            logger.info(f"MPC data for {asteroid_id} loaded successfully, shape: {df_mpc.shape}")
    logger.info(f"Fetching observations for asteroid {asteroid_id}")

    if df_mpc is not None:
        # Check if mag and obstime/obsTime columns exist
        if 'mag' not in df_mpc.columns or not 'obstime' in df_mpc.columns:
            logger.error(f"The observation data for {asteroid_id} does not contain magnitude or time data")
            return jsonify({
                "status": "error",
                "message": "The observation data does not contain magnitude or time data"
            })
        else:
            # Convert magnitude to numeric, handling any string values
            df_mpc['mag'] = pd.to_numeric(df_mpc['mag'], errors='coerce')
            
            # Convert observation time to datetime if needed
            if df_mpc['obstime'].dtype != 'datetime64[ns]':
                df_mpc['obstime'] = pd.to_datetime(df_mpc['obstime'], errors='coerce')
            
            # Drop rows with invalid dates or magnitudes
            df_mpc = df_mpc.dropna(subset=['obstime', 'mag'])
            
            logger.debug(f"Magnitude values range: {df_mpc['mag'].min()} to {df_mpc['mag'].max()}")
            
            obs_codes = load_obsevatory_codes()
            band_colors = get_band_color_map()
            station_markers = assign_marker_shapes(df_mpc['stn'])
            
            # Check if band column exists
            has_band = 'band' in df_mpc.columns
            
            # Create figure with go.Figure for better control over markers
            import plotly.graph_objects as go
            fig = go.Figure()
            
            if has_band:
                # Fill missing band values with 'Empty' to keep all data points
                df_plot = df_mpc.copy()
                df_plot['band'] = df_plot['band'].fillna('Empty')
                
                # Group by station first, then band for proper legend ordering
                grouped = df_plot.groupby(['stn', 'band'])
                
                for (stn, band), group in grouped:
                    band_str = str(band).strip()
                    color = band_colors.get(band_str, band_colors['unknown'])
                    marker_symbol = station_markers.get(stn, 'circle')
                    obs_name = obs_codes.get(stn, stn)
                    legend_name = create_band_legend_name(band_str, stn, obs_name)
                    
                    fig.add_trace(go.Scatter(
                        x=group['obstime'],
                        y=group['mag'],
                        mode='markers',
                        name=legend_name,
                        marker=dict(
                            size=8,
                            color=color,
                            symbol=marker_symbol,
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        customdata=[[stn, obs_name, band_str]] * len(group),
                        hovertemplate='Observatory: %{customdata[0]}<br>%{customdata[1]}<br>Band: %{customdata[2]}<br>Time: %{x}<br>Magnitude: %{y:.2f}<extra></extra>'
                    ))
            else:
                # Fallback to station-based grouping only
                grouped = df_mpc.groupby('stn')
                colors = px.colors.qualitative.Set1
                
                for i, (stn, group) in enumerate(grouped):
                    color = colors[i % len(colors)]
                    marker_symbol = station_markers.get(stn, 'circle')
                    obs_name = obs_codes.get(stn, stn)
                    legend_name = f"{stn} - {obs_name}" if obs_name != stn else stn
                    
                    fig.add_trace(go.Scatter(
                        x=group['obstime'],
                        y=group['mag'],
                        mode='markers',
                        name=legend_name,
                        marker=dict(
                            size=8,
                            color=color,
                            symbol=marker_symbol,
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        customdata=[[stn, obs_name]] * len(group),
                        hovertemplate='Observatory: %{customdata[0]}<br>%{customdata[1]}<br>Time: %{x}<br>Magnitude: %{y:.2f}<extra></extra>'
                    ))
            
            if show_ztf and df_ztf is not None:
                logger.debug(f"Adding ZTF data to the plot for {asteroid_id}")
                # Define colors for ZTF filter IDs
                ztf_g = df_ztf[df_ztf['i:fid'] == 1]  # g-band
                ztf_r = df_ztf[df_ztf['i:fid'] == 2]  # r-band
                
                # Use actual filter colors and star marker for ZTF
                if len(ztf_r) > 0:
                    fig.add_trace(go.Scatter(
                        x=ztf_r['obstime'],
                        y=ztf_r['i:magpsf'],
                        mode='markers',
                        name='I41 ZTF: r-band',
                        marker=dict(
                            size=8,
                            color=band_colors['r'],
                            symbol='star',
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        customdata=[['I41', 'Palomar Mountain ZTF', 'r']] * len(ztf_r),
                        hovertemplate='Observatory: I41<br>Palomar Mountain ZTF<br>Band: r<br>Time: %{x}<br>Magnitude: %{y:.2f}<extra></extra>'
                    ))
                
                if len(ztf_g) > 0:
                    fig.add_trace(go.Scatter(
                        x=ztf_g['obstime'],
                        y=ztf_g['i:magpsf'],
                        mode='markers',
                        name='I41 ZTF: g-band',
                        marker=dict(
                            size=8,
                            color=band_colors['g'],
                            symbol='star',
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        customdata=[['I41', 'Palomar Mountain ZTF', 'g']] * len(ztf_g),
                        hovertemplate='Observatory: I41<br>Palomar Mountain ZTF<br>Band: g<br>Time: %{x}<br>Magnitude: %{y:.2f}<extra></extra>'
                    ))
            
            # Invert y-axis and update layout
            fig.update_layout(
                title=f'Magnitude observations for {asteroid_id}',
                xaxis_title='Observation Time',
                yaxis_title='Magnitude',
                yaxis=dict(autorange="reversed"),
                height=600,
                legend_title_text='Observatory: Filter' if has_band else 'Observatory',
                template='plotly_white',
                hovermode='closest'
            )
            
            # Convert to JSON for sending to the client
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            logger.info(f"Successfully generated observations plot for {asteroid_id} with {len(df_mpc) if df_ztf is None else len(df_mpc) + len(df_ztf)} data points")
            return jsonify({
                "status": "success",
                "plot": graphJSON,
                "count": len(df_mpc) if df_ztf is None else len(df_mpc) + len(df_ztf)
            })
    else:
        logger.warning(f"No MPC data found for asteroid {asteroid_id}")
        return jsonify({
            "status": "error",
            "message": f"No data found for asteroid {asteroid_id}"
        })


@app.route('/plot_phase', methods=['POST'])
def plot_phase():
    asteroid_id = request.form.get('asteroid_id')

    # Replace slashes with underscores in the filename to avoid directory issues
    safe_designation = asteroid_id.replace('/', '_').replace(' ', '_') # type: ignore
    
    logger.info(f"Generating phase plot for asteroid {asteroid_id}")

    # Check if both MPC and Miriade data exist in database
    mpc_exists = db_utils.mpc_data_exists(safe_designation)
    miriade_exists = db_utils.miriade_data_exists(safe_designation)
    
    if not mpc_exists or not miriade_exists:
        missing = []
        if not mpc_exists:
            missing.append("MPC")
        if not miriade_exists:
            missing.append("Miriade")
        
        logger.warning(f"Missing data for phase plot of {asteroid_id}: {', '.join(missing)} data not found")
        return jsonify({
            "status": "error",
            "message": f"Missing data for asteroid {asteroid_id}: {', '.join(missing)} data not found"
        })
    
    try:
        # Read data from databases
        mpc_df = db_utils.load_mpc_data(safe_designation)
        miriade_df = db_utils.load_miriade_data(safe_designation)
        
        if mpc_df is None or miriade_df is None:
            logger.error(f"Failed to load data for phase plot of {asteroid_id}")
            return jsonify({
                "status": "error",
                "message": f"Failed to load data for asteroid {asteroid_id}"
            })
        
        logger.info(f"Phase plot data loaded - MPC shape: {mpc_df.shape}, Miriade shape: {miriade_df.shape}")
        
        # Convert numeric columns to proper types
        mpc_df['mag'] = pd.to_numeric(mpc_df['mag'], errors='coerce')
        miriade_df['Dhelio'] = pd.to_numeric(miriade_df['Dhelio'], errors='coerce')
        miriade_df['Dobs'] = pd.to_numeric(miriade_df['Dobs'], errors='coerce')
        miriade_df['Phase'] = pd.to_numeric(miriade_df['Phase'], errors='coerce')
        
        df_merged = pd.concat([mpc_df, miriade_df], axis=1)
        df_merged['mag_dist_corr'] = 5 * np.log10(df_merged['Dhelio'] * df_merged['Dobs'])
        
        ztf_df = None
        if db_utils.ztf_data_exists(safe_designation):
            ztf_df = db_utils.load_ztf_data(safe_designation)
            if ztf_df is not None:
                logger.info(f"Including ZTF data in phase plot, shape: {ztf_df.shape}")
                ztf_df['obstime'] = pd.to_datetime(ztf_df['Date'], origin='julian', unit='D')
                # Convert numeric columns
                ztf_df['Dhelio'] = pd.to_numeric(ztf_df['Dhelio'], errors='coerce')
                ztf_df['Dobs'] = pd.to_numeric(ztf_df['Dobs'], errors='coerce')
                ztf_df['Phase'] = pd.to_numeric(ztf_df['Phase'], errors='coerce')
                ztf_df['i:magpsf'] = pd.to_numeric(ztf_df['i:magpsf'], errors='coerce')
                ztf_df['mag_dist_corr'] = 5 * np.log10(ztf_df['Dhelio'] * ztf_df['Dobs'])
        
        if len(df_merged) == 0:
            logger.warning(f"No matching phase data found for asteroid {asteroid_id}")
            return jsonify({
                "status": "error",
                "message": f"No matching phase data found for asteroid {asteroid_id}"
            })
    except Exception as e:
        logger.error(f"Error processing phase plot data for {asteroid_id}: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error processing phase plot data: {str(e)}"
        })
    
    obs_codes = load_obsevatory_codes()
    band_colors = get_band_color_map()
    
    # Check if band column exists
    has_band = 'band' in df_merged.columns
    
    # Assign marker shapes based on station
    station_marker_map = assign_marker_shapes(df_merged['stn'].unique())
    df_merged['marker_shape'] = df_merged['stn'].map(station_marker_map)
    
    # Create figure with graph_objects for full control over markers
    fig = go.Figure()
    
    if has_band:
        # Fill missing band values with 'Empty' to keep all data points
        df_plot = df_merged.copy()
        df_plot['band'] = df_plot['band'].fillna('Empty')
        
        # Group by station first, then band for proper legend ordering
        for (stn, band), group in df_plot.groupby(['stn', 'band']):
            band_str = str(band).strip()
            color = band_colors.get(band_str, band_colors['unknown'])
            marker_symbol = station_marker_map.get(stn, 'circle')
            obs_name = obs_codes.get(stn, stn)
            
            # Create legend label with band and station
            legend_name = create_band_legend_name(band_str, stn, obs_name)
            
            # Calculate reduced magnitude
            reduced_mag = group['mag'] - group['mag_dist_corr']
            
            fig.add_trace(go.Scatter(
                x=group['Phase'],
                y=reduced_mag,
                mode='markers',
                marker=dict(size=8, color=color, symbol=marker_symbol, opacity=0.7),
                name=legend_name,
                customdata=list(zip(
                    [stn] * len(group),
                    [obs_name] * len(group),
                    [band_str] * len(group)
                )),
                hovertemplate='Observatory: %{customdata[0]}<br>Name: %{customdata[1]}<br>Band: %{customdata[2]}<br>Phase: %{x}<br>Magnitude: %{y:.2f}<extra></extra>'
            ))
    else:
        # Fallback to station-based grouping
        for stn, group in df_merged.groupby('stn'):
            marker_symbol = station_marker_map.get(stn, 'circle')
            obs_name = obs_codes.get(stn, stn)
            
            # Calculate reduced magnitude
            reduced_mag = group['mag'] - group['mag_dist_corr']
            
            fig.add_trace(go.Scatter(
                x=group['Phase'],
                y=reduced_mag,
                mode='markers',
                marker=dict(size=8, symbol=marker_symbol, opacity=0.7),
                name=f'{stn} ({obs_name})',
                customdata=list(zip(
                    [stn] * len(group),
                    [obs_name] * len(group)
                )),
                hovertemplate='Observatory: %{customdata[0]}<br>Name: %{customdata[1]}<br>Phase: %{x}<br>Magnitude: %{y:.2f}<extra></extra>'
            ))
    
    if ztf_df is not None:
        logger.debug(f"Adding ZTF data to the phase plot for {asteroid_id}")
        # Define colors for ZTF filter IDs
        ztf_g = ztf_df[ztf_df['i:fid'] == 1]  # g-band
        ztf_r = ztf_df[ztf_df['i:fid'] == 2]  # r-band

        # Add r-band with realistic red color and star marker
        fig.add_trace(go.Scatter(
            x=ztf_r['Phase'],
            y=ztf_r['i:magpsf'] - ztf_r['mag_dist_corr'],
            mode='markers',
            marker=dict(size=10, color=band_colors['r'], symbol='star', opacity=0.7),
            name='I41 ZTF: r-band',
            hovertemplate='Observatory: I41<br>Palomar Mountain ZTF<br>Band: r<br>Phase: %{x}<br>Magnitude: %{y:.2f}<extra></extra>'
        ))

        # Add g-band with realistic green/cyan color and star marker
        fig.add_trace(go.Scatter(
            x=ztf_g['Phase'],
            y=ztf_g['i:magpsf'] - ztf_g['mag_dist_corr'],
            mode='markers',
            marker=dict(size=10, color=band_colors['g'], symbol='star', opacity=0.7),
            name='I41 ZTF: g-band',
            hovertemplate='Observatory: I41<br>Palomar Mountain ZTF<br>Band: g<br>Phase: %{x}<br>Magnitude: %{y:.2f}<extra></extra>'
        ))
    
    # Update layout with inverted y-axis and styling
    fig.update_layout(
        title=f'Phase-Magnitude Relation for {asteroid_id}',
        xaxis_title='Phase Angle (degrees)',
        yaxis_title='Reduced Magnitude (H)',
        yaxis=dict(autorange="reversed"),
        height=600,
        legend_title_text='Observatory: Filter' if has_band else 'Observatory',
        template='plotly_white',
    )
    
    # Convert to JSON for sending to the client
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    logger.info(f"Successfully generated phase plot for {asteroid_id} with {len(df_merged)} data points")
    return jsonify({
        "status": "success",
        "plot": graphJSON,
        "count": len(df_merged)
    })


@app.route('/export_data', methods=['POST'])
def export_data():
    """
    Export asteroid observation data to CSV file.
    
    Request body should contain:
    - asteroid_id: The asteroid designation
    - data_source: One of 'mpc', 'miriade', 'ztf', or 'all'
    """
    from flask import send_file, make_response
    from io import BytesIO
    import zipfile
    
    data = request.get_json()
    asteroid_id = data.get('asteroid_id', '')
    data_source = data.get('data_source', 'all')
    
    logger.info(f"Exporting {data_source} data for asteroid {asteroid_id}")
    
    if not asteroid_id:
        return jsonify({
            "status": "error",
            "message": "Asteroid ID is required"
        }), 400
    
    # Replace slashes with underscores for safe designation
    safe_designation = asteroid_id.replace('/', '_').replace(' ', '_')
    
    try:
        if data_source == 'all':
            # Export all available data as a zip file
            memory_file = BytesIO()
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Export MPC data if available
                if db_utils.mpc_data_exists(safe_designation):
                    mpc_df = db_utils.load_mpc_data(safe_designation)
                    if mpc_df is not None:
                        csv_buffer = BytesIO()
                        mpc_df.to_csv(csv_buffer, index=False)
                        zf.writestr(f'{safe_designation}_mpc.csv', csv_buffer.getvalue())
                        logger.info(f"Added MPC data to export: {len(mpc_df)} rows")
                
                # Export Miriade data if available
                if db_utils.miriade_data_exists(safe_designation):
                    miriade_df = db_utils.load_miriade_data(safe_designation)
                    if miriade_df is not None:
                        csv_buffer = BytesIO()
                        miriade_df.to_csv(csv_buffer, index=False)
                        zf.writestr(f'{safe_designation}_miriade.csv', csv_buffer.getvalue())
                        logger.info(f"Added Miriade data to export: {len(miriade_df)} rows")
                
                # Export ZTF data if available
                if db_utils.ztf_data_exists(safe_designation):
                    ztf_df = db_utils.load_ztf_data(safe_designation)
                    if ztf_df is not None:
                        csv_buffer = BytesIO()
                        ztf_df.to_csv(csv_buffer, index=False)
                        zf.writestr(f'{safe_designation}_ztf.csv', csv_buffer.getvalue())
                        logger.info(f"Added ZTF data to export: {len(ztf_df)} rows")
            
            memory_file.seek(0)
            logger.info(f"Successfully created export package for {asteroid_id}")
            return send_file(
                memory_file,
                mimetype='application/zip',
                as_attachment=True,
                download_name=f'{safe_designation}_all_data.zip'
            )
        
        else:
            # Export single data source
            df = None
            filename = None
            
            if data_source == 'mpc':
                if db_utils.mpc_data_exists(safe_designation):
                    df = db_utils.load_mpc_data(safe_designation)
                    filename = f'{safe_designation}_mpc.csv'
                else:
                    return jsonify({
                        "status": "error",
                        "message": f"No MPC data found for {asteroid_id}"
                    }), 404
            
            elif data_source == 'miriade':
                if db_utils.miriade_data_exists(safe_designation):
                    df = db_utils.load_miriade_data(safe_designation)
                    filename = f'{safe_designation}_miriade.csv'
                else:
                    return jsonify({
                        "status": "error",
                        "message": f"No Miriade data found for {asteroid_id}"
                    }), 404
            
            elif data_source == 'ztf':
                if db_utils.ztf_data_exists(safe_designation):
                    df = db_utils.load_ztf_data(safe_designation)
                    filename = f'{safe_designation}_ztf.csv'
                else:
                    return jsonify({
                        "status": "error",
                        "message": f"No ZTF data found for {asteroid_id}"
                    }), 404
            
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Invalid data source: {data_source}"
                }), 400
            
            if df is None:
                return jsonify({
                    "status": "error",
                    "message": f"Failed to load {data_source} data for {asteroid_id}"
                }), 500
            
            # Create CSV in memory
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            logger.info(f"Successfully exported {data_source} data for {asteroid_id}: {len(df)} rows")
            return send_file(
                csv_buffer,
                mimetype='text/csv',
                as_attachment=True,
                download_name=filename
            )
    
    except Exception as e:
        logger.error(f"Error exporting data for {asteroid_id}: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error exporting data: {str(e)}"
        }), 500


if __name__ == '__main__':
    # Final setup for logging
    logger.info("Starting MPC Viewer application")
    
    # Uncomment the following line to run in production mode
    app.run(host='0.0.0.0', port=5000, debug=False)
    
    # Development mode with debug enabled
    logger.info("Running in debug mode on http://127.0.0.1:5000/")
    # app.run(debug=True)
