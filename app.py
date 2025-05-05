from flask import Flask, render_template, request, jsonify
import requests
import json
import pandas as pd
import os
import plotly
import plotly.express as px
import numpy as np
from astropy.time import Time
from astroquery.mpc import MPC


app = Flask(__name__)

# Create db directory if it doesn't exist
os.makedirs('./db', exist_ok=True)

# Constants for API URLs
MPC_API_IDENTIFIER_URL = "https://data.minorplanetcenter.net/api/query-identifier"
MPC_API_OBSERVATIONS_URL = "https://data.minorplanetcenter.net/api/get-obs"
IMCCE_MIRIADE_URL = "https://ssp.imcce.fr/webservices/miriade/api/ephemcc.php"

init_miriade_params = {
        '-rplane': 1,
        '-tcoor': 5,
        "-oscelem": "astorb",
        "-mime": "json",
        "-output": "--jd",
        }


def load_obsevatory_codes():
    """
    Downloads the list of observatory codes from the Minor Planet Center (MPC) 
    
    Returns:
        dict: A dictionary mapping observatory codes to their names
    """
    # 
    obs = MPC.get_observatory_codes()
    d = dict(zip(obs['Code'].tolist(), 
             obs['Name'].tolist()))
    
    return d

# obs_codes = load_obsevatory_codes

def get_id(asteroid_number):
    """
    Generate a unique ID for the asteroid based on its number.
    
    Args:
        asteroid_number (str): The asteroid number to generate an ID for
        
    Returns:
        str: The generated ID
    """
    # Use the asteroid number as the ID
    response = requests.get(MPC_API_IDENTIFIER_URL, data=str(asteroid_number))
    iau_designation = None
    if response.ok:
        mpc_data = response.json()
        iau_designation = mpc_data.get('unpacked_primary_provisional_designation')
        # print(json.dumps(response.json(), indent=4))
        filename = f"./db/designation/{iau_designation}_mpc.json"
        if not os.path.exists(filename):
            # Save the response to a file
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                json.dump(response.json(), f, indent=4)
            print(f"Asteroid {asteroid_number} has IAU designation {iau_designation}")
    else:
        print("Error: ", response.status_code, response.content)
    return iau_designation

def fetch_mpc_data(asteroid_name):
    """
    Fetch observation data for a specific asteroid from the Minor Planet Center API.
    
    Args:
        asteroid_name (str): The asteroid number to fetch data for
        
    Returns:
        dict: Parsed JSON response from the MPC API
        iau_designation (str): The asteroid ID
    """
    
    # Check asteroid id
    iau_designation = get_id(asteroid_name)

    # MPC sometimes requires specific formatting (e.g., '00001' for Ceres)
    if iau_designation is None:
        # If the asteroid number is not found, return None or raise an error
        raise Exception(f"Asteroid {asteroid_name} not found in MPC database.")
    
    filename = f"./db/mpc/{iau_designation}_mpc.csv.gz"
    if os.path.exists(filename):
        # If the file already exists, read it and return the data
        mpc_df = pd.read_csv(filename)
        print(f"Data for {iau_designation} loaded successfully, shape: {mpc_df.shape}")
        return mpc_df.to_json(), iau_designation
    
    response = requests.get(MPC_API_OBSERVATIONS_URL, 
                            json={"desigs": [iau_designation], 
                                  "output_format":["ADES_DF"]})
    if response.ok:
        # json_string = response.json()[0]
        mpc_data = response.json()[0]['ADES_DF']
        mpc_df = pd.DataFrame(mpc_data)
        mpc_df.to_csv(filename, index=False)
        # print(f"{ades_df.shape[0]} observations for {iau_designation} fetched successfully")
        return mpc_data, iau_designation
    else:
        print("Error: ", response.status_code, response.content)
        return None

def fetch_miriade_data(asteroid_name, epochs=None):
    """
    Fetches data from the IMCCE Miriade web service.
    
    Args:
        params (dict): Dictionary of parameters to be sent in the request.
        
    Returns:
        dict: Parsed JSON response from the Miriade web service.
    """
    params = {
        "-name": str(asteroid_name),
        **init_miriade_params,
    }
    print(json.dumps(params, indent=4))
    try:
        response = requests.post(IMCCE_MIRIADE_URL, params=params, 
            files=epochs, timeout=120)
        print(f"Request URL: {response.url}")  # Debugging line to check the request URL
        response.raise_for_status()  # Raise an error for bad responses
        miriade_data = response.json()
        return miriade_data
    
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except ValueError as e:
        print(f"Error parsing JSON response: {e}")
        return None

def fetch_ztf_data(asteroid_name):
    """
    Fetches ZTF data for a specific asteroid.
    
    Args:
        asteroid_name (str): The name of the asteroid to fetch data for.
        
    Returns:
        dict: Parsed JSON response from the ZTF API.
        iau_designation (str): The asteroid ID
    """
    # Check asteroid id
    iau_designation = get_id(asteroid_name)
    
    # MPC sometimes requires specific formatting (e.g., '00001' for Ceres)
    if iau_designation is None:
        # If the asteroid number is not found, return None or raise an error
        raise Exception(f"Asteroid {asteroid_name} not found in ZTF database.")
    
    filename = f"./db/ztf/{iau_designation}_ztf.csv.gz"
    if os.path.exists(filename):
        # If the file already exists, read it and return the data
        ztf_df = pd.read_csv(filename)
        print(f"Data for {iau_designation} loaded successfully, shape: {ztf_df.shape}")
        return ztf_df.to_json(), iau_designation
    
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
    print(f"Request URL: {response.url}")  # Debugging line to check the request URL
    if response.ok:
        ztf_data = response.json()
        ztf_df = pd.DataFrame(ztf_data)
        if ztf_df.empty:
            print(f"No data found for {iau_designation}")
            return None
        ztf_df.to_csv(filename, index=False)
        print(f"{ztf_df.shape[0]} observations for {iau_designation} fetched successfully")
        return ztf_df.to_json(), iau_designation
    else:
        print("Error: ", response.status_code, response.content)
        return jsonify({
            "status": "error",
            "message": f"Error fetching data for {iau_designation}: {response.status_code} {response.content}"
        })

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/fetch_mpc', methods=['POST'])
def fetch_asteroid():
    data = request.get_json()
    asteroid_name = data.get('userInput', '')
    print(f"Fetching MPC data for asteroid {asteroid_name}")
    try:
        data, iau_designation = fetch_mpc_data(asteroid_name)
        return jsonify({
            "status": "success", 
            "message": f"Data for asteroid {asteroid_name} (ID: {iau_designation}) loaded successfully from MPC",
            "data": data,
            "id": iau_designation
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    
@app.route('/fetch_ztf', methods=['POST'])
def fetch_ztf():
    data = request.get_json()
    asteroid_name = data.get('userInput', '')
    print(f"Fetching ZTF data for asteroid {asteroid_name}")
    try:
        data = fetch_ztf_data(asteroid_name)
        if data is None:
            return jsonify({
                "status": "error", 
                "message": f"No ZTF data found for asteroid {asteroid_name}"
            })
        else:
            data_obs, iau_designation = data
        return jsonify({
            "status": "success", 
            "message": f"Data for asteroid {asteroid_name} (ID: {iau_designation}) retrived successfully from ZTF",
            "data": data_obs,
            "id": iau_designation
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/fetch_miriade', methods=['POST'])
def fetch_miriade():
    """
    Fetches Miriade data for a specific asteroid.
    """
    data = request.get_json()
    asteroid_name = data.get('userInput', '')

    # Check asteroid id
    iau_designation = get_id(asteroid_name)

    # MPC sometimes requires specific formatting (e.g., '00001' for Ceres)
    if iau_designation is None:
        # If the asteroid number is not found, return None or raise an error
        raise Exception(f"Asteroid {asteroid_name} not found in MPC database.")

    filename_mpc = f"./db/mpc/{iau_designation}_mpc.csv.gz"
    if os.path.exists(filename_mpc):
        # If the file already exists, read it and return the data
        df_mpc = pd.read_csv(filename_mpc)
        print(f"Data for {iau_designation} loaded successfully, shape: {df_mpc.shape}")
    else:
        return jsonify({
            "status": "error", 
            "message": f"No MPC data found for asteroid {iau_designation}"
        })
    
    filename_midiade = f"./db/miriade/{iau_designation}_miriade.csv.gz"
    if os.path.exists(filename_midiade):
        # If the file already exists, read it and return the data
        miriade_df = pd.read_csv(filename_midiade)
        print(f"Data for {iau_designation} loaded successfully, shape: {miriade_df.shape}")
        return jsonify({
            "status": "success", 
            "message": f"Data for asteroid {asteroid_name} (ID: {iau_designation}) loaded successfully from Miriade",
            "data": miriade_df.to_json(),
            "id": iau_designation
        })
    
    # Get the epochs from the DataFrame
    epochs = df_mpc.loc[:, 'obstime'].to_list()
    # Convert to Julian Date
    epochs_jd = Time(epochs, format='isot', scale='utc').jd
    # Prepare the epochs for the request
    epochs = {'epochs':
                ('epochs', '\n'.join(['%.6f' % epoch for epoch in epochs_jd]))}
    # Send the request to Miriade
    print(f"Fetching Miriade data for asteroid {asteroid_name}")
    try:
        miriade_data = fetch_miriade_data(asteroid_name, epochs)
        miriade_df = pd.DataFrame(miriade_data["data"])
        miriade_df.to_csv(filename_midiade, index=False)
        
        return jsonify({
            "status": "success", 
            "message": f"Data for asteroid {asteroid_name} (ID: {iau_designation}) fetched successfully from MPC",
            "data": miriade_data,
            "id": iau_designation
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/plot_observations', methods=['POST'])
def plot_observations():
    asteroid_id = request.form.get('asteroid_id')
    ztf_filename = f"./db/ztf/{asteroid_id}_ztf.csv.gz"
    df_ztf = None
    if os.path.exists(ztf_filename):
        df_ztf = pd.read_csv(ztf_filename)
        print(f"ZTF data for {asteroid_id} loaded successfully, shape: {df_ztf.shape}")
        df_ztf['obstime'] = pd.to_datetime(df_ztf['Date'], origin='julian', unit='D')
        show_ztf = True
    else:
        show_ztf = False
        print(f"ZTF data for {asteroid_id} not found, skipping ZTF plot")
    
    df_mpc = None
    mpc_filename = f"./db/mpc/{asteroid_id}_mpc.csv.gz"
    if os.path.exists(mpc_filename):
        df_mpc = pd.read_csv(mpc_filename)
        print(f"MPC data for {asteroid_id} loaded successfully, shape: {df_mpc.shape}")
    print(f"Fetching observations for asteroid {asteroid_id} from {mpc_filename}")

    if df_mpc is not None:
        # Filter for columns we need
        # Check if mag and obstime/obsTime columns exist
        if 'mag' not in df_mpc.columns or not 'obstime' in df_mpc.columns:
            return jsonify({
                "status": "error",
                "message": "The observation data does not contain magnitude or time data"
            })
        else:
            # # Take only rows with magnitude values
            # df = df_mpc.dropna(subset=['mag'])
            print(f"Magnitude values range: {df_mpc['mag'].min()} to {df_mpc['mag'].max()}")
            
            
            # Convert observation time to datetime if needed
            if df_mpc['obstime'].dtype != 'datetime64[ns]':
                df_mpc['obstime'] = pd.to_datetime(df_mpc['obstime'], errors='coerce')
                # Drop rows with invalid dates
                df_mpc = df_mpc.dropna(subset=['obstime'])
            
            obs_codes = load_obsevatory_codes()
            
            # Create plot with the actual data from DataFrame
            fig = px.scatter(x=df_mpc['obstime'], 
                y=df_mpc['mag'], 
                color=df_mpc['stn'],
                title=f'Magnitude observations for {asteroid_id}',
                color_discrete_sequence=px.colors.qualitative.Set1,
                custom_data=[df_mpc['stn'], [obs_codes.get(stn, stn) for stn in df_mpc['stn']]]  # Add custom data for hover
                )

            # Update the hover template after creating the figure
            fig.update_traces(
                hovertemplate='Observatory: %{customdata[0]}<br>%{customdata[1]}<br>Time: %{x}<br>Magnitude: %{y:.2f}<extra></extra>'
            )
            if show_ztf:
                fig.add_scatter(
                    x=df_ztf['obstime'],
                    y=df_ztf['i:magpsf'],
                    mode='markers',
                    marker=dict(size=4, color='black'),
                    name='ZTF gr-band',
                    hovertemplate='Observatory: I41<br>Palomar Mountain ZTF<br>%{x}<br>Mag: %{y:.2f}<extra></extra>'
                )

            #     fig.add_scatter(
            #         x=df_ztf['obstime'],
            #         y=df_ztf['SDSS:r'],
            #         mode='markers',
            #         marker=dict(size=4, color='red'),
            #         name='ZTF r-band',
            #         hovertemplate='Observatory: I41<br>Palomar Mountain ZTF<br>%{x}<br>Mag: %{y:.2f}<extra></extra>'
            # )
            
            # Invert y-axis (astronomical convention: brighter objects have lower magnitudes)
            fig.update_layout(yaxis=dict(autorange="reversed"))
            fig.update_traces(marker=dict(size=8, opacity=0.7))
            fig.update_layout(
                title=f'Magnitude observations for {asteroid_id}',
                xaxis_title='Observation Time',
                yaxis_title='Magnitude',
                height=600,
                # margin=dict(l=50, r=50, b=100, t=100),
                # plot_bgcolor='rgba(240, 240, 240, 0.8)',
                legend_title_text='Observatory',
                template='plotly_white',
            )
            
            # Convert to JSON for sending to the client
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            
            return jsonify({
                "status": "success",
                "plot": graphJSON,
                "count": len(df_mpc) if df_ztf is None else len(df_mpc) + len(df_ztf),
            })        
    else:
        return jsonify({
            "status": "error",
            "message": f"No data found for asteroid {asteroid_id}"
        })

@app.route('/plot_phase', methods=['POST'])
def plot_phase():
    asteroid_id = request.form.get('asteroid_id')
    mpc_filename = f"./db/mpc/{asteroid_id}_mpc.csv.gz"
    miriade_filename = f"./db/miriade/{asteroid_id}_miriade.csv.gz"
    ztf_filename = f"./db/ztf/{asteroid_id}_ztf.csv.gz"
    
    print(f"Generating phase plot for asteroid {asteroid_id}")

    # Check if both files exist
    if not os.path.exists(mpc_filename) or not os.path.exists(miriade_filename):
        missing = []
        if not os.path.exists(mpc_filename):
            missing.append("MPC")
        if not os.path.exists(miriade_filename):
            missing.append("Miriade")
        
        return jsonify({
            "status": "error",
            "message": f"Missing data for asteroid {asteroid_id}: {', '.join(missing)} data not found"
        })
    
    try:
        # Read the CSV files
        mpc_df = pd.read_csv(mpc_filename)
        miriade_df = pd.read_csv(miriade_filename)
        
        print(f"MPC data shape: {mpc_df.shape}, Miriade data shape: {miriade_df.shape}")
        
        df_merged = pd.concat([mpc_df, miriade_df], axis=1)
        df_merged['mag_dist_corr'] = 5 * np.log10(df_merged['Dhelio'] * df_merged['Dobs'])
        
        ztf_df = None
        if os.path.exists(ztf_filename):
            ztf_df = pd.read_csv(ztf_filename)
            print(f"ZTF data shape: {ztf_df.shape}")
            ztf_df['obstime'] = pd.to_datetime(ztf_df['Date'], origin='julian', unit='D')
            ztf_df['mag_dist_corr'] = 5 * np.log10(ztf_df['Dhelio'] * ztf_df['Dobs'])
        
        if len(df_merged) == 0:
            return jsonify({
                "status": "error",
                "message": f"No matching phase data found for asteroid {asteroid_id}"
            })
    except Exception as e:
        print(f"Error processing phase plot data: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error processing phase plot data: {str(e)}"
        })
    
    obs_codes = load_obsevatory_codes()
    
    # Create phase plot
    fig = px.scatter(df_merged, x='Phase', 
                        y=df_merged['mag'] - df_merged['mag_dist_corr'], 
                        color='stn',
                        title=f'Phase-Magnitude Relation for {asteroid_id}',
                        color_discrete_sequence=px.colors.qualitative.Set1,
                        custom_data=[df_merged['stn'], [obs_codes.get(stn, stn) for stn in df_merged['stn']]]  # Add custom data for hover
                )
    # Update the hover template after creating the figure
    fig.update_traces(
                hovertemplate='Observatory: %{customdata[0]}<br>Name: %{customdata[1]}<br>Phase: %{x}<br>Magnitude: %{y:.2f}<extra></extra>'
            )
    if ztf_df is not None:
        fig.add_scatter(
            x=ztf_df['Phase'],
            y=ztf_df['i:magpsf'] - ztf_df['mag_dist_corr'],
            mode='markers',
            marker=dict(size=4, color='black'),
            name='ZTF gr-band',
            hovertemplate='Observatory: I41<br>Palomar Mountain ZTF<br>Phase: %{x}<br>Mag: %{y:.2f}<extra></extra>'
        )

        # fig.add_scatter(
        #     x=ztf_df['Phase'],
        #     y=ztf_df['SDSS:r'] - ztf_df['mag_dist_corr'],
        #     mode='markers',
        #     marker=dict(size=4, color='red'),
        #     name='ZTF r-band',
        #     hovertemplate='Observatory: I41<br>Phase: %{x}<br>Palomar Mountain ZTF<br>Mag: %{y:.2f}<extra></extra>'
        # )
    
    # Invert y-axis (astronomical convention: brighter objects have lower magnitudes)
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(
        title=f'Phase-Magnitude Relation for {asteroid_id}',
        xaxis_title='Phase Angle (degrees)',
        yaxis_title='Magnitude',
        height=600,
        # plot_bgcolor='rgba(240, 240, 240, 0.8)',
        legend_title_text='Observatory',
        template='plotly_white',
    )
    
    # Convert to JSON for sending to the client
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({
        "status": "success",
        "plot": graphJSON,
        "count": len(df_merged)
    })

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8080, debug=False)
    app.run(debug=True)
