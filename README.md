# Asteroid Data Viewer

A web application for visualizing asteroid observation data from multiple sources including Minor Planet Center (MPC), IMCCE Miriade, and Zwicky Transient Facility (ZTF).

![Asteroid Data Viewer](static/images/front.png)

## Features

- **Multi-Source Data Retrieval**: Fetch asteroid data from several key astronomical databases:
  - Minor Planet Center (MPC): Official observation data with magnitude information
  - IMCCE Miriade: Ephemeris data for phase angle calculations
  - Zwicky Transient Facility (ZTF): Additional photometric observations

- **Interactive Visualization**:
  - Time-series magnitude plots
  - Phase-magnitude relation plots
  - Color-coded observatory identification

- **Data Caching**: Stores retrieved data locally to avoid redundant API calls and enable offline usage

- **Responsive UI**: Bootstrap-based interface that works across device sizes

## Prerequisites

- Python 3.x
- Flask
- Pandas
- NumPy
- Plotly
- Astropy
- Astroquery

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AlexeySergeyev/mpc_viewer.git
cd mpc_viewer
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install flask pandas numpy plotly astropy astroquery requests
```
or
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python3 app.py
```

5. Open your browser and navigate to: `http://127.0.0.1:5000/`

6. The application is available at the AWS: `http://51.44.180.15:5000/`.

## Usage

1. **Enter an asteroid identifier** in the input field (name, number, or designation)
2. **Load data** using the appropriate buttons:
   - Load MPC: Retrieves observation data from Minor Planet Center
   - Load Miriade: Fetches ephemeris data (requires MPC data to be loaded first)
   - Load ZTF: Retrieves observations from Zwicky Transient Facility
3. **Generate plots**:
   - Plot Observations: Shows magnitude over time
   - Plot Phase: Shows magnitude vs. phase angle (requires both MPC and Miriade data)

## Project Structure

```
mpc_observations/
├── app.py              # Main Flask application
├── static/             # Static assets
│   ├── css/            # CSS stylesheets
│   │   └── styles.css  # Custom styles
│   └── js/             # JavaScript files
│       └── app.js      # Client-side functionality
├── templates/          # HTML templates
│   └── index.html      # Main application page
└── db/                 # Data storage directory
    ├── designation/    # Asteroid designation data
    ├── miriade/        # Cached Miriade data
    ├── mpc/            # Cached MPC data
    └── ztf/            # Cached ZTF data
```

## API Endpoints

- `/fetch_mpc`: Retrieves and stores observation data from MPC
- `/fetch_miriade`: Retrieves ephemeris data from IMCCE Miriade
- `/fetch_ztf`: Retrieves observation data from ZTF
- `/plot_observations`: Generates time-series magnitude plots
- `/plot_phase`: Generates phase-magnitude relation plots

## Data Sources

- [Minor Planet Center](https://www.minorplanetcenter.net/)
- [IMCCE Miriade](https://ssp.imcce.fr/webservices/miriade/)
- [Zwicky Transient Facility](https://www.ztf.caltech.edu/)

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgements

- Minor Planet Center for providing asteroid observation data
- IMCCE for the Miriade ephemeris service
- Zwicky Transient Facility for photometric data
- Plotly for interactive visualization capabilities