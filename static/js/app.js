document.addEventListener('DOMContentLoaded', function() {
    // Elements
    // const fetchForm = document.getElementById('fetchForm');
    const userInput = document.getElementById('userInput');
    const fetchStatus = document.getElementById('fetchStatus');
    
    const loadMPCButton = document.getElementById('loadMpcButton');
    const mpcSpinner = document.getElementById('mpcSpinner');
    const loadMiriadeButton = document.getElementById('loadMiriadeButton');
    const miriadeSpinner = document.getElementById('miriadeSpinner');
    const loadZtfButton = document.getElementById('loadZtfButton');
    const ztfSpinner = document.getElementById('ztfSpinner');

    const asteroidIdInput = document.getElementById('asteroidId');
    const plotButton = document.getElementById('plotButton');
    const plotPhaseButton = document.getElementById('plotPhaseButton');
    const plotPhaseSpinner = document.getElementById('plotPhaseSpinner');
    
    // Export elements
    const exportDropdown = document.getElementById('exportDropdown');
    const exportAll = document.getElementById('exportAll');
    const exportMpc = document.getElementById('exportMpc');
    const exportMiriade = document.getElementById('exportMiriade');
    const exportZtf = document.getElementById('exportZtf');

    const plotForm = document.getElementById('plotForm');
    const plotSpinner = document.getElementById('plotSpinner');
    const plotStatus = document.getElementById('plotStatus');
    const plotContainer = document.getElementById('plotContainer');
    
    // Unified function to show message in any status area
    function showMessage(element, message, type) {
        element.textContent = message;
        element.style.display = 'block';
        element.className = 'status-message alert mt-3 alert-' + type;
    }
    
    // Function to show spinner
    function showSpinner(spinner) {
        spinner.style.display = 'inline-block';
        return Promise.resolve();
    }

    // Function to hide spinner
    function hideSpinner(spinner) {
        spinner.style.display = 'none';
    }

    // Function to handle button clicks with AJAX
    function handleButtonClick(endpoint, buttonType, inputValue, spinner) {
        if (inputValue.trim() === '') {
            showMessage(fetchStatus, `Please enter text before loading ${buttonType} data`, 'warning');
            return;
        }
        
        showMessage(fetchStatus, `Loading ${buttonType} data for: ${inputValue}`, 'info');
        
        // Show spinner
        const spinnerPromise = showSpinner(spinner);
        
        fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ userInput: inputValue })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            spinnerPromise.then(() => {
                hideSpinner(spinner);
                showMessage(fetchStatus, data.message, 'success');
                
                // Update asteroid ID if provided in the response
                if (data.id) {
                    asteroidIdInput.value = data.id;
                    // Enable the plot buttons
                    plotButton.disabled = false;
                    plotPhaseButton.disabled = false;
                    // Enable the export dropdown
                    exportDropdown.disabled = false;
                }
            });
        })
        .catch(error => {
            spinnerPromise.then(() => {
                hideSpinner(spinner);
                showMessage(fetchStatus, `Error: ${error.message}`, 'danger');
                console.error('Error:', error);
            });
        });
    }

    // Function to handle plot requests
    function handlePlotRequest(endpoint, spinner, plotType) {
        const asteroidId = asteroidIdInput.value.trim();
        if (!asteroidId) return;
        
        // Show loading spinner and hide previous content
        showSpinner(spinner);
        plotStatus.style.display = 'none';
        plotContainer.style.display = 'none';
        
        // Create form data
        const formData = new FormData();
        formData.append('asteroid_id', asteroidId);
        
        // Fetch request
        fetch(endpoint, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Hide loading spinner
            hideSpinner(spinner);
            
            if (data.status === 'success') {
                try {
                    // Show success message
                    showMessage(plotStatus, data.count ? 
                               `Found ${data.count} observations with magnitude data` : 
                               `${plotType} plot generated successfully`, 'success');
                    
                    // Display the plot container
                    plotContainer.style.display = 'block';
                    
                    // Parse the plot data if needed
                    let plotData = typeof data.plot === 'string' ? JSON.parse(data.plot) : data.plot;
                    
                    // Create the plot
                    Plotly.newPlot(
                        'plotContainer', 
                        plotData.data, 
                        plotData.layout || {}, 
                        {responsive: true}
                    );
                } catch (error) {
                    console.error(`${plotType} plot error:`, error);
                    showMessage(plotStatus, `Error with ${plotType} plot data: ${error.message}`, 'danger');
                }
            } else {
                showMessage(plotStatus, data.message, 'danger');
            }
        })
        .catch(error => {
            hideSpinner(spinner);
            showMessage(plotStatus, `Error: ${error.message}`, 'danger');
            console.error('Fetch error:', error);
        });
    }

    // Add Enter key handling for userInput
    userInput.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            // Trigger the MPC load button when Enter is pressed
            if (loadMPCButton && !loadMPCButton.disabled) {
                loadMPCButton.click();
            }
        }
    });

    // MPC button click event
    loadMPCButton.addEventListener('click', function() {
        handleButtonClick('/fetch_mpc', 'MPC', userInput.value, mpcSpinner);
    });
    
    // Miriade button click event
    loadMiriadeButton.addEventListener('click', function() {
        handleButtonClick('/fetch_miriade', 'Miriade', userInput.value, miriadeSpinner);
    });

    // Miriade button click event
    loadZtfButton.addEventListener('click', function() {
        handleButtonClick('/fetch_ztf', 'ZTF', userInput.value, ztfSpinner);
    });

    // Plot Observations form submit
    plotForm.addEventListener('submit', function(e) {
        e.preventDefault();
        handlePlotRequest('/plot_observations', plotSpinner, 'Observations');
    });

    // Plot Phase button click event
    plotPhaseButton.addEventListener('click', function() {
        handlePlotRequest('/plot_phase', plotPhaseSpinner, 'Phase');
    });
    
    // Function to handle data export
    function handleExport(dataSource) {
        const asteroidId = asteroidIdInput.value.trim();
        if (!asteroidId) {
            showMessage(plotStatus, 'Please load asteroid data first', 'warning');
            return;
        }
        
        showMessage(plotStatus, `Preparing ${dataSource} data export...`, 'info');
        
        fetch('/export_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                asteroid_id: asteroidId,
                data_source: dataSource
            })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.message || `HTTP error! Status: ${response.status}`);
                });
            }
            // Get filename from Content-Disposition header if available
            const contentDisposition = response.headers.get('Content-Disposition');
            let filename = `${asteroidId}_${dataSource}.csv`;
            if (contentDisposition) {
                const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
                if (filenameMatch && filenameMatch[1]) {
                    filename = filenameMatch[1].replace(/['"]/g, '');
                }
            }
            return response.blob().then(blob => ({ blob, filename }));
        })
        .then(({ blob, filename }) => {
            // Create a download link
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            showMessage(plotStatus, `${dataSource.toUpperCase()} data exported successfully`, 'success');
        })
        .catch(error => {
            showMessage(plotStatus, `Export error: ${error.message}`, 'danger');
            console.error('Export error:', error);
        });
    }
    
    // Export event listeners
    exportAll.addEventListener('click', function(e) {
        e.preventDefault();
        handleExport('all');
    });
    
    exportMpc.addEventListener('click', function(e) {
        e.preventDefault();
        handleExport('mpc');
    });
    
    exportMiriade.addEventListener('click', function(e) {
        e.preventDefault();
        handleExport('miriade');
    });
    
    exportZtf.addEventListener('click', function(e) {
        e.preventDefault();
        handleExport('ztf');
    });
});