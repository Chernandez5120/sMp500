from flask import Flask, render_template, request, jsonify
import os
import sys

# Get the directory containing server.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the stockmart directory to Python path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Test core dependencies first
print("Testing core dependencies...")
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__} imported successfully")
except Exception as e:
    print(f"✗ NumPy import failed: {e}")

try:
    import pandas as pd
    print(f"✓ Pandas {pd.__version__} imported successfully")
except Exception as e:
    print(f"✗ Pandas import failed: {e}")

try:
    import yfinance as yf
    print("✓ yfinance imported successfully")
except Exception as e:
    print(f"✗ yfinance import failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib imported successfully")
except Exception as e:
    print(f"✗ matplotlib import failed: {e}")

# Now try to import GBM module
try:
    # Clear any cached imports that might be causing issues
    if 'GBM' in sys.modules:
        del sys.modules['GBM']
    
    import GBM
    print("✓ Successfully imported GBM module")
except ImportError as e:
    print(f"✗ Failed to import GBM module: {e}")
    GBM = None
except Exception as e:
    print(f"✗ Unexpected error importing GBM module: {e}")
    GBM = None

app = Flask(__name__)

# Route to serve the index.html file
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint for running the simulation
@app.route('/run_simulation', methods=['POST'])
def run_simulation_api():
    if GBM is None:
        return jsonify({"error": "Simulation module is not available. This is likely due to missing dependencies (NumPy/yfinance). Please check the console for import errors.", "success": False}), 500
    
    data = request.get_json()
    ticker = data.get('ticker')
    years = data.get('years')
    sims = data.get('sims')

    if not all([ticker, years, sims]):
        return jsonify({"error": "Missing parameters"}), 400

    try:
        # Call the simulation function from GBM.py
        img_str, output_text = GBM.run_monte_carlo_simulation(ticker, float(years), int(sims))
        return jsonify({
            "image": img_str,
            "text": output_text,
            "success": True
        })
    except ValueError as e:
        return jsonify({"error": str(e), "success": False}), 400
    except Exception as e:
        return jsonify({"error": f"An internal server error occurred: {e}", "success": False}), 500

if __name__ == '__main__':
    # Create a 'templates' directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')

    print("Serving on http://0.0.0.0:5000/")
    print("Make sure 'index.html' is in a 'templates' directory.")
    print("Press Ctrl+C to stop the server.")
    app.run(host='0.0.0.0', port=5000, debug=True) # debug=True allows for automatic reloading on code changes