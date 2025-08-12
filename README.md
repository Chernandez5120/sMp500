# sMp500: Stock Market Analysis and Prediction Toolkit

This project provides a toolkit for stock market analysis, featuring two main components: a Neural Network-based model for predicting stock price direction and a web-based simulator using Geometric Brownian Motion (GBM) for forecasting future price paths.

## Features

### 1. Neural Network Predictor

This model predicts short-term stock price movement.

* **Data Fetching**: Downloads historical stock data from Yahoo Finance.
* **Feature Engineering**: Calculates several technical indicators to be used as features, including:
    * Moving Averages (10-day and 20-day)
    * Volatility (standard deviation of returns)
    * Momentum
    * Relative Strength Index (RSI)
* **Training**: Trains a Multi-layer Perceptron (MLP) classifier to predict if a stock's price will increase by more than 1% within a 3-day window.
* **Output**: Saves the trained model as a `.pkl` file and logs results, including evaluation metrics and the final trading signal (Buy/Sell).

### 2. Geometric Brownian Motion (GBM) Simulator

A web-based tool for simulating future stock prices using a Monte Carlo method.

* **Parameter Estimation**: Calculates the annualized mean return (μ) and volatility (σ) from historical data.
* **Simulation**: Runs multiple Monte Carlo simulations to forecast future stock price paths using the GBM formula.
* **Web Interface**: An interactive web application built with Flask allows users to input a stock ticker, forecast horizon, and the number of simulations.
* **Visualization**: Plots the historical closing prices along with the mean of the simulated price paths.

## Technologies Used

* **Backend**: Python, Flask
* **Machine Learning**: Scikit-learn, Joblib
* **Data Handling**: Pandas, NumPy
* **Data Source**: yfinance
* **Plotting**: Matplotlib
* **Frontend**: HTML, CSS, JavaScript

## File Structure

```
sMp500/
├── data/
│   ├── NN.py          # Main script for training the neural network predictor
│   ├── NN2.py         # Alternative script for the NN model
│   └── ...            # Saved models (.pkl) and logs (.json) will be generated here
├── stockmarket/
│   ├── GBM.py         # Core logic for the GBM Monte Carlo simulation
│   ├── server.py      # Flask web server
│   └── templates/
│       └── index.html # Frontend for the web simulator
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd sMp500
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### GBM Web Simulator

1.  **Start the Flask server:**
    ```bash
    python stockmarket/server.py
    ```
2.  **Access the application:**
    Open your web browser and navigate to `http://127.0.0.1:5000/`.
3.  **Run a simulation:**
    Enter a stock ticker (e.g., "AAPL"), the number of years to forecast, and the number of simulations, then click "Run Simulation".

### Neural Network Predictor

1.  **Run the training script:**
    ```bash
    python data/NN.py
    ```
2.  The script will automatically download data, train models for a predefined list of tickers (`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA`), and save the resulting model files (`.pkl`) and log files (`.json`) into the project's root directory.
