import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# nltk and newspaper are not used in the provided functions, so they are commented out.
# import nltk
# from newspaper import Article

def estimate_variables(ticker):
    # Downloads the data from Yahoo Finance
    start_date = '2015-01-01'
    # Fetch data up to today's date for current analysis
    end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

    if data.empty:
        raise ValueError(f"⚠︎ No data found for TICKER ⚠︎: {ticker}")
    # Closing prices as a Pandas Series to retain the datetime index
    closing_series = data['Close'].dropna()

    # Calculates the daily returns
    returns = []
    # Use .iloc for positional indexing when looping through Series
    for i in range(1, len(closing_series)):
        daily_return = (closing_series.iloc[i] - closing_series.iloc[i-1]) / closing_series.iloc[i-1]
        returns.append(daily_return)

    returns = np.array(returns)
    # Annualized mean return
    mu = float(np.mean(returns) * 252)
    # Annualized volatility
    sigma = float(np.std(returns) * np.sqrt(252))

    # S0 for simulation is the LAST closing price, for forward-looking forecast
    S0_forecast = closing_series.iloc[-1].item()

    # Returns the starting price for forecast, expected return, volatility, and all historical closing prices (as Series)
    return S0_forecast, mu, sigma, closing_series

############################################


def simulation(ticker, T=1.0, num_simulations=10):
    # Set the variables based on the ticker inputted
    try:
        S0, mu, sigma, historical_closing_series = estimate_variables(ticker)
    except ValueError as e:
        print(e)
        return

    # Set the time steps for the forecast period (T years)
    steps_forecast = int(T * 252) # Number of steps for the future simulation
    dt = T / steps_forecast

    # Simulate multiple GBM paths for the FUTURE forecast
    simulated_paths = np.zeros((num_simulations, steps_forecast))

    for i in range(num_simulations):
        # Initialize price array for this simulation path
        S = np.zeros(steps_forecast)
        # Start the simulation from the last historical price (S0)
        S[0] = S0
        # Simulate one path
        for t in range(1, steps_forecast):
            Z = np.random.normal() # Standard normal random variable
            # GBM formula: dS = mu * S * dt + sigma * S * sqrt(dt) * Z
            dS = mu * S[t-1] * dt + sigma * S[t-1] * np.sqrt(dt) * Z
            S[t] = S[t-1] + dS
        simulated_paths[i, :] = S

    # Compute the point-wise average of the simulated paths
    mean_trajectory = np.mean(simulated_paths, axis=0)

    # --- Plotting ---
    plt.figure(figsize=(14, 7))

    # --- Historical Data Plotting ---
    historical_dates = historical_closing_series.index
    plt.plot(historical_dates, historical_closing_series.values,
             label='Historical Prices',
             color='black',
             linewidth=2)

    # --- Simulated Data Plotting ---
    last_historical_date = historical_dates[-1]

    # Create a range of business days that is *at least* steps_forecast long
    future_date_temp = pd.date_range(
        start=last_historical_date + pd.Timedelta(days=1),
        periods=steps_forecast + 30, # Add buffer for safety
        freq='B'
    )
    # Ensure forecast_dates exactly matches the length of simulation output
    forecast_dates = future_date_temp[:steps_forecast]

    if len(forecast_dates) != steps_forecast:
        print(f"Warning: Number of forecast dates ({len(forecast_dates)}) does not match simulation steps ({steps_forecast}). Using simple daily frequency.")
        forecast_dates = pd.date_range(start=last_historical_date + pd.Timedelta(days=1), periods=steps_forecast)


    # Plot the mean simulated trajectory
    plt.plot(forecast_dates, mean_trajectory,
             label=f'Mean Simulated Trajectory (n={num_simulations})',
             color='red',
             linestyle='--')

    # Plot formatting
    plt.axvline(x=last_historical_date, color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
    plt.title(f'{ticker} Stock Price Forecast using GBM\n' +
              f'S₀=${S0:.2f}, μ={mu:.2f}, σ={sigma:.2f}, Forecast Period={T} Years')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Display parameters and forecast metrics
    print(f"\n--- Forecast for {ticker} over {T} Years ---")
    print(f"Initial Price (S₀): ${S0:.2f} (based on last historical close)")
    print(f"Expected Annual Return (μ): {mu:.2%}")
    print(f"Annual Volatility (σ): {sigma:.2%}")
    print(f"Number of Simulations: {num_simulations}")
    print(f"Expected Price at End of Forecast: ${mean_trajectory[-1]:.2f}")


def run_monte_carlo_simulation(ticker, years, sims):
    """
    Web-friendly version of simulation function that returns plot as base64 string and text output
    """
    try:
        S0, mu, sigma, historical_closing_series = estimate_variables(ticker)
    except ValueError as e:
        raise ValueError(str(e))

    # Set the time steps for the forecast period (T years)
    steps_forecast = int(years * 252)
    dt = years / steps_forecast

    # Simulate multiple GBM paths for the FUTURE forecast
    simulated_paths = np.zeros((sims, steps_forecast))

    for i in range(sims):
        S = np.zeros(steps_forecast)
        S[0] = S0
        for t in range(1, steps_forecast):
            Z = np.random.normal()
            dS = mu * S[t-1] * dt + sigma * S[t-1] * np.sqrt(dt) * Z
            S[t] = S[t-1] + dS
        simulated_paths[i, :] = S

    # Compute the point-wise average of the simulated paths
    mean_trajectory = np.mean(simulated_paths, axis=0)

    # --- Plotting ---
    plt.figure(figsize=(14, 7))

    # --- Historical Data Plotting ---
    historical_dates = historical_closing_series.index
    plt.plot(historical_dates, historical_closing_series.values,
             label='Historical Prices',
             color='black',
             linewidth=2)

    # --- Simulated Data Plotting ---
    last_historical_date = historical_dates[-1]

    # Create forecast dates
    future_date_temp = pd.date_range(
        start=last_historical_date + pd.Timedelta(days=1),
        periods=steps_forecast + 30,
        freq='B'
    )
    forecast_dates = future_date_temp[:steps_forecast]

    if len(forecast_dates) != steps_forecast:
        forecast_dates = pd.date_range(start=last_historical_date + pd.Timedelta(days=1), periods=steps_forecast)

    # Plot the mean simulated trajectory
    plt.plot(forecast_dates, mean_trajectory,
             label=f'Mean Simulated Trajectory (n={sims})',
             color='red',
             linestyle='--')

    # Plot formatting
    plt.axvline(x=last_historical_date, color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
    plt.title(f'{ticker} Stock Price Forecast using GBM\n' +
              f'S₀=${S0:.2f}, μ={mu:.2f}, σ={sigma:.2f}, Forecast Period={years} Years')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Convert plot to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()

    # Generate output text
    output_text = f"""
--- Forecast for {ticker} over {years} Years ---
Initial Price (S₀): ${S0:.2f} (based on last historical close)
Expected Annual Return (μ): {mu:.2%}
Annual Volatility (σ): {sigma:.2%}
Number of Simulations: {sims}
Expected Price at End of Forecast: ${mean_trajectory[-1]:.2f}
"""

    return img_str, output_text
