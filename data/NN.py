import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib
import datetime
import json

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


def download_data(ticker, period="20y"):
    """
  Downloads historical stock data for a given ticker and period.

  Args:
      ticker (str): The stock ticker symbol (e.g., "AAPL").
      period (str): The time period for which to download data (e.g., "20y").

  Returns:
      pd.DataFrame: A DataFrame containing the historical stock data.
  """
    print(f"Downloading data for {ticker} for the last {period}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    print("Data download complete.")
    # df.to_csv(f"{ticker}.csv")
    return df


def add_technical_indicators(df):
    """
  Adds technical indicators and the target variable to the DataFrame.

  Args:
      df (pd.DataFrame): The input DataFrame with stock data.

  Returns:
      pd.DataFrame: The DataFrame with added features and target.
  """
    print("Adding technical indicators...")
    df['Return'] = df['Close'].pct_change()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Return'].rolling(window=5).std()
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    # Simplified RSI calculation to avoid potential division by zero in edge cases
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Create binary target: 1 if the price increases by more than 1% in 3 days, else 0
    df['Target'] = (df['Close'].shift(-3) / df['Close'] - 1 > 0.01).astype(int)
    df.dropna(inplace=True)
    print("Technical indicators added.")
    return df


def balance_data(df):
    """
  Balances the dataset based on the 'Target' variable by downsampling the majority class.

  Args:
      df (pd.DataFrame): The DataFrame to balance.

  Returns:
      pd.DataFrame: A balanced and shuffled DataFrame.
  """
    print("Balancing dataset...")
    df_up = df[df['Target'] == 1]
    df_down = df[df['Target'] == 0]

    # Downsample the majority class
    if len(df_up) > len(df_down):
        df_majority_downsampled = resample(df_up,
                                           replace=False,
                                           n_samples=len(df_down),
                                           random_state=42)
        df_balanced = pd.concat([df_majority_downsampled, df_down])
    else:
        df_majority_downsampled = resample(df_down,
                                           replace=False,
                                           n_samples=len(df_up),
                                           random_state=42)
        df_balanced = pd.concat([df_majority_downsampled, df_up])

    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1,
                                     random_state=42).reset_index(drop=True)
    print("Dataset balanced.")
    return df_balanced


def prepare_data_for_training(df):
    """
  Defines features, scales the data, and splits it into training and testing sets.

  Args:
      df (pd.DataFrame): The balanced DataFrame.

  Returns:
      tuple: Contains scaled training and testing data (X_train, X_test, y_train, y_test)
             and the fitted scaler object.
  """
    print("Preparing data for training...")
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA20', 'Volatility',
        'Momentum', 'RSI'
    ]
    X = df[features]
    y = df['Target'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)
    print("Data preparation complete.")
    return X_train, X_test, y_train, y_test, scaler, features


def train_model(X_train, y_train):
    """
  Initializes, trains, and returns the MLPClassifier model.

  Args:
      X_train (np.array): Scaled training features.
      y_train (np.array): Training target values.

  Returns:
      MLPClassifier: The trained model.
  """
    print("Training MLPClassifier...")
    model = MLPClassifier(hidden_layer_sizes=(256, 128, 64),
                          max_iter=3000,
                          alpha=0.01,
                          random_state=42,
                          solver='adam',
                          learning_rate_init=0.001,
                          early_stopping=True,
                          validation_fraction=0.1,
                          n_iter_no_change=20)
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model


def evaluate_model(model, X_test, y_test):
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("MSE:", mse)
    print("MAE:", mae)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    metrics = {
        "accuracy": accuracy,
        "mse": mse,
        "mae": mae,
        "classification_report": report
    }
    return metrics, y_pred


def predict_tomorrow_signal(model, df_full, features, scaler):
    """
  Predicts the trading signal for the most recent data point.

  Args:
      model (MLPClassifier): The trained model.
      df_full (pd.DataFrame): The original, un-balanced DataFrame.
      features (list): The list of feature names.
      scaler (StandardScaler): The fitted scaler object.

  Returns:
      int: The predicted signal (1 for Buy, 0 for Sell).
  """
    latest_data = df_full[features].iloc[[-1]]
    latest_scaled = scaler.transform(latest_data)
    final_signal = model.predict(latest_scaled)[0]

    print("\n--- Final Trading Signal ---")
    print("Signal:",
          "Go Long (Buy)" if final_signal == 1 else "Go Short (Sell)")
    return final_signal


def save_results(model, metrics, features, signal, ticker):
    """
  Saves the trained model and a log file with run details.

  Args:
      model (MLPClassifier): The trained model.
      metrics (dict): Evaluation metrics.
      features (list): List of features used.
      signal (int): The final predicted signal.
      ticker (str): The stock ticker.
  """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_filename = f"{ticker}_MLPClassifier_{timestamp}.pkl"
    log_filename = f"{ticker}_run_log_{timestamp}.json"

    # Save the model
    joblib.dump(model, model_filename)
    print(f"\nModel saved to {model_filename}")

    # Create and save the log
    log = {
        "timestamp": timestamp,
        "ticker": ticker,
        "model_file": model_filename,
        "features": features,
        "model_params": model.get_params(),
        "evaluation_metrics": metrics,
        "final_signal": int(signal)
    }
    with open(log_filename, "w") as f:
        json.dump(log, f, indent=4)
    print(f"Log saved to {log_filename}")


def plot_residuals(y_test, y_pred):
    """
  Plots the residuals of the model's predictions.

  Args:
      y_test (np.array): True target values.
      y_pred (np.array): Predicted target values.
  """
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(residuals)),
                residuals,
                alpha=0.5,
                label='Residuals (y_test - y_pred)')
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.title('Residual Plot', fontsize=16)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Residual Value', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_predictions(y_test, y_pred, title="Model Predictions vs Actuals"):
    plt.figure(figsize=(14, 4))
    plt.plot(y_test, label="Actual", alpha=0.7)
    plt.plot(y_pred, label="Predicted", alpha=0.7)
    plt.title(title)
    plt.xlabel("Test Sample Index")
    plt.ylabel("Direction (1 = Up, 0 = Down)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("prediction_plot.png")
    plt.close()


def main(ticker, period):
    """
  Main function to run the stock prediction workflow.

  Args:
      ticker (str): The stock ticker symbol.
      period (str): The historical data period.
  """
    # 1. Get and process data
    df_raw = download_data(ticker, period)
    df_featured = add_technical_indicators(df_raw.copy())

    # 2. Balance data for training
    df_balanced = balance_data(df_featured.copy())

    # 3. Prepare data for the model
    X_train, X_test, y_train, y_test, scaler, features = prepare_data_for_training(
        df_balanced)

    # 4. Train the model
    model = train_model(X_train, y_train)

    # 5. Evaluate the model
    metrics, y_pred = evaluate_model(model, X_test, y_test)

    # 6. Make a prediction on the latest data
    final_signal = predict_tomorrow_signal(model, df_featured, features,
                                           scaler)

    # 7. Save the results
    save_results(model, metrics, features, final_signal, ticker)

    # 8. Plot results
    # plot_residuals(y_test, y_pred)
    plot_predictions(y_test, y_pred)
    print("Saved prediction_plot.png.")


if __name__ == '__main__':
    for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]:
        # Example usage:
        TICKER = ticker
        TIME_RANGE = "max"  # e.g., "1y", "5y", "10y", "20y", "max"
        main(ticker=TICKER, period=TIME_RANGE)

