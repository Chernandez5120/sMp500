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

# Download historical data
ticker = yf.Ticker("AAPL")
df = ticker.history(period="20y")
df.to_csv("AAPL.csv")

# Add technical indicators
df['Return'] = df['Close'].pct_change()
df['MA10'] = df['Close'].rolling(window=10).mean()
df['Volatility'] = df['Return'].rolling(window=5).std()
df['MA20'] = df['Close'].rolling(window=20).mean()
df['Momentum'] = df['Close'] - df['Close'].shift(10)
df['RSI'] = 100 - (100 / (1 + df['Return'].rolling(14).mean() / df['Return'].rolling(14).std()))

# Create binary target
df['Target'] = (df['Close'].shift(-3) / df['Close'] - 1 > 0.01).astype(int)
df.dropna(inplace=True)

# Balance the dataset
df_up = df[df['Target'] == 1]
df_down = df[df['Target'] == 0]
if len(df_up) > len(df_down):
    df_up_resampled = resample(df_up, replace=False, n_samples=len(df_down), random_state=1)
    df_balanced = pd.concat([df_up_resampled, df_down], ignore_index=True)
else:
    df_down_resampled = resample(df_down, replace=False, n_samples=len(df_up), random_state=1)
    df_balanced = pd.concat([df_down_resampled, df_up], ignore_index=True)

df_balanced = df_balanced.sample(frac=1, random_state=1).reset_index(drop=True)

# Define features and target
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA20', 'Volatility', 'Momentum', 'RSI']
X = df_balanced[features]
Y = df_balanced['Target'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Final model with best alpha and fine-tuned hyperparameters
print("Training final MLPClassifier with alpha=0.01...")
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),  # deeper architecture
    max_iter=3000,
    alpha=0.01,
    random_state=42,
    solver='adam',
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20
)
model.fit(Xtrain, Ytrain)

# Predictions and evaluation
Y_pred = model.predict(Xtest)
accuracy = accuracy_score(Ytest, Y_pred)
mse = mean_squared_error(Ytest, Y_pred)
mae = mean_absolute_error(Ytest, Y_pred)
report = classification_report(Ytest, Y_pred, output_dict=True)

print("Training complete.")
print("MSE:", mse)
print("MAE:", mae)
print("Accuracy:", accuracy)
print(classification_report(Ytest, Y_pred))

# Predict tomorrow’s signal
latest_data = df[features].iloc[[-1]]
latest_scaled = scaler.transform(latest_data)
final_signal = model.predict(latest_scaled)[0]
print("\nFinal Trading Signal:", "Go Long (Buy)" if final_signal == 1 else "Go Short (Sell)")

# Save model and log
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_filename = f"MLPClassifier_Final_alpha_0.01_{timestamp}.pkl"
joblib.dump(model, model_filename)

log = {
    "timestamp": timestamp,
    "model_file": model_filename,
    "features": features,
    "alpha": 0.01,
    "architecture": [256, 128, 64],
    "solver": "adam",
    "accuracy": accuracy,
    "mse": mse,
    "mae": mae,
    "final_signal": int(final_signal)
}
with open(f"run_log_final_{timestamp}.json", "w") as f:
    json.dump(log, f, indent=4)

# Define the plotting function
def plot_predictions(Ytest, Y_pred, title="Model Predictions vs Actuals"):
    plt.figure(figsize=(14, 4))
    plt.plot(Ytest, label="Actual", alpha=0.7)
    plt.plot(Y_pred, label="Predicted", alpha=0.7)
    plt.title(title)
    plt.xlabel("Test Sample Index")
    
    plt.ylabel("Direction (1 = Up, 0 = Down)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("prediction_plot.png")
    plt.close()

# Call the function
plot_predictions(Ytest, Y_pred)
print("Saved prediction_plot.png.")