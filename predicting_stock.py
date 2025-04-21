import os
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Configuration ===
base_dir = "C:/Users/Daziel Brilliant/Desktop/CV builder/Predict SP500"
os.makedirs(base_dir, exist_ok=True)
csv_path = os.path.join(base_dir, "sp500_data.csv")
model_path = os.path.join(base_dir, "lstm_model.pth")
sequence_length = 60
future_steps = 5
features_used = ['Close', 'MA50', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'Volatility']

# === LSTM Model Class ===
class MultiStepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=2, output_size=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

# === Functions ===
def download_or_update_data():
    print("🌐 Downloading full S&P 500 data from Yahoo Finance...")
    data = yf.download("^GSPC", start="2015-01-01")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [' '.join(col).strip() for col in data.columns.values]

    if 'Close ^GSPC' in data.columns:
        data.rename(columns={
            'Close ^GSPC': 'Close',
            'Open ^GSPC': 'Open',
            'High ^GSPC': 'High',
            'Low ^GSPC': 'Low',
            'Volume ^GSPC': 'Volume'
        }, inplace=True)

    data.index = pd.to_datetime(data.index)
    data = data[~data.index.duplicated(keep='last')]
    data = data.sort_index()

    data.to_csv(csv_path)
    print(f"✅ Full dataset saved to {csv_path}")
    return data

def add_indicators(df):
    df['MA50'] = df['Close'].rolling(window=50).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['Close'].rolling(window=20).std()
    df['Volatility'] = df['Close'].pct_change().rolling(window=14).std() * np.sqrt(252)
    return df.dropna()

def prepare_data(data):
    data = add_indicators(data)
    if len(data) < sequence_length + future_steps:
        raise ValueError("📉 Not enough data after adding indicators. Wait for more trading days.")

    features = data[features_used].values
    train_size = int(len(features) * 0.8)

    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(features[:train_size])
    scaled_test = scaler.transform(features[train_size - sequence_length - future_steps:])

    def create_sequences(data):
        X, y = [], []
        for i in range(sequence_length, len(data) - future_steps + 1):
            X.append(data[i - sequence_length:i])
            y.append(data[i:i + future_steps, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(scaled_train)
    X_test, y_test = create_sequences(scaled_test)

    return X_train, y_train, X_test, y_test, scaler, features

def train_model(X_train, y_train, input_size):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    model = MultiStepLSTM(input_size=input_size, output_size=future_steps)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("🏋️ Training model...")
    for epoch in range(50):
        model.train()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50 - Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), model_path)
    print("💾 Model saved.")
    return model

def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    returns = pd.Series(np.diff(y_pred))
    volatility = returns.std()
    sharpe_ratio = returns.mean() / volatility * np.sqrt(252) if volatility != 0 else 0

    cumulative = pd.Series(y_pred).pct_change().fillna(0).add(1).cumprod()
    peak = cumulative.cummax()
    drawdown = ((cumulative - peak) / peak).min()

    # Calculate directional accuracy
    directions_pred = np.sign(np.diff(y_pred))
    directions_true = np.sign(np.diff(y_true))
    directional_matches = directions_pred == directions_true
    directional_accuracy = np.mean(directional_matches) * 100

    print(f"\n📊 RMSE: ${rmse:.2f}")
    print(f"📊 MAE:  ${mae:.2f}")
    print(f"📈 R² Score: {r2:.4f}")
    print(f"📉 Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"🔻 Max Drawdown: {drawdown:.2%}\n")
    print(f"🎯 Directional Accuracy: {directional_accuracy:.2f}%")

def evaluate_saved_model():
    if not os.path.exists(model_path):
        print("⚠️ Model not found. Train it using Option 2 first.")
        return

    data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    X_train, y_train, X_test, y_test, scaler, features = prepare_data(data)

    model = MultiStepLSTM(input_size=features.shape[1], output_size=future_steps)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_flat = y_test.reshape(-1)

    with torch.no_grad():
        y_pred = model(X_test_t).numpy().reshape(-1)

    # Pad for inverse transform
    pad_pred = np.zeros((len(y_pred), features.shape[1]))
    pad_true = np.zeros((len(y_test_flat), features.shape[1]))
    pad_pred[:, 0] = y_pred
    pad_true[:, 0] = y_test_flat

    y_pred_inv = scaler.inverse_transform(pad_pred)[:, 0]
    y_true_inv = scaler.inverse_transform(pad_true)[:, 0]

    evaluate_model(y_true_inv, y_pred_inv)
    plot_full_prediction_vs_actual(X_test, y_test, scaler, features)


def check_data_range():
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"📅 Data range: {df.index.min().date()} to {df.index.max().date()} ({len(df)} rows)")

    else:
        print("⚠️ No data file found.")

def predict_today_only():
    data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    data = add_indicators(data)

    if not os.path.exists(model_path):
        print("⚠️ Model not found. Please run option 2 first to train.")
        return

    features = data[features_used].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    latest_seq = scaled[-sequence_length:]
    latest_input = torch.tensor(latest_seq[np.newaxis, :, :], dtype=torch.float32)

    model = MultiStepLSTM(input_size=len(features_used), output_size=future_steps)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        forecast_scaled = model(latest_input).numpy()[0]

    padded = np.zeros((future_steps, features.shape[1]))
    padded[:, 0] = forecast_scaled
    forecast_prices = scaler.inverse_transform(padded)[:, 0]

    current_price = data['Close'].iloc[-1]
    print("\n--- Prediction ---")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Predicted Price (1d ahead): ${forecast_prices[0]:.2f}")
    print(f"Predicted Price (5d ahead): ${forecast_prices[-1]:.2f}")

    if forecast_prices[0] > current_price and forecast_prices[-1] > current_price:
        print("📈 Suggested Action: BUY")
    elif forecast_prices[0] < current_price and forecast_prices[-1] < current_price:
        print("📉 Suggested Action: SELL")
    else:
        print("🤔 Suggested Action: HOLD")

    compare_predictions_vs_actual(data, forecast_prices)
    return data


def plot_price_and_volume(df):
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax1.set_ylabel('Close Price')
    ax1.set_title('S&P 500 Close Price with Volume')
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.bar(df.index, df['Volume'], label='Volume', color='gray', alpha=0.3)
    ax2.set_ylabel('Volume')
    plt.tight_layout()
    save_path = os.path.join(base_dir, 'price_volume_plot.png')
    plt.savefig(save_path)
    print(f"📊 Saved price + volume plot to {save_path}")
    plt.close()

def compare_predictions_vs_actual(data, forecast_prices):
    dates = data.index[-(future_steps + 1):]  # includes today and 5 future days
    actuals = data['Close'].loc[dates[1:]].values  # skip current day

    # 1-day comparison
    plt.figure(figsize=(8, 4))
    plt.plot([dates[1]], [actuals[0]], 'ro', label='Actual')
    plt.plot([dates[1]], [forecast_prices[0]], 'bo', label='Predicted')
    plt.title('1-Day Prediction vs Actual')
    plt.legend()
    plt.tight_layout()
    save_1d = os.path.join(base_dir, 'compare_1d.png')
    plt.savefig(save_1d)
    print(f"📉 1-day comparison plot saved to {save_1d}")
    plt.close()

    # 5-day comparison
    plt.figure(figsize=(10, 5))
    plt.plot(dates[1:], actuals, label='Actual')
    plt.plot(dates[1:], forecast_prices, label='Predicted')
    plt.title('5-Day Prediction vs Actual')
    plt.legend()
    plt.tight_layout()
    save_5d = os.path.join(base_dir, 'compare_5d.png')
    plt.savefig(save_5d)
    print(f"📈 5-day comparison plot saved to {save_5d}")
    plt.close()

def plot_full_prediction_vs_actual(X_test, y_test, scaler, features, title="Full Period Prediction vs Actual"):
    model = MultiStepLSTM(input_size=features.shape[1], output_size=future_steps)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_test_t).numpy()

    actual = y_test[:, 0]  # just 1-day ahead for comparison
    pred = predictions[:, 0]

    # Inverse scale
    def inverse_close(vals):
        pad = np.zeros((len(vals), features.shape[1]))
        pad[:, 0] = vals
        return scaler.inverse_transform(pad)[:, 0]

    pred_inv = inverse_close(pred)
    actual_inv = inverse_close(actual)

    plt.figure(figsize=(12, 6))
    plt.plot(actual_inv, label="Actual", color='red')
    plt.plot(pred_inv, label="Predicted", color='blue')
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("S&P 500 Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(base_dir, "full_prediction_vs_actual.png")
    plt.savefig(path)
    print(f"📊 Full prediction plot saved to {path}")
    plt.close()

# === Main ===
if __name__ == "__main__":
    print("Choose an option:")
    print("1 - Predict Today")
    print("2 - Update Data & Retrain Model")
    print("3 - Check Stored Data Range")
    print("4 - Evaluate Model")
    choice = input("Enter 1, 2, 3 or 4: ")

    if choice == '3':
        print("📅 Checking stored data range...")
        check_data_range()

        exit()

    if choice == '4':
        evaluate_saved_model()
        exit()

    data = download_or_update_data()

    if choice == '2':
        try:
            X_train, y_train, X_test, y_test, scaler, features = prepare_data(data)
            model = train_model(X_train, y_train, input_size=features.shape[1])
        except ValueError as e:
            print(f"⚠️ Skipping training: {e}")

    elif choice == '1':
        try:
            data = predict_today_only()
            plot_price_and_volume(data)

        except ValueError as e:
            print(f"⚠️ Cannot predict: {e}")

