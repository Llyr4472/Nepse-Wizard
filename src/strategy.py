import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from Stock import *
import pandas_ta as ta
import json
import sys
import os

# Strategy Configuration
PARAMS = {
    'frama_period': 20,
    'lstm_lookback': 60,
    'atr_period': 14,
    'risk_per_trade': 0.02,
    'max_position': 0.1,
    'train_interval': 90
}


def custom_frama(df, period=20):
    """Fractal Adaptive Moving Average implementation"""
    n = period
    hh1 = df['high'].rolling(n//2).max()
    ll1 = df['low'].rolling(n//2).min()
    hh2 = df['high'].shift(n//2).rolling(n//2).max()
    ll2 = df['low'].shift(n//2).rolling(n//2).min()
    
    diff1 = (hh1 - ll1).replace(0, 1e-9)
    diff2 = (hh2 - ll2).replace(0, 1e-9)
    
    d = (np.log(diff1) - np.log(diff2)) / np.log(2)
    alpha = np.exp(-4.6 * (d - 1))
    alpha = np.clip(alpha, 0.01, 1)
    
    frama = df['close'].copy()
    for i in range(1, len(frama)):
        frama.iloc[i] = alpha.iloc[i] * df['close'].iloc[i] + (1 - alpha.iloc[i]) * frama.iloc[i-1]
    
    return frama

def calculate_vwap(df):
    """Custom VWAP calculation that works with RangeIndex"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cumulative_tpv = (typical_price * df['volume']).cumsum()
    cumulative_volume = df['volume'].cumsum()
    return cumulative_tpv / cumulative_volume

class AdvancedStrategy:
    def __init__(self, symbol):
        self.stock = Stock(symbol)
        self.symbol = self.stock.symbol
        self.model_path = f"models/{self.stock.symbol}_lstm.h5"
        self.data = pd.read_csv(self.stock.file, parse_dates=['date'])
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Fit the scaler on the full dataset during initialization
        self._fit_scaler()

    def _fit_scaler(self):
        """Fit the scaler on the full dataset"""
        df = self._calculate_features(self.data)
        if not df.empty:
            features = ['close', 'FRAMA', 'MACD', 'RSI', 'BBL_20_2.0', 'BBU_20_2.0', 'fractal_vol', 'VWAP', 'previous', 'diff', 'numtrans', 'tradedshares', 'volume']
            features = [f for f in features if f in df.columns]
            self.scaler.fit(df[features])
            print(f"Scaler fitted for {self.symbol}")
        else:
            print(f"Error: No valid data to fit scaler for {self.symbol}")

    def _calculate_features(self, df=None):
        """Feature engineering with validation"""
        if df is None:
            df = self.data.copy()
        else:
            df = df.copy()
        
        if df.empty:
            return pd.DataFrame()
                
        # Ensure required columns exist
        required_cols = ['date', 'previous', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            print(f"Missing required columns in data for {self.symbol}")
            return pd.DataFrame()
        
        # Calculate indicators
        try:
            df['FRAMA'] = custom_frama(df, PARAMS['frama_period'])
            df['MACD'] = ta.macd(df['close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
            df['RSI'] = ta.rsi(df['close'], length=14)
            bb = ta.bbands(df['close'], length=20)
            df = pd.concat([df, bb], axis=1)
            df['VWAP'] = calculate_vwap(df)
            df['fractal_vol'] = (df['high'].rolling(5).max() - df['low'].rolling(5).min()) / df['close'].rolling(5).mean()
            df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
            return df
        except Exception as e:
            print(f"Error calculating features for {self.symbol}: {str(e)}")
            return pd.DataFrame()

    def _build_model(self, input_shape):
        """LSTM model architecture"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_model(self):
        """Model training with validation"""
        df = self._calculate_features()
        if df.empty:
            print(f"Skipping training for {self.symbol} due to insufficient data")
            print(df)
            return
        features = ['close', 'FRAMA', 'MACD', 'RSI', 'BBL_20_2.0', 'BBU_20_2.0', 'fractal_vol', 'VWAP', 'previous', 'diff', 'numtrans', 'tradedshares', 'volume']
        features = [f for f in features if f in df.columns]
        
        try:
            scaled_data = self.scaler.fit_transform(df[features])
            X, y = [], []
            
            for i in range(PARAMS['lstm_lookback'], len(df)):
                X.append(scaled_data[i-PARAMS['lstm_lookback']:i])
                y.append(df['target'].iloc[i])
                
            X, y = np.array(X), np.array(y)
            if len(X) == 0:
                print(f"No valid training samples for {self.symbol}")
                return
                
            split = int(0.8 * len(X))
            
            if os.path.exists(self.model_path):
                model = load_model(self.model_path)
            else:
                model = self._build_model((X.shape[1], X.shape[2]))
                
            model.fit(
                X[:split], y[:split],
                epochs=50,
                batch_size=32,
                validation_data=(X[split:], y[split:]),
                verbose=0
            )
            model.save(self.model_path)
            print(f"Successfully trained model for {self.symbol}")
        except Exception as e:
            print(f"Error training model for {self.symbol}: {str(e)}")

    def generate_signal(self, df=None):
        """Generate trading signal with validation"""
        if df is None:
            df = self.data

        # Ensure the input DataFrame has the required features
        df = self._calculate_features(df)
        
        if df.empty or len(df) < PARAMS['lstm_lookback'] + 5:
            return 0, 0.0
            
        try:
            features = ['close', 'FRAMA', 'MACD', 'RSI', 'BBL_20_2.0', 'BBU_20_2.0', 'fractal_vol', 'VWAP', 'previous', 'diff', 'numtrans', 'tradedshares', 'volume']
            features = [f for f in features if f in df.columns]
            
            # Use the pre-fitted scaler
            scaled_data = self.scaler.transform(df[features].tail(PARAMS['lstm_lookback']))
            X = np.array([scaled_data])
            
            model = load_model(self.model_path)
            pred = model.predict(X)[0][0]
            
            # Combine with technical signals
            current = df.iloc[-1]
            frama_signal = 1 if current['close'] > current['FRAMA'] else -1
            bb_signal = 1 if current['close'] < current['BBL_20_2.0'] else -1 if current['close'] > current['BBU_20_2.0'] else 0
            rsi_signal = 1 if current['RSI'] < 30 else -1 if current['RSI'] > 70 else 0
            
            final_signal = (pred * 0.6) + (frama_signal * 0.2) + (bb_signal * 0.1) + (rsi_signal * 0.1)
            confidence = abs(final_signal)
            
            if final_signal > 0.8:
                return 1, confidence
            elif final_signal < -0.8:
                return -1, confidence
            return 0, confidence
        except Exception as e:
            print(f"Error generating signal for {self.symbol}: {str(e)}")
            return 0, 0.0

    def trailing_stop(self, entry_price, current_price):
        """Dynamic trailing stop based on fractal volatility"""
        volatility = self.data['close'].rolling(5).std().iloc[-1]
        trail_percent = 0.2 * (volatility / self.data['close'].iloc[-1])
        return 1 if current_price < entry_price * (1 - trail_percent) else 0

def main():
    if len(sys.argv) > 1 and sys.argv[1] != 'all':
        symbols = [sys.argv[1]]
    else:
        symbols = [x for x in COMPANIES if Stock(x).trade]
    
    strategy_results = []
    
    for symbol in symbols:
        strategy = AdvancedStrategy(symbol)
        
        # Retrain model periodically
        last_train = 100  # Implement date-based check in real usage
        if last_train > PARAMS['train_interval']:
            strategy.train_model()
        
        signal = strategy.generate_signal()
        strategy_results.append({
            'symbol': symbol,
            'signal': signal[0],
            'confidence': signal[1],
            'entry': strategy.data['close'].iloc[-1]
        })
    
    # Generate output compatible with existing system
    print("Advanced Strategy Signals:")
    print("Symbol\tSignal\tConfidence\tEntry Price")
    for result in strategy_results:
        print(f"{result['symbol']}\t{result['signal']:.2f}\t{result['confidence']:.2f}\t{result['entry']:.2f}")

if __name__ == "__main__":
    main()