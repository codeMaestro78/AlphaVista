import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
import yfinance as yf
import plotly.graph_objects as go
from scipy import stats

class StockML:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        
    def prepare_data(self, data, lookback=30):
        """Prepare data for ML models with technical indicators"""
        df = data.copy()
        df = df[['Close']]
        df['Close_norm'] = self.scaler.fit_transform(df[['Close']])
        X, y = [], []
        for i in range(lookback, len(df)-1):
            X.append(df['Close_norm'].values[i-lookback:i])
            y.append(df['Close_norm'].values[i+1])  # Predict next normalized price
        return np.array(X), np.array(y), df['Close'].values[lookback+1:]
    
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        return exp1 - exp2
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def prepare_price_data(self, data, lookback=30):
        df = data.copy()
        df = df[['Close']]
        df['Close_norm'] = self.scaler.fit_transform(df[['Close']])
        X, y = [], []
        for i in range(lookback, len(df)-1):
            X.append(df['Close_norm'].values[i-lookback:i])
            y.append(df['Close_norm'].values[i+1])  # Predict next normalized price
        return np.array(X), np.array(y), df['Close'].values[lookback+1:]

    def build_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(units=64, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(units=32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            Dense(units=16, activation='relu'),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict_prices(self, symbol, period='1y'):
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        X, y, actual_prices = self.prepare_price_data(hist)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = self.build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
        self.model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        pred_norm = self.model.predict(X)
        pred_prices = self.scaler.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
        return pred_prices, actual_prices
    
    def detect_anomalies(self, data):
        """Enhanced anomaly detection with multiple features"""
        df = data.copy()
        
        # Calculate features
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        
        # Detect anomalies
        features = df[['Returns', 'Volatility', 'Volume_Change', 'Price_Range']].dropna()
        anomalies = self.anomaly_detector.fit_predict(features)
        
        df['Anomaly'] = 0
        df.loc[features.index, 'Anomaly'] = anomalies
        
        return df
    
    def assess_risk(self, symbol, period='1y'):
        """Enhanced risk assessment with more metrics"""
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        returns = hist['Close'].pct_change().dropna()
        
        # Calculate risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (returns.mean() * 252) / volatility
        sortino_ratio = (returns.mean() * 252) / (returns[returns < 0].std() * np.sqrt(252))
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        max_drawdown = (hist['Close'] / hist['Close'].cummax() - 1).min()
        beta = self._calculate_beta(returns, period)
        
        return {
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'VaR (95%)': var_95,
            'CVaR (95%)': cvar_95,
            'Max Drawdown': max_drawdown,
            'Beta': beta
        }
    
    def _calculate_beta(self, returns, period='1y'):
        """Calculate beta against market (S&P 500)"""
        spy = yf.download('^GSPC', period=period)['Close'].pct_change().dropna()
        
        # Convert to Series if they're not already
        if isinstance(returns, pd.DataFrame):
            returns = returns.squeeze()
        if isinstance(spy, pd.DataFrame):
            spy = spy.squeeze()
            
        # Convert timezone-aware indices to naive
        if returns.index.tz is not None:
            returns.index = returns.index.tz_localize(None)
        if spy.index.tz is not None:
            spy.index = spy.index.tz_localize(None)
            
        # Create DataFrame with aligned dates
        df = pd.DataFrame({
            'returns': returns,
            'spy': spy
        }).dropna()
        
        # Calculate covariance and variance
        covariance = np.cov(df['returns'], df['spy'])[0][1]
        variance = np.var(df['spy'])
        
        return covariance / variance if variance != 0 else 0
    
    def optimize_portfolio(self, symbols, period='1y'):
        """Enhanced portfolio optimization with risk constraints"""
        data = pd.DataFrame()
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            data[symbol] = hist['Close']
        
        returns = data.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Generate portfolios with risk constraints
        num_portfolios = 1000
        results = np.zeros((num_portfolios, len(symbols) + 3))
        
        for i in range(num_portfolios):
            weights = np.random.random(len(symbols))
            weights = weights / np.sum(weights)
            
            portfolio_return = np.sum(mean_returns * weights) * 252
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            portfolio_sharpe = (portfolio_return - 0.02) / portfolio_std  # Assuming 2% risk-free rate
            
            results[i, 0] = portfolio_std
            results[i, 1] = portfolio_return
            results[i, 2] = portfolio_sharpe
            results[i, 3:] = weights
        
        # Find optimal portfolio (highest Sharpe ratio)
        optimal_idx = np.argmax(results[:, 2])
        optimal_weights = results[optimal_idx, 3:]
        
        return dict(zip(symbols, optimal_weights))
    
    def plot_predictions(self, predictions, actual, symbol):
        """Enhanced prediction visualization"""
        fig = go.Figure()
        
        # Plot actual prices
        fig.add_trace(go.Scatter(
            y=actual,
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        
        # Plot predictions
        fig.add_trace(go.Scatter(
            y=predictions,
            name='Predicted',
            line=dict(color='red', width=2)
        ))
        
        # Add confidence interval
        std_dev = np.std(actual - predictions)
        fig.add_trace(go.Scatter(
            y=predictions + 2*std_dev,
            name='Upper Bound',
            line=dict(color='red', width=1, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            y=predictions - 2*std_dev,
            name='Lower Bound',
            line=dict(color='red', width=1, dash='dash'),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title=f'Price Predictions for {symbol}',
            xaxis_title='Time',
            yaxis_title='Price',
            template='plotly_white',
            showlegend=True
        )
        
        return fig 

    def prepare_direction_data(self, data, lookback=30):
        df = data.copy()
        df = df[['Close']]
        df['Close_norm'] = self.scaler.fit_transform(df[['Close']])
        df['Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        X, y = [], []
        for i in range(lookback, len(df)):
            X.append(df['Close_norm'].values[i-lookback:i])
            y.append(df['Direction'].values[i])
        return np.array(X), np.array(y)

    def build_lstm_classifier(self, input_shape):
        model = Sequential([
            LSTM(units=64, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(units=32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            Dense(units=16, activation='relu'),
            Dense(units=1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def classify_direction(self, symbol, period='1y'):
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        X, y = self.prepare_direction_data(hist)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = self.build_lstm_classifier(input_shape=(X.shape[1], X.shape[2]))
        model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        return y_test, y_pred 