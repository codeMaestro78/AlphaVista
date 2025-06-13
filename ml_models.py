import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class StockML:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = 60  # Look back 60 days
        self.anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        
    def add_technical_indicators(self, df):
        """Add comprehensive technical indicators as features"""
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price change indicators
        df['Price_change'] = df['Close'].pct_change()
        df['High_Low_ratio'] = df['High'] / df['Low']
        df['Close_Open_ratio'] = df['Close'] / df['Open']
        
        # ATR (Average True Range)
        df['ATR'] = self._calculate_atr(df)
        
        # Volatility
        df['Volatility'] = df['Price_change'].rolling(window=20).std()
        
        return df
    
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
    
    def _calculate_beta(self, returns, period='1y'):
        """Calculate beta against market (S&P 500)"""
        try:
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
        except:
            return 0
    
    def prepare_enhanced_data(self, hist):
        """Prepare data with multiple features"""
        # Add technical indicators
        df = self.add_technical_indicators(hist.copy())
        
        # Select features for training
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'RSI', 'MACD', 'MACD_signal',
            'BB_position', 'Volume_ratio',
            'Price_change', 'High_Low_ratio', 'Close_Open_ratio',
            'ATR', 'Volatility'
        ]
        
        # Remove rows with NaN values
        df = df.dropna()
        
        if len(df) < self.sequence_length + 30:
            raise ValueError("Insufficient data for training. Need at least 90 days of data.")
        
        # Prepare features and target
        features = df[feature_columns].values
        target = df['Close'].values
        
        # Scale features and target separately
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.scaler.fit_transform(target.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i-self.sequence_length:i])
            y.append(target_scaled[i])
        
        X, y = np.array(X), np.array(y)
        actual_prices = target[self.sequence_length:]
        
        return X, y, actual_prices, df.index[self.sequence_length:]
    
    def prepare_direction_data(self, hist):
        """Prepare data for direction classification"""
        df = self.add_technical_indicators(hist.copy())
        
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'RSI', 'MACD', 'MACD_signal',
            'BB_position', 'Volume_ratio',
            'Price_change', 'High_Low_ratio', 'Close_Open_ratio',
            'ATR', 'Volatility'
        ]
        
        # Create direction target (1 if price goes up, 0 if down)
        df['Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        
        features = df[feature_columns].values
        features_scaled = self.feature_scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i-self.sequence_length:i])
            y.append(df['Direction'].iloc[i])
        
        return np.array(X), np.array(y)
    
    def build_enhanced_model(self, input_shape, model_type='regression'):
        """Build an enhanced model with multiple layers and regularization"""
        model = Sequential([
            # First LSTM layer with dropout
            Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
                         input_shape=input_shape),
            BatchNormalization(),
            
            # Second LSTM layer
            Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
            BatchNormalization(),
            
            # Third LSTM layer
            LSTM(32, dropout=0.2, recurrent_dropout=0.2),
            BatchNormalization(),
            
            # Dense layers with dropout
            Dense(50, activation='relu'),
            Dropout(0.3),
            Dense(25, activation='relu'),
            Dropout(0.2)
        ])
        
        if model_type == 'classification':
            model.add(Dense(1, activation='sigmoid'))
            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.add(Dense(1, activation='sigmoid'))
            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def detect_anomalies(self, data):
        """Enhanced anomaly detection with multiple features"""
        df = data.copy()
        
        # Calculate features for anomaly detection
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        
        # Detect anomalies
        features = df[['Returns', 'Volatility', 'Volume_Change', 'Price_Range']].dropna()
        if len(features) > 0:
            anomalies = self.anomaly_detector.fit_predict(features)
            df['Anomaly'] = 0
            df.loc[features.index, 'Anomaly'] = anomalies
        else:
            df['Anomaly'] = 0
        
        return df
    
    def assess_risk(self, symbol, period='1y'):
        """Enhanced risk assessment with comprehensive metrics"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            returns = hist['Close'].pct_change().dropna()
            
            # Calculate risk metrics
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (returns.mean() * 252) / volatility if volatility != 0 else 0
            sortino_ratio = (returns.mean() * 252) / (returns[returns < 0].std() * np.sqrt(252)) if len(returns[returns < 0]) > 0 else 0
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
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
        except Exception as e:
            return {'error': str(e)}
    
    def optimize_portfolio(self, symbols, period='1y'):
        """Enhanced portfolio optimization with risk constraints"""
        try:
            data = pd.DataFrame()
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                if not hist.empty:
                    data[symbol] = hist['Close']
            
            if data.empty:
                return {}
            
            returns = data.pct_change().dropna()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Generate portfolios with risk constraints
            num_portfolios = 1000
            results = np.zeros((num_portfolios, len(data.columns) + 3))
            
            for i in range(num_portfolios):
                weights = np.random.random(len(data.columns))
                weights = weights / np.sum(weights)
                
                portfolio_return = np.sum(mean_returns * weights) * 252
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
                portfolio_sharpe = (portfolio_return - 0.02) / portfolio_std if portfolio_std != 0 else 0
                
                results[i, 0] = portfolio_std
                results[i, 1] = portfolio_return
                results[i, 2] = portfolio_sharpe
                results[i, 3:] = weights
            
            # Find optimal portfolio (highest Sharpe ratio)
            optimal_idx = np.argmax(results[:, 2])
            optimal_weights = results[optimal_idx, 3:]
            
            return dict(zip(data.columns, optimal_weights))
        except Exception as e:
            return {'error': str(e)}
    
    def predict_prices(self, symbol, period='2y'):
        """Enhanced price prediction with better data preparation"""
        try:
            # Get stock data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            if len(hist) < 100:
                raise ValueError(f"Insufficient data for {symbol}. Need at least 100 days of historical data.")
            
            # Prepare enhanced data
            X, y, actual_prices, dates = self.prepare_enhanced_data(hist)
            
            # Split data chronologically (important for time series)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build and train model
            self.model = self.build_enhanced_model((X.shape[1], X.shape[2]), 'regression')
            
            # Callbacks for better training
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Make predictions
            train_pred = self.model.predict(X_train, verbose=0)
            test_pred = self.model.predict(X_test, verbose=0)
            
            # Inverse transform predictions
            train_pred_prices = self.scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
            test_pred_prices = self.scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
            
            # Combine predictions
            all_predictions = np.concatenate([train_pred_prices, test_pred_prices])
            
            # Future predictions (next 30 days)
            future_predictions = self.predict_future(X[-1:], days=30)
            
            return {
                'predictions': all_predictions,
                'actual_prices': actual_prices,
                'dates': dates,
                'train_size': len(train_pred_prices),
                'test_size': len(test_pred_prices),
                'future_predictions': future_predictions,
                'training_history': history.history
            }
            
        except Exception as e:
            raise Exception(f"Error in price prediction: {str(e)}")
    
    def classify_direction(self, symbol, period='2y'):
        """Predict price direction (up/down) using classification"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            X, y = self.prepare_direction_data(hist)
            
            # Split data chronologically
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build and train classification model
            model = self.build_enhanced_model((X.shape[1], X.shape[2]), 'classification')
            
            # Train model
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
            
            # Make predictions
            y_pred_prob = model.predict(X_test, verbose=0)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            
            return {
                'actual': y_test,
                'predicted': y_pred,
                'probabilities': y_pred_prob.flatten(),
                'accuracy': accuracy_score(y_test, y_pred)
            }
            
        except Exception as e:
            raise Exception(f"Error in direction classification: {str(e)}")
    
    def predict_future(self, last_sequence, days=30):
        """Predict future prices"""
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Predict next price
            next_pred = self.model.predict(current_sequence, verbose=0)
            future_predictions.append(next_pred[0, 0])
            
            # Update sequence - simplified approach
            new_row = current_sequence[0, -1:].copy()
            new_row[0, 3] = next_pred[0, 0]  # Update close price (index 3)
            
            # Shift sequence and add new prediction
            current_sequence = np.concatenate([
                current_sequence[:, 1:, :],
                new_row.reshape(1, 1, -1)
            ], axis=1)
        
        # Inverse transform future predictions
        future_prices = self.scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        ).flatten()
        
        return future_prices
    
    def plot_enhanced_predictions(self, results, symbol):
        """Create enhanced visualization with multiple subplots"""
        predictions = results['predictions']
        actual_prices = results['actual_prices']
        dates = results['dates']
        train_size = results['train_size']
        future_predictions = results['future_predictions']
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f'{symbol} - Price Prediction vs Actual',
                'Prediction Error Over Time',
                'Future Price Prediction (Next 30 Days)'
            ),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Main price chart
        fig.add_trace(
            go.Scatter(
                x=dates[:train_size],
                y=actual_prices[:train_size],
                mode='lines',
                name='Training Data (Actual)',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates[train_size:],
                y=actual_prices[train_size:],
                mode='lines',
                name='Test Data (Actual)',
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates[:train_size],
                y=predictions[:train_size],
                mode='lines',
                name='Training Predictions',
                line=dict(color='orange', width=1, dash='dot')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates[train_size:],
                y=predictions[train_size:],
                mode='lines',
                name='Test Predictions',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Error plot
        errors = actual_prices - predictions
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=errors,
                mode='lines',
                name='Prediction Error',
                line=dict(color='purple', width=1)
            ),
            row=2, col=1
        )
        
        # Add zero line for error
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Future predictions
        future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=30)
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=future_predictions,
                mode='lines+markers',
                name='Future Predictions',
                line=dict(color='magenta', width=2),
                marker=dict(size=4)
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            title_text=f"Comprehensive Stock Analysis for {symbol}",
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Error ($)", row=2, col=1)
        fig.update_yaxes(title_text="Future Price ($)", row=3, col=1)
        
        return fig
    
    def plot_anomalies(self, data_with_anomalies, symbol):
        """Plot anomalies in stock data"""
        fig = go.Figure()
        
        normal_data = data_with_anomalies[data_with_anomalies['Anomaly'] == 1]
        anomaly_data = data_with_anomalies[data_with_anomalies['Anomaly'] == -1]
        
        # Plot normal data
        fig.add_trace(go.Scatter(
            x=normal_data.index,
            y=normal_data['Close'],
            mode='lines',
            name='Normal',
            line=dict(color='blue', width=1)
        ))
        
        # Plot anomalies
        fig.add_trace(go.Scatter(
            x=anomaly_data.index,
            y=anomaly_data['Close'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=8, symbol='x')
        ))
        
        fig.update_layout(
            title=f'Anomaly Detection for {symbol}',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_white'
        )
        
        return fig
 