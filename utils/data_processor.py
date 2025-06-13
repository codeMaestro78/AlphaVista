import pandas as pd
import numpy as np
from datetime import datetime

class DataProcessor:
    """Handle data processing for stock analysis"""
    
    def __init__(self):
        # Indian stock exchanges and their suffixes
        self.indian_exchanges = {
            '.NS': 'NSE',  # National Stock Exchange
            '.BO': 'BSE',  # Bombay Stock Exchange
            '.AS': 'NSE',  # Additional NSE suffix
        }
        
        # Common Indian stock patterns
        self.indian_patterns = [
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 
            'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LT', 'ASIANPAINT',
            'MARUTI', 'TITAN', 'NESTLEIND', 'ULTRACEMCO', 'BAJFINANCE'
        ]
    
    def is_indian_stock(self, symbol):
        """Check if the stock symbol is from Indian market"""
        # Check for Indian exchange suffixes
        for suffix in self.indian_exchanges.keys():
            if symbol.endswith(suffix):
                return True
        
        # Check for common Indian stock patterns
        symbol_base = symbol.split('.')[0]  # Remove exchange suffix
        if symbol_base in self.indian_patterns:
            return True
            
        return False
    
    def get_currency_symbol(self, symbol, info=None):
        """Get appropriate currency symbol based on stock market"""
        if self.is_indian_stock(symbol):
            return '₹'
        
        # Check currency from stock info if available
        if info and 'currency' in info:
            currency = info['currency'].upper()
            currency_symbols = {
                'USD': '$',
                'INR': '₹',
                'EUR': '€',
                'GBP': '£',
                'JPY': '¥',
                'CAD': 'C$',
                'AUD': 'A$'
            }
            return currency_symbols.get(currency, '$')
        
        # Default to USD
        return '$'
    
    def process_stock_data(self, hist_data, info, symbol):
        """Process stock historical data and company info"""
        try:
            if hist_data is None or hist_data.empty:
                return {}
                
            # Determine currency symbol
            currency_symbol = self.get_currency_symbol(symbol, info)
            
            # Calculate additional metrics
            processed_data = {
                'symbol': symbol,
                'currency_symbol': currency_symbol,
                'is_indian_stock': self.is_indian_stock(symbol),
                'data_points': len(hist_data),
                'date_range': {
                    'start': hist_data.index[0].strftime('%Y-%m-%d'),
                    'end': hist_data.index[-1].strftime('%Y-%m-%d')
                }
            }
            
            # Calculate moving averages
            if len(hist_data) >= 20:
                hist_data['MA_20'] = hist_data['Close'].rolling(window=20).mean()
            if len(hist_data) >= 50:
                hist_data['MA_50'] = hist_data['Close'].rolling(window=50).mean()
            
            # Calculate daily returns
            hist_data['Daily_Return'] = hist_data['Close'].pct_change()
            
            # Calculate volatility (30-day rolling standard deviation)
            if len(hist_data) >= 30:
                hist_data['Volatility'] = hist_data['Daily_Return'].rolling(window=30).std()
            
            # Price performance metrics
            if len(hist_data) > 1:
                total_return = (hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[0] - 1) * 100
                processed_data['total_return'] = total_return
                
                # Calculate max drawdown
                cumulative_returns = (1 + hist_data['Daily_Return'].fillna(0)).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                processed_data['max_drawdown'] = drawdown.min() * 100
            
            # Risk metrics
            if 'Daily_Return' in hist_data.columns:
                daily_returns = hist_data['Daily_Return'].dropna()
                if len(daily_returns) > 0:
                    processed_data['avg_daily_return'] = daily_returns.mean() * 100
                    processed_data['volatility'] = daily_returns.std() * 100
                    
                    # Sharpe ratio (assuming risk-free rate of 2%)
                    risk_free_rate = 0.02 / 252  # Daily risk-free rate
                    excess_returns = daily_returns - risk_free_rate
                    if daily_returns.std() != 0:
                        processed_data['sharpe_ratio'] = excess_returns.mean() / daily_returns.std() * np.sqrt(252)
            
            return processed_data
            
        except Exception as e:
            print(f"Error processing stock data: {str(e)}")
            return {}
    
    def process_csv_data(self, df):
        """Process uploaded CSV data"""
        try:
            processed_data = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns': df.columns.tolist(),
                'data_types': df.dtypes.to_dict()
            }
            
            # Try to identify date column
            date_columns = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp']):
                    date_columns.append(col)
            
            # Try to identify price columns
            price_columns = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['price', 'close', 'open', 'high', 'low']):
                    price_columns.append(col)
            
            processed_data['date_columns'] = date_columns
            processed_data['price_columns'] = price_columns
            
            # Basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                processed_data['statistics'] = df[numeric_cols].describe().to_dict()
            
            # Prepare data for charting if possible
            if date_columns and price_columns:
                chart_data = df.copy()
                
                # Try to parse date column
                if date_columns:
                    date_col = date_columns[0]
                    try:
                        chart_data[date_col] = pd.to_datetime(chart_data[date_col])
                        chart_data = chart_data.sort_values(date_col)
                        processed_data['chart_data'] = chart_data
                    except:
                        pass
            
            # Check for missing values
            processed_data['missing_values'] = df.isnull().sum().to_dict()
            
            # Sample data for preview
            processed_data['sample'] = df.head(10).to_dict('records')
            
            return processed_data
            
        except Exception as e:
            print(f"Error processing CSV data: {str(e)}")
            return {}
    
    def validate_stock_symbol(self, symbol):
        """Validate stock symbol format"""
        if not symbol:
            return False, "Symbol cannot be empty"
        
        if len(symbol) > 10:
            return False, "Symbol too long"
        
        if not symbol.replace('.', '').replace('-', '').isalnum():
            return False, "Symbol contains invalid characters"
        
        return True, "Valid symbol"
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators"""
        try:
            if data is None or data.empty:
                return data
                
            # RSI (Relative Strength Index)
            def calculate_rsi(prices, window=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
            
            if 'Close' in data.columns and len(data) >= 14:
                data['RSI'] = calculate_rsi(data['Close'])
            
            # MACD (Moving Average Convergence Divergence)
            if 'Close' in data.columns and len(data) >= 26:
                exp1 = data['Close'].ewm(span=12).mean()
                exp2 = data['Close'].ewm(span=26).mean()
                data['MACD'] = exp1 - exp2
                data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
                data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            # Bollinger Bands
            if 'Close' in data.columns and len(data) >= 20:
                data['BB_Middle'] = data['Close'].rolling(window=20).mean()
                std = data['Close'].rolling(window=20).std()
                data['BB_Upper'] = data['BB_Middle'] + (std * 2)
                data['BB_Lower'] = data['BB_Middle'] - (std * 2)
            
            return data
            
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            return data
