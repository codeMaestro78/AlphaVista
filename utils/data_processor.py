import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class ProcessorConfig:
    """Configuration settings for DataProcessor"""
    ma_windows: list = None  # Moving average windows
    volatility_window: int = 30
    risk_free_rate: float = 0.02
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0

    def __post_init__(self):
        self.ma_windows = self.ma_windows or [20, 50]

class DataProcessor:
    """Handle data processing for stock analysis with enhanced features and validation
    
    Attributes:
        indian_exchanges (dict): Mapping of Indian exchange suffixes to names
        indian_patterns (list): Common Indian stock symbols
        config (ProcessorConfig): Configuration settings for processing
    """
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        """Initialize DataProcessor with optional configuration
        
        Args:
            config (ProcessorConfig, optional): Configuration settings
        """
        self.indian_exchanges = {
            '.NS': 'NSE',
            '.BO': 'BSE',
            '.AS': 'NSE',
        }
        self.indian_patterns = [
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
            'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LT', 'ASIANPAINT',
            'MARUTI', 'TITAN', 'NESTLEIND', 'ULTRACEMCO', 'BAJFINANCE'
        ]
        self.config = config or ProcessorConfig()

    def is_indian_stock(self, symbol: str) -> bool:
ctorConfig = None) -> Dict[str, Union[str, bool, int, dict]]:
        """Process stock historical data and company info with comprehensive metrics
        
        Args:
            hist_data (pd.DataFrame): Historical stock data
            info (dict): Company information dictionary
            symbol (str): Stock symbol
            
        Returns:
            dict: Processed stock data with calculated metrics
        """
        try:
            # Input validation
            if not isinstance(hist_data, pd.DataFrame):
                raise ValueError("hist_data must be a pandas DataFrame")
            if hist_data.empty:
                return {}
                
            # Validate symbol
            is_valid, message = self.validate_stock_symbol(symbol)
            if not is_valid:
                raise ValueError(f"Invalid symbol: {message}")
            
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
            
            # Calculate moving averages based on config
            for window in self.config.ma_windows:
                if len(hist_data) >= window:
                    hist_data[f'MA_{window}'] = hist_data['Close'].rolling(window=window).mean()
            
            # Calculate daily returns
            hist_data['Daily_Return'] = hist_data['Close'].pct_change()
            
            # Calculate volatility
            if len(hist_data) >= self.config.volatility_window:
                hist_data['Volatility'] = hist_data['Daily_Return'].rolling(
                    window=self.config.volatility_window).std()
            
            # Price performance metrics
            if len(hist_data) > 1:
                total_return = (hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[0] - 1) * 100
                processed_data['total_return'] = round(total_return, 2)
                
                # Calculate max drawdown
                cumulative_returns = (1 + hist_data['Daily_Return'].fillna(0)).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                processed_data['max_drawdown'] = round(drawdown.min() * 100, 2)
            
            # Risk metrics
            if 'Daily_Return' in hist_data.columns:
                daily_returns = hist_data['Daily_Return'].dropna()
                if len(daily_returns) > 0:
                    processed_data['avg_daily_return'] = round(daily_returns.mean() * 100, 4)
                    processed_data['volatility'] = round(daily_returns.std() * 100, 4)
                    
                    # Sharpe ratio
                    risk_free_rate = self.config.risk_free_rate / 252
                    excess_returns = daily_returns - risk_free_rate
                    if daily_returns.std() != 0:
                        processed_data['sharpe_ratio'] = round(
                            excess_returns.mean() / daily_returns.std() * np.sqrt(252), 2)
            
            # Additional performance metrics
            if len(hist_data) > 0:
                processed_data['price_stats'] = {
                    'latest_price': round(hist_data['Close'].iloc[-1], 2),
                    'high_52week': round(hist_data['CloseAnomaly'][0]
                }
            
            return processed_data
            
        except Exception as e:
            print(f"Error processing stock data: {str(e)}")
            return {}

    def process_csv_data(self, df: pd.DataFrame) -> Dict[str, Union[int, list, dict]]:
        """Process uploaded CSV data with enhanced analysis
        
        Args:
            df (pd.DataFrame): Input CSV data
            
        Returns:
            dict: Processed data with statistics and metadata
        """
        try:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
                
            processed_data = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns': df.columns.tolist(),
                'data_types': df.dtypes.to_dict()
            }
            
            # Identify key columns
            date_columns = []
            price_columns = []
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['date', 'time', 'timestamp']):
                    date_columns.append(col)
                if any(keyword in col_lower for keyword in ['price', 'close', 'open', 'high', 'low']):
                    price_columns.append(col)
            
            processed_data['date_columns'] = date_columns
            processed_data['price_columns'] = price_columns
            
            # Advanced statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats = df[numeric_cols].describe()
                processed_data['statistics'] = stats.to_dict()
                processed_data['correlation_matrix'] = df[numeric_cols].corr().to_dict()
            
            # Process date and price data
            if date_columns and price_columns:
                chart_data = df.copy()
                date_col = date_columns[0]
                try:
                    chart_data[date_col] = pd.to_datetime(chart_data[date_col])
                    chart_data = chart_data.sort_values(date_col)
                    processed_data['chart_data'] = chart_data.to_dict('records')
                    
                    # Calculate returns if price data available
                    if 'Close' in price_columns:
                        chart_data['Daily_Return'] = chart_data['Close'].pct_change()
                        processed_data['return_stats'] = {
                            'mean': chart_data['Daily_Return'].mean(),
                            'std': chart_data['Daily_Return'].std()
                        }
                except Exception as e:
                    print(f"Error processing date column: {str(e)}")
            
            # Enhanced missing value analysis
            missing = df.isnull().sum()
            processed_data['missing_values'] = {
                col: {
                    'count': int(count),
                    'percentage': round((count / len(df)) * 100, 2)
                } for col, count in missing.items()
            }
            
            # Data quality metrics
            processed_data['data_quality'] = {
                'completeness': round((1 - missing.sum() / (len(df) * len(df.columns))) * 100, 2),
                'numeric_columns': len(numeric_cols),
                'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns)
            }
            
            # Sample data
            processed_data['sample'] = df.head(10).to_dict('records')
            
            return processed_data
            
        except Exception as e:
            print(f"Error processing CSV data: {str(e)}")
            return {}

    def validate_stock_symbol(self, symbol: str) -> Tuple[bool, str]:
        """Validate stock symbol format with strict checks
        
        Args:
            symbol (str): Stock symbol to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        if not isinstance(symbol, str):
            return False, "Symbol must be a string"
        
        if not symbol:
            return False, "Symbol cannot be empty"
        
        if len(symbol) > 15:  # Extended length check
            return False, "Symbol too long (max 15 characters)"
        
        if not symbol.replace('.', '').replace('-', '').isalnum():
            return False, "Symbol contains invalid characters"
        
        # Additional validation for Indian stocks
        if any(symbol.endswith(suffix) for suffix in self.indian_exchanges):
            base_symbol = symbol.split('.')[0]
            if not base_symbol.isalpha():
                return False, "Base symbol must contain only letters"
        
        return True, "Valid symbol"

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators with configurable parameters
        
        Args:
            data (pd.DataFrame): Input price data
            
        perspective: I aim to improve the provided Python code for the DataProcessor class by adding better error handling, documentation, and additional features while maintaining the existing functionality. Here's the enhanced version:

<xaiArtifact artifact_id="65dda79d-caac-4d5b-9f89-0294c6804527" artifact_version_id="f210062d-0295-4ea3-b435-4ef7de3e3ab0" title="data_processor.py" contentType="text/python">
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class ProcessorConfig:
    """Configuration settings for DataProcessor"""
    ma_windows: list = None  # Moving average windows
    volatility_window: int = 30
    risk_free_rate: float = 0.02
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0

    def __post_init__(self):
        self.ma_windows = self.ma_windows or [20, 50]

class DataProcessor:
    """Handle data processing for stock analysis with enhanced features and validation
    
    Attributes:
        indian_exchanges (dict): Mapping of Indian exchange suffixes to names
        indian_patterns (list): Common Indian stock symbols
        config (ProcessorConfig): Configuration settings for processing
    """
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        """Initialize DataProcessor with optional configuration
        
        Args:
            config (ProcessorConfig, optional): Configuration settings
        """
        self.indian_exchanges = {
            '.NS': 'NSE',
            '.BO': 'BSE',
            '.AS': 'NSE',
        }
        self.indian_patterns = [
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
            'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LT', 'ASIANPAINT',
            'MARUTI', 'TITAN', 'NESTLEIND', 'ULTRACEMCO', 'BAJFINANCE'
        ]
        self.config = config or ProcessorConfig()

    def is_indian_stock(self, symbol: str) -> bool:
        """Check if the stock symbol is from Indian market
        
        Args:
            symbol (str): Stock symbol to check
            
        Returns:
            bool: True if Indian stock, False otherwise
        """
        if not isinstance(symbol, str):
            return False
            
        for suffix in self.indian_exchanges.keys():
            if symbol.endswith(suffix):
                return True
        
        symbol_base = symbol.split('.')[0]
        return symbol_base in self.indian_patterns

    def get_currency_symbol(self, symbol: str, info: Optional[dict] = None) -> str:
        """Get appropriate currency symbol based on stock market
        
        Args:
            symbol (str): Stock symbol
            info (dict, optional): Company information dictionary
            
        Returns:
            str: Appropriate currency symbol
        """
        if self.is_indian_stock(symbol):
            return '₹'
        
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
        
        return '$'

    def process_stock_data(self, hist_data: pd.DataFrame, info: dict, symbol: str, 
                         config: Optional[ProcessorConfig] = None) -> Dict[str, Union[str, bool, int, dict]]:
        """Process stock historical data and company info with comprehensive metrics
        
        Args:
            hist_data (pd.DataFrame): Historical stock data
            info (dict): Company information dictionary
            symbol (str): Stock symbol
            config (ProcessorConfig, optional): Configuration settings
            
        Returns:
            dict: Processed stock data with calculated metrics
        """
        try:
            # Input validation
            if not isinstance(hist_data, pd.DataFrame):
                raise ValueError("hist_data must be a pandas DataFrame")
            if hist_data.empty:
                return {}
                
            # Validate symbol
            is_valid, message = self.validate_stock_symbol(symbol)
            if not is_valid:
                raise ValueError(f"Invalid symbol: {message}")
            
            # Use provided config or class config
            config = config or self.config
            
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
            
            # Calculate moving averages based on config
            for window in config.ma_windows:
                if len(hist_data) >= window:
                    hist_data[f'MA_{window}'] = hist_data['Close'].rolling(window=window).mean()
            
            # Calculate daily returns
            hist_data['Daily_Return'] = hist_data['Close'].pct_change()
            
            # Calculate volatility
            if len(hist_data) >= config.volatility_window:
                hist_data['Volatility'] = hist_data['Daily_Return'].rolling(
                    window=config.volatility_window).std()
            
            # Price performance metrics
            if len(hist_data) > 1:
                total_return = (hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[0] - 1) * 100
                processed_data['total_return'] = round(total_return, 2)
                
                # Calculate max drawdown
                cumulative_returns = (1 + hist_data['Daily_Return'].fillna(0)).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                processed_data['max_drawdown'] = round(drawdown.min() * 100, 2)
            
            # Risk metrics
            if 'Daily_Return' in hist_data.columns:
                daily_returns = hist_data['Daily_Return'].dropna()
                if len(daily_returns) > 0:
                    processed_data['avg_daily_return'] = round(daily_returns.mean() * 100, 4)
                    processed_data['volatility'] = round(daily_returns.std() * 100, 4)
                    
                    # Sharpe ratio
                    risk_free_rate = config.risk_free_rate / 252
                    excess_returns = daily_returns - risk_free_rate
                    if daily_returns.std() != 0:
                        processed_data['sharpe_ratio'] = round(
                            excess_returns.mean() / daily_returns.std() * np.sqrt(252), 2)
            
            # Additional performance metrics
            if len(hist_data) > 0:
                processed_data['price_stats'] = {
                    'latest_price': round(hist_data['Close'].iloc[-1], 2),
                    'high_52week': round(hist_data['High'].rolling(window=252).max().iloc[-1], 2),
                    'low_52week': round(hist_data['Low'].rolling(window=252).min().iloc[-1], 2)
                }
            
            # Calculate technical indicators
            hist_data = self.calculate_technical_indicators(hist_data, config)
            processed_data['technical_indicators'] = {
                'rsi': hist_data['RSI'].iloc[-1] if 'RSI' in hist_data.columns else None,
                'macd': hist_data['MACD'].iloc[-1] if 'MACD' in hist_data.columns else None,
                'macd_signal': hist_data['MACD_Signal'].iloc[-1] if 'MACD_Signal' in hist_data.columns else None
            }
            
            return processed_data
            
        except Exception as e:
            print(f"Error processing stock data: {str(e)}")
            return {}

    def process_csv_data(self, df: pd.DataFrame) -> Dict[str, Union[int, list, dict]]:
        """Process uploaded CSV data with enhanced analysis
        
        Args:
            df (pd.DataFrame): Input CSV data
            
        Returns:
            dict: Processed data with statistics and metadata
        """
        try:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
                
            processed_data = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns': df.columns.tolist(),
                'data_types': df.dtypes.to_dict()
            }
            
            # Identify key columns
            date_columns = []
            price_columns = []
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['date', 'time', 'timestamp']):
                    date_columns.append(col)
                if any(keyword in col_lower for keyword in ['price', 'close', 'open', 'high', 'low']):
                    price_columns.append(col)
            
            processed_data['date_columns'] = date_columns
            processed_data['price_columns'] = price_columns
            
            # Advanced statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats = df[numeric_cols].describe()
                processed_data['statistics'] = stats.to_dict()
                processed_data['correlation_matrix'] = df[numeric_cols].corr().to_dict()
            
            # Process date and price data
            if date_columns and price_columns:
                chart_data = df.copy()
                date_col = date_columns[0]
                try:
                    chart_data[date_col] = pd.to_datetime(chart_data[date_col])
                    chart_data = chart_data.sort_values(date_col)
                    processed_data['chart_data'] = chart_data.to_dict('records')
                    
                    # Calculate returns if price data available
                    if 'Close' in price_columns:
                        chart_data['Daily_Return'] = chart_data['Close'].pct_change()
                        processed_data['return_stats'] = {
                            'mean': round(chart_data['Daily_Return'].mean(), 4),
                            'std': round(chart_data['Daily_Return'].std(), 4)
                        }
                except Exception as e:
                    print(f"Error processing date column: {str(e)}")
            
            # Enhanced missing value analysis
            missing = df.isnull().sum()
            processed_data['missing_values'] = {
                col: {
                    'count': int(count),
                    'percentage': round((count / len(df)) * 100, 2)
                } for col, count in missing.items()
            }
            
            # Data quality metrics
            processed_data['data_quality'] = {
                'completeness': round((1 - missing.sum() / (len(df) * len(df.columns))) * 100, 2),
                'numeric_columns': len(numeric_cols),
                'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns)
            }
            
            # Sample data
            processed_data['sample'] = df.head(10).to_dict('records')
            
            # Calculate technical indicators if price data available
            if price_columns:
                chart_data = self.calculate_technical_indicators(chart_data)
                processed_data['technical_indicators'] = {
                    'rsi': chart_data['RSI'].iloc[-1] if 'RSI' in chart_data.columns else None,
                    'macd': chart_data['MACD'].iloc[-1] if 'MACD' in chart_data.columns else None,
                    'macd_signal': chart_data['MACD_Signal'].iloc[-1] if 'MACD_Signal' in chart_data.columns else None
                }
            
            return processed_data
            
        except Exception as e:
            print(f"Error processing CSV data: {str(e)}")
            return {}

    def validate_stock_symbol(self, symbol: str) -> Tuple[bool, str]:
        """Validate stock symbol format with strict checks
        
        Args:
            symbol (str): Stock symbol to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        if not isinstance(symbol, str):
            return False, "Symbol must be a string"
        
        if not symbol:
            return False, "Symbol cannot be empty"
        
        if len(symbol) > 15:  # Extended length check
            return False, "Symbol too long (max 15 characters)"
        
        if not symbol.replace('.', '').replace('-', '').isalnum():
            return False, "Symbol contains invalid characters"
        
        # Additional validation for Indian stocks
        if any(symbol.endswith(suffix) for suffix in self.indian_exchanges):
            base_symbol = symbol.split('.')[0]
            if not base_symbol.isalpha():
                return False, "Base symbol must contain only letters"
        
        return True, "Valid symbol"

    def calculate_technical_indicators(self, data: pd.DataFrame, 
                                      config: Optional[ProcessorConfig] = None) -> pd.DataFrame:
        """Calculate technical indicators with configurable parameters
        
        Args:
            data (pd.DataFrame): Input price data
            config (ProcessorConfig, optional): Configuration settings
            
        Returns:
            pd.DataFrame: Data with added technical indicators
        """
        try:
            if not isinstance(data, pd.DataFrame) or data.empty:
                return data
                
            data = data.copy()  # Create a copy to avoid modifying input
            config = config or self.config
            
            # RSI (Relative Strength Index)
            if 'Close' in data.columns and len(data) >= config.rsi_window:
                delta = data['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=config.rsi_window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=config.rsi_window).mean()
                rs = gain / loss
                data['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD (Moving Average Convergence Divergence)
            if 'Close' in data.columns and len(data) >= config.macd_slow:
                exp1 = data['Close'].ewm(span=config.macd_fast).mean()
                exp2 = data['Close'].ewm(span=config.macd_slow).mean()
                data['MACD'] = exp1 - exp2
                data['MACD_Signal'] = data['MACD'].ewm(span=config.macd_signal).mean()
                data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            # Bollinger Bands
            if 'Close' in data.columns and len(data) >= config.bb_window:
                data['BB_Middle'] = data['Close'].rolling(window=config.bb_window).mean()
                std = data['Close'].rolling(window=config.bb_window).std()
                data['BB_Upper'] = data['BB_Middle'] + (std * config.bb_std)
                data['BB_Lower'] = data['BB_Middle'] - (std * config.bb_std)
            
            # Additional indicators: Stochastic Oscillator
            if all(col in data.columns for col in ['Close', 'High', 'Low']) and len(data) >= 14:
                lowest_low = data['Low'].rolling(window=14).min()
                highest_high = data['High'].rolling(window=14).max()
                data['Stochastic_K'] = ((data['Close'] - lowest_low) / 
                                     (highest_high - lowest_low)) * 100
                data['Stochastic_D'] = data['Stochastic_K'].rolling(window=3).mean()
            
            # Average True Range (ATR)
            if all(col in data.columns for col in ['High', 'Low', 'Close']) and len(data) >= 14:
                tr1 = data['High'] - data['Low']
                tr2 = (data['High'] - data['Close'].shift(1)).abs()
                tr3 = (data['Low'] - data['Close'].shift(1)).abs()
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                data['ATR'] = true_range.rolling(window=14).mean()
            
            return data
            
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            return data
