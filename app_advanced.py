import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentiment_analysis import fetch_news, compute_sentiment, daily_sentiment_trend
from datetime import datetime, timedelta
import io
import sys
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sentiment_analysis import fetch_news, compute_sentiment, daily_sentiment_trend
from email_utils import send_email_alert
import requests

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from data_processor import DataProcessor
from chart_generator import ChartGenerator
from ml_models import StockML

# Custom CSS for enhanced UI
def load_css():
    st.markdown("""
    <style>
        /* Overall theme */
        :root {
            --primary: #4f46e5;
            --secondary: #6366f1;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --light: #f8fafc;
            --dark: #0f172a;
            --gray: #64748b;
        }
        
        /* Main container */
        .main {
            background-color: #f1f5f9;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: var(--dark) !important;
            font-weight: 700 !important;
            margin-bottom: 1rem !important;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: var(--primary) !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 0.5rem 1.5rem !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
            border: none !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }
        
        .stButton>button:hover {
            background-color: var(--secondary) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        }
        
        /* Input fields */
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stSelectbox>div>div>div {
            border-radius: 8px !important;
            border: 1px solid #e2e8f0 !important;
            padding: 0.5rem 1rem !important;
            font-size: 1rem !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            padding: 0 1rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            background-color: #f1f5f9;
            transition: all 0.3s ease;
            color: var(--gray) !important;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--primary) !important;
            color: white !important;
            font-weight: 600;
        }
        
        /* Cards */
        .card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
            margin-bottom: 1.5rem;
            border: 1px solid #e2e8f0;
        }
        
        /* Metrics */
        .stMetric {
            background: white;
            border-radius: 12px;
            padding: 1.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-left: 4px solid var(--primary);
        }
        
        .stMetric > div {
            margin: 0 !important;
        }
        
        .stMetric > div:first-child {
            color: var(--gray) !important;
            font-size: 0.9rem !important;
        }
        
        .stMetric > div:last-child {
            color: var(--dark) !important;
            font-size: 1.5rem !important;
            font-weight: 700 !important;
        }
        
        /* Tables */
        table {
            border-radius: 8px !important;
            overflow: hidden !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        }
        
        th {
            background-color: var(--primary) !important;
            color: white !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            font-size: 0.8rem !important;
            letter-spacing: 0.05em;
        }
        
        tr:nth-child(even) {
            background-color: #f8fafc !important;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }
        
        /* Custom classes */
        .title-box {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }
        
        .info-box {
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border-left: 4px solid var(--primary);
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="StockSage Pro | Advanced Stock Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_css()

# Initialize session state
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'comparison_stocks' not in st.session_state:
    st.session_state.comparison_stocks = []
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

# Initialize utility classes
data_processor = DataProcessor()
chart_generator = ChartGenerator()

def load_local_stock_list():
    nse_path = os.path.join('data', 'nse_stocks.csv')
    nasdaq_path = os.path.join('data', 'nasdaq_stocks.csv')
    dfs = []
    
    # Process NSE stocks if the file exists
    if os.path.exists(nse_path):
        try:
            nse_df = pd.read_csv(nse_path)
            if not nse_df.empty:
                nse_df = nse_df.rename(columns={"SYMBOL": "symbol", "NAME OF COMPANY": "name"})
                if 'symbol' in nse_df.columns and 'name' in nse_df.columns:
                    dfs.append(nse_df[['symbol', 'name']])
        except Exception as e:
            st.warning(f"Error loading NSE stocks: {str(e)}")
    
    # Process NASDAQ stocks if the file exists
    if os.path.exists(nasdaq_path):
        try:
            nasdaq_df = pd.read_csv(nasdaq_path)
            if not nasdaq_df.empty:
                nasdaq_df = nasdaq_df.rename(columns={"Symbol": "symbol", "Name": "name"})
                if 'symbol' in nasdaq_df.columns and 'name' in nasdaq_df.columns:
                    dfs.append(nasdaq_df[['symbol', 'name']])
        except Exception as e:
            st.warning(f"Error loading NASDAQ stocks: {str(e)}")
    
    # Combine all dataframes if any were loaded successfully
    if dfs:
        try:
            return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['symbol']).reset_index(drop=True)
        except Exception as e:
            st.warning(f"Error combining stock data: {str(e)}")
    
    # Return empty dataframe with required columns if no data was loaded
    return pd.DataFrame(columns=['symbol', 'name'])

stock_df = load_local_stock_list()

# Helper Functions
def add_to_portfolio(symbol, quantity, purchase_price):
    """Add stock to portfolio"""
    portfolio_item = {
        'symbol': symbol,
        'quantity': quantity,
        'purchase_price': purchase_price,
        'date_added': datetime.now().strftime('%Y-%m-%d')
    }
    st.session_state.portfolio.append(portfolio_item)
    st.success(f"Added {quantity} shares of {symbol} to portfolio")

def add_to_watchlist(symbol):
    """Add stock to watchlist"""
    if symbol not in st.session_state.watchlist:
        st.session_state.watchlist.append(symbol)
        st.success(f"Added {symbol} to watchlist")
    else:
        st.info(f"{symbol} is already in your watchlist")

def display_portfolio():
    """Display portfolio holdings"""
    if not st.session_state.portfolio:
        st.info("Your portfolio is empty. Add some stocks to get started!")
        return
    
    portfolio_data = []
    for item in st.session_state.portfolio:
        try:
            ticker = yf.Ticker(item['symbol'])
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
            current_value = current_price * item['quantity']
            gain_loss = (current_price - item['purchase_price']) * item['quantity']
            gain_loss_pct = ((current_price - item['purchase_price']) / item['purchase_price']) * 100
            
            currency_symbol = data_processor.get_currency_symbol(item['symbol'])
            
            portfolio_data.append({
                'Symbol': item['symbol'],
                'Quantity': item['quantity'],
                'Purchase Price': f"{currency_symbol}{item['purchase_price']:.2f}",
                'Current Price': f"{currency_symbol}{current_price:.2f}",
                'Current Value': f"{currency_symbol}{current_value:.2f}",
                'Gain/Loss': f"{currency_symbol}{gain_loss:.2f}",
                'Gain/Loss %': f"{gain_loss_pct:.2f}%"
            })
        except:
            portfolio_data.append({
                'Symbol': item['symbol'],
                'Quantity': item['quantity'],
                'Purchase Price': f"${item['purchase_price']:.2f}",
                'Current Price': 'N/A',
                'Current Value': 'N/A',
                'Gain/Loss': 'N/A',
                'Gain/Loss %': 'N/A'
            })
    
    df = pd.DataFrame(portfolio_data)
    st.dataframe(df, use_container_width=True)

def display_watchlist():
    """Display watchlist with current prices"""
    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Add some stocks to track!")
        return
    
    watchlist_data = []
    for symbol in st.session_state.watchlist:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
            
            currency_symbol = data_processor.get_currency_symbol(symbol)
            
            watchlist_data.append({
                'Symbol': symbol,
                'Current Price': f"{currency_symbol}{current_price:.2f}",
                'Change': f"{change:.2f}",
                'Change %': f"{change_pct:.2f}%"
            })
        except:
            watchlist_data.append({
                'Symbol': symbol,
                'Current Price': 'N/A',
                'Change': 'N/A',
                'Change %': 'N/A'
            })
    
    df = pd.DataFrame(watchlist_data)
    st.dataframe(df, use_container_width=True)

def calculate_advanced_metrics(price_series, benchmark_returns=None, risk_free_rate=0.02):
    """
    Calculate advanced financial metrics for a price series
    
    Args:
        price_series: Series of stock prices
        benchmark_returns: Optional Series of benchmark returns for relative metrics
        risk_free_rate: Annual risk-free rate (default: 2%)
    """
    # Calculate returns
    returns = price_series.pct_change().dropna()
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    
    # Basic metrics
    total_return = (price_series.iloc[-1] / price_series.iloc[0] - 1) * 100
    annual_return = (1 + total_return/100) ** (252/len(price_series)) - 1
    volatility = returns.std() * np.sqrt(252) * 100
    
    # Risk-adjusted returns
    sharpe_ratio = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if excess_returns.std() != 0 else 0
    
    # Maximum drawdown
    rolling_max = price_series.cummax()
    drawdowns = (price_series - rolling_max) / rolling_max
    max_drawdown = drawdowns.min() * 100
    
    # Additional metrics
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0
    sortino_ratio = ((returns.mean() * 252 - risk_free_rate) / (downside_volatility / 100)) if downside_volatility != 0 else 0
    
    # Win rate and profit factor
    win_rate = (returns > 0).mean() * 100
    avg_win = returns[returns > 0].mean() * 100 if len(returns[returns > 0]) > 0 else 0
    avg_loss = abs(returns[returns < 0].mean() * 100) if len(returns[returns < 0]) > 0 else 0
    profit_factor = (avg_win * (win_rate/100)) / (avg_loss * (1 - win_rate/100)) if avg_loss != 0 else float('inf')
    
    metrics = {
        'Total Return %': total_return,
        'Annualized Return %': annual_return * 100,
        'Volatility %': volatility,
        'Downside Vol %': downside_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown %': max_drawdown,
        'Win Rate %': win_rate,
        'Profit Factor': profit_factor,
        'Avg Win %': avg_win,
        'Avg Loss %': avg_loss,
        'Current Price': price_series.iloc[-1],
        'Start Date': price_series.index[0].strftime('%Y-%m-%d'),
        'End Date': price_series.index[-1].strftime('%Y-%m-%d'),
        'Days': len(price_series)
    }
    
    # Calculate benchmark-relative metrics if benchmark is provided
    if benchmark_returns is not None and len(benchmark_returns) == len(returns):
        # Calculate beta
        covariance = returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        
        # Calculate alpha
        alpha = (annual_return - (risk_free_rate + beta * (benchmark_returns.mean() * 252 - risk_free_rate))) * 100
        
        # Tracking error and information ratio
        tracking_error = (returns - benchmark_returns).std() * np.sqrt(252) * 100
        information_ratio = ((returns - benchmark_returns).mean() * 252 / tracking_error) if tracking_error != 0 else 0
        
        metrics.update({
            'Beta': beta,
            'Alpha %': alpha,
            'Tracking Error %': tracking_error,
            'Information Ratio': information_ratio,
            'Correlation to Benchmark': returns.corr(benchmark_returns)
        })
    
    return metrics

def compare_stocks(symbols, period, benchmark_symbol='^GSPC'):
    """
    Compare multiple stocks with enhanced metrics and visualizations
    
    Args:
        symbols: List of stock symbols to compare
        period: Time period for analysis (e.g., '1y', '5y')
        benchmark_symbol: Symbol for benchmark (default: S&P 500)
    """
    comparison_data = {}
    volume_data = {}
    metrics_data = {}
    returns_data = {}
    
    with st.spinner('Fetching and processing stock data...'):
        # First, get benchmark data if specified
        benchmark_returns = None
        if benchmark_symbol:
            try:
                benchmark = yf.Ticker(benchmark_symbol)
                benchmark_data = benchmark.history(period=period)['Close']
                benchmark_returns = benchmark_data.pct_change().dropna()
            except Exception as e:
                st.warning(f"Could not fetch benchmark data: {str(e)}")
        
        # Process each stock
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist_data = ticker.history(period=period)
                
                if not hist_data.empty:
                    # Store price and volume data
                    comparison_data[symbol] = hist_data['Close']
                    volume_data[symbol] = hist_data['Volume']
                    
                    # Calculate returns for correlation analysis
                    returns_data[symbol] = hist_data['Close'].pct_change().dropna()
                    
                    # Calculate metrics
                    metrics_data[symbol] = calculate_advanced_metrics(
                        hist_data['Close'],
                        benchmark_returns=benchmark_returns
                    )
                    
            except Exception as e:
                st.error(f"Error processing {symbol}: {str(e)}")
    
    # Calculate correlation matrix
    if len(returns_data) > 1:
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
    else:
        correlation_matrix = pd.DataFrame()
    
    st.session_state.comparison_data = {
        'prices': comparison_data,
        'volumes': volume_data,
        'metrics': metrics_data,
        'returns': returns_data,
        'correlation': correlation_matrix,
        'period': period,
        'benchmark': benchmark_symbol
    }

def display_comparison_results():
    """Display enhanced stock comparison results with multiple visualizations"""
    if 'comparison_data' not in st.session_state or not st.session_state.comparison_data:
        return
    
    data = st.session_state.comparison_data
    comparison_df = pd.DataFrame(data['prices'])
    volume_df = pd.DataFrame(data['volumes'])
    metrics_df = pd.DataFrame(data['metrics']).T
    
    # Display analysis period and benchmark
    st.sidebar.markdown("### Analysis Period")
    st.sidebar.write(f"**Period:** {data.get('period', 'N/A')}")
    if 'benchmark' in data and data['benchmark']:
        st.sidebar.write(f"**Benchmark:** {data['benchmark']}")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        'Performance', 'Risk Analysis', 'Returns', 'Volumes', 'Correlation'
    ])
    
    with tab1:
        # Normalized price comparison
        normalized_df = comparison_df.div(comparison_df.iloc[0]) * 100 - 100
        
        fig = go.Figure()
        for symbol in normalized_df.columns:
            fig.add_trace(go.Scatter(
                x=normalized_df.index,
                y=normalized_df[symbol],
                mode='lines',
                name=symbol,
                line=dict(width=2),
                hovertemplate='%{y:.2f}%<extra></extra>'
            ))
        
        fig.update_layout(
            title='Normalized Price Performance',
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown chart
        st.subheader('Drawdowns')
        drawdown_fig = go.Figure()
        for symbol in comparison_df.columns:
            rolling_max = comparison_df[symbol].cummax()
            drawdown = (comparison_df[symbol] / rolling_max - 1) * 100
            drawdown_fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode='lines',
                name=symbol,
                line=dict(width=1.5),
                fill='tozeroy'
            ))
        
        drawdown_fig.update_layout(
            yaxis_title='Drawdown (%)',
            template='plotly_white',
            height=400,
            showlegend=True
        )
        st.plotly_chart(drawdown_fig, use_container_width=True)
    
    with tab2:
        # Daily returns distribution
        st.subheader('Daily Returns Distribution')
        returns_df = comparison_df.pct_change().dropna()
        
        fig_returns = go.Figure()
        for symbol in returns_df.columns:
            fig_returns.add_trace(go.Violin(
                y=returns_df[symbol] * 100,
                name=symbol,
                box_visible=True,
                meanline_visible=True
            ))
        
        fig_returns.update_layout(
            yaxis_title='Daily Return (%)',
            template='plotly_white',
            height=500
        )
        st.plotly_chart(fig_returns, use_container_width=True)
    
    with tab3:
        # Volume comparison
        st.subheader('Trading Volume')
        fig_volume = go.Figure()
        
        for symbol in volume_df.columns:
            fig_volume.add_trace(go.Bar(
                x=volume_df.index,
                y=volume_df[symbol],
                name=symbol,
                opacity=0.7
            ))
        
        fig_volume.update_layout(
            yaxis_title='Volume',
            template='plotly_white',
            height=500,
            barmode='group',
            showlegend=True
        )
        st.plotly_chart(fig_volume, use_container_width=True)
    
    with tab2:
        # Risk Analysis
        st.subheader('Risk-Return Profile')
        
        # Prepare data for scatter plot
        scatter_data = []
        for symbol in metrics_df.index:
            scatter_data.append({
                'Symbol': symbol,
                'Volatility %': metrics_df.loc[symbol, 'Volatility %'],
                'Annualized Return %': metrics_df.loc[symbol, 'Annualized Return %'],
                'Sharpe Ratio': metrics_df.loc[symbol, 'Sharpe Ratio'],
                'Max Drawdown %': abs(metrics_df.loc[symbol, 'Max Drawdown %'])
            })
        
        scatter_df = pd.DataFrame(scatter_data)
        
        # Create bubble chart for risk-return
        fig_risk_return = px.scatter(
            scatter_df,
            x='Volatility %',
            y='Annualized Return %',
            size='Max Drawdown %',
            color='Sharpe Ratio',
            hover_name='Symbol',
            title='Risk-Return Profile',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        # Add market line (simplified)
        if 'benchmark' in data and data['benchmark']:
            fig_risk_return.add_hline(
                y=0,
                line_dash="dash",
                line_color="red",
                annotation_text="Risk-Free Rate",
                annotation_position="bottom right"
            )
        
        fig_risk_return.update_layout(
            xaxis_title='Volatility (Annualized %)',
            yaxis_title='Annualized Return (%)',
            template='plotly_white',
            height=600
        )
        st.plotly_chart(fig_risk_return, use_container_width=True)
        
        # Rolling volatility
        st.subheader('Rolling Volatility (21-day)')
        rolling_vol = comparison_df.pct_change().rolling(21).std() * np.sqrt(252) * 100
        
        fig_rolling_vol = go.Figure()
        for col in rolling_vol.columns:
            fig_rolling_vol.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol[col],
                mode='lines',
                name=col
            ))
        
        fig_rolling_vol.update_layout(
            yaxis_title='Volatility (Annualized %)',
            template='plotly_white',
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_rolling_vol, use_container_width=True)
    
    with tab5:
        # Correlation heatmap
        st.subheader('Returns Correlation')
        if 'correlation' in data and not data['correlation'].empty:
            corr_matrix = data['correlation']
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                text=np.around(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size":12}
            ))
            
            fig_corr.update_layout(
                title='Returns Correlation Matrix',
                template='plotly_white',
                height=500
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Cluster map of correlations
            st.subheader('Correlation Clustering')
            try:
                import seaborn as sns
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.clustermap(
                    corr_matrix,
                    cmap='coolwarm',
                    center=0,
                    annot=True,
                    fmt=".2f",
                    linewidths=.5,
                    figsize=(10, 8)
                )
                plt.title('Hierarchical Clustering of Correlations')
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate cluster map: {str(e)}")
        else:
            st.warning("Not enough data to calculate correlations")
    
    # Define color formatting function
    def color_negative_red(val):
        if isinstance(val, str) and val != 'N/A':
            if val.endswith('%'):
                num = float(val.rstrip('%'))
                color = 'red' if num < 0 else 'green' if num > 0 else 'black'
                return f'color: {color}'
            elif val.startswith('-'):
                return 'color: red'
        return 'color: black'
    
    # Display metrics table with tabs for different metric categories
    st.subheader('Performance & Risk Metrics')
    
    # Categorize metrics
    return_metrics = [
        'Total Return %', 'Annualized Return %', 'Alpha %',
        'Win Rate %', 'Avg Win %', 'Avg Loss %', 'Profit Factor'
    ]
    
    risk_metrics = [
        'Volatility %', 'Downside Vol %', 'Max Drawdown %',
        'Sharpe Ratio', 'Sortino Ratio', 'Information Ratio',
        'Beta', 'Tracking Error %', 'Correlation to Benchmark'
    ]
    
    # Format metrics for display
    display_metrics = metrics_df.copy()
    for col in display_metrics.columns:
        if col == 'Current Price':
            display_metrics[col] = display_metrics[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else 'N/A')
        elif isinstance(display_metrics[col].iloc[0], (int, float)):
            if '%' in col:
                display_metrics[col] = display_metrics[col].apply(
                    lambda x: f"{x:,.2f}%" if pd.notnull(x) else 'N/A')
            else:
                display_metrics[col] = display_metrics[col].apply(
                    lambda x: f"{x:,.2f}" if pd.notnull(x) else 'N/A')
    
    # Create tabs for different metric categories
    tab_returns, tab_risk, tab_all = st.tabs(['Returns', 'Risk', 'All Metrics'])
    
    with tab_returns:
        st.dataframe(
            display_metrics[[col for col in return_metrics if col in display_metrics.columns]]
            .style.applymap(color_negative_red),
            use_container_width=True
        )
    
    with tab_risk:
        st.dataframe(
            display_metrics[[col for col in risk_metrics if col in display_metrics.columns]]
            .style.applymap(color_negative_red),
            use_container_width=True
        )
    
    with tab_all:
        st.dataframe(
            display_metrics.style.applymap(color_negative_red),
            use_container_width=True
        )
    
    # Add download button
    csv = metrics_df.to_csv().encode('utf-8')
    st.download_button(
        label="Download Metrics as CSV",
        data=csv,
        file_name='stock_comparison_metrics.csv',
        mime='text/csv'
    )

@st.cache_data(ttl=3600)  # Cache results for 1 hour
def get_stock_data(symbol, sector_filter=None, **kwargs):
    """
    Fetch and process data for a single stock with comprehensive metrics
    
    Args:
        symbol: Stock ticker symbol
        sector_filter: Optional sector to filter by
        **kwargs: Additional parameters for extended data fetching
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Get basic info with timeout
        info = ticker.info
        if not info:
            return None, f"{symbol}: No info available"
        
        # Get historical data (5 days for more reliable volume)
        hist = ticker.history(period="5d", interval="1d")
        if hist.empty:
            return None, f"{symbol}: No historical data"
        
        # Get latest price and volume
        latest = hist.iloc[-1]
        price = latest['Close']
        volume = hist['Volume'].mean() / 1e6  # Average volume in millions
        
        # Skip if no price or volume data
        if pd.isna(price) or pd.isna(volume) or volume <= 0:
            return None, f"{symbol}: Missing price/volume data"
        
        # Get market cap and handle different currencies
        market_cap_raw = info.get('marketCap', info.get('totalAssets', 0))
        if market_cap_raw is None or market_cap_raw <= 0:
            return None, f"{symbol}: Missing market cap"
            
        if symbol.endswith('.NS'):
            # For Indian stocks (NSE), convert to crores (1e7)
            market_cap_bn = float(market_cap_raw) / 1e7  # Convert to crores for Indian stocks
            market_cap = f"₹{market_cap_bn:,.2f}"  # Format with commas and 2 decimal places
        else:
            # For US stocks, convert to billions (1e9)
            market_cap_bn = float(market_cap_raw) / 1e9  # Convert to billions for US stocks
            market_cap = f"${market_cap_bn:,.2f}"  # Format with commas and 2 decimal places
        
        # Get key financial metrics with fallbacks
        pe_ratio = info.get('trailingPE', info.get('forwardPE', 0)) or 0
        dividend_yield = info.get('dividendYield', 0) * 100  # Convert to percentage
        beta = info.get('beta', 1.0)
        profit_margins = info.get('profitMargins', 0) * 100  # Convert to percentage
        
        # Calculate ROI (Return on Investment)
        total_assets = info.get('totalAssets', 1)
        net_income = info.get('netIncomeToCommon', 0)
        roi =  (net_income / total_assets) * 100 if total_assets > 0 else 0
        
        # Get sector and apply filter if needed
        stock_sector = info.get('sector', 'N/A')
        if sector_filter and sector_filter != 'All' and stock_sector != sector_filter:
            return None, f"{symbol}: Sector doesn't match"
        
        # Calculate price changes
        if len(hist) > 1:
            prev_close = hist['Close'].iloc[-2]
            daily_change = ((price - prev_close) / prev_close) * 100
        else:
            daily_change = 0
            
        # Calculate 52-week performance
        fifty_two_week_high = info.get('fiftyTwoWeekHigh')
        fifty_two_week_low = info.get('fiftyTwoWeekLow')
        
        # Calculate distance from 52-week high/low
        if fifty_two_week_high and fifty_two_week_high > 0:
            from_52w_high = ((fifty_two_week_high - price) / fifty_two_week_high) * 100
        else:
            from_52w_high = None
            
        if fifty_two_week_low and fifty_two_week_low > 0:
            from_52w_low = ((price - fifty_two_week_low) / fifty_two_week_low) * 100
        else:
            from_52w_low = None
        
        # Prepare comprehensive result
        result = {
            # Basic Info
            'Symbol': symbol,
            'Name': info.get('shortName', symbol),
            'Exchange': info.get('exchange', 'N/A'),
            'Sector': stock_sector,
            'Industry': info.get('industry', 'N/A'),
            'Currency': '₹' if symbol.endswith('.NS') else '$',
            
            # Price Data
            'Price': price,
            'Previous Close': info.get('previousClose', 0),
            'Open': latest.get('Open', 0),
            'Day High': latest.get('High', 0),
            'Day Low': latest.get('Low', 0),
            'Change %': daily_change,
            'Volume (M)': round(volume, 2),
            'Avg Volume (M)': round(info.get('averageVolume', 0) / 1e6, 2) if info.get('averageVolume') else 0,
            
            # Valuation
            'Market Cap (B)': market_cap_bn,
            'Market Cap': market_cap,
            'P/E': round(pe_ratio, 2) if pe_ratio > 0 else None,
            'PEG': info.get('pegRatio'),
            'P/S': info.get('priceToSalesTrailing12Months'),
            'P/B': info.get('priceToBook'),
            
            # Performance
            '52W High': fifty_two_week_high,
            '52W Low': fifty_two_week_low,
            'From 52W High %': round(from_52w_high, 2) if from_52w_high is not None else None,
            'From 52W Low %': round(from_52w_low, 2) if from_52w_low is not None else None,
            'Beta': round(beta, 2) if beta else None,
            
            # Dividends
            'Dividend Yield %': round(dividend_yield, 2) if dividend_yield > 0 else 0,
            'Payout Ratio': info.get('payoutRatio'),
            'Dividend Rate': info.get('dividendRate'),
            
            # Financial Health
            'Profit Margin %': round(profit_margins, 2) if profit_margins else None,
            'ROI %': round(roi*100, 2) if roi else None,
            'ROE': info.get('returnOnEquity'),
            'ROA': info.get('returnOnAssets'),
            'Current Ratio': info.get('currentRatio'),
            'Debt/Equity': info.get('debtToEquity'),
            
            # Growth
            'Revenue Growth (YOY)': info.get('revenueGrowth'),
            'Earnings Growth (YOY)': info.get('earningsGrowth'),
            
            # Technical
            'RSI (14)': None,  # Can be calculated separately
            '50D MA': info.get('fiftyDayAverage'),
            '200D MA': info.get('twoHundredDayAverage'),
            '50D/200D MA Ratio': info.get('fiftyDayAverage') / info.get('twoHundredDayAverage') 
                                if info.get('fiftyDayAverage') and info.get('twoHundredDayAverage') else None
        }
        
        return result, None
        
    except Exception as e:
        error_msg = f"{symbol}: {str(e)[:150]}"
        # Cache negative results briefly to avoid repeated failures
        st.session_state.stock_data_cache[cache_key] = (None, error_msg)
        return None, error_msg
    
    # Cache successful results
    st.session_state.stock_data_cache[cache_key] = (result, None)
    return result, None

def get_default_screener_params():
    """Return default parameters for the stock screener"""
    return {
        'market_cap_min': 10,  # $10B minimum market cap
        'pe_max': 30,          # Maximum P/E ratio
        'price_min': 10,       # Minimum price
        'price_max': 1000,     # Maximum price
        'volume_min': 1,       # 1M minimum volume
        'sector': 'All',       # All sectors
        'dividend_yield_min': 0,  # Minimum dividend yield
        'beta_max': 2.0,       # Maximum beta (volatility)
        'profit_margin_min': 0,  # Minimum profit margin
        'roi_min': 0,          # Minimum Return on Investment
    }

def process_stock_batch(stock_batch, sector_filter, **filters):
    """Process a batch of stocks in parallel"""
    results = []
    errors = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Start the load operations and mark each future with its symbol
        future_to_symbol = {
            executor.submit(
                get_stock_data, 
                symbol, 
                sector_filter=sector_filter,
                **filters
            ): symbol 
            for symbol in stock_batch
        }
        
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result, error = future.result()
                if error:
                    errors.append(error)
                elif result:
                    results.append(result)
            except Exception as e:
                errors.append(f"{symbol}: {str(e)[:150]}")
    
    return results, errors

def screen_stocks(
    market_cap_min=None, 
    pe_max=None, 
    price_min=None, 
    price_max=None, 
    volume_min=None, 
    sector=None,
    dividend_yield_min=None,
    beta_max=None,
    profit_margin_min=None,
    roi_min=None,
    batch_size=10  # Process stocks in batches
):
    """
    Screen stocks based on fundamental and technical criteria
    
    Parameters:
    - market_cap_min: Minimum market cap in billions (e.g., 10 for $10B)
    - pe_max: Maximum P/E ratio (0 to disable)
    - price_min: Minimum stock price
    - price_max: Maximum stock price
    - volume_min: Minimum average volume in millions (e.g., 1 for 1M shares)
    - sector: Filter by sector (e.g., 'Technology', 'Financial')
    - dividend_yield_min: Minimum dividend yield (%)
    - beta_max: Maximum beta (volatility relative to market)
    - profit_margin_min: Minimum profit margin (%)
    - roi_min: Minimum Return on Investment (%)
    """
    # Set default values if not provided
    defaults = get_default_screener_params()
    market_cap_min = market_cap_min if market_cap_min is not None else defaults['market_cap_min']
    pe_max = pe_max if pe_max is not None else defaults['pe_max']
    price_min = price_min if price_min is not None else defaults['price_min']
    price_max = price_max if price_max is not None else defaults['price_max']
    volume_min = volume_min if volume_min is not None else defaults['volume_min']
    sector = sector if sector is not None else defaults['sector']
    dividend_yield_min = dividend_yield_min if dividend_yield_min is not None else defaults['dividend_yield_min']
    beta_max = beta_max if beta_max is not None else defaults['beta_max']
    profit_margin_min = profit_margin_min if profit_margin_min is not None else defaults['profit_margin_min']
    roi_min = roi_min if roi_min is not None else defaults['roi_min']
    
    # Expanded list of stocks from different markets and sectors
    sample_stocks = [
        # US Large Cap
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V',
        'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE', 'NFLX', 'CRM', 'INTC',
        # Indian Large Cap
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'LT.NS', 'ITC.NS',
        # Tech
        'AMD', 'QCOM', 'AVGO', 'TXN', 'ADSK', 'INTU', 'NOW', 'SHOP', 'MELI', 'ASML', 'TSM',
        # Consumer
        'KO', 'PEP', 'NKE', 'MCD', 'SBUX', 'LOW', 'COST', 'TGT', 'WMT',
        # Financial
        'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 'SCHW', 'SPGI', 'MCO',
        # Healthcare
        'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'UNH', 'LLY', 'ISRG'
    ]
    
    # Remove duplicates while preserving order
    sample_stocks = list(dict.fromkeys(sample_stocks))
    
    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    screened_stocks = []
    errors = []
    
    # Prepare filters
    filters = {
        'pe_max': pe_max,
        'market_cap_min': market_cap_min,
        'price_min': price_min,
        'price_max': price_max,
        'volume_min': volume_min,
        'dividend_yield_min': dividend_yield_min,
        'beta_max': beta_max,
        'profit_margin_min': profit_margin_min,
        'roi_min': roi_min
    }
    
    # Process stocks in batches
    for i in range(0, len(sample_stocks), batch_size):
        batch = sample_stocks[i:i + batch_size]
        
        # Update progress
        progress = min((i + len(batch)) / len(sample_stocks), 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processing batch {i//batch_size + 1}/{(len(sample_stocks)-1)//batch_size + 1}")
        
        # Process batch in parallel
        batch_results, batch_errors = process_stock_batch(
            batch, 
            sector_filter=sector,
            **filters
        )
        
        # Filter results
        for result in batch_results:
            filters_passed = [
                price_min <= result.get('Price', 0) <= price_max,
                (market_cap_min <= 0 or (result.get('Market Cap (B)', 0) >= market_cap_min)),
                (pe_max <= 0 or (result.get('P/E') and 0 < result['P/E'] <= pe_max)),
                result.get('Volume (M)', 0) >= volume_min,
                (dividend_yield_min <= 0 or (result.get('Dividend Yield %', 0) or 0) >= dividend_yield_min),
                (beta_max <= 0 or (result.get('Beta', 0) or 0) <= beta_max),
                (profit_margin_min <= 0 or (result.get('Profit Margin %', 0) or 0) >= profit_margin_min),
                (roi_min <= 0 or (result.get('ROI %', 0) or 0) >= roi_min)
            ]
            
            if all(filters_passed):
                screened_stocks.append(result)
        
        errors.extend(batch_errors)
        
        # Early exit if we have enough results
        if len(screened_stocks) >= 50:  # Limit to top 50 results
            break
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Show summary of errors if any
    if errors and len(screened_stocks) < 5:  # Only show errors if we have very few results
        with st.expander("⚠️ Data retrieval issues (click to view)"):
            st.warning(f"Encountered {len(errors)} issues while fetching data. Here are some examples:")
            for error in errors[:5]:  # Show first 5 errors to avoid cluttering
                st.code(error, language='text')
            if len(errors) > 5:
                st.info(f"... and {len(errors) - 5} more issues not shown")
    
    # Get the list of stocks that were checked
    stocks_checked = sample_stocks if 'sample_stocks' in locals() else []
    stocks_checked_count = len(stocks_checked)
    
    # Display results or no results message
    if not screened_stocks:
        st.warning("""
        ## No stocks found matching your criteria. Try these adjustments:
        
        ### Quick Fixes:
        - Increase the maximum P/E ratio (current: {pe_max})
        - Lower the minimum market cap (current: ${market_cap_min}B)
        - Widen the price range (current: ${price_min} - ${price_max})
        - Try a different sector (current: {sector})
        - Reduce the minimum volume (current: {volume_min}M)
        - Lower the minimum dividend yield (current: {dividend_yield_min}%)
        - Increase the maximum beta (current: {beta_max})
        - Lower the minimum profit margin (current: {profit_margin_min}%)
        - Lower the minimum ROI (current: {roi_min}%)
        
        ### Common Issues:
        - Some stocks may have missing data points
        - International stocks may have different data availability
        - Market data might be delayed
        
        Note: This screener checked {stocks_count} stocks.
        For a more comprehensive scan, consider using a dedicated stock screener API.
        """.format(
            pe_max=pe_max,
            market_cap_min=market_cap_min,
            price_min=price_min,
            price_max=price_max,
            sector=sector if sector else 'All',
            volume_min=volume_min,
            dividend_yield_min=dividend_yield_min,
            beta_max=beta_max,
            profit_margin_min=profit_margin_min,
            roi_min=roi_min,
            stocks_count=stocks_checked_count
        ))
        
        # Show sample of available stocks if we have any
        if stocks_checked_count > 0 and len(stocks_checked) > 0:
            with st.expander("🔍 View sample of stocks being checked"):
                cols = 4
                rows = (stocks_checked_count + cols - 1) // cols
                for i in range(rows):
                    row_cols = st.columns(cols)
                    for j in range(cols):
                        idx = i * cols + j
                        if idx < stocks_checked_count and idx < len(stocks_checked):
                            with row_cols[j]:
                                st.code(stocks_checked[idx], language='text')
        else:
            st.info("No stock symbols were available to check. Please verify your data source.")
        
        # Add a button to reset filters
        if st.button("🔄 Reset to Default Filters"):
            st.session_state.screener_reset = True
            st.experimental_rerun()
            
        return
    
    # Convert to DataFrame for display
    df = pd.DataFrame(screened_stocks)
    
    # Ensure all required columns exist with default values if missing
    for col in ['Currency', 'Price', 'Change %', 'Market Cap (B)', 'P/E', 'Volume (M)']:
        if col not in df.columns:
            df[col] = None
    
    # Format numeric columns
    def format_price(row):
        if pd.isna(row['Price']) or not isinstance(row['Price'], (int, float)):
            return 'N/A'
        currency = row.get('Currency', '$')
        return f"{currency}{row['Price']:.2f}"
        
    def format_change(x):
        if pd.isna(x) or not isinstance(x, (int, float)):
            return 'N/A'
        color = '🔴' if x < 0 else '🟢'
        return f"{color}{abs(x):.2f}%"
    
    # Apply formatting with error handling
    try:
        df['Price'] = df.apply(format_price, axis=1)
        df['Change %'] = df['Change %'].apply(format_change)
        df['Market Cap (B)'] = df['Market Cap (B)'].apply(
            lambda x: f"${float(x):,.2f}B" if pd.notnull(x) and str(x).replace('.', '').isdigit() else 'N/A'
        )
        df['P/E'] = df['P/E'].apply(
            lambda x: f"{float(x):.1f}" if pd.notnull(x) and str(x).replace('.', '').replace('-', '').isdigit() and float(x) > 0 else 'N/A'
        )
        df['Volume (M)'] = df['Volume (M)'].apply(
            lambda x: f"{float(x):,.1f}M" if pd.notnull(x) and str(x).replace('.', '').isdigit() else 'N/A'
        )
    except Exception as e:
        st.error(f"Error formatting data: {str(e)}")
        st.stop()
    
    # Sort by Market Cap (descending) by default
    df = df.sort_values(by='Market Cap (B)', key=lambda x: x.str.replace('[$,B]', '', regex=True).astype(float), ascending=False)
    
    # Display results
    st.success(f"🎉 Found {len(df)} stocks matching your criteria")
    
    # Show summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_pe = df[df['P/E'] != 'N/A']['P/E'].astype(float).mean()
        st.metric("Average P/E", f"{avg_pe:.1f}" if not pd.isna(avg_pe) else 'N/A')
    with col2:
        total_market_cap = df['Market Cap (B)'].str.replace('[$,B]', '', regex=True).astype(float).sum()
        st.metric("Total Market Cap", f"${total_market_cap:,.2f}B")
    with col3:
        st.metric("Sectors Found", f"{df['Sector'].nunique()}")
    
    # Add tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Performance", "💼 Fundamentals"])
    
    with tab1:  # Overview tab
        view_cols = st.columns(2)
        with view_cols[0]:
            sort_by = st.selectbox(
                "Sort by",
                ["Market Cap (High to Low)", "Price (High to Low)", "P/E (Low to High)", 
                 "Dividend Yield (High to Low)", "Volume (High to Low)"],
                index=0
            )
        
        # Apply sorting
        if sort_by == "Market Cap (High to Low)":
            df_sorted = df.sort_values(
                by='Market Cap (B)', 
                key=lambda x: x.str.replace('[$,B]', '', regex=True).astype(float), 
                ascending=False
            )
        elif sort_by == "Price (High to Low)":
            df_sorted = df.sort_values(
                by='Price', 
                key=lambda x: x.str.replace('[^\d.]', '', regex=True).astype(float), 
                ascending=False
            )
        elif sort_by == "P/E (Low to High)":
            df_sorted = df.sort_values(
                by='P/E', 
                key=lambda x: x.replace('N/A', '999999').astype(float), 
                ascending=True
            )
        elif sort_by == "Dividend Yield (High to Low)":
            df_sorted = df.sort_values(
                by='Dividend Yield %', 
                ascending=False
            )
        else:  # Volume High to Low
            df_sorted = df.sort_values(
                by='Volume (M)', 
                key=lambda x: x.str.replace('[M,]', '', regex=True).astype(float), 
                ascending=False
            )
        
        # Display the dataframe with better formatting
        st.dataframe(
            df_sorted[[
                'Symbol', 'Name', 'Price', 'Change %', 'Market Cap (B)', 
                'P/E', 'Volume (M)', 'Sector'
            ]],
            use_container_width=True,
            column_config={
                'Symbol': st.column_config.TextColumn("Symbol", width="small"),
                'Name': st.column_config.TextColumn("Company"),
                'Price': st.column_config.TextColumn("Price"),
                'Change %': st.column_config.TextColumn("Change"),
                'Market Cap (B)': st.column_config.TextColumn("Market Cap"),
                'P/E': st.column_config.TextColumn("P/E"),
                'Volume (M)': st.column_config.TextColumn("Volume"),
                'Sector': st.column_config.TextColumn("Sector")
            },
            hide_index=True,
            height=min(600, 50 + len(df_sorted) * 35)  # Dynamic height
        )
    
    with tab2:  # Performance tab
        st.dataframe(
            df[[
                'Symbol', 'Name', 'Price', 'Change %', '52W High', '52W Low',
                'From 52W High %', 'From 52W Low %', 'Beta', '50D MA', '200D MA'
            ]],
            use_container_width=True,
            column_config={
                'Symbol': st.column_config.TextColumn("Symbol"),
                'Name': st.column_config.TextColumn("Company"),
                'Price': st.column_config.TextColumn("Price"),
                'Change %': st.column_config.TextColumn("Change"),
                '52W High': st.column_config.NumberColumn("52W High", format="%.2f"),
                '52W Low': st.column_config.NumberColumn("52W Low", format="%.2f"),
                'From 52W High %': st.column_config.NumberColumn("From High %", format="%.1f%%"),
                'From 52W Low %': st.column_config.NumberColumn("From Low %", format="%.1f%%"),
                'Beta': st.column_config.NumberColumn("Beta", format="%.2f"),
                '50D MA': st.column_config.NumberColumn("50D MA", format="%.2f"),
                '200D MA': st.column_config.NumberColumn("200D MA", format="%.2f")
            },
            hide_index=True
        )
    
    with tab3:  # Fundamentals tab
        st.dataframe(
            df[[
                'Symbol', 'Name', 'Market Cap (B)', 'P/E', 'PEG', 'P/S', 'P/B',
                'Profit Margin %', 'ROI %', 'ROE', 'Debt/Equity', 'Dividend Yield %',
                'Payout Ratio', 'Revenue Growth (YOY)', 'Earnings Growth (YOY)'
            ]],
            use_container_width=True,
            column_config={
                'Symbol': st.column_config.TextColumn("Symbol"),
                'Name': st.column_config.TextColumn("Company"),
                'Market Cap (B)': st.column_config.TextColumn("Market Cap"),
                'P/E': st.column_config.NumberColumn("P/E", format="%.1f"),
                'PEG': st.column_config.NumberColumn("PEG", format="%.2f"),
                'P/S': st.column_config.NumberColumn("P/S", format="%.2f"),
                'P/B': st.column_config.NumberColumn("P/B", format="%.2f"),
                'Profit Margin %': st.column_config.NumberColumn("Profit %", format="%.1f%%"),
                'ROI %': st.column_config.NumberColumn("ROI %", format="%.1f%%"),
                'ROE': st.column_config.NumberColumn("ROE", format="%.1f%%"),
                'Debt/Equity': st.column_config.NumberColumn("Debt/Eq", format="%.2f"),
                'Dividend Yield %': st.column_config.NumberColumn("Div. Yield", format="%.2f%%"),
                'Payout Ratio': st.column_config.NumberColumn("Payout %", format="%.1f%%"),
                'Revenue Growth (YOY)': st.column_config.NumberColumn("Rev. Growth", format="%.1f%%"),
                'Earnings Growth (YOY)': st.column_config.NumberColumn("EPS Growth", format="%.1f%%")
            },
            hide_index=True
        )
    
    # Add download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Results as CSV",
        data=csv,
        file_name=f'stockscreener_results_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
        mime='text/csv',
        key='download_screener_results',
        help="Download the full dataset including all available metrics"
    )
    
    # Add a button to reset filters
    if st.button("🔄 Reset to Default Filters", key="reset_filters_bottom"):
        st.session_state.screener_reset = True
        st.experimental_rerun()
    else:
        st.warning(f"""
        ## No stocks found matching your criteria. Try these adjustments:
        
        ### Quick Fixes:
        - Increase the maximum P/E ratio (current: {pe_max})
        - Lower the minimum market cap (current: ${market_cap_min}B)
        - Widen the price range (current: ${price_min} - ${price_max})
        - Try a different sector (current: {sector})
        - Reduce the minimum volume (current: {volume_min}M)
        
        ### Common Issues:
        - Some stocks may have missing data points
        - International stocks may have different data availability
        - Market data might be delayed
        
        Note: This screener checks a diverse set of {len(sample_stocks)} stocks.
        For a more comprehensive scan, consider using a dedicated stock screener API.
        """)
        
        # Show sample of available stocks
        with st.expander("🔍 View sample of stocks being checked"):
            cols = 3
            rows = (len(sample_stocks) + cols - 1) // cols
            for i in range(rows):
                cols = st.columns(3)
                for j in range(3):
                    idx = i * 3 + j
                    if idx < len(sample_stocks):
                        with cols[j]:
                            st.code(sample_stocks[idx], language='text')
        
        # Add a button to reset filters
        if st.button("🔄 Reset to Default Filters"):
            st.session_state.screener_reset = True
            st.experimental_rerun()

def analyze_stock(symbol, period, show_advanced=False):
    """Analyze stock data from Yahoo Finance"""
    try:
        with st.spinner(f"Fetching data for  {symbol}..."):
            # Fetch stock data
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            hist_data = ticker.history(period=period)
            
            if hist_data.empty:
                st.error(f"No data found for symbol '{symbol}'. Please check the symbol and try again.")
                return
            
            # Get stock info
            info = ticker.info
            
            # Calculate advanced technical indicators if requested
            if show_advanced:
                hist_data = data_processor.calculate_technical_indicators(hist_data)
            
            # Process the data
            processed_data = data_processor.process_stock_data(hist_data, info, symbol)
            
            # Store in session state
            st.session_state.analyzed_data = {
                'symbol': symbol,
                'historical_data': hist_data,
                'processed_data': processed_data,
                'info': info,
                'period': period,
                'show_advanced': show_advanced
            }
            
            st.success(f"Successfully fetched data for {symbol}")
            
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        st.info("Please check if the symbol is correct and try again.")

def display_analysis_results():
    """Display stock analysis results"""
    if st.session_state.analyzed_data is None:
        st.info("Enter a stock symbol and click 'Analyze Stock' to get started! ")
        return
    
    data = st.session_state.analyzed_data
    symbol = data['symbol']
    hist_data = data['historical_data']
    processed_data = data['processed_data']
    info = data['info']
    show_advanced = data.get('show_advanced', False)
    
    # Display stock information
    st.header(f"Analysis Results for {symbol}")
    
    # Get currency symbol from processed data first
    currency_symbol = processed_data.get('currency_symbol', '$')
    
    # Show currency and market informations
    is_indian = processed_data.get('is_indian_stock', False)
    if is_indian:
        st.info(f"Indian Stock - Prices displayed in Indian Rupees ({currency_symbol})")
    else:
        currency_info = info.get('currency', 'USD') if info else 'USD'
        st.info(f"International Stock - Currency: {currency_info} ({currency_symbol})")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = hist_data['Close'].iloc[-1]
    previous_price = hist_data['Close'].iloc[-2] if len(hist_data) > 1 else current_price
    price_change = current_price - previous_price
    price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0
    
    with col1:
        st.metric(
            "Current Price",
            f"{currency_symbol}{current_price:.2f}",
            delta=f"{price_change:.2f} ({price_change_pct:.2f}%)"
        )
    
    with col2:
        st.metric("52W High", f"{currency_symbol}{hist_data['High'].max():.2f}")
    
    with col3:
        st.metric("52W Low", f"{currency_symbol}{hist_data['Low'].min():.2f}")
    
    with col4:
        volume = hist_data['Volume'].iloc[-1]
        st.metric("Latest Volume", f"{volume:,.0f}")
    
    # Company information
    if info:
        with st.expander("Company Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Company:** {info.get('longName', 'N/A')}")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            with col2:
                market_cap = info.get('marketCap', 0)
                if market_cap:
                    st.write(f"**Market Cap:** {currency_symbol}{market_cap:,.0f}")
                else:
                    st.write("**Market Cap:** N/A")
                st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
                st.write(f"**Dividend Yield:** {info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else "N/A")
    
    # Charts
    st.subheader("Price Charts")
    
    # Price chart
    price_chart = chart_generator.create_price_chart(hist_data, symbol)
    st.plotly_chart(price_chart, use_container_width=True)
    
    # Volume chart
    volume_chart = chart_generator.create_volume_chart(hist_data, symbol)
    st.plotly_chart(volume_chart, use_container_width=True)
    
    # Advanced technical indicators
    if show_advanced and ('RSI' in hist_data.columns or 'MACD' in hist_data.columns):
        st.subheader("Advanced Technical Analysis")
        
        if 'RSI' in hist_data.columns:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=hist_data.index, y=hist_data['RSI'], 
                                       mode='lines', name='RSI', line=dict(color='purple')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5)
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5)
            fig_rsi.update_layout(title='RSI (Relative Strength Index)', 
                                yaxis_title='RSI', template='plotly_white', height=300)
            st.plotly_chart(fig_rsi, use_container_width=True)
    
    # Data table
    st.subheader("Historical Data")
    
    # Display data with formatting
    display_data = hist_data.copy()
    display_data = display_data.round(2)
    display_data.index = display_data.index.strftime('%Y-%m-%d')
    
    # Dynamic column formatting based on currency
    price_format = f"{currency_symbol}%.2f"
    
    st.dataframe(
        display_data,
        use_container_width=True,
        column_config={
            "Open": st.column_config.NumberColumn("Open", format=price_format),
            "High": st.column_config.NumberColumn("High", format=price_format),
            "Low": st.column_config.NumberColumn("Low", format=price_format),
            "Close": st.column_config.NumberColumn("Close", format=price_format),
            "Volume": st.column_config.NumberColumn("Volume", format="%d"),
        }
    )
    
    # Download button outside form to prevent conflicts
    csv_data = display_data.to_csv(index=True).encode('utf-8')
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv_data,
        file_name=f"{symbol}_stock_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key=f"download_{symbol}",
        type="primary"
    )
    st.caption("Note: Downloading data will not affect your current session")

def calculate_portfolio_value():
    """Calculate total portfolio value"""
    total_value = 0
    for item in st.session_state.portfolio:
        try:
            ticker = yf.Ticker(item['symbol'])
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
            total_value += current_price * item['quantity']
        except:
            continue
    return total_value

def main():
    # Header with logo and title
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://img.icons8.com/?size=100&id=7721&format=png&color=000000", width=60)
    with col2:
        st.markdown("<h1 style='margin-bottom: 0;'>StockSage Pro</h1>", unsafe_allow_html=True)
        st.markdown("<div style='color: #4f46e5; font-size: 1.1rem; font-weight: 500;'>Advanced Stock Analysis & Portfolio Management</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main navigation tabs with icons
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Analysis", 
        "🔍 Comparison", 
        "💼 Portfolio", 
        "👀 Watchlist", 
        "🔍 Screener", 
        "🤖 ML Analysis"
    ])
    
    with tab1:
        st.markdown("<div class='card'><h3>📈 Stock Analysis</h3></div>", unsafe_allow_html=True)
        
        # Stock input and controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            stock_symbol = st.text_input(
                "Enter Stock Symbol",
                placeholder="e.g., AAPL, GOOGL, RELIANCE.NS, TCS.NS",
                help="Enter a valid stock ticker symbol. For Indian stocks, use .NS (NSE) or .BO (BSE) suffix"
            ).strip().upper()
        
        with col2:
            period_options = {
                "1 Month": "1mo",
                "3 Months": "3mo",
                "6 Months": "6mo",
                "1 Year": "1y",
                "2 Years": "2y",
                "5 Years": "5y"
            }
            selected_period = st.selectbox(
                "Select Time Period",
                options=list(period_options.keys()),
                index=2
            )
        
        with col3:
            show_advanced = st.checkbox("Show Technical Indicators")
        
        # Analyze button and portfolio actions
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🔍 Analyze Stock", type="primary") and stock_symbol:
                analyze_stock(stock_symbol, period_options[selected_period], show_advanced)
        
        if stock_symbol:
            with col2:
                if st.button("➕ Add to Portfolio"):
                    st.session_state.show_portfolio_modal = True
            with col3:
                if st.button("👁️ Add to Watchlist"):
                    add_to_watchlist(stock_symbol)
        
        # Portfolio modal
        if stock_symbol and st.session_state.get('show_portfolio_modal', False):
            with st.form("add_to_portfolio_form"):
                st.subheader(f"Add {stock_symbol} to Portfolio")
                quantity = st.number_input("Quantity", min_value=1, value=1)
                purchase_price = st.number_input("Purchase Price", min_value=0.01, value=100.0, step=0.01)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("Add to Portfolio"):
                        add_to_portfolio(stock_symbol, quantity, purchase_price)
                        st.session_state.show_portfolio_modal = False
                with col2:
                    if st.form_submit_button("Cancel"):
                        st.session_state.show_portfolio_modal = False
        
        # Display results
        display_analysis_results()
        
        # Stock examples
        with st.expander("📝 Stock Symbol Examples"):
            st.write("**US Stocks:**")
            st.write("• AAPL - Apple Inc.")
            st.write("• GOOGL - Alphabet Inc.")
            st.write("• MSFT - Microsoft Corp.")
            st.write("• TSLA - Tesla Inc.")
            st.write("• AMZN - Amazon.com Inc.")
            st.write("**Indian Stocks (NSE):**")
            st.write("• RELIANCE.NS - Reliance Industries")
            st.write("• TCS.NS - Tata Consultancy Services")
            st.write("• INFY.NS - Infosys Ltd.")
            st.write("• HDFCBANK.NS - HDFC Bank Ltd.")
            st.write("**Indian Stocks (BSE):**")
            st.write("• RELIANCE.BO - Reliance Industries")
            st.write("• TCS.BO - Tata Consultancy Services")
    
    with tab2:
        st.markdown("<div class='card'><h3>📊 Stock Comparison</h3></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Multi-stock input
            symbols_input = st.text_input(
                "Enter Stock Symbols (comma-separated)",
                placeholder="e.g., AAPL, GOOGL, MSFT or RELIANCE.NS, TCS.NS",
                help="Enter multiple stock symbols separated by commas"
            )
            
            period = st.selectbox(
                "Comparison Period",
                ["1mo", "3mo", "6mo", "1y", "2y"],
                index=2
            )
            
            if st.button("Compare Stocks") and symbols_input:
                symbols = [s.strip().upper() for s in symbols_input.split(',')]
                compare_stocks(symbols, period)
        
        with col2:
            st.info("Compare multiple stocks side by side to analyze their relative performance, correlations, and key metrics.")
        
        # Display comparison results
        display_comparison_results()
    
    with tab3:
        st.markdown("<div class='card'><h3>💼 Portfolio Management</h3></div>", unsafe_allow_html=True)
        
        # Portfolio summary
        if st.session_state.portfolio:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_value = calculate_portfolio_value()
                st.metric("Total Portfolio Value", f"${total_value:,.2f}")
            
            with col2:
                st.metric("Number of Stocks", len(st.session_state.portfolio))
            
            with col3:
                if st.button("🗑️ Clear Portfolio"):
                    st.session_state.portfolio = []
                    st.rerun()
        
        # Display portfolio
        display_portfolio()
    
    with tab4:
        st.markdown("<div class='card'><h3>👀 Stock Watchlist</h3></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Add to watchlist
            symbol = st.text_input("Add Stock to Watchlist", key="watchlist_symbol")
            if st.button("Add to Watchlist", key="add_to_watchlist_tab") and symbol:
                add_to_watchlist(symbol.upper())
        
        with col2:
            if st.session_state.watchlist:
                st.write(f"**Watching {len(st.session_state.watchlist)} stocks**")
                if st.button("Clear Watchlist"):
                    st.session_state.watchlist = []
                    st.rerun()
        
        # Display watchlist
        display_watchlist()
    
    with tab5:
        st.markdown("<div class='card'><h3>🔍 Stock Screener</h3></div>", unsafe_allow_html=True)
        
        # Create a form to prevent automatic reloading
        with st.form("screener_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                market_cap_min = st.number_input("Min Market Cap (B)", value=0.0, step=1.0, key="market_cap_min")
                pe_max = st.number_input("Max P/E Ratio", value=50.0, step=1.0, key="pe_max")
            
            with col2:
                price_min = st.number_input("Min Price", value=0.0, step=1.0, key="price_min")
                price_max = st.number_input("Max Price", value=1000.0, step=10.0, key="price_max")
            
            with col3:
                volume_min = st.number_input("Min Volume (M)", value=0.0, step=0.1, key="volume_min")
                sector = st.selectbox("Sector", ["All", "Technology", "Healthcare", "Financial", "Energy"], key="sector")
            
            submitted = st.form_submit_button("Screen Stocks")
        
        # Only run the screener when the form is submitted
        if submitted:
            screen_stocks(market_cap_min, pe_max, price_min, price_max, volume_min, sector)
    
    with tab6:
        st.markdown("<div class='card'><h3>🤖 Machine Learning Analysis</h3></div>", unsafe_allow_html=True)
        
        # Initialize ML model
        ml_model = StockML()
        
        # ML Analysis Options
        ml_option = st.selectbox(
        "Select ML Analysis Type",
        ["Price Prediction", "Anomaly Detection", "Risk Assessment", "Portfolio Optimization", "Direction Classification", "Sentiment Analysis", "User Customization"]
         )
        
        if ml_option == "Price Prediction":
            symbol = st.text_input("Enter stock symbol (e.g., AAPL)", key="ml_symbol_price", 
                 help="The ticker symbol for the stock you want to predict prices for.")
            if symbol:
                with st.spinner("Training LSTM model for price prediction... This might take a one or two minutes"):
                    try:
                        # Get predictions using the enhanced method
                        results = ml_model.predict_prices(symbol, period='2y')
                
                        # Create and display the enhanced plot
                        fig = ml_model.plot_enhanced_predictions(results, symbol)
                        st.plotly_chart(fig, use_container_width=True)
                
                            # Show prediction metrics
                        predictions = results['predictions']
                        actual_prices = results['actual_prices']
                        train_size = results['train_size']
                
                        # Calculate metrics for test set only
                        test_predictions = predictions[train_size:]
                        test_actual = actual_prices[train_size:]
                
                        mse = mean_squared_error(test_actual, test_predictions)
                        mae = mean_absolute_error(test_actual, test_predictions)
                        r2 = r2_score(test_actual, test_predictions)
                
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean Squared Error", f"{mse:.2f}")
                        with col2:
                            st.metric("Mean Absolute Error", f"{mae:.2f}")
                        with col3:
                            st.metric("R² Score", f"{r2:.3f}")
                
                        # Add disclaimer
                        st.warning("""
                        **Note on Price Predictions:**
                        - These predictions are based on historical price patterns and technical indicators
                        - Past performance is not indicative of future results
                        - Always conduct your own research and consider multiple factors before making investment decisions
                        - The model has inherent limitations and may not account for all market conditions
                        """)
                
                    except Exception as e:
                        st.error(f"Error generating predictions: {str(e)}")
                        st.info("Please check the stock symbol and try again. The stock may not have enough historical data.")
                        st.markdown("**Learn more:** [Why price prediction is hard](https://www.investopedia.com/ask/answers/06/stockprices.asp)")
        elif ml_option == "Direction Classification":
            symbol = st.text_input("Enter stock symbol (e.g., AAPL)", key="ml_symbol_direction", 
                          help="The ticker symbol for the stock you want to analyze. E.g., AAPL for Apple.")
            if symbol:
                with st.spinner("Training LSTM classifier for direction (up/down)..."):
                    try:
                        results = ml_model.classify_direction(symbol, period='2y')
                
                        y_test = results['actual']
                        y_pred = results['predicted']
                        acc = results['accuracy']
                        prec = precision_score(y_test, y_pred)
                        rec = recall_score(y_test, y_pred)
                        cm = confusion_matrix(y_test, y_pred)
                
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{acc:.3f}", help="Proportion of correct up/down predictions. [Learn more](https://en.wikipedia.org/wiki/Accuracy_and_precision)")
                        with col2:
                            st.metric("Precision", f"{prec:.3f}", help="How many predicted 'up' days were actually up. [Learn more](https://en.wikipedia.org/wiki/Precision_and_recall)")
                        with col3:
                            st.metric("Recall", f"{rec:.3f}", help="How many actual 'up' days were correctly predicted. [Learn more](https://en.wikipedia.org/wiki/Precision_and_recall)")
                
                        st.write("#### Confusion Matrix")
                        st.markdown("Shows the number of correct and incorrect up/down predictions. [Learn more](https://en.wikipedia.org/wiki/Confusion_matrix)")
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        st.pyplot(fig)
                
                    except Exception as e:
                        st.error(f"Error in direction classification: {str(e)}")
        
        elif ml_option == "Anomaly Detection":
            symbol = st.text_input("Enter Stock Symbol for Anomaly Detection", key="anomaly_symbol")
            if symbol:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1y")
                
                    if hist.empty:
                        st.error(f"No data found for symbol {symbol}")
                    else:
                        anomalies = ml_model.detect_anomalies(hist)
                
                        # Plot anomalies using the enhanced method
                        fig = ml_model.plot_anomalies(anomalies, symbol)
                        st.plotly_chart(fig, use_container_width=True)
                
                        # Show anomaly statistics
                        num_anomalies = len(anomalies[anomalies['Anomaly'] == -1])
                        total_days = len(anomalies)
                        anomaly_rate = (num_anomalies / total_days) * 100
                
                        st.info(f"Detected {num_anomalies} anomalies out of {total_days} trading days ({anomaly_rate:.1f}%)")
                
                except Exception as e:
                    st.error(f"Error in anomaly detection: {str(e)}")
        
        elif ml_option == "Risk Assessment":
            symbol = st.text_input("Enter Stock Symbol for Risk Assessment", key="risk_symbol")
            if symbol:
                try:
                    risk_metrics = ml_model.assess_risk(symbol, period='1y')
            
                    if 'error' in risk_metrics:
                        st.error(f"Error calculating risk metrics: {risk_metrics['error']}")
                    else:
                        # Display risk metrics in a more organized way
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Volatility", f"{risk_metrics['Volatility']:.2%}")
                            st.metric("Sharpe Ratio", f"{risk_metrics['Sharpe Ratio']:.2f}")
                            st.metric("Sortino Ratio", f"{risk_metrics['Sortino Ratio']:.2f}")
                            st.metric("Beta", f"{risk_metrics['Beta']:.2f}")
                        with col2:
                            st.metric("VaR (95%)", f"{risk_metrics['VaR (95%)']:.2%}")
                            st.metric("CVaR (95%)", f"{risk_metrics['CVaR (95%)']:.2%}")
                            st.metric("Max Drawdown", f"{risk_metrics['Max Drawdown']:.2%}")
                
                        # Risk interpretation
                        st.subheader("Risk Interpretation")
                        if risk_metrics['Volatility'] > 0.3:
                            st.warning("⚠️ High volatility stock - higher risk")
                        elif risk_metrics['Volatility'] < 0.15:
                            st.success("✅ Low volatility stock - lower risk")
                        else:
                            st.info("ℹ️ Moderate volatility stock")
                
                except Exception as e:
                    st.error(f"Error in risk assessment: {str(e)}")
        
        elif ml_option == "Portfolio Optimization":
            symbols_input = st.text_input(
                "Enter Stock Symbols (comma-separated)",
                placeholder="e.g., AAPL, GOOGL, MSFT",
                key="optimize_symbols"
            )
            if symbols_input:
                symbols = [s.strip().upper() for s in symbols_input.split(',')]
                with st.spinner("Optimizing portfolio..."):
                    try:
                        optimal_weights = ml_model.optimize_portfolio(symbols, period='1y')
                        
                        if 'error' in optimal_weights:
                            st.error(f"Error in portfolio optimization: {optimal_weights['error']}")
                        elif optimal_weights:
                            # Display optimal weights
                            st.subheader("Optimal Portfolio Weights")
                            weights_df = pd.DataFrame({
                                'Symbol': list(optimal_weights.keys()),
                                'Weight': [f"{w:.2%}" for w in optimal_weights.values()]
                            })
                            st.dataframe(weights_df, use_container_width=True)
                    
                        # Plot pie chart of weights
                            fig = go.Figure(data=[go.Pie(
                                labels=list(optimal_weights.keys()),
                                values=list(optimal_weights.values()),
                                hole=.3
                            )])
                            fig.update_layout(title="Optimal Portfolio Allocation")
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown("**Learn more:** [Portfolio optimization explained](https://www.investopedia.com/terms/m/modernportfoliotheory.asp)")
                        else:
                            st.error("No valid data found for the provided symbols")
                    except Exception as e:
                        st.error(f"Error in portfolio optimization: {str(e)}")
        
        elif ml_option == "Sentiment Analysis":
            symbol = st.text_input("Enter stock symbol for sentiment analysis (e.g., AAPL)", key="ml_symbol_sentiment")
            if symbol:
                with st.spinner("Fetching news and analyzing sentiment..."):
                    try:
                        news_df = fetch_news(symbol)
                        news_df = compute_sentiment(news_df)
                        trend = daily_sentiment_trend(news_df)
                
                        st.write("### Recent News Headlines")
                        expected_cols = ['publishedAt', 'title', 'sentiment', 'url']
                        available_cols = [col for col in expected_cols if col in news_df.columns]
                
                        if not news_df.empty and available_cols:
                            st.dataframe(news_df[available_cols])
                        else:
                            st.info("No news data available or NewsAPI/mediastack limit reached.")
                
                        if not trend.empty:
                            st.write("### Daily Sentiment Trend")
                            fig = px.line(trend, x='publishedAt', y='sentiment', title=f"Sentiment Trend for {symbol}")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No sentiment data available for this period.")
                
                        st.markdown("**Learn more:** [What is news sentiment analysis?](https://www.investopedia.com/terms/s/sentiment-analysis.asp)")
                
                    except Exception as e:
                        st.error(f"Error in sentiment analysis: {str(e)}")
        
        elif ml_option == "User Customization":
            st.subheader("Custom Alerts & Indicator Builder")
            symbol = st.text_input("Enter stock symbol for customization (e.g., AAPL)", key="custom_symbol")
            if symbol:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="6mo")
                if not hist.empty:
                    # Custom Alerts
                    st.markdown("**Set Price/Indicator Alert**")
                    alert_type = st.selectbox("Alert Type", ["Price", "Moving Average (MA_20)"], key="alert_type_1")
                    alert_value = st.number_input("Alert Value", value=float(hist['Close'].iloc[-1]), key="alert_value_1")
                    triggered = False
                    if alert_type == "Price" and hist['Close'].iloc[-1] >= alert_value:
                        triggered = True
                    elif alert_type == "Moving Average (MA_20)":
                        hist['MA_20'] = hist['Close'].rolling(window=20).mean()
                        if hist['MA_20'].iloc[-1] >= alert_value:
                            triggered = True
                    email = st.text_input("Enter your email for alerts (optional)", key="email_1")
                    if st.button("Check Alert & Send Email", key="check_alert_1"):
                        if triggered:
                            st.success(f"Alert triggered! {alert_type} >= {alert_value}")
                            if email:
                                try:
                                    send_email_alert(
                                        to_email=email,
                                        subject=f"Stock Alert for {symbol}",
                                        message=f"Your alert for {symbol} was triggered: {alert_type} >= {alert_value}",
                                        from_email=st.secrets["EMAIL_USER"],
                                        from_password=st.secrets["EMAIL_PASS"]
                                    )
                                    st.info("Email sent!")
                                except Exception as e:
                                    st.warning(f"Failed to send email: {e}")
                        else:
                            st.info(f"Alert not triggered. {alert_type} < {alert_value}")
                    # Custom Indicator Builder
                    st.markdown("**Custom Indicator Builder**")
                    st.write("You can use columns like Close, Open, High, Low, Volume, MA_20, etc.")
                    formula = st.text_input("Enter formula (e.g., Close - MA_20)", value="Close - MA_20", key="formula_1")
                    # Compute MA_20 for convenience
                    hist['MA_20'] = hist['Close'].rolling(window=20).mean()
                    try:
                        hist['Custom_Indicator'] = eval(formula, {}, hist)
                        st.line_chart(hist[['Close', 'Custom_Indicator']].dropna())
                    except Exception as e:
                        st.warning(f"Error in formula: {e}")
                else:
                    st.info("No data found for this symbol.")
                    email = st.text_input("Enter your email for alerts (optional)", key="email_2")
                    if st.button("Check Alert & Send Email", key="check_alert_2"):
                        if triggered:
                            st.success(f"Alert triggered! {alert_type} >= {alert_value}")
                            if email:
                                try:
                                    send_email_alert(
                                        to_email=email,
                                        subject=f"Stock Alert for {symbol}",
                                        message=f"Your alert for {symbol} was triggered: {alert_type} >= {alert_value}",
                                        from_email=st.secrets["EMAIL_USER"],
                                        from_password=st.secrets["EMAIL_PASS"]
                                    )
                                    st.info("Email sent!")
                                except Exception as e:
                                    st.warning(f"Failed to send email: {e}")
                        else:
                            st.info(f"Alert not triggered. {alert_type} < {alert_value}")
                    # Custom Indicator Builder
                    st.markdown("**Custom Indicator Builder**")
                    st.write("You can use columns like Close, Open, High, Low, Volume, MA_20, etc.")
                    formula = st.text_input("Enter formula (e.g., Close - MA_20)", value="Close - MA_20", key="formula_2")
                    # Compute MA_20 for convenience
                    hist['MA_20'] = hist['Close'].rolling(window=20).mean()
                    try:
                        hist['Custom_Indicator'] = eval(formula, {}, hist)
                        st.line_chart(hist[['Close', 'Custom_Indicator']].dropna())
                    except Exception as e:
                        st.warning(f"Error in formula: {e}")
            else:
                st.info("No sentiment data available for this period.")
            st.markdown("**Learn more:** [What is news sentiment analysis?](https://www.investopedia.com/terms/s/sentiment-analysis.asp)")
        
        elif ml_option == "User Customization":
            st.subheader("Custom Alerts & Indicator Builder")
            symbol = st.text_input("Enter stock symbol for customization (e.g., AAPL)", key="custom_symbol")
            if symbol:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="6mo")
            
                    if not hist.empty:
                        # Custom Alerts
                        st.markdown("**Set Price/Indicator Alert**")
                        alert_type = st.selectbox("Alert Type", ["Price", "Moving Average (MA_20), Moving Average (MA_50)"], key="alert_type_3")
                        alert_value = st.number_input("Alert Value", value=float(hist['Close'].iloc[-1]), key="alert_value_3")
                        triggered = False
                
                        if alert_type == "Price" and hist['Close'].iloc[-1] >= alert_value:
                            triggered = True
                        elif alert_type == "Moving Average (MA_20)":
                            hist['MA_20'] = hist['Close'].rolling(window=20).mean()
                            if hist['MA_20'].iloc[-1] >= alert_value:
                                triggered = True
                        elif alert_type == "Moving Average (MA_50)":
                            hist['MA_50'] = hist['Close'].rolling(window=50).mean()
                            if hist['MA_50'].iloc[-1] >= alert_value:
                                triggered = True
                        
                
                        email = st.text_input("Enter your email for alerts (optional)", key="email_3")
                
                        if st.button("Check Alert & Send Email", key="check_alert_3"):
                            if triggered:
                                st.success(f"Alert triggered! {alert_type} >= {alert_value}")
                                if email:
                                    try:
                                        send_email_alert(
                                            to_email=email,
                                            subject=f"Stock Alert for {symbol}",
                                            message=f"Your alert for {symbol} was triggered: {alert_type} >= {alert_value}",
                                            from_email=st.secrets["EMAIL_USER"],
                                            from_password=st.secrets["EMAIL_PASS"]
                                        )
                                        st.info("Email sent!")
                                    except Exception as e:
                                        st.warning(f"Failed to send email: {e}")
                                else:
                                    st.info(f"Alert not triggered. {alert_type} < {alert_value}")
                
                # Custom Indicator Builder
                    st.markdown("**Custom Indicator Builder**")
                    st.write("You can use columns like Close, Open, High, Low, Volume, MA_20, etc.")
                    formula = st.text_input("Enter formula (e.g., Close - MA_20)", value="Close - MA_20")
                
                # Compute MA_20 for convenience
                    hist['MA_20'] = hist['Close'].rolling(window=20).mean()
                
                    try:
                        hist['Custom_Indicator'] = eval(formula, {}, hist)
                        st.line_chart(hist[['Close', 'Custom_Indicator']].dropna())
                    except Exception as e:
                        st.warning(f"Error in formula: {e}")
                    else:
                        st.info("No data found for this symbol.")
                
                except Exception as e:
                    st.error(f"Error in user customization: {str(e)}")
    
    st.markdown("**Learn more:** [How to build custom indicators](https://www.investopedia.com/terms/t/technicalindicator.asp)")

if __name__ == "__main__":
    main()