import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

def compare_stocks(symbols, period):
    """Compare multiple stocks"""
    comparison_data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period=period)
            if not hist_data.empty:
                comparison_data[symbol] = hist_data['Close']
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
    
    st.session_state.comparison_data = comparison_data

def display_comparison_results():
    """Display stock comparison results"""
    if 'comparison_data' not in st.session_state or not st.session_state.comparison_data:
        return
    
    comparison_df = pd.DataFrame(st.session_state.comparison_data)
    
    # Normalize prices to show percentage changes
    normalized_df = comparison_df.div(comparison_df.iloc[0]) * 100 - 100
    
    # Create comparison chart
    fig = go.Figure()
    for symbol in normalized_df.columns:
        fig.add_trace(go.Scatter(
            x=normalized_df.index,
            y=normalized_df[symbol],
            mode='lines',
            name=symbol,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title='Stock Performance Comparison (Normalized %)',
        xaxis_title='Date',
        yaxis_title='Performance (%)',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show comparison table
    st.subheader("Performance Summary")
    summary_data = []
    for symbol in comparison_df.columns:
        total_return = ((comparison_df[symbol].iloc[-1] / comparison_df[symbol].iloc[0]) - 1) * 100
        volatility = comparison_df[symbol].pct_change().std() * np.sqrt(252) * 100
        
        summary_data.append({
            'Symbol': symbol,
            'Total Return %': f"{total_return:.2f}%",
            'Volatility %': f"{volatility:.2f}%",
            'Current Price': f"${comparison_df[symbol].iloc[-1]:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

def screen_stocks(market_cap_min, pe_max, price_min, price_max, volume_min, sector):
    """Screen stocks based on criteria"""
    sample_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']
    screened_stocks = []
    
    with st.spinner("Screening stocks..."):
        for symbol in sample_stocks:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.fast_info
                hist = ticker.history(period="1d")
                
                price = hist['Close'].iloc[-1] if not hist.empty else 0
                market_cap = info.get('marketCap', 0) / 1e9
                pe_ratio = info.get('trailingPE', 0) or 0
                volume = hist['Volume'].iloc[-1] / 1e6 if not hist.empty else 0
                
                if (price >= price_min and price <= price_max and
                    market_cap >= market_cap_min and
                    pe_ratio <= pe_max and pe_ratio > 0 and
                    volume >= volume_min):
                    
                    currency_symbol = data_processor.get_currency_symbol(symbol)
                    
                    screened_stocks.append({
                        'Symbol': symbol,
                        'Price': f"{currency_symbol}{price:.2f}",
                        'Market Cap (B)': f"{market_cap:.2f}",
                        'P/E Ratio': f"{pe_ratio:.2f}",
                        'Volume (M)': f"{volume:.2f}",
                        'Sector': info.get('sector', 'N/A')
                    })
            except:
                continue
    
    if screened_stocks:
        st.success(f"Found {len(screened_stocks)} stocks matching your criteria:")
        df = pd.DataFrame(screened_stocks)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No stocks found matching your criteria. Try adjusting the filters.")

def analyze_stock(symbol, period, show_advanced=False):
    """Analyze stock data from Yahoo Finance"""
    try:
        with st.spinner(f"Fetching data for {symbol}..."):
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
        st.info("Enter a stock symbol and click 'Analyze Stock' to get started!")
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
    
    # Show currency and market information
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
    
    # Download button
    csv_data = display_data.to_csv(index=True).encode('utf-8')
    st.download_button(
        label="Download Data as CSV",
        data=csv_data,
        file_name=f"{symbol}_stock_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

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
        st.image("https://img.icons8.com/color/96/000000/stock-share.png", width=60)
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
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            market_cap_min = st.number_input("Min Market Cap (B)", value=0.0, step=1.0)
            pe_max = st.number_input("Max P/E Ratio", value=50.0, step=1.0)
        
        with col2:
            price_min = st.number_input("Min Price", value=0.0, step=1.0)
            price_max = st.number_input("Max Price", value=1000.0, step=10.0)
        
        with col3:
            volume_min = st.number_input("Min Volume (M)", value=0.0, step=0.1)
            sector = st.selectbox("Sector", ["All", "Technology", "Healthcare", "Financial", "Energy"])
        
        if st.button("Screen Stocks"):
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
                with st.spinner("Training LSTM model for price prediction..."):
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
                    alert_type = st.selectbox("Alert Type", ["Price", "Moving Average (MA_20)"])
                    alert_value = st.number_input("Alert Value", value=float(hist['Close'].iloc[-1]))
                    triggered = False
                    if alert_type == "Price" and hist['Close'].iloc[-1] >= alert_value:
                        triggered = True
                    elif alert_type == "Moving Average (MA_20)":
                        hist['MA_20'] = hist['Close'].rolling(window=20).mean()
                        if hist['MA_20'].iloc[-1] >= alert_value:
                            triggered = True
                    email = st.text_input("Enter your email for alerts (optional)")
                    if st.button("Check Alert & Send Email"):
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
                email = st.text_input("Enter your email for alerts (optional)")
                if st.button("Check Alert & Send Email"):
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
                        alert_type = st.selectbox("Alert Type", ["Price", "Moving Average (MA_20)"])
                        alert_value = st.number_input("Alert Value", value=float(hist['Close'].iloc[-1]))
                        triggered = False
                
                        if alert_type == "Price" and hist['Close'].iloc[-1] >= alert_value:
                            triggered = True
                        elif alert_type == "Moving Average (MA_20)":
                            hist['MA_20'] = hist['Close'].rolling(window=20).mean()
                            if hist['MA_20'].iloc[-1] >= alert_value:
                                triggered = True
                
                        email = st.text_input("Enter your email for alerts (optional)")
                
                        if st.button("Check Alert & Send Email"):
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