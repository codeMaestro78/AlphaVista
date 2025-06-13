import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
import sys
import os
import numpy as np

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from data_processor import DataProcessor
from chart_generator import ChartGenerator

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Portfolio Management Functions
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

def add_to_portfolio_modal(symbol):
    """Modal for adding stock to portfolio"""
    add_to_portfolio(symbol, 1, 100.0)

def calculate_portfolio_value():
    """Calculate total portfolio value"""
    total_value = 0
    for item in st.session_state.portfolio:
        try:
            ticker = yf.Ticker(item['symbol'])
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
            total_value += current_price * item['quantity']
        except:
            total_value += item['purchase_price'] * item['quantity']
    return total_value

def analyze_portfolio():
    """Analyze portfolio performance"""
    st.session_state.portfolio_analysis = True

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
            
            portfolio_data.append({
                'Symbol': item['symbol'],
                'Quantity': item['quantity'],
                'Purchase Price': f"${item['purchase_price']:.2f}",
                'Current Price': f"${current_price:.2f}",
                'Current Value': f"${current_value:.2f}",
                'Gain/Loss': f"${gain_loss:.2f}",
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

# Watchlist Functions
def add_to_watchlist(symbol):
    """Add stock to watchlist"""
    if symbol not in st.session_state.watchlist:
        st.session_state.watchlist.append(symbol)
        st.success(f"Added {symbol} to watchlist")
    else:
        st.info(f"{symbol} is already in your watchlist")

def update_watchlist_prices():
    """Update prices for all watchlist stocks"""
    st.session_state.watchlist_updated = True

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

# Comparison Functions
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

# Stock Screener Function
def screen_stocks(market_cap_min, pe_max, price_min, price_max, volume_min, sector):
    """Screen stocks based on criteria"""
    # Sample screening logic (in a real app, you'd use a stock screening API)
    sample_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'RELIANCE.NS', 'TCS.NS']
    screened_stocks = []
    
    for symbol in sample_stocks:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d")
            
            price = hist['Close'].iloc[-1] if not hist.empty else 0
            market_cap = info.get('marketCap', 0) / 1e9  # Convert to billions
            pe_ratio = info.get('trailingPE', 0) or 0
            volume = hist['Volume'].iloc[-1] / 1e6 if not hist.empty else 0  # Convert to millions
            
            # Apply filters
            if (price >= price_min and price <= price_max and
                market_cap >= market_cap_min and
                pe_ratio <= pe_max and pe_ratio > 0 and
                volume >= volume_min):
                
                screened_stocks.append({
                    'Symbol': symbol,
                    'Price': f"${price:.2f}",
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

def main():
    st.title("📈 Advanced Stock Analysis Dashboard")
    st.markdown("Comprehensive stock analysis with advanced features, portfolio tracking, and comparison tools")
    
    # Main navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Analysis", "📈 Comparison", "💼 Portfolio", "👁️ Watchlist", "🔍 Screener"])
    
    with tab1:
        stock_analysis_tab()
    
    with tab2:
        stock_comparison_tab()
    
    with tab3:
        portfolio_tab()
    
    with tab4:
        watchlist_tab()
    
    with tab5:
        stock_screener_tab()

def stock_analysis_tab():
    st.subheader("Individual Stock Analysis")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Analysis Controls")
        
        # Stock symbol input
        stock_symbol = st.text_input(
            "Enter Stock Symbol",
            placeholder="e.g., AAPL, GOOGL, RELIANCE.NS, TCS.NS",
            help="Enter a valid stock ticker symbol. For Indian stocks, use .NS (NSE) or .BO (BSE) suffix"
        ).upper()
        
        # Time period selection
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
            index=2  # Default to 6 months
        )
        
        # Advanced options
        show_advanced = st.checkbox("Show Advanced Technical Indicators")
        
        # Analyze button
        analyze_button = st.button("🔍 Analyze Stock", type="primary")
        
        # Quick actions
        if stock_symbol and st.button("Add to Watchlist", key="add_to_watchlist_sidebar"):
            add_to_watchlist(stock_symbol)
        
        if stock_symbol and st.button("Add to Portfolio", key="add_to_portfolio_sidebar"):
            add_to_portfolio_modal(stock_symbol)
        
        # Stock examples
        with st.expander("📝 Stock Symbol Examples"):
            st.write("**US Stocks:**")
            st.write("• AAPL, GOOGL, MSFT, TSLA, AMZN")
            st.write("**Indian Stocks (NSE):**")
            st.write("• RELIANCE.NS, TCS.NS, INFY.NS")
            st.write("• HDFCBANK.NS, ICICIBANK.NS")
            st.write("**Indian Stocks (BSE):**")
            st.write("• RELIANCE.BO, TCS.BO, INFY.BO")
        
        st.divider()
        
        # CSV Upload section
        st.header("📂 CSV Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload a CSV file with stock data for analysis"
        )
        
        if uploaded_file is not None:
            if st.button("📊 Process Uploaded CSV"):
                process_uploaded_csv(uploaded_file)
    
    # Main content area
    if analyze_button and stock_symbol:
        analyze_stock(stock_symbol, period_options[selected_period], show_advanced)
    
    # Display results if available
    display_analysis_results()
    
    # Display uploaded CSV results if available
    display_uploaded_csv_results()

def stock_comparison_tab():
    st.subheader("Stock Comparison Analysis")
    
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
        
        if st.button("📊 Compare Stocks") and symbols_input:
            symbols = [s.strip().upper() for s in symbols_input.split(',')]
            compare_stocks(symbols, period)
    
    with col2:
        st.info("Compare multiple stocks side by side to analyze their relative performance, correlations, and key metrics.")
    
    # Display comparison results
    display_comparison_results()

def portfolio_tab():
    st.subheader("Portfolio Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Add stock to portfolio
        st.write("**Add Stock to Portfolio**")
        symbol = st.text_input("Stock Symbol", key="portfolio_symbol")
        col_a, col_b = st.columns(2)
        with col_a:
            quantity = st.number_input("Quantity", min_value=1, value=1)
        with col_b:
            purchase_price = st.number_input("Purchase Price", min_value=0.01, value=100.0, step=0.01)
        
        if st.button("Add to Portfolio", key="add_to_portfolio_main") and symbol:
            add_to_portfolio(symbol.upper(), quantity, purchase_price)
    
    with col2:
        # Portfolio summary
        if st.session_state.portfolio:
            total_value = calculate_portfolio_value()
            st.metric("Portfolio Value", f"${total_value:,.2f}")
            
            if st.button("📈 Analyze Portfolio"):
                analyze_portfolio()
    
    # Display portfolio
    display_portfolio()

def watchlist_tab():
    st.subheader("Stock Watchlist")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Add to watchlist
        symbol = st.text_input("Add Stock to Watchlist", key="watchlist_symbol")
        if st.button("Add to Watchlist", key="add_to_watchlist_tab") and symbol:
            add_to_watchlist(symbol.upper())
    
    with col2:
        if st.session_state.watchlist:
            st.write(f"**Watching {len(st.session_state.watchlist)} stocks**")
            if st.button("📊 Update All Prices"):
                update_watchlist_prices()
    
    # Display watchlist
    display_watchlist()

def stock_screener_tab():
    st.subheader("Stock Screener")
    
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
    
    if st.button("🔍 Screen Stocks"):
        screen_stocks(market_cap_min, pe_max, price_min, price_max, volume_min, sector)

def analyze_stock(symbol, period, show_advanced=False):
    """Analyze stock data from Yahoo Finance"""
    with st.spinner(f"Fetching data for {symbol}..."):
        try:
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

def process_uploaded_csv(uploaded_file):
    """Process uploaded CSV file"""
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        
        # Validate CSV structure
        if df.empty:
            st.error("The uploaded CSV file is empty.")
            return
        
        # Process the uploaded data
        processed_csv_data = data_processor.process_csv_data(df)
        
        # Store in session state
        st.session_state.uploaded_data = {
            'filename': uploaded_file.name,
            'raw_data': df,
            'processed_data': processed_csv_data
        }
        
        st.success(f"Successfully processed {uploaded_file.name}")
        
    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")
        st.info("Please ensure the CSV file has proper format with columns like Date, Open, High, Low, Close, Volume.")

def display_analysis_results():
    """Display stock analysis results"""
    if st.session_state.analyzed_data is None:
        st.info("👆 Enter a stock symbol and click 'Analyze Stock' to get started!")
        return
    
    data = st.session_state.analyzed_data
    symbol = data['symbol']
    hist_data = data['historical_data']
    processed_data = data['processed_data']
    info = data['info']
    
    # Display stock information
    st.header(f"📊 Analysis Results for {symbol}")
    
    # Get currency symbol from processed data first
    currency_symbol = processed_data.get('currency_symbol', '$')
    
    # Show currency and market information
    is_indian = processed_data.get('is_indian_stock', False)
    if is_indian:
        st.info(f"🇮🇳 Indian Stock - Prices displayed in Indian Rupees ({currency_symbol})")
    else:
        currency_info = info.get('currency', 'USD') if info else 'USD'
        st.info(f"🌍 International Stock - Currency: {currency_info} ({currency_symbol})")
    
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
        with st.expander("📋 Company Information"):
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
    st.subheader("📈 Price Charts")
    
    # Price chart
    price_chart = chart_generator.create_price_chart(hist_data, symbol)
    st.plotly_chart(price_chart, use_container_width=True)
    
    # Volume chart
    volume_chart = chart_generator.create_volume_chart(hist_data, symbol)
    st.plotly_chart(volume_chart, use_container_width=True)
    
    # Data table
    st.subheader("📋 Historical Data")
    
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
    csv_data = convert_df_to_csv(display_data)
    st.download_button(
        label="💾 Download Data as CSV",
        data=csv_data,
        file_name=f"{symbol}_stock_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

def display_uploaded_csv_results():
    """Display results from uploaded CSV file"""
    if st.session_state.uploaded_data is None:
        return
    
    data = st.session_state.uploaded_data
    filename = data['filename']
    raw_data = data['raw_data']
    processed_data = data['processed_data']
    
    st.header(f"📂 Uploaded CSV Analysis: {filename}")
    
    # Basic statistics
    if processed_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(raw_data))
        
        with col2:
            if 'Close' in raw_data.columns:
                st.metric("Average Close", f"${raw_data['Close'].mean():.2f}")
        
        with col3:
            if 'High' in raw_data.columns:
                st.metric("Highest Price", f"${raw_data['High'].max():.2f}")
        
        with col4:
            if 'Low' in raw_data.columns:
                st.metric("Lowest Price", f"${raw_data['Low'].min():.2f}")
    
    # Display the data
    st.subheader("📊 Uploaded Data")
    st.dataframe(raw_data, use_container_width=True)
    
    # Create chart if possible
    if processed_data and 'chart_data' in processed_data:
        st.subheader("📈 Data Visualization")
        chart = chart_generator.create_csv_chart(processed_data['chart_data'])
        st.plotly_chart(chart, use_container_width=True)
    
    # Download processed data
    csv_data = convert_df_to_csv(raw_data)
    st.download_button(
        label="💾 Download Processed Data as CSV",
        data=csv_data,
        file_name=f"processed_{filename}",
        mime="text/csv"
    )

def convert_df_to_csv(df):
    """Convert dataframe to CSV string"""
    return df.to_csv(index=True).encode('utf-8')

if __name__ == "__main__":
    main()
