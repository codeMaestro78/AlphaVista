import streamlit as st
import pandas as pd
from fuzzywuzzy import process
import re

@st.cache_data
def load_us_symbols():
    return pd.read_csv("data/nasdaq_stocks.csv").rename(columns={"Symbol": "symbol", "Security Name": "name"})

@st.cache_data
def load_india_symbols():
    return pd.read_csv("data/nse_stocks.csv").rename(columns={"SYMBOL": "symbol", "NAME OF COMPANY": "name"})

def highlight_match(text, query):
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return pattern.sub(lambda m: f"**{m.group(0)}**", text)

def country_stock_autocomplete(label="Search for a stock", key="stock_search"):
    country = st.selectbox("Select Market", ["US Stocks", "Indian Stocks"], key=f"{key}_country")
    if country == "US Stocks":
        symbols_df = load_us_symbols()
    else:
        symbols_df = load_india_symbols()

    query = st.text_input(label, key=key)
    suggestion = None
    if query:
        choices = symbols_df.apply(lambda row: f"{row['symbol']} - {row['name']}", axis=1).tolist()
        matches = process.extract(query, choices, limit=10)
        matches = [m for m in matches if m[1] > 60]
        if matches:
            st.write("Suggestions:")
            for match, score in matches:
                highlighted = highlight_match(match, query)
                if st.button(highlighted, key=f"suggestion_{match}"):
                    suggestion = match.split(" - ")[0]
    return suggestion, country