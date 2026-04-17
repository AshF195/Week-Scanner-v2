import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import os

# Suppress pandas warnings for cleaner Streamlit execution
warnings.filterwarnings('ignore')

# ==========================================
# 1. MARKET DATA UNIVERSE (Local CSV Files)
# ==========================================
@st.cache_data
def get_tickers_and_names(markets):
    """Loads tickers and company names from local CSV files."""
    tickers = []
    ticker_map = {} 
    
    file_map = {
        "S&P 500": "sp500.csv",
        "S&P 400 (MidCap)": "sp400.csv",
        "S&P 600 (SmallCap)": "sp600.csv",
        "NASDAQ 100": "nasdaq100.csv",
        "Dow Jones": "dow_jones.csv",
        "FTSE 100": "ftse100.csv",
        "FTSE 250": "ftse250.csv",
        "CAC 40": "cac40.csv",
        "DAX 40": "dax.csv",
        "GETTEX (Manual)": "gettex.csv"
    }
    
    for market in markets:
        filename = file_map.get(market)
        if filename:
            try:
                df = pd.read_csv(filename)
                for _, row in df.iterrows():
                    t = str(row['Ticker'])
                    tickers.append(t)
                    ticker_map[t] = str(row['Company'])
            except FileNotFoundError:
                st.error(f"⚠️ Could not find '{filename}'. Ensure it is uploaded to your GitHub repo.")
                
    # Remove duplicate tickers, but keep our dictionary map
    tickers = list(set(tickers))
    return tickers, ticker_map

# ==========================================
# 2. DATA FETCHING (4 Workers) & INDICATORS
# ==========================================
@st.cache_data(ttl=3600) # Cache live data for 1 hour to prevent API bans
def fetch_latest_data(tickers):
    """Fetches the last 6 months of data using exactly 4 concurrent workers."""
    # yfinance natively supports ThreadPoolExecutor via threads=4
    data = yf.download(tickers, period="6mo", group_by='ticker', threads=4, progress=False)
    
    latest_rows = []
    
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = data.copy()
            else:
                df = data[ticker].copy()
                
            df.dropna(inplace=True)
            if df.empty or len(df) < 50: 
                continue # Skip if not enough data for 50-day MAs
                
            # Calculate Indicators
            df['ma_20'] = df['Close'].rolling(window=20).mean()
            df['ma_50'] = df['Close'].rolling(window=50).mean()
            df['ema_8'] = df['Close'].ewm(span=8, adjust=False).mean()
            df['ema_21'] = df['Close'].ewm(span=21, adjust=False).mean()
            df['ma_20_slope'] = df['ma_20'].diff(5)
            
            ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            df['volume_avg_20'] = df['Volume'].rolling(window=20).mean()
            df['rvol'] = df['Volume'] / df['volume_avg_20']
            df['volume_trend'] = df['volume_avg_20'].diff(5)
            
            df['ret_5d'] = df['Close'].pct_change(5)
            df['ret_10d'] = df['Close'].pct_change(10)
            
            df['high_50d'] = df['High'].rolling(window=50).max()
            df['near_high'] = df['Close'] >= (df['high_50d'] * 0.95)
            df['close_near_high'] = df['Close'] >= (df['High'] - 0.2 * (df['High'] - df['Low']))
            
            df['post_earnings'] = (df['rvol'] > 3.0) & (df['ret_5d'] > 0.05)
            df['short_interest_proxy'] = (df['Close'] < df['ma_50']) & (df['rvol'] > 2.0)
            
            # Extract ONLY the most recent trading day for the scanner
            latest_day = df.iloc[-1:].copy()
            latest_day['Ticker'] = ticker
            latest_rows.append(latest_day)
            
        except Exception:
            continue
            
    if not latest_rows:
        return pd.DataFrame()
        
    final_df = pd.concat(latest_rows)
    # Liquidity filter
    final_df = final_df[(final_df['Close'] >= 5) & (final_df['volume_avg_20'] >= 1000000)]
    return final_df

# ==========================================
# 3. SCORING MODELS
# ==========================================
def score_chatgpt(df):
    s = pd.Series(0, index=df.index)
    s += np.where(df['Close'] > df['ma_20'], 10, 0)
    s += np.where(df['ma_20_slope'] > 0, 10, 0)
    s += np.where((df['Close'] - df['ma_20']) / df['ma_20'] < 0.08, 10, 0)
    s += np.where((df['rsi'] >= 55) & (df['rsi'] <= 70), 10, 0)
    s += np.where(df['ret_5d'] > 0, 5, 0)
    s += np.where(df['macd'] > df['macd_signal'], 5, 0)
    s += np.where(df['rvol'] >= 1.5, 15, 0)
    s += np.where(df['volume_trend'] > 0, 10, 0)
    s -= np.where(df['ret_5d'] > 0.15, 15, 0)
    return s

def score_grok(df):
    s = pd.Series(0, index=df.index)
    s += np.where(df['ret_5d'] > 0, 10, 0)
    s += np.where(df['ret_10d'] > df['ret_5d'], 5, 0)
    s += np.where(df['rvol'] >= 1.5, 10, 0)
    s += np.where(df['Close'] > df['ma_20'], 5, 0)
    s += np.where(df['ma_20'] > df['ma_50'], 5, 0)
    s += np.where(df['near_high'], 10, 0)
    return s

def score_gemini(df):
    s = pd.Series(0, index=df.index)
    s += np.where(df['ema_8'] > df['ema_21'], 15, 0)
    s += np.where(df['macd'] > df['macd_signal'], 15, 0)
    s += np.where((df['rsi'] >= 50) & (df['rsi'] <= 70), 10, 0)
    s += np.where(df['rvol'] >= 1.5, 20, np.where(df['rvol'] >= 1.2, 10, 0))
    s += np.where(df['close_near_high'], 20, 0)
    s += np.where(df['post_earnings'], 10, 0)
    s += np.where(df['short_interest_proxy'], 10, 0)
    return s

def score_hybrid(df):
    s = pd.Series(0, index=df.index)
    s += np.where(df['Close'] > df['ma_20'], 5, 0)
    s += np.where(df['ma_20_slope'] > 0, 5, 0)
    s += np.where((df['rsi'] >= 55) & (df['rsi'] <= 70), 8, 0)
    s += np.where(df['macd'] > df['macd_signal'], 6, 0)
    s += np.where(df['ret_10d'] > df['ret_5d'], 6, 0)
    s += np.where(df['rvol'] >= 1.5, 10, 0)
    s += np.where(df['close_near_high'], 5, 0)
    s += np.where(df['near_high'], 5, 0)
    s += np.where(df['post_earnings'], 10, 0)
    s -= np.where(df['ret_5d'] > 0.15, 10, 0)
    return s

# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="V2 Market Scanner", layout="wide")

st.title("⚡ V2 Live Market Scanner")
st.markdown("Scan major global markets and generate consensus momentum picks using 4 AI models.")

# Sidebar Settings
st.sidebar.header("Scanner Settings")
market_options = [
    "S&P 500", "S&P 400 (MidCap)", "S&P 600 (SmallCap)", 
    "NASDAQ 100", "Dow Jones", 
    "FTSE 100", "FTSE 250", "CAC 40", "DAX 40", "GETTEX (Manual)"
]
selected_markets = st.sidebar.multiselect(
    "Select Markets to Scan:",
    market_options,
    default=["NASDAQ 100"]
)

if st.sidebar.button("🚀 Run Live Scan"):
    if not selected_markets:
        st.warning("Please select at least one market.")
    else:
        with st.spinner("Loading tickers from local files..."):
            tickers, ticker_map = get_tickers_and_names(selected_markets)
            
        if not tickers:
            st.error("No tickers loaded. Check that your .csv files are uploaded.")
        else:
            st.sidebar.success(f"Loaded {len(tickers)} tickers.")
            
            with st.spinner("Scraping live data (4 workers)... This may take a minute."):
                live_data = fetch_latest_data(tickers)
                
            if live_data.empty:
                st.error("Failed to fetch data or no stocks met the minimum liquidity requirements.")
            else:
                with st.spinner("Calculating AI Scores..."):
                    # Map the company names to the dataframe
                    live_data['Company'] = live_data['Ticker'].map(ticker_map)
                    
                    live_data['ChatGPT_Score'] = score_chatgpt(live_data)
                    live_data['Grok_Score'] = score_grok(live_data)
                    live_data['Gemini_Score'] = score_gemini(live_data)
                    live_data['Hybrid_Score'] = score_hybrid(live_data)
                    
                    # Ranks (method='min' means ties get the same top rank)
                    live_data['Rank_ChatGPT'] = live_data['ChatGPT_Score'].rank(ascending=False, method='min')
                    live_data['Rank_Grok'] = live_data['Grok_Score'].rank(ascending=False, method='min')
                    live_data['Rank_Gemini'] = live_data['Gemini_Score'].rank(ascending=False, method='min')
                    live_data['Rank_Hybrid'] = live_data['Hybrid_Score'].rank(ascending=False, method='min')
                    
                    # Master Average Rank
                    live_data['Average_Rank'] = live_data[['Rank_ChatGPT', 'Rank_Grok', 'Rank_Gemini', 'Rank_Hybrid']].mean(axis=1)
                    
                    # Formatting
                    live_data['Close'] = live_data['Close'].round(2)
                    live_data['rsi'] = live_data['rsi'].round(2)
                    live_data['rvol'] = live_data['rvol'].round(2)
                    live_data['ret_5d'] = (live_data['ret_5d'] * 100).round(2).astype(str) + '%'
                    
                    display_cols = ['Ticker', 'Company', 'Close', 'rsi', 'rvol', 'ret_5d']
                    
                st.success(f"Scan complete for {len(live_data)} qualifying stocks. Prices as of last close.")
                
                # --- UI TABS ---
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "👑 Master Consensus", "🤖 ChatGPT", "🌌 Grok", "✨ Gemini", "🧬 Hybrid"
                ])
                
                with tab1:
                    st.subheader("Top 20: Master Consensus Ranking")
                    st.markdown("Sorted by the lowest average rank across all four models.")
                    master = live_data.sort_values('Average_Rank', ascending=True).head(20)
                    st.dataframe(master[['Ticker', 'Company', 'Average_Rank', 'Rank_ChatGPT', 'Rank_Grok', 'Rank_Gemini', 'Rank_Hybrid', 'Close', 'rvol']].reset_index(drop=True), use_container_width=True)
                    
                with tab2:
                    st.subheader("Top 20: ChatGPT Model (Trend Focus)")
                    chatgpt_top = live_data.sort_values('ChatGPT_Score', ascending=False).head(20)
                    st.dataframe(chatgpt_top[['Ticker', 'Company', 'ChatGPT_Score', 'Close', 'rsi', 'rvol', 'ret_5d']].reset_index(drop=True), use_container_width=True)
                    
                with tab3:
                    st.subheader("Top 20: Grok Model (Breakout Focus)")
                    grok_top = live_data.sort_values('Grok_Score', ascending=False).head(20)
                    st.dataframe(grok_top[['Ticker', 'Company', 'Grok_Score', 'Close', 'rsi', 'rvol', 'ret_5d']].reset_index(drop=True), use_container_width=True)
                    
                with tab4:
                    st.subheader("Top 20: Gemini Model (Volume & Catalyst Focus)")
                    gemini_top = live_data.sort_values('Gemini_Score', ascending=False).head(20)
                    st.dataframe(gemini_top[['Ticker', 'Company', 'Gemini_Score', 'Close', 'rsi', 'rvol', 'ret_5d']].reset_index(drop=True), use_container_width=True)
                    
                with tab5:
                    st.subheader("Top 20: Hybrid V2 Model (Best-of-All)")
                    hybrid_top = live_data.sort_values('Hybrid_Score', ascending=False).head(20)
                    st.dataframe(hybrid_top[['Ticker', 'Company', 'Hybrid_Score', 'Close', 'rsi', 'rvol', 'ret_5d']].reset_index(drop=True), use_container_width=True)
