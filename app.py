import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings

# Try to import FinBERT libraries, catch gracefully if not installed
try:
    from transformers import pipeline
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==========================================
# 1. FINBERT NLP SETUP (Cached)
# ==========================================
@st.cache_resource
def load_finbert():
    """Loads the FinBERT model into memory once to prevent reloading."""
    if not FINBERT_AVAILABLE:
        return None
    # ProsusAI/finbert is the industry standard for financial sentiment
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment(ticker, nlp_pipe):
    """Fetches recent headlines and returns an aggregated FinBERT sentiment."""
    if not nlp_pipe:
        return "No FinBERT ⚠️"
        
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return "No News ⚪"
            
        # Grab the top 5 most recent headlines
        headlines = [article['title'] for article in news[:5]]
        results = nlp_pipe(headlines)
        
        # FinBERT outputs labels: 'positive', 'negative', 'neutral'
        score = 0
        for res in results:
            if res['label'] == 'positive':
                score += 1
            elif res['label'] == 'negative':
                score -= 1
                
        if score >= 1:
            return "Bullish 🟢"
        elif score <= -1:
            return "Bearish 🔴"
        else:
            return "Neutral 🟡"
    except Exception:
        return "Error ⚪"

# ==========================================
# 2. MARKET DATA UNIVERSE
# ==========================================
@st.cache_data
def get_tickers_and_names(markets):
    tickers, ticker_map = [], {}
    file_map = {
        "S&P 500": "sp500.csv", "S&P 400 (MidCap)": "sp400.csv", "S&P 600 (SmallCap)": "sp600.csv",
        "NASDAQ 100": "nasdaq100.csv", "Dow Jones": "dow_jones.csv", "FTSE 100": "ftse100.csv",
        "FTSE 250": "ftse250.csv", "CAC 40": "cac40.csv", "DAX 40": "dax.csv", "GETTEX (Manual)": "gettex.csv"
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
                st.error(f"⚠️ Could not find '{filename}'.")
    return list(set(tickers)), ticker_map

# ==========================================
# 3. DATA FETCHING & INDICATORS (STEALTH MODE)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_latest_data(tickers):
    latest_rows = []
    
    # 1. Break the massive ticker list into safe chunks of 50 
    # to avoid triggering Yahoo's aggressive rate limiters.
    chunk_size = 50
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    # 2. We remove 'threads=4' because concurrent requests trigger Yahoo's IP ban
    for chunk in chunks:
        # Download data for this chunk sequentially
        data = yf.download(chunk, period="6mo", progress=False)
        
        if data.empty:
            continue # If this chunk fails/rate-limits, skip and try the next one
            
        for ticker in chunk:
            try:
                # Bulletproof MultiIndex extraction (Handles both old and new YFinance updates)
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker in data.columns.get_level_values(1):
                        df = data.xs(ticker, axis=1, level=1).copy()
                    else:
                        continue
                else:
                    df = data.copy() if len(chunk) == 1 else data[ticker].copy()
                    
                df.dropna(inplace=True)
                if df.empty or len(df) < 50: continue
                    
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
                df['rsi'] = 100 - (100 / (1 + (gain / loss)))
                
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
                
                latest_day = df.iloc[-1:].copy()
                latest_day['Ticker'] = ticker
                latest_rows.append(latest_day)
            except Exception: 
                continue
                
    if not latest_rows: 
        st.error("⚠️ Yahoo Finance returned no data! You are still temporarily rate-limited. Wait 15-30 minutes and try again.")
        return pd.DataFrame()
        
    final_df = pd.concat(latest_rows)
    return final_df[(final_df['Close'] >= 1) & (final_df['volume_avg_20'] >= 50000)]

# ==========================================
# 4. SCORING MODELS
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
    return s

def score_hybrid(df):
    s = pd.Series(0, index=df.index)
    s += np.where(df['Close'] > df['ma_20'], 5, 0)
    s += np.where(df['ma_20_slope'] > 0, 5, 0)
    s += np.where((df['rsi'] >= 55) & (df['rsi'] <= 70), 8, 0)
    s += np.where(df['macd'] > df['macd_signal'], 6, 0)
    s += np.where(df['ret_10d'] > df['ret_5d'], 6, 0)
    s += np.where(df['rvol'] >= 1.5, 10, 0)
    s += np.where(df['near_high'], 5, 0)
    s += np.where(df['post_earnings'], 10, 0)
    return s

# ==========================================
# 5. RAG PANDAS FORMATTING
# ==========================================
def color_rsi(val):
    if pd.isna(val): return ''
    if 50 <= val <= 70: return 'color: #00FF00' # Green (Bullish zone)
    elif val > 70 or 40 <= val < 50: return 'color: #FFA500' # Amber (Overbought/Neutral)
    return 'color: #FF0000' # Red (Bearish)

def color_rvol(val):
    if pd.isna(val): return ''
    if val >= 1.5: return 'color: #00FF00' # Green (High vol)
    elif 1.0 <= val < 1.5: return 'color: #FFA500' # Amber (Avg vol)
    return 'color: #FF0000' # Red (Low vol)

def color_ret(val):
    if pd.isna(val): return ''
    if val >= 0.02: return 'color: #00FF00' # Green (> 2%)
    elif val <= -0.02: return 'color: #FF0000' # Red (< -2%)
    return 'color: #FFA500' # Amber (Chop)

def apply_rag_formatting(df):
    """Applies RAG colors and formats floats for display."""
    return df.style.map(color_rsi, subset=['rsi']) \
                   .map(color_rvol, subset=['rvol']) \
                   .map(color_ret, subset=['ret_5d']) \
                   .format({
                       'Close': '{:.2f}',
                       'rsi': '{:.1f}',
                       'rvol': '{:.2f}',
                       'ret_5d': '{:.2%}',
                       'ret_10d': '{:.2%}',
                       'ma_20': '{:.2f}',
                       'ema_8': '{:.2f}'
                   })

# ==========================================
# 6. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="V2 Market Scanner", layout="wide")

st.title("⚡ V2 Live Market Scanner")
st.markdown("Scan major global markets and generate consensus momentum picks using 4 AI models.")

# --- EXPLANATION UI ---
with st.expander("📚 How Scoring & FinBERT Sentiment Works (Click to Expand)", expanded=False):
    st.markdown("""
    ### FinBERT News Sentiment
    The Master Consensus top 20 runs its recent headlines through **FinBERT**, a financial NLP neural network.
    * 🟢 **Bullish:** Earnings beats, upgrades, new contracts, or positive guidance.
    * 🟡 **Neutral:** Routine market reporting, sector-wide news, or mixed results.
    * 🔴 **Bearish:** Missed earnings, downgrades, lawsuits, or lowered guidance.

    ### The 4 Models
    * 🤖 **ChatGPT (Trend Focus):** Looks for steady momentum. Rewards stocks trading cleanly above their 20-day moving average with an RSI in the "sweet spot" (55-70). Punishes overextended stocks.
    * 🌌 **Grok (Breakout Focus):** Aggressive momentum. Cares purely about accelerating price action (10-day returns > 5-day returns) and stocks breaking into new 50-day highs on volume.
    * ✨ **Gemini (Volume/Catalyst Focus):** Looks for explosive action under the hood. Heavily rewards massive relative volume (RVOL > 1.5), EMA crossovers, and post-earnings drift setups.
    * 🧬 **Hybrid (Best-of-All):** A balanced blend combining ChatGPT's safety, Grok's breakout triggers, and Gemini's volume requirements.
    
    ### RAG Metric Formatting (Red / Amber / Green)
    * **RSI:** 🟢 50-70 (Trend) | 🟡 40-50 or >70 (Chop/Overbought) | 🔴 < 40 (Bearish)
    * **RVOL:** 🟢 > 1.5 (High Interest) | 🟡 1.0 - 1.5 (Normal) | 🔴 < 1.0 (Low Interest)
    * **5-Day Return:** 🟢 > +2% | 🟡 -2% to +2% | 🔴 < -2%
    """)

# Sidebar Settings
st.sidebar.header("Scanner Settings")
if not FINBERT_AVAILABLE:
    st.sidebar.error("⚠️ FinBERT libraries not found. Run `pip install transformers torch` in your terminal to enable news sentiment.")
    
market_options = [
    "S&P 500", "S&P 400 (MidCap)", "S&P 600 (SmallCap)", 
    "NASDAQ 100", "Dow Jones", "FTSE 100", "FTSE 250", 
    "CAC 40", "DAX 40", "GETTEX (Manual)"
]
selected_markets = st.sidebar.multiselect("Select Markets to Scan:", market_options, default=["NASDAQ 100"])

if st.sidebar.button("🚀 Run Live Scan"):
    if not selected_markets:
        st.warning("Please select at least one market.")
    else:
        with st.spinner("Loading tickers from local files..."):
            tickers, ticker_map = get_tickers_and_names(selected_markets)
            
        if not tickers:
            st.error("No tickers loaded. Check that your .csv files are uploaded.")
        else:
            with st.spinner("Scraping live data (4 workers)... This may take a minute."):
                live_data = fetch_latest_data(tickers)
                
            if live_data.empty:
                st.error("Failed to fetch data or no stocks met liquidity requirements.")
            else:
                with st.spinner("Calculating AI Scores..."):
                    live_data['Company'] = live_data['Ticker'].map(ticker_map)
                    
                    live_data['ChatGPT_Score'] = score_chatgpt(live_data)
                    live_data['Grok_Score'] = score_grok(live_data)
                    live_data['Gemini_Score'] = score_gemini(live_data)
                    live_data['Hybrid_Score'] = score_hybrid(live_data)
                    
                    live_data['Rank_ChatGPT'] = live_data['ChatGPT_Score'].rank(ascending=False, method='min')
                    live_data['Rank_Grok'] = live_data['Grok_Score'].rank(ascending=False, method='min')
                    live_data['Rank_Gemini'] = live_data['Gemini_Score'].rank(ascending=False, method='min')
                    live_data['Rank_Hybrid'] = live_data['Hybrid_Score'].rank(ascending=False, method='min')
                    live_data['Average_Rank'] = live_data[['Rank_ChatGPT', 'Rank_Grok', 'Rank_Gemini', 'Rank_Hybrid']].mean(axis=1)
                    
                # Master list processing + FinBERT
                master = live_data.sort_values('Average_Rank', ascending=True).head(20).copy()
                
                with st.spinner("Running FinBERT AI on Top 20 News Headlines..."):
                    nlp = load_finbert()
                    master['FinBERT_Sentiment'] = master['Ticker'].apply(lambda t: analyze_sentiment(t, nlp))
                
                st.success(f"Scan complete for {len(live_data)} qualifying stocks.")
                
                # --- UI TABS ---
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "👑 Master Consensus", "🤖 ChatGPT", "🌌 Grok", "✨ Gemini", "🧬 Hybrid"
                ])
                
                with tab1:
                    st.subheader("Top 20: Master Consensus + News Sentiment")
                    st.markdown("Sorted by the lowest average rank. **Sentiment is calculated from the last 5 news headlines.**")
                    master_cols = ['Ticker', 'Company', 'FinBERT_Sentiment', 'Average_Rank', 'Rank_ChatGPT', 'Rank_Grok', 'Rank_Gemini', 'Close', 'rsi', 'rvol', 'ret_5d']
                    st.dataframe(apply_rag_formatting(master[master_cols]), use_container_width=True, hide_index=True)
                    
                with tab2:
                    st.subheader("🤖 ChatGPT Model (Trend Focus)")
                    st.markdown("Rewards moving average strength and healthy RSI. Punishes over-extension.")
                    chatgpt_top = live_data.sort_values(by=['ChatGPT_Score', 'Average_Rank'], ascending=[False, True]).head(20)
                    cg_cols = ['Ticker', 'Company', 'ChatGPT_Score', 'Close', 'ma_20', 'rsi', 'rvol', 'ret_5d']
                    st.dataframe(apply_rag_formatting(chatgpt_top[cg_cols]), use_container_width=True, hide_index=True)
                    
                with tab3:
                    st.subheader("🌌 Grok Model (Breakout Focus)")
                    st.markdown("Rewards strong short-term breakouts and proximity to the 50-day high.")
                    grok_top = live_data.sort_values(by=['Grok_Score', 'Average_Rank'], ascending=[False, True]).head(20)
                    grok_cols = ['Ticker', 'Company', 'Grok_Score', 'Close', 'near_high', 'ret_10d', 'ret_5d', 'rvol', 'rsi']
                    st.dataframe(apply_rag_formatting(grok_top[grok_cols]), use_container_width=True, hide_index=True)
                    
                with tab4:
                    st.subheader("✨ Gemini Model (Volume & Catalyst Focus)")
                    st.markdown("Rewards aggressive relative volume, EMA crossovers, and post-earnings setups.")
                    gemini_top = live_data.sort_values(by=['Gemini_Score', 'Average_Rank'], ascending=[False, True]).head(20)
                    gem_cols = ['Ticker', 'Company', 'Gemini_Score', 'Close', 'ema_8', 'rvol', 'post_earnings', 'rsi', 'ret_5d']
                    st.dataframe(apply_rag_formatting(gemini_top[gem_cols]), use_container_width=True, hide_index=True)
                    
                with tab5:
                    st.subheader("🧬 Hybrid V2 Model")
                    hybrid_top = live_data.sort_values(by=['Hybrid_Score', 'Average_Rank'], ascending=[False, True]).head(20)
                    hyb_cols = ['Ticker', 'Company', 'Hybrid_Score', 'Close', 'rsi', 'rvol', 'ret_10d', 'ret_5d']
                    st.dataframe(apply_rag_formatting(hybrid_top[hyb_cols]), use_container_width=True, hide_index=True)
