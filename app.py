import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import time
import urllib.request
import xml.etree.ElementTree as ET

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
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment(ticker, nlp_pipe):
    """Fetches recent headlines from Yahoo, falls back to Google News, and scores with FinBERT."""
    if not nlp_pipe:
        return "No FinBERT ⚠️"
        
    headlines = []
    
    # --- SOURCE 1: Try Yahoo Finance ---
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        time.sleep(0.5) 
        
        if news and isinstance(news, list):
            headlines = [
                article.get('title') 
                for article in news[:5] 
                if isinstance(article, dict) and article.get('title')
            ]
    except Exception as e:
        print(f"[{ticker}] YF news fetch failed: {e}")

    # --- SOURCE 2: Fallback to Google News RSS ---
    if not headlines:
        try:
            url = f"https://news.google.com/rss/search?q={ticker}+stock+news&hl=en-US&gl=US&ceid=US:en"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            
            with urllib.request.urlopen(req, timeout=5) as response:
                xml_data = response.read()
                
            root = ET.fromstring(xml_data)
            
            for item in root.findall('.//item')[:5]:
                title_node = item.find('title')
                if title_node is not None and title_node.text:
                    clean_title = title_node.text.rsplit(' - ', 1)[0]
                    clean_title = clean_title[:200] # Prevent FinBERT size mismatch crash
                    headlines.append(clean_title)
        except Exception as e:
            print(f"[{ticker}] Google News fallback failed: {e}")

    if not headlines:
        return "No Headlines ⚪"
        
    # --- RUN FINBERT ---
    try:
        results = nlp_pipe(headlines)
        
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
            
    except Exception as e:
        print(f"[{ticker}] FinBERT Pipeline Error: {e}")
        return "Error ⚪"

# ==========================================
# 2. MARKET DATA UNIVERSE
# ==========================================
@st.cache_data
def get_tickers_and_names(markets):
    tickers, ticker_map = [], {}
    file_map = {
        "S&P 500": ("sp500.csv", ""), "S&P 400 (MidCap)": ("sp400.csv", ""), "S&P 600 (SmallCap)": ("sp600.csv", ""),
        "NASDAQ 100": ("nasdaq100.csv", ""), "Dow Jones": ("dow_jones.csv", ""), 
        "FTSE 100": ("ftse100.csv", ".L"), "FTSE 250": ("ftse250.csv", ".L"), 
        "CAC 40": ("cac40.csv", ".PA"), "DAX 40": ("dax.csv", ".DE"), "GETTEX (Manual)": ("gettex.csv", ".DE")
    }
    
    for market in markets:
        market_info = file_map.get(market)
        if market_info:
            filename, suffix = market_info
            try:
                df = pd.read_csv(filename)
                for _, row in df.iterrows():
                    t = str(row['Ticker']).strip().upper()
                    
                    if suffix:
                        t = t.split('-')[0]
                        t = t.split('.')[0]
                        t = f"{t}{suffix}"
                        
                    tickers.append(t)
                    ticker_map[t] = str(row['Company'])
            except FileNotFoundError:
                st.error(f"⚠️ Could not find '{filename}'.")
                
    return list(set(tickers)), ticker_map

# ==========================================
# 3. DATA FETCHING & INDICATORS (ROBUST MODE)
# ==========================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_latest_data(tickers):
    latest_rows = []
    chunk_size = 10
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    for chunk in chunks:
        data = pd.DataFrame()
        max_retries = 3
        
        for attempt in range(max_retries):
            data = yf.download(chunk, period="3mo", progress=False)
            if not data.empty:
                break 
            time.sleep(2) 
            
        if data.empty:
            continue 
            
        time.sleep(1.0)
        
        for ticker in chunk:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker in data.columns.get_level_values(1):
                        df = data.xs(ticker, axis=1, level=1).copy()
                    else:
                        continue
                else:
                    df = data.copy() if len(chunk) == 1 else data[ticker].copy()
                    
                df.ffill(inplace=True)
                df.dropna(subset=['Close', 'Volume', 'High', 'Low'], inplace=True)
                
                if df.empty or len(df) < 50: 
                    continue
                    
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
                df['rvol'] = df['Volume'] / (df['volume_avg_20'] + 1e-9)
                df['volume_trend'] = df['volume_avg_20'].diff(5)
                
                df['ret_5d'] = df['Close'].pct_change(5)
                df['ret_10d'] = df['Close'].pct_change(10)
                df['ret_21d'] = df['Close'].pct_change(21) # 1-Month Return
                
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
    return final_df[(final_df['Close'] >= 1) & (final_df['volume_avg_20'] >= 20000)]

# ==========================================
# 4. SCORING MODELS (SCALED / CONTINUOUS)
# ==========================================
def score_chatgpt(df):
    s = pd.Series(0.0, index=df.index)
    dist_ma = (df['Close'] - df['ma_20']) / (df['ma_20'] + 1e-9)
    s += np.clip(dist_ma * 100, 0, 10)
    s += np.clip(df['ma_20_slope'] * 10, 0, 10)
    s -= np.clip((dist_ma - 0.08) * 100, 0, 10)
    s += 10 - np.abs(df['rsi'] - 62) * 0.4
    s += np.clip(df['ret_5d'] * 100, 0, 10)
    macd_diff = df['macd'] - df['macd_signal']
    s += np.clip(macd_diff * 50, 0, 5)
    s += np.clip((df['rvol'] - 1) * 10, 0, 15)
    s += np.clip(df['volume_trend'] * 10, 0, 10)
    return s

def score_grok(df):
    s = pd.Series(0.0, index=df.index)
    s += np.clip(df['ret_5d'] * 100, 0, 15)
    s += np.clip((df['ret_10d'] - df['ret_5d']) * 100, 0, 10)
    s += np.clip((df['rvol'] - 1) * 10, 0, 15)
    s += np.clip((df['Close'] - df['ma_20']) / (df['ma_20'] + 1e-9) * 50, 0, 10)
    s += np.clip((df['ma_20'] - df['ma_50']) / (df['ma_50'] + 1e-9) * 50, 0, 10)
    dist_high = df['Close'] / (df['high_50d'] + 1e-9)
    s += np.clip((dist_high - 0.9) * 50, 0, 10)
    return s

def score_gemini(df):
    s = pd.Series(0.0, index=df.index)
    ema_gap = (df['ema_8'] - df['ema_21']) / (df['ema_21'] + 1e-9)
    s += np.clip(ema_gap * 100, 0, 15)
    macd_diff = df['macd'] - df['macd_signal']
    s += np.clip(macd_diff * 50, 0, 15)
    s += 10 - np.abs(df['rsi'] - 60) * 0.3
    s += np.clip((df['rvol'] - 1) * 20, 0, 25)
    close_pos = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-9)
    s += close_pos * 20
    s += np.where(df['post_earnings'], 10, 0)
    return s

def score_hybrid(df):
    s = pd.Series(0.0, index=df.index)
    dist_ma = (df['Close'] - df['ma_20']) / (df['ma_20'] + 1e-9)
    s += np.clip(dist_ma * 50, 0, 10)
    s += np.clip(df['ma_20_slope'] * 10, 0, 10)
    s += np.clip(df['ret_5d'] * 100, 0, 10)
    s += np.clip((df['ret_10d'] - df['ret_5d']) * 100, 0, 10)
    s += 10 - np.abs(df['rsi'] - 60) * 0.3
    s += np.clip((df['rvol'] - 1) * 15, 0, 15)
    dist_high = df['Close'] / (df['high_50d'] + 1e-9)
    s += np.clip((dist_high - 0.9) * 50, 0, 10)
    s += np.where(df['post_earnings'], 10, 0)
    return s

# --- 1-MONTH MODELS (FULLY SCALED) ---

def score_chatgpt_1m(df):
    s = pd.Series(0.0, index=df.index)
    
    # Trend Quality
    dist_ma50 = (df['Close'] - df['ma_50']) / (df['ma_50'] + 1e-9)
    ma_alignment = (df['ma_20'] - df['ma_50']) / (df['ma_50'] + 1e-9)

    s += np.clip(dist_ma50 * 100, 0, 12)
    s -= np.clip((dist_ma50 - 0.12) * 120, 0, 12)
    s += np.clip(ma_alignment * 120, 0, 12)
    
    # Sustainable Momentum
    s += np.clip(df['ret_21d'] * 100, 0, 12)
    momentum_balance = df['ret_21d'] - df['ret_10d']
    s += np.clip(momentum_balance * 120, 0, 10)
    
    # RSI Stability (smooth curve)
    s += 12 - np.abs(df['rsi'] - 55) * 0.4
    
    # Volume Consistency
    vol_trend_norm = df['volume_trend'] / (df['volume_avg_20'] + 1e-9)
    s += np.clip(vol_trend_norm * 60, 0, 10)
    s += np.clip((df['rvol'] - 1) * 8, 0, 8)
    
    # Positioning
    dist_high = df['Close'] / (df['high_50d'] + 1e-9)
    s += np.clip((dist_high - 0.85) * 60, 0, 10)
    s -= np.clip((dist_high - 0.98) * 120, 0, 8)
    
    return s


def score_grok_1m(df):
    s = pd.Series(0.0, index=df.index)
    
    # Strong Momentum (primary driver)
    s += np.clip(df['ret_21d'] * 140, 0, 30)
    
    # Acceleration
    momentum_acceleration = df['ret_21d'] - df['ret_10d']
    s += np.clip(momentum_acceleration * 140, 0, 15)
    
    # Trend Strength
    dist_ma50 = (df['Close'] - df['ma_50']) / (df['ma_50'] + 1e-9)
    ma_alignment = (df['ma_20'] - df['ma_50']) / (df['ma_50'] + 1e-9)
    s += np.clip(dist_ma50 * 100, 0, 12)
    s += np.clip(ma_alignment * 120, 0, 12)
    
    # Volume Confirmation
    s += np.clip((df['rvol'] - 1) * 10, 0, 12)
    vol_trend_norm = df['volume_trend'] / (df['volume_avg_20'] + 1e-9)
    s += np.clip(vol_trend_norm * 80, -5, 10)
    
    # Positioning (more aggressive)
    dist_high = df['Close'] / (df['high_50d'] + 1e-9)
    s += np.clip((dist_high - 0.80) * 80, 0, 15)
    s -= np.clip((dist_high - 0.97) * 100, 0, 10)
    
    # RSI (slightly higher tolerance)
    s += 12 - np.abs(df['rsi'] - 58) * 0.35
    
    return s


def score_gemini_1m(df):
    s = pd.Series(0.0, index=df.index)

    # Trend Structure (EMA vs MA)
    ema_gap = (df['ema_21'] - df['ma_50']) / (df['ma_50'] + 1e-9)
    s += np.clip(ema_gap * 140, 0, 18)
    
    # Sustained Momentum
    s += np.clip(df['ret_21d'] * 110, 0, 18)
    
    # RSI Stability
    s += 12 - np.abs(df['rsi'] - 55) * 0.35
    
    # Volume Consistency (core feature)
    vol_trend_norm = df['volume_trend'] / (df['volume_avg_20'] + 1e-9)
    s += np.clip(vol_trend_norm * 70, 0, 15)
    
    # Light RVOL confirmation
    s += np.clip((df['rvol'] - 1) * 6, 0, 8)
    
    # Positioning (prefers controlled trends)
    dist_high = df['Close'] / (df['high_50d'] + 1e-9)
    s += np.clip((dist_high - 0.85) * 50, 0, 8)
    
    return s


def score_hybrid_1m(df):
    s = pd.Series(0.0, index=df.index)
    
    # Trend (balanced)
    dist_ma50 = (df['Close'] - df['ma_50']) / (df['ma_50'] + 1e-9)
    ma_alignment = (df['ma_20'] - df['ma_50']) / (df['ma_50'] + 1e-9)

    s += np.clip(dist_ma50 * 80, 0, 10)
    s += np.clip(ma_alignment * 100, 0, 10)
    
    # Momentum + Acceleration
    s += np.clip(df['ret_21d'] * 110, 0, 15)
    momentum_balance = df['ret_21d'] - df['ret_10d']
    s += np.clip(momentum_balance * 100, 0, 12)
    
    # Volume Flow
    vol_trend_norm = df['volume_trend'] / (df['volume_avg_20'] + 1e-9)
    s += np.clip(vol_trend_norm * 60, 0, 12)
    s += np.clip((df['rvol'] - 1) * 8, 0, 10)
    
    # Positioning
    dist_high = df['Close'] / (df['high_50d'] + 1e-9)
    s += np.clip((dist_high - 0.85) * 60, 0, 10)
    s -= np.clip((dist_high - 0.98) * 100, 0, 6)
    
    # RSI Balance
    s += 12 - np.abs(df['rsi'] - 56) * 0.35
    
    return s

# ==========================================
# 5. RAG PANDAS FORMATTING
# ==========================================
def color_rsi(val):
    if pd.isna(val): return ''
    if 50 <= val <= 70: return 'color: #00FF00' 
    elif val > 70 or 40 <= val < 50: return 'color: #FFA500' 
    return 'color: #FF0000' 

def color_rvol(val):
    if pd.isna(val): return ''
    if val >= 1.5: return 'color: #00FF00' 
    elif 1.0 <= val < 1.5: return 'color: #FFA500' 
    return 'color: #FF0000' 

def color_ret(val):
    if pd.isna(val): return ''
    if val >= 0.02: return 'color: #00FF00' 
    elif val <= -0.02: return 'color: #FF0000' 
    return 'color: #FFA500' 

def apply_rag_formatting(df):
    df = df.reset_index(drop=True)
    styler = df.style
    
    if 'rsi' in df.columns:
        styler = styler.map(color_rsi, subset=['rsi'])
    if 'rvol' in df.columns:
        styler = styler.map(color_rvol, subset=['rvol'])
    if 'ret_5d' in df.columns:
        styler = styler.map(color_ret, subset=['ret_5d'])
    if 'ret_21d' in df.columns:
        styler = styler.map(color_ret, subset=['ret_21d'])
        
    format_dict = {
        'Close': '{:.2f}', 'rsi': '{:.1f}', 'rvol': '{:.2f}',
        'ret_5d': '{:.2%}', 'ret_10d': '{:.2%}', 'ret_21d': '{:.2%}', 
        'ma_20': '{:.2f}', 'ma_50': '{:.2f}', 
        'ema_8': '{:.2f}', 'ema_21': '{:.2f}',
        'ma_20_slope': '{:.3f}', 'macd': '{:.3f}', 'macd_signal': '{:.3f}',
        'volume_trend': '{:.0f}', 'vol_trend_norm': '{:.2f}',
        'dist_ma50': '{:.3f}', 'ma_alignment': '{:.3f}', 'dist_high': '{:.3f}',
        'Average_Rank': '{:.0f}', 'Rank_ChatGPT': '{:.0f}',
        'Rank_Grok': '{:.0f}', 'Rank_Gemini': '{:.0f}', 'Rank_Hybrid': '{:.0f}',
        'Average_Rank_1M': '{:.0f}', 'Rank_ChatGPT_1M': '{:.0f}',
        'Rank_Grok_1M': '{:.0f}', 'Rank_Gemini_1M': '{:.0f}', 'Rank_Hybrid_1M': '{:.0f}'
    }
    
    safe_format_dict = {k: v for k, v in format_dict.items() if k in df.columns}
    return styler.format(safe_format_dict)


# ==========================================
# 6. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="V2 Market Scanner", layout="wide")

st.title("⚡ V2 Live Market Scanner (Dual Timeframe)")
st.markdown("Scan major global markets and generate consensus momentum picks using 4 AI models.")

with st.expander("📚 How Scoring & FinBERT Sentiment Works (Click to Expand)", expanded=False):
    st.markdown("""
    ### FinBERT News Sentiment
    The Master Consensus top 20 runs its recent headlines through **FinBERT**, a financial NLP neural network.
    * 🟢 **Bullish:** Earnings beats, upgrades, new contracts, or positive guidance.
    * 🟡 **Neutral:** Routine market reporting, sector-wide news, or mixed results.
    * 🔴 **Bearish:** Missed earnings, downgrades, lawsuits, or lowered guidance.

    ### The 4 Models (Dual Timeframe)
    Each AI now computes two distinct score sets: **Short-Term (1-2 weeks)** and **Medium-Term (1-Month Swing)**.
    * 🤖 **ChatGPT (Trend Focus):** Rewards moving average strength and RSI. The 1M model tracks the 50-day MA.
    * 🌌 **Grok (Breakout Focus):** Cares purely about accelerating price action and high proximity breakouts.
    * ✨ **Gemini (Volume/Catalyst Focus):** Looks for explosive RVOL and EMA crossovers. The 1M model targets rising volume trends.
    * 🧬 **Hybrid (Best-of-All):** A balanced blend combining all distinct logic triggers.
    """)

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
            live_data = fetch_latest_data(tickers)
                
            if live_data.empty:
                st.error("Failed to fetch data or no stocks met liquidity requirements.")
            else:
                with st.spinner("Calculating AI Scores & Timeframes..."):
                    live_data['Company'] = live_data['Ticker'].map(ticker_map)
                    
                    # Short Term Scoring
                    live_data['ChatGPT_Score'] = score_chatgpt(live_data)
                    live_data['Grok_Score'] = score_grok(live_data)
                    live_data['Gemini_Score'] = score_gemini(live_data)
                    live_data['Hybrid_Score'] = score_hybrid(live_data)
                    
                    live_data['Rank_ChatGPT'] = live_data['ChatGPT_Score'].rank(ascending=False, method='min')
                    live_data['Rank_Grok'] = live_data['Grok_Score'].rank(ascending=False, method='min')
                    live_data['Rank_Gemini'] = live_data['Gemini_Score'].rank(ascending=False, method='min')
                    live_data['Rank_Hybrid'] = live_data['Hybrid_Score'].rank(ascending=False, method='min')
                    live_data['Average_Rank'] = live_data[['Rank_ChatGPT', 'Rank_Grok', 'Rank_Gemini', 'Rank_Hybrid']].mean(axis=1)
                    
                    # 1-Month Scoring
                    live_data['ChatGPT_Score_1M'] = score_chatgpt_1m(live_data)
                    live_data['Grok_Score_1M'] = score_grok_1m(live_data)
                    live_data['Gemini_Score_1M'] = score_gemini_1m(live_data)
                    live_data['Hybrid_Score_1M'] = score_hybrid_1m(live_data)

                    live_data['Rank_ChatGPT_1M'] = live_data['ChatGPT_Score_1M'].rank(ascending=False, method='min')
                    live_data['Rank_Grok_1M'] = live_data['Grok_Score_1M'].rank(ascending=False, method='min')
                    live_data['Rank_Gemini_1M'] = live_data['Gemini_Score_1M'].rank(ascending=False, method='min')
                    live_data['Rank_Hybrid_1M'] = live_data['Hybrid_Score_1M'].rank(ascending=False, method='min')
                    live_data['Average_Rank_1M'] = live_data[['Rank_ChatGPT_1M', 'Rank_Grok_1M', 'Rank_Gemini_1M', 'Rank_Hybrid_1M']].mean(axis=1)

                master = live_data.sort_values('Average_Rank', ascending=True).head(20).copy()
                master_1m = live_data.sort_values('Average_Rank_1M', ascending=True).head(20).copy()
                
                # NLP Sentiment (Applied to Short-Term and 1-Month Master Top 20s)
                nlp = load_finbert()
                
                # 1. Short-Term FinBERT Loop
                sentiments = []
                sentiment_bar = st.progress(0, text="Analyzing Short-Term Master News Sentiment...")
                for idx, row in master.iterrows():
                    sent = analyze_sentiment(row['Ticker'], nlp)
                    sentiments.append(sent)
                    current_step = len(sentiments)
                    sentiment_bar.progress(current_step / 20, text=f"Analyzing News for {row['Ticker']} ({current_step}/20)...")
                master['FinBERT_Sentiment'] = sentiments
                sentiment_bar.empty()

                # 2. 1-Month FinBERT Loop
                sentiments_1m = []
                sentiment_bar_1m = st.progress(0, text="Analyzing 1-Month Master News Sentiment...")
                for idx, row in master_1m.iterrows():
                    sent = analyze_sentiment(row['Ticker'], nlp)
                    sentiments_1m.append(sent)
                    current_step = len(sentiments_1m)
                    sentiment_bar_1m.progress(current_step / 20, text=f"Analyzing News for {row['Ticker']} ({current_step}/20)...")
                master_1m['FinBERT_Sentiment'] = sentiments_1m
                sentiment_bar_1m.empty()
                
                st.success(f"Scan complete for {len(live_data)} qualifying stocks.")
                
                # --- UI TABS ---
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "👑 Master Consensus", "🤖 ChatGPT", "🌌 Grok", "✨ Gemini", "🧬 Hybrid"
                ])
                
                with tab1:
                    st.subheader("⚡ Short-Term (1-2 Weeks) Top 20")
                    master_cols = ['Ticker', 'Company', 'FinBERT_Sentiment', 'Average_Rank', 'Rank_ChatGPT', 'Rank_Grok', 'Rank_Gemini', 'Rank_Hybrid', 'Close', 'rsi', 'rvol', 'ret_5d']
                    safe_master_cols = [c for c in master_cols if c in master.columns]
                    st.dataframe(apply_rag_formatting(master[safe_master_cols]), use_container_width=True, hide_index=True)
                    
                    st.divider() 
                    
                    st.subheader("📅 Medium-Term (1 Month) Top 20")
                    master_cols_1m = ['Ticker', 'Company', 'FinBERT_Sentiment', 'Average_Rank_1M', 'Rank_ChatGPT_1M', 'Rank_Grok_1M', 'Rank_Gemini_1M', 'Rank_Hybrid_1M', 'Close', 'rsi', 'rvol', 'ret_21d']
                    safe_master_cols_1m = [c for c in master_cols_1m if c in master_1m.columns]
                    st.dataframe(apply_rag_formatting(master_1m[safe_master_cols_1m]), use_container_width=True, hide_index=True)
                    
                with tab2:
                    st.subheader("⚡ Short-Term Trend Focus")
                    chatgpt_top = live_data.sort_values(by=['ChatGPT_Score', 'Average_Rank'], ascending=[False, True]).head(20)
                    cg_cols = ['Ticker', 'Company', 'Rank_ChatGPT', 'ChatGPT_Score', 'Close', 'ma_20', 'ma_20_slope', 'rsi', 'macd', 'macd_signal', 'rvol', 'volume_trend', 'ret_5d']
                    safe_cg_cols = [c for c in cg_cols if c in chatgpt_top.columns]
                    st.dataframe(apply_rag_formatting(chatgpt_top[safe_cg_cols]), use_container_width=True, hide_index=True)
                    
                    st.divider()
                    
                    st.subheader("📅 1-Month Trend Focus (Transparent Scoring)")
                    chatgpt_top_1m = live_data.sort_values(by=['ChatGPT_Score_1M', 'Average_Rank_1M'], ascending=[False, True]).head(20)
                    cg_cols_1m = ['Ticker', 'Company', 'Rank_ChatGPT_1M', 'ChatGPT_Score_1M', 'Close', 'ma_50', 'dist_ma50', 'ma_alignment', 'ret_21d', 'momentum_balance', 'rsi', 'rvol', 'volume_trend', 'vol_trend_norm', 'dist_high']
                    safe_cg_cols_1m = [c for c in cg_cols_1m if c in chatgpt_top_1m.columns]
                    st.dataframe(apply_rag_formatting(chatgpt_top_1m[safe_cg_cols_1m]), use_container_width=True, hide_index=True)
                    
                with tab3:
                    st.subheader("⚡ Short-Term Breakout Focus")
                    grok_top = live_data.sort_values(by=['Grok_Score', 'Average_Rank'], ascending=[False, True]).head(20)
                    grok_cols = ['Ticker', 'Company', 'Rank_Grok', 'Grok_Score', 'Close', 'ma_20', 'ma_50', 'near_high', 'rvol', 'ret_5d', 'ret_10d']
                    safe_grok_cols = [c for c in grok_cols if c in grok_top.columns]
                    st.dataframe(apply_rag_formatting(grok_top[safe_grok_cols]), use_container_width=True, hide_index=True)
                    
                    st.divider()
                    
                    st.subheader("📅 1-Month Breakout Focus (Transparent Scoring)")
                    grok_top_1m = live_data.sort_values(by=['Grok_Score_1M', 'Average_Rank_1M'], ascending=[False, True]).head(20)
                    grok_cols_1m = ['Ticker', 'Company', 'Rank_Grok_1M', 'Grok_Score_1M', 'Close', 'ret_21d', 'momentum_acceleration', 'ma_50', 'dist_ma50', 'ma_alignment', 'near_high', 'dist_high', 'rsi', 'rvol', 'vol_trend_norm']
                    safe_grok_cols_1m = [c for c in grok_cols_1m if c in grok_top_1m.columns]
                    st.dataframe(apply_rag_formatting(grok_top_1m[safe_grok_cols_1m]), use_container_width=True, hide_index=True)
                    
                with tab4:
                    st.subheader("⚡ Short-Term Catalyst Focus")
                    gemini_top = live_data.sort_values(by=['Gemini_Score', 'Average_Rank'], ascending=[False, True]).head(20)
                    gem_cols = ['Ticker', 'Company', 'Rank_Gemini', 'Gemini_Score', 'Close', 'ema_8', 'ema_21', 'macd', 'macd_signal', 'rsi', 'rvol', 'close_near_high', 'post_earnings']
                    safe_gem_cols = [c for c in gem_cols if c in gemini_top.columns]
                    st.dataframe(apply_rag_formatting(gemini_top[safe_gem_cols]), use_container_width=True, hide_index=True)

                    st.divider()
                    
                    st.subheader("📅 1-Month Squeeze Focus (Transparent Scoring)")
                    gemini_top_1m = live_data.sort_values(by=['Gemini_Score_1M', 'Average_Rank_1M'], ascending=[False, True]).head(20)
                    gem_cols_1m = ['Ticker', 'Company', 'Rank_Gemini_1M', 'Gemini_Score_1M', 'Close', 'ema_21', 'ma_50', 'dist_ma50', 'ma_alignment', 'rsi', 'volume_trend', 'vol_trend_norm', 'rvol', 'ret_21d']
                    safe_gem_cols_1m = [c for c in gem_cols_1m if c in gemini_top_1m.columns]
                    st.dataframe(apply_rag_formatting(gemini_top_1m[safe_gem_cols_1m]), use_container_width=True, hide_index=True)
                    
                with tab5:
                    st.subheader("⚡ Short-Term Best-of-All")
                    hybrid_top = live_data.sort_values(by=['Hybrid_Score', 'Average_Rank'], ascending=[False, True]).head(20)
                    hyb_cols = ['Ticker', 'Company', 'Rank_Hybrid', 'Hybrid_Score', 'Close', 'ma_20', 'ma_20_slope', 'macd', 'macd_signal', 'rsi', 'rvol', 'near_high', 'post_earnings', 'ret_5d', 'ret_10d']
                    safe_hyb_cols = [c for c in hyb_cols if c in hybrid_top.columns]
                    st.dataframe(apply_rag_formatting(hybrid_top[safe_hyb_cols]), use_container_width=True, hide_index=True)

                    st.divider()
                    
                    st.subheader("📅 1-Month Best-of-All (Transparent Scoring)")
                    hybrid_top_1m = live_data.sort_values(by=['Hybrid_Score_1M', 'Average_Rank_1M'], ascending=[False, True]).head(20)
                    hyb_cols_1m = ['Ticker', 'Company', 'Rank_Hybrid_1M', 'Hybrid_Score_1M', 'Close', 'ma_50', 'dist_ma50', 'ma_alignment', 'ret_21d', 'momentum_balance', 'rsi', 'rvol', 'vol_trend_norm', 'dist_high']
                    safe_hyb_cols_1m = [c for c in hyb_cols_1m if c in hybrid_top_1m.columns]
                    st.dataframe(apply_rag_formatting(hybrid_top_1m[safe_hyb_cols_1m]), use_container_width=True, hide_index=True)
