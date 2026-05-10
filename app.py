import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import time
import urllib.request
import xml.etree.ElementTree as ET

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
    if not FINBERT_AVAILABLE:
        return None
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment(ticker, nlp_pipe):
    if not nlp_pipe:
        return "No FinBERT ⚠️"
    headlines = []
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

    if not headlines:
        try:
            url = f"https://news.google.com/rss/search?q={ticker}+stock+news&hl=en-US&gl=US&ceid=US:en"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                xml_data = response.read()
            root = ET.fromstring(xml_data)
            for item in root.findall('.//item')[:5]:
                title_node = item.find('title')
                if title_node is not None and title_node.text:
                    clean_title = title_node.text.rsplit(' - ', 1)[0][:200]
                    headlines.append(clean_title)
        except Exception as e:
            print(f"[{ticker}] Google News fallback failed: {e}")

    if not headlines:
        return "No Headlines ⚪"
    try:
        results = nlp_pipe(headlines)
        score = 0
        for res in results:
            if res['label'] == 'positive': score += 1
            elif res['label'] == 'negative': score -= 1
        if score >= 1: return "Bullish 🟢"
        elif score <= -1: return "Bearish 🔴"
        else: return "Neutral 🟡"
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
        "S&P 500": ("sp500.csv", ""), "S&P 400 (MidCap)": ("sp400.csv", ""),
        "S&P 600 (SmallCap)": ("sp600.csv", ""), "NASDAQ 100": ("nasdaq100.csv", ""),
        "Dow Jones": ("dow_jones.csv", ""), "FTSE 100": ("ftse100.csv", ".L"),
        "FTSE 250": ("ftse250.csv", ".L"), "CAC 40": ("cac40.csv", ".PA"),
        "DAX 40": ("dax.csv", ".DE"), "GETTEX (Manual)": ("gettex.csv", ".DE")
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
                        t = t.split('-')[0].split('.')[0]
                        t = f"{t}{suffix}"
                        if t == "BT.L": t = "BT-A.L"
                    tickers.append(t)
                    ticker_map[t] = str(row['Company'])
            except FileNotFoundError:
                st.error(f"⚠️ Could not find '{filename}'.")
    return list(set(tickers)), ticker_map


# ==========================================
# 3. DATA FETCHING & INDICATORS
#    Fetches 1 year to support 3M/6M returns,
#    200-day MA, and 126-day high lookbacks.
# ==========================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_latest_data(tickers):
    latest_rows = []
    chunk_size = 10
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]

    for chunk in chunks:
        data = pd.DataFrame()
        for attempt in range(3):
            # Extended to 1y to power 3M/6M scoring
            data = yf.download(chunk, period="1y", progress=False)
            if not data.empty: break
            time.sleep(2)
        if data.empty: continue
        time.sleep(1.0)

        for ticker in chunk:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker in data.columns.get_level_values(1):
                        df = data.xs(ticker, axis=1, level=1).copy()
                    else: continue
                else:
                    df = data.copy() if len(chunk) == 1 else data[ticker].copy()

                df.ffill(inplace=True)
                df.dropna(subset=['Close', 'Volume', 'High', 'Low'], inplace=True)
                if df.empty or len(df) < 21: continue

                # ---- SHORT-TERM INDICATORS ----
                df['ma_20']         = df['Close'].rolling(20, min_periods=1).mean()
                df['ma_50']         = df['Close'].rolling(50, min_periods=1).mean()
                df['ema_8']         = df['Close'].ewm(span=8, adjust=False).mean()
                df['ema_21']        = df['Close'].ewm(span=21, adjust=False).mean()
                df['ma_20_slope']   = df['ma_20'].diff(5).fillna(0)

                ema12 = df['Close'].ewm(span=12, adjust=False).mean()
                ema26 = df['Close'].ewm(span=26, adjust=False).mean()
                df['macd']          = ema12 - ema26
                df['macd_signal']   = df['macd'].ewm(span=9, adjust=False).mean()

                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
                df['rsi']           = (100 - (100 / (1 + (gain / loss)))).fillna(50)

                df['volume_avg_20'] = df['Volume'].rolling(20, min_periods=1).mean()
                df['rvol']          = df['Volume'] / (df['volume_avg_20'] + 1e-9)
                df['volume_trend']  = df['volume_avg_20'].diff(5).fillna(0)

                df['ret_5d']        = df['Close'].pct_change(5).fillna(0)
                df['ret_10d']       = df['Close'].pct_change(10).fillna(0)
                df['ret_21d']       = df['Close'].pct_change(21).fillna(0)

                df['high_50d']      = df['High'].rolling(50, min_periods=1).max()
                df['near_high']     = df['Close'] >= (df['high_50d'] * 0.95)
                df['close_near_high'] = df['Close'] >= (df['High'] - 0.2 * (df['High'] - df['Low']))
                df['post_earnings'] = (df['rvol'] > 3.0) & (df['ret_5d'] > 0.05)
                df['short_interest_proxy'] = (df['Close'] < df['ma_50']) & (df['rvol'] > 2.0)

                # ---- MEDIUM/LONG-TERM INDICATORS (new) ----
                df['ret_63d']       = df['Close'].pct_change(63).fillna(0)   # ~3 months
                df['ret_126d']      = df['Close'].pct_change(126).fillna(0)  # ~6 months
                df['ma_200']        = df['Close'].rolling(200, min_periods=1).mean()
                df['high_126d']     = df['High'].rolling(126, min_periods=1).max()
                df['ma_50_slope']   = df['ma_50'].diff(21).fillna(0)   # 1-month slope of 50MA
                df['ma_200_slope']  = df['ma_200'].diff(63).fillna(0)  # 3-month slope of 200MA

                latest_day = df.iloc[-1:].copy()
                latest_day['Ticker'] = ticker
                latest_rows.append(latest_day)
            except Exception:
                continue

    if not latest_rows:
        st.error("⚠️ Yahoo Finance returned no data! You may be rate-limited. Wait 15–30 min and try again.")
        return pd.DataFrame()

    final_df = pd.concat(latest_rows)
    final_df = final_df[(final_df['Close'] >= 0.5) & (final_df['volume_avg_20'] >= 5000)]

    # ---- PRE-COMPUTE DERIVED DISPLAY METRICS ----
    # These fix the original bug where "transparent" columns didn't show in the tabs.
    final_df['dist_ma50']           = (final_df['Close'] - final_df['ma_50'])   / (final_df['ma_50']   + 1e-9)
    final_df['dist_ma200']          = (final_df['Close'] - final_df['ma_200'])  / (final_df['ma_200']  + 1e-9)
    final_df['ma_alignment']        = (final_df['ma_20'] - final_df['ma_50'])   / (final_df['ma_50']   + 1e-9)
    final_df['ma_alignment_50_200'] = (final_df['ma_50'] - final_df['ma_200'])  / (final_df['ma_200']  + 1e-9)
    final_df['dist_high']           = final_df['Close'] / (final_df['high_50d']  + 1e-9)
    final_df['dist_126d_high']      = final_df['Close'] / (final_df['high_126d'] + 1e-9)
    final_df['vol_trend_norm']      = final_df['volume_trend'] / (final_df['volume_avg_20'] + 1e-9)
    final_df['momentum_balance']    = final_df['ret_21d'] - final_df['ret_10d']
    final_df['momentum_acceleration']= final_df['ret_21d'] - final_df['ret_10d']  # alias
    final_df['momentum_3m_vs_1m']   = final_df['ret_63d']  - final_df['ret_21d']
    final_df['momentum_6m_vs_3m']   = final_df['ret_126d'] - final_df['ret_63d']
    final_df['ema_gap_8_21']        = (final_df['ema_8']   - final_df['ema_21']) / (final_df['ema_21'] + 1e-9)
    final_df['ema_gap_21_ma200']    = (final_df['ema_21']  - final_df['ma_200']) / (final_df['ma_200'] + 1e-9)

    return final_df


# ==========================================
# 4. SHORT-TERM SCORING MODELS (unchanged)
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


# ==========================================
# 5. 1-MONTH SCORING MODELS (unchanged)
# ==========================================
def score_chatgpt_1m(df):
    s = pd.Series(0.0, index=df.index)
    dist_ma50 = (df['Close'] - df['ma_50']) / (df['ma_50'] + 1e-9)
    ma_alignment = (df['ma_20'] - df['ma_50']) / (df['ma_50'] + 1e-9)
    s += np.clip(dist_ma50 * 100, 0, 12)
    s -= np.clip((dist_ma50 - 0.12) * 120, 0, 12)
    s += np.clip(ma_alignment * 120, 0, 12)
    s += np.clip(df['ret_21d'] * 100, 0, 12)
    momentum_balance = df['ret_21d'] - df['ret_10d']
    s += np.clip(momentum_balance * 120, 0, 10)
    s += 12 - np.abs(df['rsi'] - 55) * 0.4
    vol_trend_norm = df['volume_trend'] / (df['volume_avg_20'] + 1e-9)
    s += np.clip(vol_trend_norm * 60, 0, 10)
    s += np.clip((df['rvol'] - 1) * 8, 0, 8)
    dist_high = df['Close'] / (df['high_50d'] + 1e-9)
    s += np.clip((dist_high - 0.85) * 60, 0, 10)
    s -= np.clip((dist_high - 0.98) * 120, 0, 8)
    return s

def score_grok_1m(df):
    s = pd.Series(0.0, index=df.index)
    s += np.clip(df['ret_21d'] * 140, 0, 30)
    momentum_acceleration = df['ret_21d'] - df['ret_10d']
    s += np.clip(momentum_acceleration * 140, 0, 15)
    dist_ma50 = (df['Close'] - df['ma_50']) / (df['ma_50'] + 1e-9)
    ma_alignment = (df['ma_20'] - df['ma_50']) / (df['ma_50'] + 1e-9)
    s += np.clip(dist_ma50 * 100, 0, 12)
    s += np.clip(ma_alignment * 120, 0, 12)
    s += np.clip((df['rvol'] - 1) * 10, 0, 12)
    vol_trend_norm = df['volume_trend'] / (df['volume_avg_20'] + 1e-9)
    s += np.clip(vol_trend_norm * 80, -5, 10)
    dist_high = df['Close'] / (df['high_50d'] + 1e-9)
    s += np.clip((dist_high - 0.80) * 80, 0, 15)
    s -= np.clip((dist_high - 0.97) * 100, 0, 10)
    s += 12 - np.abs(df['rsi'] - 58) * 0.35
    return s

def score_gemini_1m(df):
    s = pd.Series(0.0, index=df.index)
    ema_gap = (df['ema_21'] - df['ma_50']) / (df['ma_50'] + 1e-9)
    s += np.clip(ema_gap * 140, 0, 18)
    s += np.clip(df['ret_21d'] * 110, 0, 18)
    s += 12 - np.abs(df['rsi'] - 55) * 0.35
    vol_trend_norm = df['volume_trend'] / (df['volume_avg_20'] + 1e-9)
    s += np.clip(vol_trend_norm * 70, 0, 15)
    s += np.clip((df['rvol'] - 1) * 6, 0, 8)
    dist_high = df['Close'] / (df['high_50d'] + 1e-9)
    s += np.clip((dist_high - 0.85) * 50, 0, 8)
    return s

def score_hybrid_1m(df):
    s = pd.Series(0.0, index=df.index)
    dist_ma50 = (df['Close'] - df['ma_50']) / (df['ma_50'] + 1e-9)
    ma_alignment = (df['ma_20'] - df['ma_50']) / (df['ma_50'] + 1e-9)
    s += np.clip(dist_ma50 * 80, 0, 10)
    s += np.clip(ma_alignment * 100, 0, 10)
    s += np.clip(df['ret_21d'] * 110, 0, 15)
    momentum_balance = df['ret_21d'] - df['ret_10d']
    s += np.clip(momentum_balance * 100, 0, 12)
    vol_trend_norm = df['volume_trend'] / (df['volume_avg_20'] + 1e-9)
    s += np.clip(vol_trend_norm * 60, 0, 12)
    s += np.clip((df['rvol'] - 1) * 8, 0, 10)
    dist_high = df['Close'] / (df['high_50d'] + 1e-9)
    s += np.clip((dist_high - 0.85) * 60, 0, 10)
    s -= np.clip((dist_high - 0.98) * 100, 0, 6)
    s += 12 - np.abs(df['rsi'] - 56) * 0.35
    return s


# ==========================================
# 6. 3-MONTH SCORING MODELS
#    *** PLACEHOLDER — send AI template to ChatGPT / Grok / Gemini ***
#    Drop in their returned functions below to activate custom methodology.
# ==========================================

def score_chatgpt_3m(df):
    """PLACEHOLDER: 3-Month Sustained Trend (ChatGPT methodology)"""
    s = pd.Series(0.0, index=df.index)

    # --- Trend Structure (MA50 vs MA200) ---
    dist_ma200      = (df['Close'] - df['ma_200']) / (df['ma_200'] + 1e-9)
    ma_align_50_200 = (df['ma_50']  - df['ma_200']) / (df['ma_200'] + 1e-9)
    s += np.clip(dist_ma200 * 60, 0, 15)
    s -= np.clip((dist_ma200 - 0.20) * 80, 0, 15)
    s += np.clip(ma_align_50_200 * 100, 0, 15)

    # --- 3M Momentum ---
    s += np.clip(df['ret_63d'] * 80, 0, 15)

    # --- MA200 Slope (secular trend direction) ---
    s += np.clip(df['ma_200_slope'] * 5, 0, 10)

    # --- RSI Stability (55 is the sweet spot for sustained uptrends) ---
    s += 10 - np.abs(df['rsi'] - 55) * 0.35

    # --- Positioning vs 126d high ---
    s += np.clip((df['dist_126d_high'] - 0.80) * 40, 0, 10)

    # --- Volume Consistency ---
    s += np.clip(df['vol_trend_norm'] * 60, 0, 10)

    return s


def score_grok_3m(df):
    """PLACEHOLDER: 3-Month Breakout & Momentum Continuation (Grok methodology)"""
    s = pd.Series(0.0, index=df.index)

    # --- Primary: 3M Momentum (Grok goes aggressive) ---
    s += np.clip(df['ret_63d'] * 100, 0, 25)

    # --- Acceleration: 3M minus 1M ---
    s += np.clip(df['momentum_3m_vs_1m'] * 80, 0, 15)

    # --- Trend Alignment ---
    dist_ma200      = (df['Close'] - df['ma_200']) / (df['ma_200'] + 1e-9)
    ma_align_50_200 = (df['ma_50']  - df['ma_200']) / (df['ma_200'] + 1e-9)
    s += np.clip(dist_ma200 * 60, 0, 12)
    s += np.clip(ma_align_50_200 * 80, 0, 12)

    # --- 126d High Proximity ---
    s += np.clip((df['dist_126d_high'] - 0.80) * 60, 0, 12)

    # --- Volume Confirmation ---
    s += np.clip((df['rvol'] - 1) * 8, 0, 10)

    # --- RSI (higher tolerance for momentum names) ---
    s += 8 - np.abs(df['rsi'] - 58) * 0.25

    return s


def score_gemini_3m(df):
    """PLACEHOLDER: 3-Month Volume Structure & Catalyst (Gemini methodology)"""
    s = pd.Series(0.0, index=df.index)

    # --- EMA21 vs MA200 (structural long trend) ---
    s += np.clip(df['ema_gap_21_ma200'] * 100, 0, 20)

    # --- 3M Momentum ---
    s += np.clip(df['ret_63d'] * 90, 0, 20)

    # --- RSI Stability ---
    s += 10 - np.abs(df['rsi'] - 55) * 0.30

    # --- Volume Trend (core feature for Gemini) ---
    s += np.clip(df['vol_trend_norm'] * 50, 0, 15)

    # --- MA200 Slope ---
    s += np.clip(df['ma_200_slope'] * 4, 0, 8)

    # --- Light RVOL ---
    s += np.clip((df['rvol'] - 1) * 5, 0, 8)

    return s


def score_hybrid_3m(df):
    """PLACEHOLDER: 3-Month Best-of-All (Hybrid methodology)"""
    s = pd.Series(0.0, index=df.index)

    # --- Trend ---
    dist_ma200      = (df['Close'] - df['ma_200']) / (df['ma_200'] + 1e-9)
    ma_align_50_200 = (df['ma_50']  - df['ma_200']) / (df['ma_200'] + 1e-9)
    s += np.clip(dist_ma200 * 60, 0, 12)
    s += np.clip(ma_align_50_200 * 80, 0, 12)

    # --- Momentum + Acceleration ---
    s += np.clip(df['ret_63d'] * 90, 0, 15)
    s += np.clip(df['momentum_3m_vs_1m'] * 80, 0, 12)

    # --- Volume ---
    s += np.clip(df['vol_trend_norm'] * 50, 0, 10)
    s += np.clip((df['rvol'] - 1) * 6, 0, 8)

    # --- Positioning ---
    s += np.clip((df['dist_126d_high'] - 0.80) * 50, 0, 10)

    # --- RSI ---
    s += 10 - np.abs(df['rsi'] - 56) * 0.30

    # --- MA200 Slope ---
    s += np.clip(df['ma_200_slope'] * 4, 0, 8)

    return s


# ==========================================
# 7. 6-MONTH SCORING MODELS
#    *** PLACEHOLDER — send AI template to ChatGPT / Grok / Gemini ***
#    Drop in their returned functions below to activate custom methodology.
# ==========================================

def score_chatgpt_6m(df):
    """PLACEHOLDER: 6-Month Long-Term Trend Quality (ChatGPT methodology)"""
    s = pd.Series(0.0, index=df.index)

    # --- MA200 as primary trend anchor ---
    dist_ma200      = (df['Close'] - df['ma_200']) / (df['ma_200'] + 1e-9)
    ma_align_50_200 = (df['ma_50']  - df['ma_200']) / (df['ma_200'] + 1e-9)
    s += np.clip(dist_ma200 * 50, 0, 15)
    s -= np.clip((dist_ma200 - 0.30) * 60, 0, 15)
    s += np.clip(ma_align_50_200 * 80, 0, 15)

    # --- 6M Return (primary momentum driver) ---
    s += np.clip(df['ret_126d'] * 60, 0, 20)

    # --- MA200 Slope (is the secular trend accelerating?) ---
    s += np.clip(df['ma_200_slope'] * 4, 0, 12)

    # --- RSI Long-Term Stability ---
    s += 12 - np.abs(df['rsi'] - 55) * 0.30

    # --- 126d High Positioning ---
    s += np.clip((df['dist_126d_high'] - 0.75) * 35, 0, 8)

    # --- Volume Consistency ---
    s += np.clip(df['vol_trend_norm'] * 40, 0, 8)

    return s


def score_grok_6m(df):
    """PLACEHOLDER: 6-Month Secular Momentum (Grok methodology)"""
    s = pd.Series(0.0, index=df.index)

    # --- Primary: Dominant 6M momentum ---
    s += np.clip(df['ret_126d'] * 80, 0, 30)

    # --- Acceleration: 6M vs 3M ---
    s += np.clip(df['momentum_6m_vs_3m'] * 60, 0, 15)

    # --- Trend Structure ---
    dist_ma200      = (df['Close'] - df['ma_200']) / (df['ma_200'] + 1e-9)
    ma_align_50_200 = (df['ma_50']  - df['ma_200']) / (df['ma_200'] + 1e-9)
    s += np.clip(dist_ma200 * 50, 0, 12)
    s += np.clip(ma_align_50_200 * 70, 0, 12)

    # --- MA200 Slope (momentum of the trend itself) ---
    s += np.clip(df['ma_200_slope'] * 5, 0, 10)

    # --- Positioning ---
    s += np.clip((df['dist_126d_high'] - 0.70) * 40, 0, 10)

    # --- RSI (higher tolerance for secular movers) ---
    s += 8 - np.abs(df['rsi'] - 60) * 0.20

    return s


def score_gemini_6m(df):
    """PLACEHOLDER: 6-Month Macro Structure & Volume (Gemini methodology)"""
    s = pd.Series(0.0, index=df.index)

    # --- EMA21 vs MA200 (long-term structural trend) ---
    s += np.clip(df['ema_gap_21_ma200'] * 80, 0, 20)

    # --- 6M Momentum ---
    s += np.clip(df['ret_126d'] * 70, 0, 20)

    # --- MA200 Slope (key signal for 6M view) ---
    s += np.clip(df['ma_200_slope'] * 5, 0, 15)

    # --- RSI Stability ---
    s += 10 - np.abs(df['rsi'] - 55) * 0.25

    # --- Sustained Volume Accumulation ---
    s += np.clip(df['vol_trend_norm'] * 40, 0, 12)

    # --- MA Alignment ---
    s += np.clip(df['ma_alignment_50_200'] * 60, 0, 12)

    return s


def score_hybrid_6m(df):
    """PLACEHOLDER: 6-Month Best-of-All (Hybrid methodology)"""
    s = pd.Series(0.0, index=df.index)

    # --- Trend Structure ---
    dist_ma200      = (df['Close'] - df['ma_200']) / (df['ma_200'] + 1e-9)
    ma_align_50_200 = (df['ma_50']  - df['ma_200']) / (df['ma_200'] + 1e-9)
    s += np.clip(dist_ma200 * 50, 0, 12)
    s += np.clip(ma_align_50_200 * 70, 0, 12)

    # --- 6M Momentum + Acceleration ---
    s += np.clip(df['ret_126d'] * 70, 0, 18)
    s += np.clip(df['momentum_6m_vs_3m'] * 50, 0, 12)

    # --- MA200 Slope ---
    s += np.clip(df['ma_200_slope'] * 4, 0, 10)

    # --- Volume ---
    s += np.clip(df['vol_trend_norm'] * 40, 0, 10)

    # --- Positioning ---
    s += np.clip((df['dist_126d_high'] - 0.75) * 40, 0, 8)

    # --- RSI ---
    s += 10 - np.abs(df['rsi'] - 56) * 0.25

    return s


# ==========================================
# 8. RAG / COLOUR FORMATTING
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

def color_avg_rank(val):
    """Green ≤ 20, orange ≤ 50, grey beyond — for rank columns."""
    if pd.isna(val): return ''
    if val <= 20: return 'color: #00FF00'
    elif val <= 50: return 'color: #FFA500'
    return 'color: #888888'

def apply_rag_formatting(df):
    df = df.reset_index(drop=True)
    styler = df.style

    for col, fn in [
        ('rsi', color_rsi), ('rvol', color_rvol),
        ('ret_5d', color_ret), ('ret_21d', color_ret),
        ('ret_63d', color_ret), ('ret_126d', color_ret),
    ]:
        if col in df.columns:
            styler = styler.map(fn, subset=[col])

    fmt = {
        'Close': '{:.2f}', 'rsi': '{:.1f}', 'rvol': '{:.2f}',
        'ret_5d': '{:.2%}', 'ret_10d': '{:.2%}', 'ret_21d': '{:.2%}',
        'ret_63d': '{:.2%}', 'ret_126d': '{:.2%}',
        'ma_20': '{:.2f}', 'ma_50': '{:.2f}', 'ma_200': '{:.2f}',
        'ema_8': '{:.2f}', 'ema_21': '{:.2f}',
        'ma_20_slope': '{:.3f}', 'ma_50_slope': '{:.4f}', 'ma_200_slope': '{:.4f}',
        'macd': '{:.3f}', 'macd_signal': '{:.3f}',
        'volume_trend': '{:.0f}', 'vol_trend_norm': '{:.2f}',
        'dist_ma50': '{:.3f}', 'dist_ma200': '{:.3f}',
        'ma_alignment': '{:.3f}', 'ma_alignment_50_200': '{:.3f}',
        'dist_high': '{:.3f}', 'dist_126d_high': '{:.3f}',
        'momentum_balance': '{:.3f}', 'momentum_acceleration': '{:.3f}',
        'momentum_3m_vs_1m': '{:.3f}', 'momentum_6m_vs_3m': '{:.3f}',
        'ema_gap_8_21': '{:.3f}', 'ema_gap_21_ma200': '{:.3f}',
        # Ranks — short-term
        'Average_Rank': '{:.0f}', 'Rank_ChatGPT': '{:.0f}',
        'Rank_Grok': '{:.0f}', 'Rank_Gemini': '{:.0f}', 'Rank_Hybrid': '{:.0f}',
        # Ranks — 1M
        'Average_Rank_1M': '{:.0f}', 'Rank_ChatGPT_1M': '{:.0f}',
        'Rank_Grok_1M': '{:.0f}', 'Rank_Gemini_1M': '{:.0f}', 'Rank_Hybrid_1M': '{:.0f}',
        # Ranks — 3M
        'Average_Rank_3M': '{:.0f}', 'Rank_ChatGPT_3M': '{:.0f}',
        'Rank_Grok_3M': '{:.0f}', 'Rank_Gemini_3M': '{:.0f}', 'Rank_Hybrid_3M': '{:.0f}',
        # Ranks — 6M
        'Average_Rank_6M': '{:.0f}', 'Rank_ChatGPT_6M': '{:.0f}',
        'Rank_Grok_6M': '{:.0f}', 'Rank_Gemini_6M': '{:.0f}', 'Rank_Hybrid_6M': '{:.0f}',
        # Overview
        'Overall_Rank': '{:.0f}',
    }
    safe_fmt = {k: v for k, v in fmt.items() if k in df.columns}
    return styler.format(safe_fmt)


def apply_overview_formatting(df):
    """Special formatting for the All Stocks overview tab with rank colour coding."""
    df = df.reset_index(drop=True)
    rank_cols = [c for c in ['Average_Rank', 'Average_Rank_1M', 'Average_Rank_3M',
                              'Average_Rank_6M', 'Overall_Rank'] if c in df.columns]
    styler = df.style
    for col in rank_cols:
        styler = styler.map(color_avg_rank, subset=[col])

    ret_cols = [c for c in ['ret_5d', 'ret_21d', 'ret_63d', 'ret_126d'] if c in df.columns]
    for col in ret_cols:
        styler = styler.map(color_ret, subset=[col])

    fmt = {
        'Close': '{:.2f}', 'rsi': '{:.1f}', 'rvol': '{:.2f}',
        'ret_5d': '{:.2%}', 'ret_21d': '{:.2%}', 'ret_63d': '{:.2%}', 'ret_126d': '{:.2%}',
        'Average_Rank': '{:.0f}', 'Average_Rank_1M': '{:.0f}',
        'Average_Rank_3M': '{:.0f}', 'Average_Rank_6M': '{:.0f}', 'Overall_Rank': '{:.0f}',
    }
    safe_fmt = {k: v for k, v in fmt.items() if k in df.columns}
    return styler.format(safe_fmt)


# ==========================================
# 9. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="V3 Market Scanner", layout="wide")
st.title("⚡ V3 Live Market Scanner (Quad Timeframe)")
st.markdown("Scan major global markets using 4 AI models across **Short-Term, 1M, 3M, and 6M** timeframes.")

with st.expander("📚 How Scoring & FinBERT Sentiment Works (Click to Expand)", expanded=False):
    st.markdown("""
    ### FinBERT News Sentiment
    Runs recent headlines through **FinBERT**, a financial NLP neural network. Enabled for the Short-Term and 1M Master lists by default. Toggle *Extended FinBERT* in the sidebar to also score 3M and 6M lists (adds ~2 min).

    ### The 4 Models × 4 Timeframes
    | | Short-Term (1-2w) | 1 Month | 3 Months | 6 Months |
    |---|---|---|---|---|
    | 🤖 ChatGPT | MA20 / RSI / MACD | MA50 trend | MA200 trend | Secular MA200 |
    | 🌌 Grok | Breakout / RVOL | Momentum burst | 3M acceleration | 6M secular momentum |
    | ✨ Gemini | RVOL / EMA crossover | Volume squeeze | EMA21 vs MA200 | Macro volume structure |
    | 🧬 Hybrid | Best-of-all | Balanced blend | Balanced blend | Balanced blend |

    > **3M and 6M scoring functions are placeholders.** Use the AI Scoring Template (downloadable from sidebar) to get custom methodologies from ChatGPT, Grok, and Gemini, then paste their functions into the source code.
    """)

# ---- SIDEBAR ----
st.sidebar.header("Scanner Settings")

if not FINBERT_AVAILABLE:
    st.sidebar.error("⚠️ FinBERT not found. Run `pip install transformers torch`.")

market_options = [
    "S&P 500", "S&P 400 (MidCap)", "S&P 600 (SmallCap)",
    "NASDAQ 100", "Dow Jones", "FTSE 100", "FTSE 250",
    "CAC 40", "DAX 40", "GETTEX (Manual)"
]
selected_markets = st.sidebar.multiselect("Select Markets to Scan:", market_options, default=["NASDAQ 100"])
run_finbert_extended = st.sidebar.checkbox(
    "🧠 Extended FinBERT (3M + 6M master lists)",
    value=False,
    help="Runs FinBERT sentiment on 3M and 6M top-20 lists too. Adds ~2 minutes to scan time."
)
top_n = st.sidebar.slider("Top N picks per timeframe", min_value=10, max_value=50, value=20, step=5)

# ---- MAIN SCAN ----
if st.sidebar.button("🚀 Run Live Scan"):
    if not selected_markets:
        st.warning("Please select at least one market.")
    else:
        with st.spinner("Loading tickers..."):
            tickers, ticker_map = get_tickers_and_names(selected_markets)

        if not tickers:
            st.error("No tickers loaded. Check your .csv files.")
        else:
            live_data = fetch_latest_data(tickers)

            if live_data.empty:
                st.error("Failed to fetch data or no stocks met liquidity requirements.")
            else:
                missing = set(tickers) - set(live_data['Ticker'].unique())
                if missing:
                    st.warning(f"🕵️ Missing Tickers: {', '.join(missing)}")

                with st.spinner("Calculating AI Scores across all 4 timeframes..."):
                    live_data['Company'] = live_data['Ticker'].map(ticker_map)

                    # -- Short-Term --
                    live_data['ChatGPT_Score']  = score_chatgpt(live_data)
                    live_data['Grok_Score']     = score_grok(live_data)
                    live_data['Gemini_Score']   = score_gemini(live_data)
                    live_data['Hybrid_Score']   = score_hybrid(live_data)
                    live_data['Rank_ChatGPT']   = live_data['ChatGPT_Score'].rank(ascending=False, method='min')
                    live_data['Rank_Grok']      = live_data['Grok_Score'].rank(ascending=False, method='min')
                    live_data['Rank_Gemini']    = live_data['Gemini_Score'].rank(ascending=False, method='min')
                    live_data['Rank_Hybrid']    = live_data['Hybrid_Score'].rank(ascending=False, method='min')
                    live_data['Average_Rank']   = live_data[['Rank_ChatGPT','Rank_Grok','Rank_Gemini','Rank_Hybrid']].mean(axis=1)

                    # -- 1-Month --
                    live_data['ChatGPT_Score_1M']  = score_chatgpt_1m(live_data)
                    live_data['Grok_Score_1M']     = score_grok_1m(live_data)
                    live_data['Gemini_Score_1M']   = score_gemini_1m(live_data)
                    live_data['Hybrid_Score_1M']   = score_hybrid_1m(live_data)
                    live_data['Rank_ChatGPT_1M']   = live_data['ChatGPT_Score_1M'].rank(ascending=False, method='min')
                    live_data['Rank_Grok_1M']      = live_data['Grok_Score_1M'].rank(ascending=False, method='min')
                    live_data['Rank_Gemini_1M']    = live_data['Gemini_Score_1M'].rank(ascending=False, method='min')
                    live_data['Rank_Hybrid_1M']    = live_data['Hybrid_Score_1M'].rank(ascending=False, method='min')
                    live_data['Average_Rank_1M']   = live_data[['Rank_ChatGPT_1M','Rank_Grok_1M','Rank_Gemini_1M','Rank_Hybrid_1M']].mean(axis=1)

                    # -- 3-Month --
                    live_data['ChatGPT_Score_3M']  = score_chatgpt_3m(live_data)
                    live_data['Grok_Score_3M']     = score_grok_3m(live_data)
                    live_data['Gemini_Score_3M']   = score_gemini_3m(live_data)
                    live_data['Hybrid_Score_3M']   = score_hybrid_3m(live_data)
                    live_data['Rank_ChatGPT_3M']   = live_data['ChatGPT_Score_3M'].rank(ascending=False, method='min')
                    live_data['Rank_Grok_3M']      = live_data['Grok_Score_3M'].rank(ascending=False, method='min')
                    live_data['Rank_Gemini_3M']    = live_data['Gemini_Score_3M'].rank(ascending=False, method='min')
                    live_data['Rank_Hybrid_3M']    = live_data['Hybrid_Score_3M'].rank(ascending=False, method='min')
                    live_data['Average_Rank_3M']   = live_data[['Rank_ChatGPT_3M','Rank_Grok_3M','Rank_Gemini_3M','Rank_Hybrid_3M']].mean(axis=1)

                    # -- 6-Month --
                    live_data['ChatGPT_Score_6M']  = score_chatgpt_6m(live_data)
                    live_data['Grok_Score_6M']     = score_grok_6m(live_data)
                    live_data['Gemini_Score_6M']   = score_gemini_6m(live_data)
                    live_data['Hybrid_Score_6M']   = score_hybrid_6m(live_data)
                    live_data['Rank_ChatGPT_6M']   = live_data['ChatGPT_Score_6M'].rank(ascending=False, method='min')
                    live_data['Rank_Grok_6M']      = live_data['Grok_Score_6M'].rank(ascending=False, method='min')
                    live_data['Rank_Gemini_6M']    = live_data['Gemini_Score_6M'].rank(ascending=False, method='min')
                    live_data['Rank_Hybrid_6M']    = live_data['Hybrid_Score_6M'].rank(ascending=False, method='min')
                    live_data['Average_Rank_6M']   = live_data[['Rank_ChatGPT_6M','Rank_Grok_6M','Rank_Gemini_6M','Rank_Hybrid_6M']].mean(axis=1)

                    # -- Overall Cross-Timeframe Composite --
                    live_data['Overall_Rank'] = live_data[['Average_Rank','Average_Rank_1M','Average_Rank_3M','Average_Rank_6M']].mean(axis=1)

                # -- Build top-N master tables --
                master    = live_data.sort_values('Average_Rank',    ascending=True).head(top_n).copy()
                master_1m = live_data.sort_values('Average_Rank_1M', ascending=True).head(top_n).copy()
                master_3m = live_data.sort_values('Average_Rank_3M', ascending=True).head(top_n).copy()
                master_6m = live_data.sort_values('Average_Rank_6M', ascending=True).head(top_n).copy()

                # -- FinBERT: always run for ST and 1M --
                nlp = load_finbert()

                def run_finbert_loop(df_top, label):
                    sentiments = []
                    bar = st.progress(0, text=f"Analysing {label} news sentiment...")
                    n = len(df_top)
                    for i, (_, row) in enumerate(df_top.iterrows()):
                        sentiments.append(analyze_sentiment(row['Ticker'], nlp))
                        bar.progress((i + 1) / n, text=f"[{label}] {row['Ticker']} ({i+1}/{n})")
                    bar.empty()
                    return sentiments

                master['FinBERT_Sentiment']    = run_finbert_loop(master,    "Short-Term")
                master_1m['FinBERT_Sentiment'] = run_finbert_loop(master_1m, "1-Month")

                if run_finbert_extended:
                    master_3m['FinBERT_Sentiment'] = run_finbert_loop(master_3m, "3-Month")
                    master_6m['FinBERT_Sentiment'] = run_finbert_loop(master_6m, "6-Month")

                st.success(f"✅ Scan complete — {len(live_data)} qualifying stocks across {len(selected_markets)} market(s).")

                # ==============================================
                # TABS
                # ==============================================
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "👑 Master Consensus",
                    "🤖 ChatGPT",
                    "🌌 Grok",
                    "✨ Gemini",
                    "🧬 Hybrid",
                    "📊 All Stocks Overview",
                ])

                # -----------------------------------------------
                # TAB 1 — MASTER CONSENSUS
                # -----------------------------------------------
                with tab1:
                    st.subheader(f"⚡ Short-Term (1-2 Weeks) Top {top_n}")
                    m_cols = ['Ticker','Company','FinBERT_Sentiment','Average_Rank',
                              'Rank_ChatGPT','Rank_Grok','Rank_Gemini','Rank_Hybrid',
                              'Close','rsi','rvol','ret_5d']
                    st.dataframe(apply_rag_formatting(master[[c for c in m_cols if c in master.columns]]),
                                 use_container_width=True, hide_index=True)
                    st.divider()

                    st.subheader(f"📅 Medium-Term (1 Month) Top {top_n}")
                    m1_cols = ['Ticker','Company','FinBERT_Sentiment','Average_Rank_1M',
                               'Rank_ChatGPT_1M','Rank_Grok_1M','Rank_Gemini_1M','Rank_Hybrid_1M',
                               'Close','rsi','rvol','ret_21d']
                    st.dataframe(apply_rag_formatting(master_1m[[c for c in m1_cols if c in master_1m.columns]]),
                                 use_container_width=True, hide_index=True)
                    st.divider()

                    st.subheader(f"🗓️ 3-Month Swing Top {top_n}")
                    if not run_finbert_extended:
                        st.caption("💡 FinBERT not run for 3M/6M — enable *Extended FinBERT* in sidebar.")
                    m3_cols = ['Ticker','Company'] + (['FinBERT_Sentiment'] if run_finbert_extended else []) + \
                              ['Average_Rank_3M','Rank_ChatGPT_3M','Rank_Grok_3M','Rank_Gemini_3M','Rank_Hybrid_3M',
                               'Close','rsi','rvol','ret_63d']
                    st.dataframe(apply_rag_formatting(master_3m[[c for c in m3_cols if c in master_3m.columns]]),
                                 use_container_width=True, hide_index=True)
                    st.divider()

                    st.subheader(f"📆 6-Month Position Top {top_n}")
                    m6_cols = ['Ticker','Company'] + (['FinBERT_Sentiment'] if run_finbert_extended else []) + \
                              ['Average_Rank_6M','Rank_ChatGPT_6M','Rank_Grok_6M','Rank_Gemini_6M','Rank_Hybrid_6M',
                               'Close','rsi','rvol','ret_126d']
                    st.dataframe(apply_rag_formatting(master_6m[[c for c in m6_cols if c in master_6m.columns]]),
                                 use_container_width=True, hide_index=True)

                # -----------------------------------------------
                # TAB 2 — CHATGPT
                # -----------------------------------------------
                with tab2:
                    st.subheader("⚡ Short-Term — Trend Focus")
                    cg = live_data.sort_values(by=['ChatGPT_Score','Average_Rank'], ascending=[False,True]).head(top_n)
                    cg_cols = ['Ticker','Company','Rank_ChatGPT','ChatGPT_Score','Close',
                               'ma_20','ma_20_slope','rsi','macd','macd_signal','rvol','volume_trend','ret_5d']
                    st.dataframe(apply_rag_formatting(cg[[c for c in cg_cols if c in cg.columns]]),
                                 use_container_width=True, hide_index=True)
                    st.divider()

                    st.subheader("📅 1-Month — Trend Focus (Transparent)")
                    cg1m = live_data.sort_values(by=['ChatGPT_Score_1M','Average_Rank_1M'], ascending=[False,True]).head(top_n)
                    cg1m_cols = ['Ticker','Company','Rank_ChatGPT_1M','ChatGPT_Score_1M','Close',
                                 'ma_50','dist_ma50','ma_alignment','ret_21d','momentum_balance',
                                 'rsi','rvol','vol_trend_norm','dist_high']
                    st.dataframe(apply_rag_formatting(cg1m[[c for c in cg1m_cols if c in cg1m.columns]]),
                                 use_container_width=True, hide_index=True)
                    st.divider()

                    st.subheader("🗓️ 3-Month — Sustained Trend (Transparent)")
                    cg3m = live_data.sort_values(by=['ChatGPT_Score_3M','Average_Rank_3M'], ascending=[False,True]).head(top_n)
                    cg3m_cols = ['Ticker','Company','Rank_ChatGPT_3M','ChatGPT_Score_3M','Close',
                                 'ma_200','dist_ma200','ma_alignment_50_200','ret_63d','momentum_3m_vs_1m',
                                 'ma_200_slope','rsi','vol_trend_norm','dist_126d_high']
                    st.dataframe(apply_rag_formatting(cg3m[[c for c in cg3m_cols if c in cg3m.columns]]),
                                 use_container_width=True, hide_index=True)
                    st.divider()

                    st.subheader("📆 6-Month — Long-Term Trend (Transparent)")
                    cg6m = live_data.sort_values(by=['ChatGPT_Score_6M','Average_Rank_6M'], ascending=[False,True]).head(top_n)
                    cg6m_cols = ['Ticker','Company','Rank_ChatGPT_6M','ChatGPT_Score_6M','Close',
                                 'ma_200','dist_ma200','ma_alignment_50_200','ret_126d','momentum_6m_vs_3m',
                                 'ma_200_slope','rsi','vol_trend_norm','dist_126d_high']
                    st.dataframe(apply_rag_formatting(cg6m[[c for c in cg6m_cols if c in cg6m.columns]]),
                                 use_container_width=True, hide_index=True)

                # -----------------------------------------------
                # TAB 3 — GROK
                # -----------------------------------------------
                with tab3:
                    st.subheader("⚡ Short-Term — Breakout Focus")
                    gk = live_data.sort_values(by=['Grok_Score','Average_Rank'], ascending=[False,True]).head(top_n)
                    gk_cols = ['Ticker','Company','Rank_Grok','Grok_Score','Close',
                               'ma_20','ma_50','near_high','rvol','ret_5d','ret_10d']
                    st.dataframe(apply_rag_formatting(gk[[c for c in gk_cols if c in gk.columns]]),
                                 use_container_width=True, hide_index=True)
                    st.divider()

                    st.subheader("📅 1-Month — Momentum Burst (Transparent)")
                    gk1m = live_data.sort_values(by=['Grok_Score_1M','Average_Rank_1M'], ascending=[False,True]).head(top_n)
                    gk1m_cols = ['Ticker','Company','Rank_Grok_1M','Grok_Score_1M','Close',
                                 'ret_21d','momentum_acceleration','ma_50','dist_ma50','ma_alignment',
                                 'near_high','dist_high','rsi','rvol','vol_trend_norm']
                    st.dataframe(apply_rag_formatting(gk1m[[c for c in gk1m_cols if c in gk1m.columns]]),
                                 use_container_width=True, hide_index=True)
                    st.divider()

                    st.subheader("🗓️ 3-Month — Breakout Continuation (Transparent)")
                    gk3m = live_data.sort_values(by=['Grok_Score_3M','Average_Rank_3M'], ascending=[False,True]).head(top_n)
                    gk3m_cols = ['Ticker','Company','Rank_Grok_3M','Grok_Score_3M','Close',
                                 'ret_63d','momentum_3m_vs_1m','ma_200','dist_ma200','ma_alignment_50_200',
                                 'dist_126d_high','rsi','rvol']
                    st.dataframe(apply_rag_formatting(gk3m[[c for c in gk3m_cols if c in gk3m.columns]]),
                                 use_container_width=True, hide_index=True)
                    st.divider()

                    st.subheader("📆 6-Month — Secular Momentum (Transparent)")
                    gk6m = live_data.sort_values(by=['Grok_Score_6M','Average_Rank_6M'], ascending=[False,True]).head(top_n)
                    gk6m_cols = ['Ticker','Company','Rank_Grok_6M','Grok_Score_6M','Close',
                                 'ret_126d','momentum_6m_vs_3m','ma_200','dist_ma200','ma_alignment_50_200',
                                 'ma_200_slope','dist_126d_high','rsi']
                    st.dataframe(apply_rag_formatting(gk6m[[c for c in gk6m_cols if c in gk6m.columns]]),
                                 use_container_width=True, hide_index=True)

                # -----------------------------------------------
                # TAB 4 — GEMINI
                # -----------------------------------------------
                with tab4:
                    st.subheader("⚡ Short-Term — Catalyst / RVOL Focus")
                    gm = live_data.sort_values(by=['Gemini_Score','Average_Rank'], ascending=[False,True]).head(top_n)
                    gm_cols = ['Ticker','Company','Rank_Gemini','Gemini_Score','Close',
                               'ema_8','ema_21','macd','macd_signal','rsi','rvol','close_near_high','post_earnings']
                    st.dataframe(apply_rag_formatting(gm[[c for c in gm_cols if c in gm.columns]]),
                                 use_container_width=True, hide_index=True)
                    st.divider()

                    st.subheader("📅 1-Month — Volume Squeeze (Transparent)")
                    gm1m = live_data.sort_values(by=['Gemini_Score_1M','Average_Rank_1M'], ascending=[False,True]).head(top_n)
                    gm1m_cols = ['Ticker','Company','Rank_Gemini_1M','Gemini_Score_1M','Close',
                                 'ema_21','ma_50','dist_ma50','ma_alignment','rsi',
                                 'volume_trend','vol_trend_norm','rvol','ret_21d']
                    st.dataframe(apply_rag_formatting(gm1m[[c for c in gm1m_cols if c in gm1m.columns]]),
                                 use_container_width=True, hide_index=True)
                    st.divider()

                    st.subheader("🗓️ 3-Month — Volume Structure (Transparent)")
                    gm3m = live_data.sort_values(by=['Gemini_Score_3M','Average_Rank_3M'], ascending=[False,True]).head(top_n)
                    gm3m_cols = ['Ticker','Company','Rank_Gemini_3M','Gemini_Score_3M','Close',
                                 'ema_21','ma_200','ema_gap_21_ma200','ret_63d','rsi',
                                 'vol_trend_norm','ma_200_slope','rvol']
                    st.dataframe(apply_rag_formatting(gm3m[[c for c in gm3m_cols if c in gm3m.columns]]),
                                 use_container_width=True, hide_index=True)
                    st.divider()

                    st.subheader("📆 6-Month — Macro Volume Structure (Transparent)")
                    gm6m = live_data.sort_values(by=['Gemini_Score_6M','Average_Rank_6M'], ascending=[False,True]).head(top_n)
                    gm6m_cols = ['Ticker','Company','Rank_Gemini_6M','Gemini_Score_6M','Close',
                                 'ema_21','ma_200','ema_gap_21_ma200','ret_126d','rsi',
                                 'vol_trend_norm','ma_200_slope','ma_alignment_50_200']
                    st.dataframe(apply_rag_formatting(gm6m[[c for c in gm6m_cols if c in gm6m.columns]]),
                                 use_container_width=True, hide_index=True)

                # -----------------------------------------------
                # TAB 5 — HYBRID
                # -----------------------------------------------
                with tab5:
                    st.subheader("⚡ Short-Term — Best-of-All")
                    hy = live_data.sort_values(by=['Hybrid_Score','Average_Rank'], ascending=[False,True]).head(top_n)
                    hy_cols = ['Ticker','Company','Rank_Hybrid','Hybrid_Score','Close',
                               'ma_20','ma_20_slope','macd','macd_signal','rsi','rvol',
                               'near_high','post_earnings','ret_5d','ret_10d']
                    st.dataframe(apply_rag_formatting(hy[[c for c in hy_cols if c in hy.columns]]),
                                 use_container_width=True, hide_index=True)
                    st.divider()

                    st.subheader("📅 1-Month — Best-of-All (Transparent)")
                    hy1m = live_data.sort_values(by=['Hybrid_Score_1M','Average_Rank_1M'], ascending=[False,True]).head(top_n)
                    hy1m_cols = ['Ticker','Company','Rank_Hybrid_1M','Hybrid_Score_1M','Close',
                                 'ma_50','dist_ma50','ma_alignment','ret_21d','momentum_balance',
                                 'rsi','rvol','vol_trend_norm','dist_high']
                    st.dataframe(apply_rag_formatting(hy1m[[c for c in hy1m_cols if c in hy1m.columns]]),
                                 use_container_width=True, hide_index=True)
                    st.divider()

                    st.subheader("🗓️ 3-Month — Best-of-All (Transparent)")
                    hy3m = live_data.sort_values(by=['Hybrid_Score_3M','Average_Rank_3M'], ascending=[False,True]).head(top_n)
                    hy3m_cols = ['Ticker','Company','Rank_Hybrid_3M','Hybrid_Score_3M','Close',
                                 'ma_200','dist_ma200','ma_alignment_50_200','ret_63d','momentum_3m_vs_1m',
                                 'vol_trend_norm','rvol','dist_126d_high','rsi','ma_200_slope']
                    st.dataframe(apply_rag_formatting(hy3m[[c for c in hy3m_cols if c in hy3m.columns]]),
                                 use_container_width=True, hide_index=True)
                    st.divider()

                    st.subheader("📆 6-Month — Best-of-All (Transparent)")
                    hy6m = live_data.sort_values(by=['Hybrid_Score_6M','Average_Rank_6M'], ascending=[False,True]).head(top_n)
                    hy6m_cols = ['Ticker','Company','Rank_Hybrid_6M','Hybrid_Score_6M','Close',
                                 'ma_200','dist_ma200','ma_alignment_50_200','ret_126d','momentum_6m_vs_3m',
                                 'ma_200_slope','vol_trend_norm','rvol','dist_126d_high','rsi']
                    st.dataframe(apply_rag_formatting(hy6m[[c for c in hy6m_cols if c in hy6m.columns]]),
                                 use_container_width=True, hide_index=True)

                # -----------------------------------------------
                # TAB 6 — ALL STOCKS OVERVIEW
                # -----------------------------------------------
                with tab6:
                    st.subheader("📊 All Stocks — Multi-Timeframe Consensus Rank")
                    st.markdown("""
                    Every qualifying stock shown with its **average consensus rank** across each timeframe and
                    an **Overall Rank** (mean of all four). Lower = better. Colour: 🟢 top 20 · 🟡 top 50 · ⚪ rest.
                    
                    Use this table to spot stocks appearing consistently across all timeframes — a strong signal
                    that the setup is structural, not just a short-term spike.
                    """)

                    overview_sort = st.selectbox(
                        "Sort by timeframe:",
                        ['Overall_Rank','Average_Rank','Average_Rank_1M','Average_Rank_3M','Average_Rank_6M'],
                        index=0
                    )

                    overview_cols = ['Ticker','Company','Overall_Rank',
                                     'Average_Rank','Average_Rank_1M','Average_Rank_3M','Average_Rank_6M',
                                     'Close','rsi','rvol','ret_5d','ret_21d','ret_63d','ret_126d']
                    overview_df = live_data[[c for c in overview_cols if c in live_data.columns]].sort_values(overview_sort)

                    st.dataframe(apply_overview_formatting(overview_df),
                                 use_container_width=True, hide_index=True, height=600)

                    st.caption(f"Showing all {len(overview_df)} qualifying stocks. "
                               f"Sorted by **{overview_sort}** (ascending = higher ranked).")