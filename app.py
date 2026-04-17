import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION (config.py)
# ==========================================
TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL", "AMD", 
    "NFLX", "PLTR", "SOFI", "HOOD", "COIN", "UBER", "CRWD"
] # Simplified universe for fast testing
START_DATE = "2023-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')
REBALANCE_FREQ = "W-FRI"
HOLD_DAYS = 5
TOP_N = 3
MIN_PRICE = 5
MIN_VOLUME = 1_000_000

# ==========================================
# 2. DATA & INDICATORS (data.py & indicators.py)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_and_prepare_data(tickers, start, end):
    """Fetches data and calculates all vectorized indicators."""
    df_list = []
    
    # Download data
    data = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False)
    
    for ticker in tickers:
        try:
            # Handle single ticker vs multi-ticker yfinance output
            if len(tickers) == 1:
                df = data.copy()
            else:
                df = data[ticker].copy()
                
            df.dropna(inplace=True)
            if df.empty: continue
            
            df['Ticker'] = ticker
            
            # --- Indicators ---
            # Moving Averages
            df['ma_20'] = df['Close'].rolling(window=20).mean()
            df['ma_50'] = df['Close'].rolling(window=50).mean()
            df['ema_8'] = df['Close'].ewm(span=8, adjust=False).mean()
            df['ema_21'] = df['Close'].ewm(span=21, adjust=False).mean()
            
            # Slopes (Momentum of MAs)
            df['ma_20_slope'] = df['ma_20'].diff(5)
            
            # MACD
            ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            # RSI (14-day)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Volume & Price Action
            df['volume_avg_20'] = df['Volume'].rolling(window=20).mean()
            df['rvol'] = df['Volume'] / df['volume_avg_20']
            df['volume_trend'] = df['volume_avg_20'].diff(5)
            
            # Returns
            df['ret_5d'] = df['Close'].pct_change(5)
            df['ret_10d'] = df['Close'].pct_change(10)
            df['fwd_ret_5d'] = df['Close'].shift(-5) / df['Close'] - 1 # What we are trying to predict
            
            # Breakout/Highs
            df['high_50d'] = df['High'].rolling(window=50).max()
            df['near_high'] = df['Close'] >= (df['high_50d'] * 0.95)
            df['close_near_high'] = df['Close'] >= (df['High'] - 0.2 * (df['High'] - df['Low']))
            
            # Proxies for missing fundamental data
            # Proxy for post-earnings: Massive volume spike + gap up
            df['post_earnings'] = (df['rvol'] > 3.0) & (df['ret_5d'] > 0.05)
            df['short_interest_proxy'] = (df['Close'] < df['ma_50']) & (df['rvol'] > 2.0) # Down trend, sudden high volume
            
            df_list.append(df)
        except Exception as e:
            continue
            
    full_df = pd.concat(df_list)
    full_df = full_df[(full_df['Close'] >= MIN_PRICE) & (full_df['volume_avg_20'] >= MIN_VOLUME)]
    return full_df

# ==========================================
# 3. SCORING MODELS (Vectorized)
# ==========================================
def score_chatgpt(df):
    score = pd.Series(0, index=df.index)
    score += np.where(df['Close'] > df['ma_20'], 10, 0)
    score += np.where(df['ma_20_slope'] > 0, 10, 0)
    score += np.where((df['Close'] - df['ma_20']) / df['ma_20'] < 0.08, 10, 0)
    score += np.where((df['rsi'] >= 55) & (df['rsi'] <= 70), 10, 0)
    score += np.where(df['ret_5d'] > 0, 5, 0)
    score += np.where(df['macd'] > df['macd_signal'], 5, 0)
    score += np.where(df['rvol'] >= 1.5, 15, 0)
    score += np.where(df['volume_trend'] > 0, 10, 0)
    score -= np.where(df['ret_5d'] > 0.15, 15, 0) # Risk penalty
    return score

def score_grok(df):
    score = pd.Series(0, index=df.index)
    score += np.where(df['ret_5d'] > 0, 10, 0)
    score += np.where(df['ret_10d'] > df['ret_5d'], 5, 0)
    score += np.where(df['rvol'] >= 1.5, 10, 0)
    score += np.where(df['Close'] > df['ma_20'], 5, 0)
    score += np.where(df['ma_20'] > df['ma_50'], 5, 0)
    score += np.where(df['near_high'], 10, 0)
    return score

def score_gemini(df):
    score = pd.Series(0, index=df.index)
    score += np.where(df['ema_8'] > df['ema_21'], 15, 0)
    score += np.where(df['macd'] > df['macd_signal'], 15, 0)
    score += np.where((df['rsi'] >= 50) & (df['rsi'] <= 70), 10, 0)
    score += np.where(df['rvol'] >= 1.5, 20, np.where(df['rvol'] >= 1.2, 10, 0))
    score += np.where(df['close_near_high'], 20, 0)
    score += np.where(df['post_earnings'], 10, 0)
    score += np.where(df['short_interest_proxy'], 10, 0)
    return score

def score_hybrid(df):
    score = pd.Series(0, index=df.index)
    score += np.where(df['Close'] > df['ma_20'], 5, 0)
    score += np.where(df['ma_20_slope'] > 0, 5, 0)
    score += np.where((df['rsi'] >= 55) & (df['rsi'] <= 70), 8, 0)
    score += np.where(df['macd'] > df['macd_signal'], 6, 0)
    score += np.where(df['ret_10d'] > df['ret_5d'], 6, 0)
    score += np.where(df['rvol'] >= 1.5, 10, 0)
    score += np.where(df['close_near_high'], 5, 0)
    score += np.where(df['near_high'], 5, 0)
    score += np.where(df['post_earnings'], 10, 0)
    score -= np.where(df['ret_5d'] > 0.15, 10, 0)
    return score

# ==========================================
# 4. BACKTEST ENGINE
# ==========================================
def run_backtest(df, model_func):
    """Runs a vectorized backtest by sorting scores on Fridays."""
    df = df.copy()
    df['Score'] = model_func(df)
    
    # Filter only to Fridays (Rebalance days)
    df.reset_index(inplace=True)
    fridays = df[df['Date'].dt.dayofweek == 4].copy()
    
    # Rank and pick Top N
    fridays['Rank'] = fridays.groupby('Date')['Score'].rank(method='first', ascending=False)
    picks = fridays[fridays['Rank'] <= TOP_N].copy()
    
    # Calculate portfolio returns per week
    weekly_returns = picks.groupby('Date')['fwd_ret_5d'].mean().reset_index()
    weekly_returns['fwd_ret_5d'] = weekly_returns['fwd_ret_5d'].fillna(0)
    weekly_returns['Cumulative_Return'] = (1 + weekly_returns['fwd_ret_5d']).cumprod()
    
    return weekly_returns, picks

# ==========================================
# 5. STREAMLIT UI (main.py)
# ==========================================
st.set_page_config(page_title="V2 AI Momentum Scanner", layout="wide")

st.title("📈 V2 AI Momentum Scanner & Backtester")
st.markdown("Comparing methodologies from ChatGPT, Grok, and Gemini.")

if st.button("Run Engine & Backtest"):
    with st.spinner("Fetching data and calculating vectorized indicators..."):
        df = fetch_and_prepare_data(TICKERS, START_DATE, END_DATE)
        
    models = {
        "ChatGPT (Trend)": score_chatgpt,
        "Grok (Breakout)": score_grok,
        "Gemini (Volume+Cat)": score_gemini,
        "Hybrid (V2)": score_hybrid
    }
    
    results = {}
    picks_dict = {}
    
    with st.spinner("Running vectorized backtests..."):
        for name, func in models.items():
            returns, picks = run_backtest(df, func)
            results[name] = returns
            picks_dict[name] = picks

    # --- RESULTS DASHBOARD ---
    st.header("📊 1. Equity Curves")
    
    # Combine curves for plotting
    plot_df = pd.DataFrame()
    for name, res in results.items():
        temp = res[['Date', 'Cumulative_Return']].copy()
        temp['Model'] = name
        plot_df = pd.concat([plot_df, temp])
        
    fig = px.line(plot_df, x='Date', y='Cumulative_Return', color='Model', 
                  title="Cumulative Returns by AI Model (Top 3 Picks Weekly)")
    st.plotly_chart(fig, use_container_width=True)
    
    # --- SUMMARY STATS ---
    st.header("📋 2. Summary Statistics")
    stats = []
    for name, res in results.items():
        total_return = (res['Cumulative_Return'].iloc[-1] - 1) * 100 if not res.empty else 0
        win_rate = (res['fwd_ret_5d'] > 0).mean() * 100
        stats.append({"Model": name, "Total Return (%)": round(total_return, 2), "Win Rate (%)": round(win_rate, 2)})
        
    st.dataframe(pd.DataFrame(stats), use_container_width=True)
    
    # --- WEEKLY OVERLAP (The Real Insight) ---
    st.header("🔍 3. Current Week Picks & Overlap")
    st.markdown("If multiple AI models select the same stock, conviction is high.")
    
    latest_date = picks_dict["Hybrid (V2)"]['Date'].max()
    st.write(f"**Latest Rebalance Date Analyzed:** {latest_date.strftime('%Y-%m-%d')}")
    
    overlap_data = []
    for name, picks in picks_dict.items():
        recent_picks = picks[picks['Date'] == latest_date]['Ticker'].tolist()
        overlap_data.append({"Model": name, "Selected Tickers": ", ".join(recent_picks)})
        
    st.table(pd.DataFrame(overlap_data))

    # Calculate literal overlap
    all_recent_picks = []
    for picks in picks_dict.values():
        all_recent_picks.extend(picks[picks['Date'] == latest_date]['Ticker'].tolist())
    
    overlap_counts = pd.Series(all_recent_picks).value_counts()
    high_conviction = overlap_counts[overlap_counts > 1]
    
    if not high_conviction.empty:
        st.success(f"🔥 **High Conviction Overlaps:** {', '.join(high_conviction.index.tolist())}")
    else:
        st.info("No overlaps found for the current week.")