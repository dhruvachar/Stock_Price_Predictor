import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings('ignore')

# Enhanced CSS with improved styling
st.markdown(
    """
    <style>
    .main {
        background-color: #000000;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background: none;
    }
    .sidebar .sidebar-content {
        background-color: #1a1a2e;
        color: white;
    }
    .stButton>button {
        background-color: #6b48ff;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #5a3ce6;
        transform: scale(1.05);
    }
    .card {
        background-color: #6b48ff;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .card-positive {
        background-color: #28a745;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .card-negative {
        background-color: #dc3545;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .card-neutral {
        background-color: #6c757d;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stHeader {
        background: linear-gradient(90deg, #2c2c54, #474787);
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 24px;
    }
    .alert-box {
        background-color: #ffc107;
        color: #000;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
    }
    .status-online {
        color: #28a745;
        font-weight: bold;
    }
    .status-offline {
        color: #dc3545;
        font-weight: bold;
    }
    .metric-label {
        font-size: 12px;
        opacity: 0.8;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 18px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set page configuration
st.set_page_config(page_title="Enhanced Stock Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

# Initialize session state for alerts and portfolio
if 'price_alerts' not in st.session_state:
    st.session_state.price_alerts = {}
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}

# Title and description
st.markdown('<div class="stHeader"><h1>📈 Advanced Stock Analytics Dashboard</h1></div>', unsafe_allow_html=True)
st.markdown("**Professional-grade stock analysis with technical indicators, alerts, and portfolio tracking**")

# Enhanced Sidebar
st.sidebar.header("📊 Configuration")
ticker = st.sidebar.text_input("Stock Symbol", value="AAPL",
                               help="Enter stock ticker (e.g., AAPL, GOOGL, MSFT)").upper()


# Market status check
def check_market_status():
    try:
        now = datetime.now()
        # Simple market hours check (9:30 AM - 4:00 PM EST, Mon-Fri)
        if now.weekday() < 5 and 9 <= now.hour < 16:
            return "🟢 Market Open"
        else:
            return "🔴 Market Closed"
    except:
        return "❓ Unknown"


market_status = check_market_status()
st.sidebar.markdown(f"**Market Status:** {market_status}")

# Time period and intervals
col1, col2 = st.sidebar.columns(2)
with col1:
    period = st.selectbox("Time Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=3)
with col2:
    ma_periods = st.multiselect("Moving Averages", [10, 20, 50, 100, 200], default=[20, 50])

# Technical analysis options
st.sidebar.subheader("📈 Technical Analysis")
show_volume = st.sidebar.checkbox("Show Volume", value=True)
show_rsi = st.sidebar.checkbox("Show RSI", value=True)
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", value=False)
show_macd = st.sidebar.checkbox("Show MACD", value=False)

# Price alerts
st.sidebar.subheader("🚨 Price Alerts")
alert_price = st.sidebar.number_input(f"Set alert for {ticker}", value=0.0,
                                      help="Get notified when price reaches this level")
if st.sidebar.button("Add Alert") and alert_price > 0:
    st.session_state.price_alerts[ticker] = alert_price
    st.sidebar.success(f"Alert set for {ticker} at ${alert_price:.2f}")

# Display active alerts
if st.session_state.price_alerts:
    st.sidebar.write("**Active Alerts:**")
    for symbol, price in st.session_state.price_alerts.items():
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.sidebar.write(f"{symbol}: ${price:.2f}")
        with col2:
            if st.sidebar.button("❌", key=f"del_{symbol}"):
                del st.session_state.price_alerts[symbol]
                st.rerun()

# Refresh settings
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", min_value=10, max_value=120, value=30)


def get_interval(period):
    intervals = {
        "1d": "1m", "5d": "5m", "1mo": "1h",
        "3mo": "1h", "6mo": "1d", "1y": "1d"
    }
    return intervals.get(period, "1d")


def calculate_rsi(prices, window=14):
    """Calculate RSI manually"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_ema(prices, window):
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=window, adjust=False).mean()


def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_technical_indicators(df):
    """Calculate various technical indicators"""
    if df.empty or len(df) < 20:
        return df

    try:
        # RSI
        df['RSI'] = calculate_rsi(df['Close'])

        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])

        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = calculate_macd(df['Close'])

        # Moving Averages
        for period in ma_periods:
            df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()

        # Support and Resistance levels
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()

    except Exception as e:
        st.error(f"Error calculating indicators: {e}")

    return df


def calculate_volatility(df, window=30):
    """Calculate various volatility metrics"""
    if len(df) < window:
        return {}

    returns = df['Close'].pct_change().dropna()

    return {
        'daily_volatility': returns.std() * 100,
        'annualized_volatility': returns.std() * np.sqrt(252) * 100,
        'volatility_trend': 'Increasing' if returns.tail(10).std() > returns.head(10).std() else 'Decreasing'
    }


@st.cache_data(ttl=300)
def fetch_enhanced_stock_data(ticker, period):
    """Fetch and process stock data with enhanced analytics"""
    try:
        interval = get_interval(period)
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            return None, None, None

        # Calculate technical indicators
        df = calculate_technical_indicators(df)

        # Get company info
        info = stock.info

        # Calculate additional metrics
        volatility_metrics = calculate_volatility(df)

        return df, info, volatility_metrics

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None, None


def check_price_alerts(current_price, ticker):
    """Check if any price alerts are triggered"""
    alerts_triggered = []
    if ticker in st.session_state.price_alerts:
        alert_price = st.session_state.price_alerts[ticker]
        if (current_price >= alert_price and current_price <= alert_price * 1.02) or \
                (current_price <= alert_price and current_price >= alert_price * 0.98):
            alerts_triggered.append(f"🚨 ALERT: {ticker} reached ${current_price:.2f} (Target: ${alert_price:.2f})")

    return alerts_triggered


def create_enhanced_chart(df, ticker):
    """Create enhanced multi-panel chart"""
    # Determine number of subplots
    subplot_count = 1
    if show_volume: subplot_count += 1
    if show_rsi: subplot_count += 1
    if show_macd: subplot_count += 1

    # Create subplot titles
    subplot_titles = ['Price']
    if show_volume: subplot_titles.append('Volume')
    if show_rsi: subplot_titles.append('RSI')
    if show_macd: subplot_titles.append('MACD')

    # Create subplots
    fig = make_subplots(
        rows=subplot_count, cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.05,
        row_width=[0.7] + [0.3] * (subplot_count - 1) if subplot_count > 1 else [1.0]
    )

    current_row = 1

    # Main price chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ), row=current_row, col=1)

    # Add moving averages
    colors = ['orange', 'cyan', 'yellow', 'magenta', 'lime']
    for i, period in enumerate(ma_periods):
        if f'MA_{period}' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[f'MA_{period}'],
                name=f'MA{period}',
                line=dict(color=colors[i % len(colors)], width=1)
            ), row=current_row, col=1)

    # Add Bollinger Bands
    if show_bollinger and 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Upper'],
            name='BB Upper', line=dict(color='rgba(255,255,255,0.3)')
        ), row=current_row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Lower'],
            name='BB Lower', line=dict(color='rgba(255,255,255,0.3)'),
            fill='tonexty', fillcolor='rgba(255,255,255,0.1)'
        ), row=current_row, col=1)

    current_row += 1

    # Volume chart
    if show_volume:
        colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' for i in range(len(df))]
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ), row=current_row, col=1)
        current_row += 1

    # RSI
    if show_rsi and 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI'],
            name='RSI',
            line=dict(color='purple', width=2)
        ), row=current_row, col=1)

        # RSI overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)
        current_row += 1

    # MACD
    if show_macd and 'MACD' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MACD'],
            name='MACD',
            line=dict(color='blue', width=2)
        ), row=current_row, col=1)

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MACD_Signal'],
            name='Signal',
            line=dict(color='red', width=1)
        ), row=current_row, col=1)

        # MACD Histogram
        colors = ['green' if val >= 0 else 'red' for val in df['MACD_Histogram']]
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['MACD_Histogram'],
            name='Histogram',
            marker_color=colors,
            opacity=0.6
        ), row=current_row, col=1)

    # Update layout
    fig.update_layout(
        title=f'{ticker} - Advanced Technical Analysis',
        template="plotly_dark",
        height=600 + (subplot_count - 1) * 200,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    return fig


# Main application logic
placeholder = st.empty()


def update_dashboard():
    """Update the entire dashboard"""
    with placeholder.container():
        # Fetch data
        stock_data, stock_info, volatility_metrics = fetch_enhanced_stock_data(ticker, period)

        if stock_data is None or stock_data.empty:
            st.error(f"❌ Unable to fetch data for {ticker}. Please check the symbol and try again.")
            return

        # Current data
        latest_data = stock_data.iloc[-1]
        latest_price = latest_data['Close']
        prev_price = stock_data.iloc[-2]['Close'] if len(stock_data) > 1 else latest_price
        price_change = latest_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0

        # Check price alerts
        alerts = check_price_alerts(latest_price, ticker)
        for alert in alerts:
            st.markdown(f'<div class="alert-box">{alert}</div>', unsafe_allow_html=True)

        # Enhanced metric cards with color coding
        st.subheader("📊 Key Metrics")
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        # Determine card colors based on performance
        price_card_class = "card-positive" if price_change >= 0 else "card-negative"

        with col1:
            st.markdown(f'''
                <div class="card">
                    <div class="metric-label">Current Price</div>
                    <div class="metric-value">${latest_price:.2f}</div>
                </div>
            ''', unsafe_allow_html=True)

        with col2:
            st.markdown(f'''
                <div class="{price_card_class}">
                    <div class="metric-label">Change</div>
                    <div class="metric-value">${price_change:+.2f}<br>({price_change_pct:+.2f}%)</div>
                </div>
            ''', unsafe_allow_html=True)

        with col3:
            st.markdown(f'''
                <div class="card">
                    <div class="metric-label">Day High</div>
                    <div class="metric-value">${stock_data["High"].max():.2f}</div>
                </div>
            ''', unsafe_allow_html=True)

        with col4:
            st.markdown(f'''
                <div class="card">
                    <div class="metric-label">Day Low</div>
                    <div class="metric-value">${stock_data["Low"].min():.2f}</div>
                </div>
            ''', unsafe_allow_html=True)

        with col5:
            volume_latest = latest_data['Volume']
            volume_avg = stock_data['Volume'].mean()
            volume_ratio = volume_latest / volume_avg if volume_avg > 0 else 1
            volume_card_class = "card-positive" if volume_ratio > 1.2 else "card-neutral"

            st.markdown(f'''
                <div class="{volume_card_class}">
                    <div class="metric-label">Volume</div>
                    <div class="metric-value">{volume_latest:,.0f}<br>({volume_ratio:.1f}x avg)</div>
                </div>
            ''', unsafe_allow_html=True)

        with col6:
            if volatility_metrics:
                vol_color = "card-negative" if volatility_metrics['daily_volatility'] > 3 else "card-neutral"
                st.markdown(f'''
                    <div class="{vol_color}">
                        <div class="metric-label">Volatility</div>
                        <div class="metric-value">{volatility_metrics['daily_volatility']:.1f}%<br>{volatility_metrics['volatility_trend']}</div>
                    </div>
                ''', unsafe_allow_html=True)

        # Company overview
        if stock_info:
            st.subheader("🏢 Company Overview")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")

            with col2:
                market_cap = stock_info.get('marketCap', 0)
                if market_cap:
                    st.write(f"**Market Cap:** ${market_cap:,.0f}")
                pe_ratio = stock_info.get('trailingPE', 'N/A')
                st.write(f"**P/E Ratio:** {pe_ratio if pe_ratio != 'N/A' else 'N/A'}")

            with col3:
                beta = stock_info.get('beta', 'N/A')
                st.write(f"**Beta:** {beta if beta != 'N/A' else 'N/A'}")
                div_yield = stock_info.get('dividendYield', 0)
                if div_yield:
                    st.write(f"**Dividend Yield:** {div_yield * 100:.2f}%")
                else:
                    st.write("**Dividend Yield:** N/A")

        # Technical indicators summary
        if 'RSI' in stock_data.columns:
            st.subheader("📈 Technical Analysis Summary")
            latest_rsi = stock_data['RSI'].iloc[-1]

            col1, col2, col3 = st.columns(3)
            with col1:
                rsi_signal = "Overbought" if latest_rsi > 70 else "Oversold" if latest_rsi < 30 else "Neutral"
                rsi_color = "🔴" if latest_rsi > 70 else "🟢" if latest_rsi < 30 else "🟡"
                st.write(f"**RSI (14):** {latest_rsi:.1f} {rsi_color} {rsi_signal}")

            with col2:
                # Moving average trend
                if len(ma_periods) >= 2:
                    ma_short = stock_data[f'MA_{min(ma_periods)}'].iloc[-1]
                    ma_long = stock_data[f'MA_{max(ma_periods)}'].iloc[-1]
                    trend = "Bullish" if ma_short > ma_long else "Bearish"
                    trend_color = "🟢" if trend == "Bullish" else "🔴"
                    st.write(f"**MA Trend:** {trend_color} {trend}")

            with col3:
                # Price vs MA
                if ma_periods:
                    ma_key = f'MA_{ma_periods[0]}'
                    if ma_key in stock_data.columns:
                        ma_value = stock_data[ma_key].iloc[-1]
                        position = "Above" if latest_price > ma_value else "Below"
                        pos_color = "🟢" if position == "Above" else "🔴"
                        st.write(f"**Price vs MA{ma_periods[0]}:** {pos_color} {position}")

        # Enhanced chart
        st.subheader("📊 Advanced Price Chart")
        fig = create_enhanced_chart(stock_data, ticker)
        st.plotly_chart(fig, use_container_width=True)

        # Performance metrics
        st.subheader("📈 Performance Analysis")
        col1, col2 = st.columns(2)

        with col1:
            # Period performance
            period_start_price = stock_data['Close'].iloc[0]
            period_return = ((latest_price - period_start_price) / period_start_price) * 100

            st.write(f"**{period.upper()} Return:** {period_return:+.2f}%")

            # High/Low analysis
            period_high = stock_data['High'].max()
            period_low = stock_data['Low'].min()
            current_from_high = ((latest_price - period_high) / period_high) * 100
            current_from_low = ((latest_price - period_low) / period_low) * 100

            st.write(f"**From Period High:** {current_from_high:+.2f}%")
            st.write(f"**From Period Low:** {current_from_low:+.2f}%")

        with col2:
            if volatility_metrics:
                st.write(f"**Daily Volatility:** {volatility_metrics['daily_volatility']:.2f}%")
                st.write(f"**Annualized Volatility:** {volatility_metrics['annualized_volatility']:.2f}%")
                st.write(f"**Volatility Trend:** {volatility_metrics['volatility_trend']}")

        # Data export section
        st.subheader("💾 Data Export")
        col1, col2, col3 = st.columns(3)

        with col1:
            csv_data = stock_data.to_csv()
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"{ticker}_{period}_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

        with col2:
            json_data = stock_data.to_json(orient='records', date_format='iso')
            st.download_button(
                label="📋 Download JSON",
                data=json_data,
                file_name=f"{ticker}_{period}_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )

        # Status and timestamp
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST")
        with col2:
            data_points = len(stock_data)
            st.markdown(f"**Data Points:** {data_points:,} | **Interval:** {get_interval(period)}")


# Auto-refresh toggle and manual refresh
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    auto_refresh = st.checkbox("🔄 Auto-refresh", value=False)
with col2:
    if st.button("🔄 Refresh Now"):
        update_dashboard()
with col3:
    if st.button("🛑 Stop"):
        auto_refresh = False

# Initial load
if not auto_refresh:
    update_dashboard()

# Auto-refresh loop
if auto_refresh:
    while auto_refresh:
        update_dashboard()
        time.sleep(refresh_interval)

# Footer with instructions
st.markdown("""
---
### 🚀 Dashboard Features
- **Real-time Price Tracking:** Live updates with customizable intervals
- **Advanced Technical Analysis:** RSI, MACD, Bollinger Bands, Multiple MAs
- **Price Alerts:** Get notified when stocks hit target prices
- **Volume Analysis:** Compare current volume with historical averages  
- **Volatility Metrics:** Daily and annualized volatility calculations
- **Performance Analytics:** Period returns and drawdown analysis
- **Data Export:** Download data in CSV and JSON formats
- **Market Status:** Real-time market open/close indicator
- **Professional Styling:** Dark theme optimized for financial data

**Usage Tips:**
- Use shorter periods (1d, 5d) for intraday analysis with minute-level data
- Enable multiple moving averages to identify trend strength
- Set price alerts for key support/resistance levels
- Monitor RSI for overbought (>70) and oversold (<30) conditions
- Volume spikes often precede significant price movements
""")