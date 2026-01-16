"""
comoney 
Utilise uniquement: streamlit, pandas, numpy, yfinance, plotly, scipy
Fonctionnalités:
- Récupération données via yfinance
- Rendements arithm./log
- SMA, EMA, Bollinger, RSI, MACD
- Statistiques (mean, std, skew, kurtosis, VaR paramétrique)
- Backtest simple SMA crossover
- Trading manuel 
- Graphiques interactifs avec Plotly
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

st.set_page_config(page_title="comoney", page_icon="📈", layout="wide")


def get_data(symbol: str, start: datetime, end: datetime, interval: str = "1d") -> pd.DataFrame:
  
    try:
        df = yf.download(symbol, start=start, end=end + timedelta(days=1), interval=interval, progress=False, threads=False)
    except Exception as e:
        st.error(f"yfinance error: {e}")
        return pd.DataFrame()

    if df is None or (hasattr(df, "empty") and df.empty):
        return pd.DataFrame()

    if isinstance(df, pd.Series):
        df = df.to_frame()

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = list(df.columns.get_level_values(0))
        lvl1 = list(df.columns.get_level_values(1))

        # Case A: symbol is in level 0 (e.g., df['AAPL']['Close'])
        if symbol in lvl0:
            try:
                df = pd.DataFrame(df[symbol]).copy()
            except Exception:
                df.columns = ["_".join(map(str, c)) for c in df.columns]
        # Case B: symbol is in level 1 (e.g., ('Close','AAPL'))
        elif symbol in lvl1:
            try:
                df = df.xs(symbol, axis=1, level=1).copy()
                if isinstance(df, pd.Series):
                    df = df.to_frame(name="Close" if df.name in ["Close", "Adj Close"] else df.name)
            except Exception:
                df.columns = ["_".join(map(str, c)) for c in df.columns]
        else:
            cols_with_symbol = [c for c in df.columns if symbol in c]
            if cols_with_symbol:
                newd = {}
                for c in cols_with_symbol:
                    if isinstance(c, tuple):
                        other = c[0] if c[1] == symbol else c[1]
                        newd[str(other)] = df[c]
                    else:
                        newd[str(c)] = df[c]
                df = pd.DataFrame(newd)
            else:
                df.columns = ["_".join(map(str, c)) for c in df.columns]

    # Ensure DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    if "Close" not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df["Close"] = df[numeric_cols[-1]]
        else:
            return pd.DataFrame()

    # dropna on Close safely
    if "Close" in df.columns:
        try:
            df = df.dropna(subset=["Close"])
        except Exception:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

    # ensure datetime index
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass

    return df


# -----------------------------
# Financial calculations
# -----------------------------
def arithmetic_returns(prices: pd.Series) -> pd.Series:
    values = prices.values
    returns = []

    for i in range(1, len(values)):
        r = (values[i] - values[i - 1]) / values[i - 1]
        returns.append(r)

    return pd.Series(returns, index=prices.index[1:])


import math

def logarithmic_returns(prices: pd.Series) -> pd.Series:
    values = prices.values
    returns = []

    for i in range(1, len(values)):
        lr = math.log(values[i] / values[i - 1])
        returns.append(lr)

    return pd.Series(returns, index=prices.index[1:])


def sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window=window, min_periods=1).mean()


def ema(prices: pd.Series, span: int) -> pd.Series:
    return prices.ewm(span=span, adjust=False).mean()


def bollinger_bands(prices: pd.Series, window: int = 20, n_std: float = 2.0):
    mid = sma(prices, window)
    std = prices.rolling(window=window, min_periods=1).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    # Use exponential moving average of gains/losses (Wilder's smoothing)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(50)


def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def volatility_annualized(returns: pd.Series, periods_per_year: int = 252) -> float:
    std_daily = np.sqrt(((returns - returns.mean())**2).sum() / len(returns))
    vol_ann = std_daily * (periods_per_year ** 0.5)
    return vol_ann


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    mean_daily = returns.mean()
    ann_return = mean_daily * periods_per_year
    ann_vol = volatility_annualized(returns, periods_per_year)
    if ann_vol == 0:
        return 0.0
    return (ann_return - risk_free_rate) / ann_vol


def var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    # mean daily
    mu = returns.mean()
    sigma = np.sqrt(((returns - mu)**2).sum() / len(returns))
    # z-score for 1-confidence
    z = stats.norm.ppf(1 - confidence)
    var = -(mu + z * sigma)
    return var

# -----------------------------
# SMA Crossover Backtest
# -----------------------------
def backtest_sma_crossover(prices: pd.Series, short: int = 20, long: int = 50, initial_capital: float = 10000.0):
    """
    Backtest simple: Buy full position when SMA_short crosses above SMA_long, sell when crosses below.
    No transaction costs, no partial sizing (all-in/all-out).
    Returns: dict with portfolio value timeseries, trades, metrics
    """
    if long <= short:
        raise ValueError("long must be > short")

    df = pd.DataFrame({"close": prices})
    df["sma_short"] = sma(df["close"], short)
    df["sma_long"] = sma(df["close"], long)
    df = df.dropna().copy()
    df["position"] = 0 

    # generate signals
    df["signal"] = np.where(df["sma_short"] > df["sma_long"], 1, 0)
    df["signal_shift"] = df["signal"].shift(1).fillna(0)
    df["trade"] = df["signal"] - df["signal_shift"]  # +1 = buy, -1 = sell

    cash = initial_capital
    shares = 0
    portfolio_values = []
    trades = []

    for idx, row in df.iterrows():
        price = row["close"]
        if row["trade"] == 1:  # buy
            if cash >= price:
                shares = int(cash // price)
                spent = shares * price
                cash -= spent
                trades.append({"date": idx, "type": "BUY", "price": float(price), "shares": int(shares)})
        elif row["trade"] == -1:  # sell
            if shares > 0:
                proceeds = shares * price
                cash += proceeds
                trades.append({"date": idx, "type": "SELL", "price": float(price), "shares": int(shares)})
                shares = 0
        # portfolio value 
        total = cash + shares * price
        portfolio_values.append({"date": idx, "value": total})

    pv_df = pd.DataFrame(portfolio_values).set_index("date")
    pv_df.index = pd.to_datetime(pv_df.index)
    pv_df = pv_df.sort_index()

    start_value = initial_capital
    end_value = float(pv_df["value"].iloc[-1]) if not pv_df.empty else start_value
    total_return_pct = (end_value / start_value - 1) * 100

    # daily returns of portfolio
    pv_df["ret"] = pv_df["value"].pct_change().fillna(0)

    log_rets = np.log(1 + pv_df["ret"].replace(-1, 0))
    sr = sharpe_ratio(log_rets)

    # drawdown
    pv_df["peak"] = pv_df["value"].cummax()
    pv_df["drawdown"] = (pv_df["peak"] - pv_df["value"]) / pv_df["peak"]
    max_drawdown_pct = pv_df["drawdown"].max() * 100 if "drawdown" in pv_df.columns else 0.0

    # trades metrics
    trades_df = pd.DataFrame(trades)
    wins = 0
    profits = []
    if not trades_df.empty:
        buy = None
        for _, t in trades_df.iterrows():
            if t["type"] == "BUY":
                buy = t
            elif t["type"] == "SELL" and buy is not None:
                profit = (t["price"] - buy["price"]) * t["shares"]
                profits.append(profit)
                if profit > 0:
                    wins += 1
                buy = None
    win_rate = (wins / (len(profits)) * 100) if profits else 0.0
    avg_profit = np.mean(profits) if profits else 0.0
    profit_factor = (sum([p for p in profits if p > 0]) / abs(sum([p for p in profits if p < 0]))) if any(np.array(profits) < 0) else (np.inf if sum(profits) > 0 else 0)

    result = {
        "pv": pv_df,
        "trades": trades_df.to_dict(orient="records") if not trades_df.empty else [],
        "final_value": end_value,
        "total_return": total_return_pct,
        "sharpe_ratio": sr,
        "max_drawdown": max_drawdown_pct,
        "num_trades": len(trades_df),
        "win_rate": win_rate,
        "avg_profit": avg_profit,
        "profit_factor": profit_factor,
    }
    return result


# -----------------------------
# Plot helpers (Plotly)
# -----------------------------
def plot_price_with_indicators(df: pd.DataFrame, symbol: str, show_sma: list = None, show_ema: list = None, bb: tuple = None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(
        go.Candlestick(x=df.index, open=df.get("Open"), high=df.get("High"), low=df.get("Low"), close=df.get("Close"), name=f"{symbol} OHLC"),
        row=1, col=1
    )
    fig.add_trace(go.Bar(x=df.index, y=df.get("Volume"), name="Volume", marker_color="rgba(100,100,100,0.4)"), row=2, col=1)

    if show_sma:
        for period in show_sma:
            s = sma(df["Close"], period)
            fig.add_trace(go.Scatter(x=df.index, y=s, name=f"SMA {period}", line=dict(width=1.5)), row=1, col=1)

    if show_ema:
        for period in show_ema:
            e = ema(df["Close"], period)
            fig.add_trace(go.Scatter(x=df.index, y=e, name=f"EMA {period}", line=dict(dash="dash", width=1.2)), row=1, col=1)

    if bb:
        upper, mid, lower = bb
        fig.add_trace(go.Scatter(x=df.index, y=upper, name="BB Upper", line=dict(color="gray", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=lower, name="BB Lower", line=dict(color="gray", width=1), fill="tonexty", fillcolor="rgba(128,128,128,0.08)"), row=1, col=1)

    fig.update_layout(height=700, title=f"{symbol} Price & Indicators", xaxis_rangeslider_visible=False, hovermode="x unified")
    return fig


def plot_returns_distribution(returns_arith: pd.Series, returns_log: pd.Series, symbol: str):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Arithmetic Returns", "Log Returns"))
    fig.add_trace(go.Histogram(x=returns_arith, nbinsx=60, name="arith"), row=1, col=1)
    fig.add_trace(go.Histogram(x=returns_log, nbinsx=60, name="log", marker_color="orange"), row=1, col=2)
    fig.update_layout(title=f"Returns Distribution - {symbol}", bargap=0.1, height=400)
    return fig


def plot_rsi(series: pd.Series, period: int = 14):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series, name=f"RSI ({period})", line=dict(color="purple")))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(height=350, yaxis=dict(range=[0, 100]), title="RSI")
    return fig


def plot_macd(macd_line: pd.Series, signal_line: pd.Series, histogram: pd.Series):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=macd_line.index, y=macd_line, name="MACD"), row=1, col=1)
    fig.add_trace(go.Scatter(x=signal_line.index, y=signal_line, name="Signal", line=dict(color="red")), row=1, col=1)
    fig.add_trace(go.Bar(x=histogram.index, y=histogram, name="Histogram", marker_color=np.where(histogram >= 0, "green", "red")), row=2, col=1)
    fig.update_layout(height=400, title="MACD")
    return fig


def plot_portfolio(pv_df: pd.DataFrame, symbol: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pv_df.index, y=pv_df["value"], name="Portfolio Value", line=dict(color="blue")))
    fig.update_layout(title=f"Portfolio Value (Backtest) vs Time", height=400)
    return fig


def plot_drawdown(pv_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pv_df.index, y=pv_df["drawdown"] * 100, name="Drawdown (%)", line=dict(color="red")))
    fig.update_layout(title="Drawdown (%)", height=300, yaxis_title="Drawdown %")
    return fig


def qq_plot(returns: pd.Series):
    (theoretical, ordered), (slope, intercept, r) = stats.probplot(returns, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theoretical, y=ordered, mode="markers", name="Data"))
    fig.add_trace(go.Line(x=theoretical, y=slope * theoretical + intercept, name="Fit", line=dict(color="red")))
    fig.update_layout(title="QQ-Plot vs Normal", height=400, xaxis_title="Theoretical Quantiles", yaxis_title="Ordered Values")
    return fig


# -----------------------------
# Trading UI & Simple Portfolio (integrated)
# -----------------------------
def _init_portfolio(initial_cash: float = 10000.0):
    if 'portfolio' not in st.session_state:
        st.session_state['portfolio'] = {
            'cash': float(initial_cash),
            'positions': {},       # symbol -> shares (int)
            'trades': [],          # list of trades
            'nav_history': []      # list of dicts {'date': ..., 'nav': ...}
        }


def _record_trade(symbol: str, side: str, qty: int, price: float, cash_after: float, value_after: float):
    st.session_state['portfolio']['trades'].append({
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'symbol': symbol,
        'type': side,
        'price': float(price),
        'qty': int(qty),
        'cash_after': float(cash_after),
        'value_after': float(value_after)
    })


def _update_nav_history(date, nav):
    st.session_state['portfolio']['nav_history'].append({'date': date, 'nav': float(nav)})


def place_order(symbol: str, side: str, qty: int, price: float):
    """
    side: 'BUY' or 'SELL'
    qty: integer number of shares
    price: executed price (market or custom)
    """
    _init_portfolio()
    pf = st.session_state['portfolio']
    cash = float(pf['cash'])
    positions = pf['positions']

    if qty <= 0:
        st.error("Quantity must be > 0")
        return False

    cost = qty * price

    if side == 'BUY':
        if cash < cost:
            st.error("Insufficient cash to buy")
            return False
        # update cash and position
        cash -= cost
        positions[symbol] = positions.get(symbol, 0) + int(qty)
    else:  # SELL
        held = positions.get(symbol, 0)
        if held < qty:
            st.error(f"Not enough shares to sell (held={held})")
            return False
        # update cash and position
        cash += cost
        positions[symbol] = held - int(qty)
        if positions[symbol] == 0:
            del positions[symbol]

    pf['cash'] = float(cash)

    # compute current portfolio value (NAV) using current price for this symbol
    pos_val = positions.get(symbol, 0) * price
    nav = pf['cash'] + pos_val
    _record_trade(symbol, side, qty, price, pf['cash'], nav)
    _update_nav_history(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), nav)

    # developer console trace
    print(f"place_order executed: {side} {qty} {symbol} @ {price:.2f} | cash: {pf['cash']:.2f} | positions: {pf['positions']}")
    st.success(f"{side} {qty} {symbol} @ {price:.2f} — NAV: ${nav:,.2f}")
    return True


def trading_ui(symbol: str, close_series: pd.Series, initial_cash: float = 10000.0):
    """
    Clean trading UI: show portfolio as table and provide a stable form in the main area.
    Uses existing helpers: _init_portfolio(...) and place_order(...).
    """
    st.markdown("### 🛒 Trading manuel")
    _init_portfolio(initial_cash)
    pf = st.session_state['portfolio']

    # safe last_price extraction
    try:
        last_price = float(close_series.iloc[-1])
    except Exception:
        last_price = 0.0

    # Snapshot
    st.subheader("Snapshot")
    st.markdown(f"- Cash: **${pf['cash']:,.2f}**")
    st.markdown(f"- Last price for **{symbol}**: **${last_price:.2f}**")

    # Positions as a nice table
    st.subheader("Positions")
    if pf['positions']:
        pos_rows = []
        for s, shares in pf['positions'].items():
            price = float(close_series.iloc[-1]) if s == symbol and last_price != 0.0 else np.nan
            value = shares * price if not np.isnan(price) else np.nan
            pos_rows.append({"symbol": s, "shares": int(shares), "price": price, "value": value})
        pos_df = pd.DataFrame(pos_rows).set_index("symbol")
        st.table(pos_df)
    else:
        st.info("No positions currently held")

    st.markdown("---")

    # Order form (main area) using st.form to avoid intermediate reruns
    st.subheader("Passer un ordre")
    with st.form("order_form_clean", clear_on_submit=False):
        side = st.radio("Side", ["BUY", "SELL"], index=0)
        mode = st.radio("Qty mode", ["Quantity", "Percent of Cash"], index=0)

        if mode == "Quantity":
            qty = st.number_input("Quantity (shares)", min_value=1, value=1, step=1)
        else:
            pct = st.slider("Percent of cash to use", min_value=1, max_value=100, value=10)
            usd_to_use = (pct / 100.0) * pf['cash']
            qty = int(usd_to_use // last_price) if last_price > 0 else 0
            st.write(f"Will use ~${usd_to_use:,.2f} → qty = {qty} shares")

        price_mode = st.selectbox("Price", ["Market (last close)", "Custom (limit)"], index=0)
        if price_mode == "Market (last close)":
            exec_price = last_price
            st.write(f"Market price = ${exec_price:.2f}")
        else:
            exec_price = float(st.number_input("Limit price", value=float(last_price), format="%.4f"))

        submit = st.form_submit_button("Execute Order")

        if submit:
            # call the existing order execution function
            success = place_order(symbol, side, int(qty), float(exec_price))
            if success:
                st.success(f"Order executed: {side} {qty} @ ${exec_price:.2f}")
            else:
                st.error("Order failed (see console for details)")

    st.markdown("---")

    # Trade history
    st.subheader("Trade history")
    if pf['trades']:
        trades_df = pd.DataFrame(pf['trades']).sort_values("date", ascending=False)
        st.dataframe(trades_df)
    else:
        st.info("No trades yet")

    # NAV history
    st.subheader("NAV history")
    if pf['nav_history']:
        nav_df = pd.DataFrame(pf['nav_history']).set_index('date')
        st.line_chart(nav_df['nav'])
    else:
        st.write("No NAV history")

    return st.session_state['portfolio']


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.markdown("<h1 style='text-align:center;color:#1f77b4'>📈 comoney </h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar inputs
    with st.sidebar:
        st.header("⚙️ Configuration")
        source = st.selectbox("Source", ["Yahoo Finance (Stocks)", "Yahoo Finance (Crypto)"])
        if "Crypto" in source:
            symbol = st.selectbox("Symbol", ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD"], index=0)
        else:
            symbol = st.text_input("Symbol", value="AAPL")
        st.subheader("📅 Dates")
        default_start = datetime.now() - timedelta(days=365)
        start_date = st.date_input("Start date", default_start)
        end_date = st.date_input("End date", datetime.now())
        interval = st.selectbox("Interval", ["1d", "1wk", "1mo", "1h"], index=0)
        st.subheader("Indicators & Backtest")
        show_sma = st.multiselect("SMA (periods)", [10, 20, 50, 100, 200], default=[20, 50])
        show_ema = st.multiselect("EMA (periods)", [10, 20, 50], default=[20])
        show_bb = st.checkbox("Show Bollinger Bands", value=True)
        rsi_period = st.slider("RSI period", 7, 30, 14)
        run_btn = st.button("🚀 Load Data & Analyze")

    # If user clicked Load Data, fetch and persist in session_state
    if run_btn:
        with st.spinner("Downloading data..."):
            data = get_data(symbol, pd.to_datetime(start_date), pd.to_datetime(end_date), interval)
        if data is None or data.empty:
            st.error("No data returned for this symbol / period.")
            return

        # persist data and symbol so trading UI survives reruns
        st.session_state['data'] = data
        st.session_state['symbol'] = symbol
        st.session_state['initial_cash'] = st.session_state.get('initial_cash', 10000.0)

        st.success(f"Loaded {len(data)} rows for {symbol}")

    # If data is present in session_state, show trading UI (persists between reruns)
    if 'data' in st.session_state and isinstance(st.session_state['data'], pd.DataFrame) and not st.session_state['data'].empty:
        data = st.session_state['data']
        symbol_session = st.session_state.get('symbol', symbol)
        close = data['Close']
        # show trading UI (main area)
        trading_ui(symbol_session, close, initial_cash=st.session_state.get('initial_cash', 10000.0))

        # Continue to show metrics & charts below trading UI
        st.header(f"Overview - {symbol_session}")

        # Metrics
        latest = close.iloc[-1]
        price_change = latest - close.iloc[0]
        price_change_pct = price_change / close.iloc[0] * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last Price", f"${latest:.2f}")
        c2.metric("Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
        try:
            high_val = data.get('High').max()
            low_val = data.get('Low').min()
            c3.metric("High", f"${high_val:.2f}")
            c4.metric("Low", f"${low_val:.2f}")
        except Exception:
            c3.metric("High", "-")
            c4.metric("Low", "-")

        # Price chart
        bb_tuple = None
        if show_bb:
            upper, mid, lower = bollinger_bands(close, window=20, n_std=2.0)
            bb_tuple = (upper, mid, lower)

        fig_price = plot_price_with_indicators(data, symbol_session, show_sma=show_sma, show_ema=show_ema, bb=bb_tuple)
        st.plotly_chart(fig_price, use_container_width=True)

        # Returns analysis
        st.subheader("Returns Analysis")
        returns_arith = arithmetic_returns(close)
        returns_log = logarithmic_returns(close)

        ra_col1, ra_col2 = st.columns(2)
        with ra_col1:
            st.markdown("#### Arithmetic returns (sample)")
            st.metric("Mean (daily)", f"{returns_arith.mean()*100:.4f}%")
            st.metric("Std (daily)", f"{returns_arith.std()*100:.4f}%")
        with ra_col2:
            st.markdown("#### Log returns (sample)")
            st.metric("Mean (daily)", f"{returns_log.mean()*100:.4f}%")
            st.metric("Std (daily)", f"{returns_log.std()*100:.4f}%")

        fig_dist = plot_returns_distribution(returns_arith, returns_log, symbol_session)
        st.plotly_chart(fig_dist, use_container_width=True)

        # Technical indicators: RSI & MACD
        st.subheader("Technical Indicators")
        rsi_series = rsi(close, rsi_period)
        st.plotly_chart(plot_rsi(rsi_series, rsi_period), use_container_width=True)

        macd_line, signal_line, hist = macd(close)
        st.plotly_chart(plot_macd(macd_line, signal_line, hist), use_container_width=True)

        # Stats & risk metrics
        st.subheader("Statistical & Risk Metrics")
        mean_log = returns_log.mean()
        std_log = returns_log.std(ddof=1)
        skew = returns_log.skew()
        kurt = returns_log.kurtosis()
        vol_ann = volatility_annualized(returns_log, periods_per_year=252)
        sharpe = sharpe_ratio(returns_log)
        var95 = var_parametric(returns_log, confidence=0.95)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Mean daily (log)", f"{mean_log*100:.4f}%")
        s2.metric("Std daily (log)", f"{std_log*100:.4f}%")
        s3.metric("Skewness", f"{skew:.4f}")
        s4.metric("Kurtosis (excess)", f"{kurt:.4f}")

        r1, r2, r3 = st.columns(3)
        r1.metric("Vol Annualized", f"{vol_ann*100:.2f}%")
        r2.metric("Sharpe Ratio", f"{sharpe:.4f}")
        r3.metric("VaR 95%", f"{var95*100:.2f}%")

        # QQ-plot
        try:
            st.plotly_chart(qq_plot(returns_log.values), use_container_width=True)
        except Exception:
            pass

        # Backtest area
        st.subheader("Backtesting - SMA Crossover")
        bt_col1, bt_col2, bt_col3 = st.columns([1,1,1])
        with bt_col1:
            sma_short = st.number_input("SMA short period", min_value=2, max_value=200, value=20, step=1)
        with bt_col2:
            sma_long = st.number_input("SMA long period", min_value=3, max_value=400, value=50, step=1)
        with bt_col3:
            initial_capital = st.number_input("Initial capital ($)", min_value=100.0, value=10000.0, step=100.0)
            # persist chosen capital for trading_ui
            st.session_state['initial_cash'] = float(initial_capital)

        if st.button("Run Backtest"):
            if sma_long <= sma_short:
                st.error("SMA long must be greater than SMA short")
            else:
                with st.spinner("Running backtest..."):
                    bt = backtest_sma_crossover(close, short=int(sma_short), long=int(sma_long), initial_capital=float(initial_capital))
                st.success("Backtest finished")
                st.metric("Final Value", f"${bt['final_value']:,.2f}", f"{bt['total_return']:.2f}%")
                st.metric("Sharpe", f"{bt['sharpe_ratio']:.4f}")
                st.metric("Max Drawdown", f"{bt['max_drawdown']:.2f}%")

                st.plotly_chart(plot_portfolio(bt["pv"], symbol_session), use_container_width=True)
                st.plotly_chart(plot_drawdown(bt["pv"]), use_container_width=True)

                if bt["trades"]:
                    trades_df = pd.DataFrame(bt["trades"])
                    trades_df["date"] = pd.to_datetime(trades_df["date"]).dt.strftime("%Y-%m-%d")
                    st.dataframe(trades_df)
                else:
                    st.info("No trades were generated by this SMA crossover on the given period.")
    else:
        st.info("Configure the parameters on the left and click 'Load Data & Analyze'")

if __name__ == "__main__":
    main()