"""
Markov Chain Market Regime Modelling
=====================================
Author : Channabasava H
Article: https://medium.com/insiderfinance/modelling-stock-market-regimes-with-markov-chains-a-practical-data-driven-study-712d98300ccf

A data-driven system that classifies daily stock market behaviour into
regimes (Uptrend, Downtrend, Neutral, Reversal, Volatile, Calm), builds
a Markov transition probability matrix, backtests next-day predictions,
and generates trading signals and multi-day forecasts.

Usage
-----
    python markov_market_regimes.py

    To change the stock or lookback period, edit the CONFIG section below.
"""

# ─────────────────────────────────────────────
#  DEPENDENCIES
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  CONFIG  ← edit here to change stock / period
# ─────────────────────────────────────────────
TICKER          = "SUNPHARMA.NS"   # Any Yahoo Finance ticker
PERIOD          = "24mo"           # yfinance period string
BACKTEST_FROM   = "2025-01-01"     # Start date for backtesting window
ATR_PERIOD      = 14               # EWM span for ATR smoothing
MA_PERIOD       = 20               # Rolling window for moving average
N_FORECAST_DAYS = 5                # Days ahead for multi-step forecast


# ─────────────────────────────────────────────
#  COLOUR MAP  (used across all Plotly charts)
# ─────────────────────────────────────────────
COLOR_MAP = {
    "Uptrend"  : "#00b050",
    "Downtrend": "#c00000",
    "Neutral"  : "#808080",
    "Reversal" : "#ff9900",
    "Volatile" : "#7030a0",
    "Calm"     : "#00bcd4",
}


# ═══════════════════════════════════════════════════════
#  STEP 1 — DATA COLLECTION
# ═══════════════════════════════════════════════════════

def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance, clean it, and return a
    flat DataFrame with consistent column names.

    Parameters
    ----------
    ticker : str  — Yahoo Finance ticker symbol (e.g. "SUNPHARMA.NS")
    period : str  — Lookback period accepted by yfinance (e.g. "24mo")

    Returns
    -------
    pd.DataFrame  with columns: Date, Close, High, Low, Open, Volume
    """
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    raw.reset_index(inplace=True)

    # Flatten any multi-level column headers produced by yfinance
    raw.columns = raw.columns.map("".join)
    raw.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

    # Sort chronologically and drop zero-volume rows (non-trading days)
    raw.sort_values("Date", inplace=True)
    raw = raw.loc[raw["Volume"] > 0].reset_index(drop=True)

    print(f"[Data] Downloaded {len(raw)} trading days for {ticker}.")
    return raw


# ═══════════════════════════════════════════════════════
#  STEP 2 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators used for regime classification:

    - Pct_Change : daily percentage return
    - MA20       : 20-day simple moving average of Close
    - Vol_Change : daily percentage change in volume
    - TR         : True Range = max(H-L, |H-PrevClose|, |L-PrevClose|)
    - ATR        : Exponentially smoothed TR (14-period EWM) — more
                   stable than a simple rolling ATR during fast moves
    - ATR_Pct    : ATR expressed as % of Close (normalised volatility)
    - Vol_MA20   : 20-day rolling mean of Volume
    - Vol_Ratio  : Volume / Vol_MA20  (>1 = unusually high volume)

    Parameters
    ----------
    df : pd.DataFrame — raw OHLCV frame from fetch_data()

    Returns
    -------
    pd.DataFrame with additional feature columns (modifies a copy)
    """
    df = df.copy()

    # Daily return
    df["Pct_Change"] = round(df["Close"].pct_change() * 100, 2)

    # 20-day moving average (momentum anchor)
    df[f"MA{MA_PERIOD}"] = df["Close"].rolling(MA_PERIOD).mean()

    # Volume change
    df["Vol_Change"] = round(df["Volume"].pct_change() * 100, 2)

    # True Range components
    df["H-L"]  = df["High"] - df["Low"]
    df["H-PC"] = (df["High"] - df["Close"].shift(1)).abs()
    df["L-PC"] = (df["Low"]  - df["Close"].shift(1)).abs()
    df["TR"]   = df[["H-L", "H-PC", "L-PC"]].max(axis=1)

    # EWM-smoothed ATR — more responsive than a simple rolling mean
    df["ATR"]     = df["TR"].ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
    df["ATR_Pct"] = round((df["ATR"] / df["Close"]) * 100, 2)

    # Volume relative to its own 20-day average
    df["Vol_MA20"]  = df["Volume"].rolling(MA_PERIOD).mean()
    df["Vol_Ratio"] = df["Volume"] / df["Vol_MA20"]

    return df


# ═══════════════════════════════════════════════════════
#  STEP 3 — REGIME CLASSIFICATION
# ═══════════════════════════════════════════════════════

def market_state(row: pd.Series) -> str:
    """
    Classify a single trading day into one of six market regimes using
    a priority-ordered rule set.

    Rule priority (highest → lowest):
    1. Volatile  — ATR% > 2.2  (very wide daily range, overrides all)
    2. Uptrend   — positive day AND price above MA20
    3. Downtrend — negative day AND price below MA20
    4. Calm      — very low volatility AND near-flat movement
    5. Reversal  — strong move that contradicts the current MA position
    6. Neutral   — everything else (sideways consolidation)

    Parameters
    ----------
    row : pd.Series — one row from the feature-engineered DataFrame

    Returns
    -------
    str — regime label
    """
    pct   = row["Pct_Change"]
    atr_p = row["ATR_Pct"]
    close = row["Close"]
    ma20  = row[f"MA{MA_PERIOD}"]

    # 1. Volatile: extreme intraday range
    if atr_p > 2.2:
        return "Volatile"

    # 2. Uptrend: positive momentum + price above trend
    if pct > 0.4 and close > ma20:
        return "Uptrend"

    # 3. Downtrend: negative momentum + price below trend
    if pct < -0.4 and close < ma20:
        return "Downtrend"

    # 4. Calm: very low volatility and minimal movement
    if atr_p < 1.0 and abs(pct) < 0.4:
        return "Calm"

    # 5. Reversal: large move that contradicts the MA trend
    if abs(pct) > 0.8:
        if pct < 0 and close > ma20:   # strong drop but still above MA — likely reversal
            return "Reversal"
        if pct > 0 and close < ma20:   # strong rally but still below MA — likely reversal
            return "Reversal"

    # 6. Default: sideways / no clear signal
    return "Neutral"


def classify_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply market_state() row-wise to produce the 'State' column.
    Uses .loc[] to avoid SettingWithCopyWarning.
    """
    df = df.copy()
    df.loc[:, "State"] = df.apply(market_state, axis=1)
    return df


# ═══════════════════════════════════════════════════════
#  STEP 4 — MARKOV TRANSITION MATRIX
# ═══════════════════════════════════════════════════════

def build_transition_matrix(states: list) -> pd.DataFrame:
    """
    Count all consecutive state-to-state transitions and normalise each
    row to produce a stochastic (row-sums-to-1) transition probability
    matrix.

    Parameters
    ----------
    states : list of str — ordered sequence of regime labels

    Returns
    -------
    pd.DataFrame — rows = current state, columns = next state
    """
    unique = sorted(set(states))
    matrix = pd.DataFrame(0, index=unique, columns=unique)

    for s1, s2 in zip(states[:-1], states[1:]):
        matrix.loc[s1, s2] += 1

    # Normalise each row so probabilities sum to 1
    matrix = matrix.div(matrix.sum(axis=1), axis=0)

    return matrix


# ═══════════════════════════════════════════════════════
#  STEP 5 — BACKTESTING
# ═══════════════════════════════════════════════════════

def backtest_markov(df_states: pd.DataFrame,
                    transition_matrix: pd.DataFrame) -> tuple:
    """
    Walk forward through df_states day-by-day. For each day, predict
    tomorrow's regime as the argmax of the current state's row in the
    transition matrix, then compare against the actual next state.

    Also computes:
    - Top-1 / Top-2 predicted states + their probabilities
    - Worst-case probability = P(Reversal) + P(Volatile)

    Parameters
    ----------
    df_states        : pd.DataFrame — must have columns [Date, State]
    transition_matrix: pd.DataFrame — stochastic matrix from build_transition_matrix()

    Returns
    -------
    results_df : pd.DataFrame — per-day prediction log
    accuracy   : float — fraction of correct next-day predictions
    """
    df_states = df_states.sort_values("Date").reset_index(drop=True)

    results = []
    correct = 0

    for i in range(len(df_states) - 1):
        today_state       = df_states.loc[i,   "State"]
        actual_next_state = df_states.loc[i+1, "State"]

        # Skip if today's state isn't in the matrix (e.g. very rare state)
        if today_state not in transition_matrix.index:
            continue

        probs        = transition_matrix.loc[today_state]
        sorted_probs = probs.sort_values(ascending=False)

        predicted = sorted_probs.index[0]   # argmax

        top1_state, top1_prob = sorted_probs.index[0], sorted_probs.iloc[0]
        top2_state, top2_prob = sorted_probs.index[1], sorted_probs.iloc[1]

        # Combined probability of an adverse regime
        worst_prob = sum(
            probs.get(s, 0) for s in ["Reversal", "Volatile"]
        )

        is_correct = (predicted == actual_next_state)
        correct   += int(is_correct)

        results.append({
            "Date_t"                   : df_states.loc[i, "Date"],
            "State_t"                  : today_state,
            "Predicted_State"          : predicted,
            "Actual_State"             : actual_next_state,
            "Correct"                  : is_correct,
            "Top1_State"               : top1_state,
            "Top1_Prob"                : round(top1_prob, 4),
            "Top2_State"               : top2_state,
            "Top2_Prob"                : round(top2_prob, 4),
            "Worst_Prob(Reversal+Volatile)": round(worst_prob, 4),
        })

    results_df = pd.DataFrame(results)
    accuracy   = correct / len(results_df) if len(results_df) > 0 else 0.0

    print(f"[Backtest] Accuracy: {round(accuracy * 100, 2)}%  "
          f"({correct}/{len(results_df)} correct predictions)")
    return results_df, accuracy


# ═══════════════════════════════════════════════════════
#  STEP 6 — TRADING SIGNAL GENERATION
# ═══════════════════════════════════════════════════════

def classify_signal(row: pd.Series) -> str:
    """
    Map the next-day probability distribution to a simple trading signal.

    Signal logic
    ------------
    BUY         — Uptrend probability dominates clearly (>40% and
                  greater than the sum of bearish probabilities)
    SELL        — High combined risk (Reversal+Volatile > 25%) or
                  Downtrend probability > 35%
    BUY-the-DIP — Elevated downtrend probability but low risk of a
                  sharp adverse move; indicates a possible dip-buying
                  opportunity
    HOLD        — Neutral dominates with limited upside or downside
                  probability; no clear edge either way

    Parameters
    ----------
    row : pd.Series — one row from results_df (must contain *_Prob columns)

    Returns
    -------
    str — one of: BUY, SELL, BUY-the-DIP, HOLD
    """
    up   = row.get("Uptrend_Prob",   0)
    down = row.get("Downtrend_Prob", 0)
    rev  = row.get("Reversal_Prob",  0)
    vol  = row.get("Volatile_Prob",  0)
    neu  = row.get("Neutral_Prob",   0)

    if up > 0.40 and up > (down + rev + vol):
        return "BUY"
    if (rev + vol) > 0.25 or down > 0.35:
        return "SELL"
    if down > 0.30 and vol < 0.10 and rev < 0.10:
        return "BUY-the-DIP"
    if neu > 0.50 and up < 0.30 and (rev + vol) < 0.15:
        return "HOLD"

    return "HOLD"


def add_signals(results_df: pd.DataFrame,
                transition_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Append per-state probability columns and a Signal column to results_df.
    """
    results_df = results_df.copy()

    for state in transition_matrix.columns:
        results_df[f"{state}_Prob"] = results_df["State_t"].apply(
            lambda s: transition_matrix.loc[s, state] if s in transition_matrix.index else 0
        )

    results_df["Signal"] = results_df.apply(classify_signal, axis=1)
    return results_df


# ═══════════════════════════════════════════════════════
#  STEP 7 — N-STEP FORECAST
# ═══════════════════════════════════════════════════════

def n_step_markov(matrix: pd.DataFrame,
                  current_state: str,
                  n_steps: int = 5) -> pd.DataFrame:
    """
    Project the probability distribution forward n_steps days by
    repeatedly left-multiplying the state vector by the transition matrix.

    Parameters
    ----------
    matrix        : pd.DataFrame — stochastic transition matrix
    current_state : str          — today's regime label
    n_steps       : int          — number of days to forecast ahead

    Returns
    -------
    pd.DataFrame — shape (n_steps, n_states) with probability for each
                   state on each future day
    """
    mat = matrix.values.copy()
    idx = list(matrix.index).index(current_state)

    state_vec = np.zeros(len(matrix))
    state_vec[idx] = 1.0       # one-hot encode today's state

    history = []
    for _ in range(n_steps):
        state_vec = state_vec.dot(mat)
        history.append(state_vec.copy())

    return pd.DataFrame(history, columns=matrix.columns,
                        index=[f"Day +{i+1}" for i in range(n_steps)])


# ═══════════════════════════════════════════════════════
#  STEP 8 — VISUALISATIONS
# ═══════════════════════════════════════════════════════

def plot_regime_timeline(df: pd.DataFrame) -> None:
    """Scatter plot of market regime labels over the full date range."""
    fig = px.scatter(
        df, x="Date", y="State", color="State",
        color_discrete_map=COLOR_MAP,
        title=f"Market Regime Timeline — {TICKER}",
        height=350,
    )
    fig.update_traces(marker=dict(size=10))
    fig.show()


def plot_transition_matrix(matrix: pd.DataFrame) -> None:
    """Heatmap of the Markov transition probability matrix."""
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale="Blues",
            text=matrix.round(2).astype(str),
            texttemplate="%{text}",
            showscale=True,
        )
    )
    fig.update_layout(
        title=dict(text="Markov Transition Probability Matrix",
                   x=0.5, xanchor="center"),
        xaxis_title="Next State",
        yaxis_title="Current State",
        height=500,
    )
    fig.show()


def plot_backtest_dashboard(results_df: pd.DataFrame,
                            accuracy: float) -> None:
    """
    Three-panel dashboard:
    1. Risk gauge — combined Reversal + Volatile probability
    2. Top-1 / Top-2 prediction probabilities over time
    3. Predicted vs actual state timeline
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "Risk Indicator: P(Reversal) + P(Volatile)",
            "Top-1 and Top-2 Prediction Probabilities",
            "Predicted vs Actual State Timeline",
        ),
    )

    # Panel 1 — Risk gauge
    fig.add_trace(
        go.Scatter(
            x=results_df["Date_t"],
            y=results_df["Worst_Prob(Reversal+Volatile)"],
            mode="lines+markers",
            marker=dict(size=8, color="#ff3300"),
            name="Worst Probability",
        ),
        row=1, col=1,
    )

    # Panel 2 — Confidence
    for prob_col, label, colour in [
        ("Top1_Prob", "Top-1 Probability", "#1f77b4"),
        ("Top2_Prob", "Top-2 Probability", "#ff7f0e"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=results_df["Date_t"],
                y=results_df[prob_col],
                mode="lines+markers",
                marker=dict(size=7, color=colour),
                name=label,
            ),
            row=2, col=1,
        )

    # Panel 3 — Predicted vs actual
    fig.add_trace(
        go.Scatter(
            x=results_df["Date_t"],
            y=results_df["Actual_State"],
            mode="markers",
            marker=dict(size=11,
                        color=[COLOR_MAP.get(s, "#888") for s in results_df["Actual_State"]]),
            name="Actual State",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=results_df["Date_t"],
            y=results_df["Predicted_State"],
            mode="markers+lines",
            marker=dict(size=9, symbol="diamond",
                        color=[COLOR_MAP.get(s, "#888") for s in results_df["Predicted_State"]]),
            name="Predicted State",
        ),
        row=3, col=1,
    )

    fig.update_layout(
        height=900,
        title=f"Markov Model Backtest Dashboard — Accuracy: {round(accuracy * 100, 2)}%",
        showlegend=True,
    )
    fig.update_yaxes(title_text="Worst Prob", row=1, col=1, tickformat=".0%")
    fig.update_yaxes(title_text="Probability", row=2, col=1, tickformat=".0%")
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.show()


def plot_confusion_matrix(results_df: pd.DataFrame,
                          matrix: pd.DataFrame) -> None:
    """Confusion matrix of predicted vs actual states."""
    labels = matrix.index.tolist()
    cm = confusion_matrix(results_df["Actual_State"],
                          results_df["Predicted_State"],
                          labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {s}" for s in labels],
        columns=[f"Predicted: {s}" for s in labels],
    )
    fig = px.imshow(
        cm_df, text_auto=True, color_continuous_scale="Blues",
        title="Confusion Matrix — Predicted vs Actual States",
    )
    fig.update_layout(height=500)
    fig.show()


def plot_state_durations(df_states: pd.DataFrame) -> None:
    """Box plot showing how many consecutive days each regime typically lasts."""
    durations = []
    current, count = None, 0
    for s in df_states["State"]:
        if s == current:
            count += 1
        else:
            if current is not None:
                durations.append((current, count))
            current, count = s, 1
    if current:
        durations.append((current, count))

    dur_df = pd.DataFrame(durations, columns=["State", "Days"])
    fig = px.box(
        dur_df, x="State", y="Days",
        color="State", color_discrete_map=COLOR_MAP,
        title="State Duration Distribution — How long each regime lasts",
    )
    fig.update_layout(height=400)
    fig.show()


def plot_forecast(forecast_df: pd.DataFrame, current_state: str) -> None:
    """Line chart of n-step forward probability distribution."""
    fig = px.line(
        forecast_df,
        title=f"{N_FORECAST_DAYS}-Day Markov Forecast (starting from '{current_state}')",
        markers=True,
        color_discrete_map=COLOR_MAP,
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=450)
    fig.show()


def plot_confidence_vs_accuracy(results_df: pd.DataFrame) -> None:
    """Scatter: Top-1 prediction probability vs whether that prediction was correct."""
    df_plot = results_df.copy()
    df_plot["Correct_Num"] = df_plot["Correct"].astype(int)
    fig = px.scatter(
        df_plot,
        x="Top1_Prob",
        y="Correct_Num",
        color="Predicted_State",
        color_discrete_map=COLOR_MAP,
        title="Prediction Confidence vs Accuracy",
        labels={"Correct_Num": "Correct (1) / Incorrect (0)"},
    )
    fig.update_layout(height=400)
    fig.show()


# ═══════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  Markov Chain Market Regime Modelling")
    print(f"  Ticker : {TICKER}   Period : {PERIOD}")
    print("=" * 55)

    # 1. Fetch data
    df = fetch_data(TICKER, PERIOD)

    # 2. Engineer features
    df = engineer_features(df)

    # 3. Classify regimes
    df = classify_regimes(df)
    print(f"[Regimes] Distribution:\n{df['State'].value_counts().to_string()}\n")

    # 4. Build transition matrix
    states_list       = df["State"].dropna().tolist()
    transition_matrix = build_transition_matrix(states_list)
    print("[Transition Matrix]\n", transition_matrix.round(3), "\n")

    # 5. Visualise full timeline + transition matrix
    plot_regime_timeline(df)
    plot_transition_matrix(transition_matrix)

    # 6. Backtest on the specified window
    df_states           = df.loc[df["Date"] >= BACKTEST_FROM, ["Date", "State"]].drop_duplicates()
    results_df, accuracy = backtest_markov(df_states, transition_matrix)

    # 7. Add trading signals
    results_df = add_signals(results_df, transition_matrix)
    print("[Signals] Sample output (last 10 rows):")
    print(results_df[["Date_t", "State_t", "Predicted_State",
                       "Actual_State", "Correct", "Signal"]].tail(10).to_string(index=False))
    print()

    # 8. Visualise backtest results
    plot_backtest_dashboard(results_df, accuracy)
    plot_confusion_matrix(results_df, transition_matrix)
    plot_state_durations(df_states)
    plot_confidence_vs_accuracy(results_df)

    # 9. N-step forward forecast
    current_state = df_states.iloc[-1]["State"]
    forecast_df   = n_step_markov(transition_matrix, current_state, N_FORECAST_DAYS)
    print(f"[Forecast] {N_FORECAST_DAYS}-day probability forecast from '{current_state}':")
    print(forecast_df.round(3).to_string(), "\n")
    plot_forecast(forecast_df, current_state)

    print("Done.")


if __name__ == "__main__":
    main()
