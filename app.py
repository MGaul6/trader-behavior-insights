import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Trader vs Sentiment", layout="wide", page_icon="📊")

DATA_DIR = Path("data")


@st.cache_data
def load_data():
    df_sentiment = pd.read_csv(DATA_DIR / "sentiment_data.csv")
    df_trades = pd.read_csv(DATA_DIR / "trader_data.csv")

    df_sentiment["date"] = pd.to_datetime(df_sentiment["date"])
    df_sentiment = df_sentiment.sort_values("date").reset_index(drop=True)

    def _map(cls):
        if cls in ("Fear", "Extreme Fear"):
            return "Fear"
        elif cls in ("Greed", "Extreme Greed"):
            return "Greed"
        return "Neutral"

    df_sentiment["sentiment_binary"] = df_sentiment["classification"].map(_map)

    df_trades.columns = df_trades.columns.str.strip().str.lower().str.replace(" ", "_")
    df_trades["datetime"] = pd.to_datetime(df_trades["timestamp"], unit="ms")
    df_trades["date"] = df_trades["datetime"].dt.normalize()

    df = df_trades.merge(
        df_sentiment[["date", "value", "classification", "sentiment_binary"]],
        on="date",
        how="inner",
    )
    return df_sentiment, df_trades, df


@st.cache_data
def build_daily_metrics(_df):
    df = _df.copy()

    def compute(group):
        pnl = group["closed_pnl"]
        size = group["size_usd"]
        direction = group["direction"]
        long_c = direction.str.contains("Long", case=False, na=False).sum()
        short_c = direction.str.contains("Short", case=False, na=False).sum()
        cum = pnl.cumsum()
        dd = (cum - cum.cummax()).min()
        closing = pnl[pnl != 0]
        return pd.Series({
            "daily_pnl": pnl.sum(),
            "win_rate": (closing > 0).mean() if len(closing) > 0 else np.nan,
            "avg_trade_size": size.mean(),
            "trade_count": len(group),
            "long_count": long_c,
            "short_count": short_c,
            "long_short_ratio": long_c / short_c if short_c > 0 else np.nan,
            "max_drawdown": dd,
            "total_volume": size.sum(),
        })

    daily = df.groupby(["account", "date"]).apply(compute, include_groups=False).reset_index()
    sent_cols = df.groupby("date")[["value", "classification", "sentiment_binary"]].first().reset_index()
    daily = daily.merge(sent_cols, on="date", how="left")
    return daily


df_sentiment, df_trades, df_merged = load_data()
daily = build_daily_metrics(df_merged)

# --- Sidebar ---
st.sidebar.title("Filters")

date_min = daily["date"].min().date()
date_max = daily["date"].max().date()
date_range = st.sidebar.date_input("Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)

if len(date_range) == 2:
    mask = (daily["date"].dt.date >= date_range[0]) & (daily["date"].dt.date <= date_range[1])
    daily_f = daily[mask]
else:
    daily_f = daily

sent_options = ["All"] + sorted(daily_f["sentiment_binary"].dropna().unique().tolist())
sent_filter = st.sidebar.selectbox("Sentiment regime", sent_options)
if sent_filter != "All":
    daily_f = daily_f[daily_f["sentiment_binary"] == sent_filter]

accounts = sorted(daily_f["account"].unique())
short_accounts = {a: a[:6] + "..." + a[-4:] for a in accounts}
acct_labels = ["All"] + [short_accounts[a] for a in accounts]
acct_choice = st.sidebar.selectbox("Trader account", acct_labels)
if acct_choice != "All":
    full_acct = [a for a, s in short_accounts.items() if s == acct_choice][0]
    daily_f = daily_f[daily_f["account"] == full_acct]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing {len(daily_f):,} trader-day observations")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "PnL vs Sentiment", "Segments", "Model Insights"])

# ===== TAB 1: OVERVIEW =====
with tab1:
    st.header("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trader-Days", f"{len(daily_f):,}")
    c2.metric("Unique Traders", f"{daily_f['account'].nunique()}")
    c3.metric("Avg Daily PnL", f"${daily_f['daily_pnl'].mean():,.0f}")
    c4.metric("Avg Win Rate", f"{daily_f['win_rate'].mean():.1%}")

    col1, col2 = st.columns(2)
    with col1:
        agg = daily_f.groupby("date").agg(total_pnl=("daily_pnl", "sum")).reset_index()
        agg["rolling_7d"] = agg["total_pnl"].rolling(7, min_periods=1).mean()
        fig = px.area(agg, x="date", y="rolling_7d", title="7-Day Rolling Aggregate PnL",
                      labels={"rolling_7d": "Rolling PnL (USD)", "date": "Date"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sent_dist = daily_f["sentiment_binary"].value_counts().reset_index()
        sent_dist.columns = ["Sentiment", "Count"]
        fig2 = px.pie(sent_dist, values="Count", names="Sentiment", title="Sentiment Distribution",
                      color="Sentiment", color_discrete_map={"Fear": "#ef5350", "Neutral": "#fdd835", "Greed": "#66bb6a"})
        st.plotly_chart(fig2, use_container_width=True)

# ===== TAB 2: PNL VS SENTIMENT =====
with tab2:
    st.header("Performance by Sentiment Regime")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(daily_f, x="sentiment_binary", y="daily_pnl",
                     category_orders={"sentiment_binary": ["Fear", "Neutral", "Greed"]},
                     title="Daily PnL Distribution",
                     labels={"sentiment_binary": "Sentiment", "daily_pnl": "Daily PnL (USD)"})
        fig.update_traces(boxpoints=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(daily_f, x="sentiment_binary", y="win_rate",
                     category_orders={"sentiment_binary": ["Fear", "Neutral", "Greed"]},
                     title="Win Rate Distribution",
                     labels={"sentiment_binary": "Sentiment", "win_rate": "Win Rate"})
        fig.update_traces(boxpoints=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Behavior Shifts")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(daily_f, x="sentiment_binary", y="trade_count",
                     category_orders={"sentiment_binary": ["Fear", "Neutral", "Greed"]},
                     title="Trade Count per Day",
                     labels={"sentiment_binary": "Sentiment", "trade_count": "Trades"})
        fig.update_traces(boxpoints=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ls_data = daily_f.dropna(subset=["long_short_ratio"])
        ls_data = ls_data[ls_data["long_short_ratio"] < ls_data["long_short_ratio"].quantile(0.95)]
        fig = px.box(ls_data, x="sentiment_binary", y="long_short_ratio",
                     category_orders={"sentiment_binary": ["Fear", "Neutral", "Greed"]},
                     title="Long/Short Ratio",
                     labels={"sentiment_binary": "Sentiment", "long_short_ratio": "L/S Ratio"})
        fig.update_traces(boxpoints=False)
        fig.add_hline(y=1, line_dash="dash", line_color="red", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mean Metrics by Sentiment (5-class)")
    class_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    perf_5 = daily_f.groupby("classification").agg(
        mean_pnl=("daily_pnl", "mean"), mean_wr=("win_rate", "mean"), obs=("daily_pnl", "count")
    ).reindex(class_order).reset_index()
    fig = px.bar(perf_5, x="classification", y="mean_pnl", title="Mean Daily PnL by 5-Class Sentiment",
                 labels={"classification": "Sentiment", "mean_pnl": "Mean PnL (USD)"},
                 color="classification",
                 color_discrete_map={"Extreme Fear": "#d32f2f", "Fear": "#ff7043", "Neutral": "#fdd835",
                                     "Greed": "#66bb6a", "Extreme Greed": "#2e7d32"})
    st.plotly_chart(fig, use_container_width=True)

# ===== TAB 3: SEGMENTS =====
with tab3:
    st.header("Trader Segmentation")

    tp = daily_f.groupby("account").agg(
        mean_pnl=("daily_pnl", "mean"), pnl_std=("daily_pnl", "std"),
        mean_wr=("win_rate", "mean"), mean_tc=("trade_count", "mean"),
        mean_size=("avg_trade_size", "mean"),
    ).reset_index()
    tp["pnl_cv"] = (tp["pnl_std"] / tp["mean_pnl"].abs()).replace([np.inf, -np.inf], np.nan)

    size_med = tp["mean_size"].median()
    tp["size_seg"] = np.where(tp["mean_size"] >= size_med, "High Exposure", "Low Exposure")
    freq_med = tp["mean_tc"].median()
    tp["freq_seg"] = np.where(tp["mean_tc"] >= freq_med, "Frequent", "Infrequent")
    cv_med = tp["pnl_cv"].median()
    tp["cons_seg"] = np.where(
        (tp["mean_pnl"] > 0) & (tp["pnl_cv"] <= cv_med), "Consistent Winner",
        np.where(tp["mean_pnl"] > 0, "Inconsistent Winner", "Net Loser"))

    daily_seg = daily_f.merge(tp[["account", "size_seg", "freq_seg", "cons_seg"]], on="account", how="left")

    seg_choice = st.radio("Segment type", ["Exposure", "Frequency", "Consistency"], horizontal=True)
    seg_col = {"Exposure": "size_seg", "Frequency": "freq_seg", "Consistency": "cons_seg"}[seg_choice]

    col1, col2 = st.columns(2)
    with col1:
        seg_data = daily_seg.groupby([seg_col, "sentiment_binary"]).agg(
            mean_pnl=("daily_pnl", "mean")).reset_index()
        fig = px.bar(seg_data, x=seg_col, y="mean_pnl", color="sentiment_binary",
                     barmode="group", title=f"Mean PnL by {seg_choice} Segment & Sentiment",
                     labels={seg_col: "Segment", "mean_pnl": "Mean PnL (USD)", "sentiment_binary": "Sentiment"},
                     color_discrete_map={"Fear": "#ef5350", "Neutral": "#fdd835", "Greed": "#66bb6a"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_wr = daily_seg.groupby([seg_col, "sentiment_binary"]).agg(
            mean_wr=("win_rate", "mean")).reset_index()
        fig = px.bar(seg_wr, x=seg_col, y="mean_wr", color="sentiment_binary",
                     barmode="group", title=f"Win Rate by {seg_choice} Segment & Sentiment",
                     labels={seg_col: "Segment", "mean_wr": "Mean Win Rate", "sentiment_binary": "Sentiment"},
                     color_discrete_map={"Fear": "#ef5350", "Neutral": "#fdd835", "Greed": "#66bb6a"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Trader Profiles")
    display_tp = tp.copy()
    display_tp["account"] = display_tp["account"].apply(lambda x: x[:6] + "..." + x[-4:])
    st.dataframe(display_tp.round(2), use_container_width=True, hide_index=True)

# ===== TAB 4: MODEL INSIGHTS =====
with tab4:
    st.header("Predictive Model & Clustering")

    st.subheader("Feature Importance (Random Forest)")
    st.image("output/charts/feature_importance.png", use_container_width=True)

    st.subheader("Trader Behavioral Archetypes (K-Means + PCA)")
    st.image("output/charts/trader_clusters_pca.png", use_container_width=True)

    st.subheader("Strategy Recommendations")
    st.markdown("""
**Strategy 1: Reduce Exposure on Fear Days for High-Exposure Traders**

High-exposure traders experience significantly worse drawdowns during Fear periods
(mean drawdown ~$-43,776) compared to Greed periods (~$-19,950). Low-exposure traders
remain insulated. **When the F&G Index < 40, high-exposure traders should reduce
position sizes by 30-50%.**

**Strategy 2: Consistent Winners Should Lean Into Extreme Fear**

Consistent winners maintain strong win rates (~90%) during Fear and see elevated mean
PnL (~$134,570 on Fear vs ~$105,588 on Greed). **Proven profitable traders should
maintain or increase activity during Extreme Fear, while inconsistent traders should
reduce exposure.**
""")
