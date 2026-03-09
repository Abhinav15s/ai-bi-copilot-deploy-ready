"""
Streamlit dashboard for AI BI Copilot.

Run with::

    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# Allow imports from the project root when running from any working directory
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.db import run_query
from modules.process_mining import (
    SLA_MINUTES,
    calculate_cycle_times,
    detect_bottlenecks,
    load_event_log,
)
from modules.sentiment_engine import analyze_reviews_df, get_sentiment_summary

# ---------------------------------------------------------------------------
# Secrets: inject from st.secrets into os.environ so all modules pick them up
# Works on Streamlit Cloud (st.secrets) and locally (.env via dotenv).
# ---------------------------------------------------------------------------
load_dotenv()  # local dev convenience — no-op on Cloud

import os  # noqa: E402

for _key in ("GROQ_API_KEY", "GROQ_MODEL"):
    if _key not in os.environ:
        try:
            _val = st.secrets.get(_key, "")
            if _val:
                os.environ[_key] = _val
        except Exception:  # noqa: BLE001
            pass

# ---------------------------------------------------------------------------
# Database bootstrap: generate synthetic data if DB is missing (Cloud deploy)
# ---------------------------------------------------------------------------
_DB_PATH = Path(__file__).parent.parent / "data" / "business_data.db"

if not _DB_PATH.exists():
    with st.spinner("⏳ First run: generating synthetic business data (takes ~10 s) …"):
        try:
            import sys as _sys
            _data_dir = str(Path(__file__).parent.parent)
            if _data_dir not in _sys.path:
                _sys.path.insert(0, _data_dir)
            from data.generate_data import main as _gen_data
            _gen_data()
        except Exception as _exc:
            st.error(f"Failed to generate data: {_exc}")
            st.stop()

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI BI Copilot", layout="wide", page_icon="🤖")

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_transactions() -> pd.DataFrame:
    """Load the transactions table from the database."""
    return run_query("SELECT * FROM transactions")


@st.cache_data(ttl=300)
def load_process_data():
    """Load process events and compute derived metrics."""
    df, _ = load_event_log()
    cycle = calculate_cycle_times(df)
    bottlenecks = detect_bottlenecks(df)
    summary = {
        "avg_cycle_time_minutes": round(cycle["total_cycle_time_minutes"].mean(), 2),
        "median_cycle_time_minutes": round(cycle["total_cycle_time_minutes"].median(), 2),
        "slowest_activity": bottlenecks.iloc[0]["activity"] if len(bottlenecks) > 0 else "N/A",
        "pct_cases_over_sla": round(
            (cycle["total_cycle_time_minutes"] > SLA_MINUTES).sum() / len(cycle) * 100, 2
        ),
    }
    return df, cycle, bottlenecks, summary


@st.cache_data(ttl=300)
def load_reviews() -> pd.DataFrame:
    """Load and sentiment-enrich the customer reviews table."""
    raw = run_query("SELECT * FROM customer_reviews")
    return analyze_reviews_df(raw)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🤖 AI BI Copilot")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Business Overview",
        "⚙️ Process Mining",
        "💬 Sentiment Analysis",
        "🤖 Ask the Copilot",
    ],
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def kpi_card(label: str, value: str, col) -> None:
    """Render a single KPI metric card."""
    col.metric(label=label, value=value)


# ---------------------------------------------------------------------------
# Page 1 — Business Overview
# ---------------------------------------------------------------------------
if page == "📊 Business Overview":
    st.title("📊 Business Overview")

    try:
        txn = load_transactions()
    except Exception as exc:
        st.error(f"Could not load data: {exc}.  Run `python data/generate_data.py` first.")
        st.stop()

    # KPI cards
    total_revenue = txn["revenue"].sum()
    total_transactions = len(txn)
    avg_order_value = txn["revenue"].mean()
    top_region = txn.groupby("region")["revenue"].sum().idxmax()

    c1, c2, c3, c4 = st.columns(4)
    kpi_card("💰 Total Revenue", f"${total_revenue:,.0f}", c1)
    kpi_card("🧾 Total Transactions", f"{total_transactions:,}", c2)
    kpi_card("📦 Avg Order Value", f"${avg_order_value:,.0f}", c3)
    kpi_card("🌍 Top Region", top_region, c4)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Revenue by Region")
        rev_region = txn.groupby("region")["revenue"].sum().reset_index()
        fig = px.bar(
            rev_region,
            x="region",
            y="revenue",
            color="region",
            labels={"revenue": "Total Revenue ($)", "region": "Region"},
        )
        st.plotly_chart(fig, width="stretch")

    with col_right:
        st.subheader("Monthly Revenue Trend")
        txn["year_month"] = txn["date"].str[:7]
        monthly = txn.groupby("year_month")["revenue"].sum().reset_index()
        fig2 = px.line(
            monthly,
            x="year_month",
            y="revenue",
            markers=True,
            labels={"revenue": "Revenue ($)", "year_month": "Month"},
        )
        st.plotly_chart(fig2, width="stretch")


# ---------------------------------------------------------------------------
# Page 2 — Process Mining
# ---------------------------------------------------------------------------
elif page == "⚙️ Process Mining":
    st.title("⚙️ Process Mining")

    try:
        _, cycle, bottlenecks, summary = load_process_data()
    except Exception as exc:
        st.error(f"Could not load process data: {exc}.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    kpi_card("⏱️ Avg Cycle Time", f"{summary['avg_cycle_time_minutes']:,.0f} min", c1)
    kpi_card("📊 Median Cycle Time", f"{summary['median_cycle_time_minutes']:,.0f} min", c2)
    kpi_card("🚨 Cases Over SLA", f"{summary['pct_cases_over_sla']:.1f}%", c3)

    st.markdown("---")

    st.subheader("Average Duration by Activity")
    bottlenecks["color"] = bottlenecks["is_bottleneck"].map(
        {True: "Bottleneck", False: "Normal"}
    )
    fig = px.bar(
        bottlenecks,
        x="activity",
        y="avg_duration_minutes",
        color="color",
        color_discrete_map={"Bottleneck": "crimson", "Normal": "steelblue"},
        labels={"avg_duration_minutes": "Avg Duration (min)", "activity": "Activity"},
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Top 10 Slowest Cases")
    top_slow = cycle.sort_values("total_cycle_time_minutes", ascending=False).head(10)
    st.dataframe(top_slow.reset_index(drop=True), width="stretch")


# ---------------------------------------------------------------------------
# Page 3 — Sentiment Analysis
# ---------------------------------------------------------------------------
elif page == "💬 Sentiment Analysis":
    st.title("💬 Sentiment Analysis")

    try:
        reviews = load_reviews()
    except Exception as exc:
        st.error(f"Could not load reviews: {exc}.")
        st.stop()

    sent_summary = get_sentiment_summary(reviews)
    counts = sent_summary["counts"]
    pcts = sent_summary["percentages"]
    total = sum(counts.values())

    c1, c2, c3 = st.columns(3)
    kpi_card("😊 % Positive", f"{pcts.get('positive', 0):.1f}%", c1)
    kpi_card("😐 % Neutral", f"{pcts.get('neutral', 0):.1f}%", c2)
    kpi_card("😞 % Negative", f"{pcts.get('negative', 0):.1f}%", c3)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Sentiment Distribution")
        pie_df = pd.DataFrame(
            {"Sentiment": list(counts.keys()), "Count": list(counts.values())}
        )
        fig = px.pie(pie_df, names="Sentiment", values="Count", hole=0.35)
        st.plotly_chart(fig, width="stretch")

    with col_right:
        st.subheader("Avg Sentiment Score by Category")
        avg_score = (
            reviews.groupby("product_category")["sentiment_score"].mean().reset_index()
        )
        avg_score.columns = ["Category", "Avg Sentiment Score"]
        fig2 = px.bar(
            avg_score,
            x="Category",
            y="Avg Sentiment Score",
            color="Category",
        )
        st.plotly_chart(fig2, width="stretch")

    st.subheader("Sample Negative Reviews")
    neg = reviews[reviews["sentiment_label"] == "negative"][
        ["date", "product_category", "customer_segment", "review_text", "rating", "sentiment_score"]
    ].head(20)
    st.dataframe(neg.reset_index(drop=True), width="stretch")


# ---------------------------------------------------------------------------
# Page 4 — Ask the Copilot
# ---------------------------------------------------------------------------
elif page == "🤖 Ask the Copilot":
    st.title("🤖 Ask the Copilot")
    st.markdown(
        "Ask any business question in plain English and the AI will query the database for you."
    )

    example_questions = [
        "Which region had the highest revenue last quarter?",
        "What is the average cycle time for order fulfilment?",
        "Show me the top 3 product categories with the most negative customer reviews.",
        "What percentage of orders were delayed by more than 2 days?",
        "Who are the top 5 sales reps by total revenue?",
        "What is the revenue vs cost margin for each product category?",
    ]

    st.markdown("**💡 Example questions — click to copy:**")
    cols = st.columns(2)
    for i, q in enumerate(example_questions):
        cols[i % 2].code(q)

    st.markdown("---")
    question = st.text_input("Your question:", placeholder="e.g. Which region had the highest revenue?")

    if st.button("Ask 🚀") and question.strip():
        import os
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key or api_key == "your-groq-api-key-here":
            st.warning(
                "⚠️ Groq API key not found.  "
                "Copy `.env.example` to `.env` and add your key, then restart the app."
            )
        else:
            with st.spinner("Thinking …"):
                from modules.genai_query import ask_question as ai_ask
                answer = ai_ask(question)
            st.success(answer)
