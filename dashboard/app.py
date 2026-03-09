"""
Streamlit dashboard for AI BI Copilot.

Run with::

    streamlit run dashboard/app.py
"""

import json
import re
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
# AI Smart Charts helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _generate_chart_specs(schema_key: str, sample_json: str) -> list:
    """Ask Groq to return declarative chart specs for the given schema.

    Parameters
    ----------
    schema_key : str
        Serialised column→dtype mapping (used as cache key).
    sample_json : str
        JSON string of the first 5 rows (sent to LLM for context).

    Returns
    -------
    list of dict
        Chart specification dicts, or [] on error.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or api_key == "your-groq-api-key-here":
        return []
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0,
            api_key=api_key,
        )
        prompt = f"""You are a data visualisation expert. Given the dataset schema and sample below,
sugggest 4-6 meaningful charts that reveal useful patterns.

Column schema (name: dtype):
{schema_key}

Sample rows (JSON):
{sample_json}

Return ONLY a valid JSON array. Each element must be an object with:
- "chart_type": one of "bar", "line", "pie", "scatter", "histogram"
- "title": a short, descriptive chart title
- For "bar" or "line": "x" (column), "y" (column), "agg" ("sum"|"mean"|"count"), optionally "color" (column)
- For "pie": "names" (column), "values" (column), "agg" ("sum"|"count")
- For "scatter": "x" (column), "y" (column), optionally "color" (column)
- For "histogram": "x" (column)

Only use column names that exist exactly in the schema. Output valid JSON only — no markdown, no explanation."""
        response = llm.invoke(prompt)
        content = response.content.strip()
        # Strip markdown code fences if present
        match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        content = match.group(1).strip() if match else content
        return json.loads(content)
    except Exception:  # noqa: BLE001
        return []


def _render_chart(spec: dict, df: pd.DataFrame) -> None:
    """Render a single Plotly chart from a declarative *spec* dict."""
    chart_type = spec.get("chart_type", "bar")
    title = spec.get("title", "Chart")
    try:
        if chart_type in ("bar", "line"):
            x, y = spec["x"], spec["y"]
            agg = spec.get("agg", "sum")
            color = spec.get("color")
            if agg == "count":
                data = df.groupby(x).size().reset_index(name=y)
            elif agg == "mean":
                data = df.groupby(x)[y].mean().reset_index()
            else:
                data = df.groupby(x)[y].sum().reset_index()
            if chart_type == "bar":
                fig = px.bar(data, x=x, y=y, color=color, title=title)
            else:
                fig = px.line(data, x=x, y=y, title=title, markers=True)
        elif chart_type == "pie":
            names, values = spec["names"], spec["values"]
            agg = spec.get("agg", "sum")
            if agg == "count":
                data = df.groupby(names).size().reset_index(name=values)
            else:
                data = df.groupby(names)[values].sum().reset_index()
            fig = px.pie(data, names=names, values=values, title=title, hole=0.3)
        elif chart_type == "scatter":
            x, y = spec["x"], spec["y"]
            color = spec.get("color")
            fig = px.scatter(df, x=x, y=y, color=color, title=title)
        elif chart_type == "histogram":
            x = spec["x"]
            fig = px.histogram(df, x=x, title=title)
        else:
            return
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"⚠️ Could not render '{title}': {exc}")


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
        "✨ AI Smart Charts",
    ],
)

# ---------------------------------------------------------------------------
# Sidebar — CSV Upload
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("📂 Upload Your Data")

_REQUIRED_COLS = [
    "date", "region", "product_category", "customer_segment",
    "revenue", "cost", "units_sold", "sales_rep", "country",
]

_uploaded_file = st.sidebar.file_uploader(
    "Upload transactions CSV", type="csv", key="csv_uploader"
)

if _uploaded_file is not None:
    _df_upload = pd.read_csv(_uploaded_file)
    # Normalize: strip whitespace and lowercase so "Date " and "DATE" both match
    _df_upload.columns = _df_upload.columns.str.strip().str.lower()
    _missing = [c for c in _REQUIRED_COLS if c not in _df_upload.columns]
    if _missing:
        _found = ", ".join(_df_upload.columns.tolist()) or "(none)"
        st.sidebar.error(
            f"**Missing columns:** {', '.join(_missing)}\n\n"
            f"**Columns found in your file:** {_found}\n\n"
            "Download the template below to see the exact headers required."
        )
        st.stop()
    st.session_state["user_transactions"] = _df_upload
    st.sidebar.success(f"✅ Loaded {len(_df_upload):,} rows")

# Template download button
_template_csv = ",".join(_REQUIRED_COLS) + "\n"
st.sidebar.download_button(
    label="⬇️ Download CSV template",
    data=_template_csv,
    file_name="transactions_template.csv",
    mime="text/csv",
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

    if "user_transactions" in st.session_state:
        txn = st.session_state["user_transactions"]
    else:
        try:
            txn = load_transactions()
        except Exception as exc:
            st.error(f"Could not load data: {exc}.  Run `python data/generate_data.py` first.")
            st.stop()

    if txn.empty:
        st.warning("The uploaded CSV has no data rows. Please add data and re-upload.")
        st.stop()

    # KPI cards
    total_revenue = txn["revenue"].sum()
    total_transactions = len(txn)
    avg_order_value = txn["revenue"].mean() if not txn.empty else 0
    _region_revenue = txn.groupby("region")["revenue"].sum()
    top_region = _region_revenue.idxmax() if not _region_revenue.empty else "N/A"

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


# ---------------------------------------------------------------------------
# Page 5 — AI Smart Charts
# ---------------------------------------------------------------------------
elif page == "✨ AI Smart Charts":
    st.title("✨ AI Smart Charts")
    st.markdown(
        "Upload a CSV in the sidebar and the AI will analyse the dataset structure "
        "and automatically generate the most insightful visualisations for you."
    )

    if "user_transactions" not in st.session_state:
        st.info(
            "👈 Upload a CSV file in the sidebar to get started.  \n"
            "The AI will inspect the columns and sample rows, then decide which "
            "charts best reveal the patterns in your data."
        )
        st.stop()

    _df = st.session_state["user_transactions"]

    # Build schema string + sample for the LLM
    _schema_key = "\n".join(f"  {col}: {dtype}" for col, dtype in _df.dtypes.items())
    _sample_json = _df.head(5).to_json(orient="records", date_format="iso")

    # Check API key availability
    _api_key = os.environ.get("GROQ_API_KEY", "")
    if not _api_key or _api_key == "your-groq-api-key-here":
        st.warning(
            "⚠️ GROQ_API_KEY not set — AI chart generation is unavailable.  \n"
            "Add your key to Streamlit Secrets (Cloud) or `.env` (local)."
        )
        st.stop()

    # Regenerate button clears the cache for this schema
    if st.button("🔄 Regenerate charts"):
        _generate_chart_specs.clear()

    with st.spinner("🤖 Analysing your dataset and generating chart ideas …"):
        _specs = _generate_chart_specs(_schema_key, _sample_json)

    if not _specs:
        st.error(
            "The AI could not generate chart specifications.  \n"
            "Check your GROQ_API_KEY and try again, or verify the LLM logs."
        )
        st.stop()

    st.success(f"✅ Generated **{len(_specs)}** charts based on your dataset")
    st.markdown("---")

    # Render charts in a 2-column grid
    _cols = st.columns(2)
    for i, _spec in enumerate(_specs):
        with _cols[i % 2]:
            _render_chart(_spec, _df)
