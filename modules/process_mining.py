"""
Process mining module for AI BI Copilot.

Loads the ``process_events`` table, converts it to a pm4py event log, and
exposes helpers for cycle time analysis and bottleneck detection.

Usage::

    from modules.process_mining import get_process_summary, detect_bottlenecks
"""

import sys
from pathlib import Path

import pandas as pd

# Support running this file directly as a script as well as package import
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.db import run_query

# SLA threshold: 2 days in minutes
SLA_MINUTES = 2880


def load_event_log() -> tuple:
    """Load ``process_events`` from the database.

    Returns a tuple of ``(raw_df, pm4py_event_log)``.  The pm4py event log is
    only returned when pm4py is available; otherwise ``None`` is returned.

    Returns
    -------
    tuple
        ``(pd.DataFrame, pm4py.objects.log.obj.EventLog | None)``
    """
    df = run_query("SELECT * FROM process_events ORDER BY case_id, timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    try:
        import pm4py

        event_log = pm4py.format_dataframe(
            df.copy(),
            case_id="case_id",
            activity_key="activity",
            timestamp_key="timestamp",
        )
        event_log = pm4py.convert_to_event_log(event_log)
    except ImportError:
        event_log = None

    return df, event_log


def calculate_cycle_times(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a DataFrame with ``case_id`` and ``total_cycle_time_minutes``.

    Parameters
    ----------
    df:
        Optional pre-loaded events DataFrame.  If ``None`` the data is loaded
        from the database.
    """
    if df is None:
        df, _ = load_event_log()

    cycle = (
        df.groupby("case_id")["timestamp"]
        .agg(["min", "max"])
        .reset_index()
    )
    cycle.columns = ["case_id", "start_time", "end_time"]
    cycle["total_cycle_time_minutes"] = (
        (cycle["end_time"] - cycle["start_time"]).dt.total_seconds() / 60
    ).round(2)
    return cycle[["case_id", "total_cycle_time_minutes"]]


def detect_bottlenecks(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return activities ranked by average duration, flagging outliers.

    An activity is flagged as a bottleneck when its average duration exceeds
    ``mean + 1.5 * std`` across all activities.

    Parameters
    ----------
    df:
        Optional pre-loaded events DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: ``activity``, ``avg_duration_minutes``, ``max_duration_minutes``,
        ``event_count``, ``is_bottleneck``.
    """
    if df is None:
        df, _ = load_event_log()

    agg = (
        df.groupby("activity")["duration_minutes"]
        .agg(avg_duration_minutes="mean", max_duration_minutes="max", event_count="count")
        .reset_index()
    )
    agg["avg_duration_minutes"] = agg["avg_duration_minutes"].round(2)
    agg["max_duration_minutes"] = agg["max_duration_minutes"].round(2)

    mean = agg["avg_duration_minutes"].mean()
    std = agg["avg_duration_minutes"].std()
    threshold = mean + 1.5 * std
    agg["is_bottleneck"] = agg["avg_duration_minutes"] > threshold

    return agg.sort_values("avg_duration_minutes", ascending=False).reset_index(drop=True)


def get_process_summary(df: pd.DataFrame | None = None) -> dict:
    """Return a summary dict of key process mining metrics.

    Keys
    ----
    avg_cycle_time_minutes : float
    median_cycle_time_minutes : float
    slowest_activity : str
    pct_cases_over_sla : float
        Percentage of cases whose total cycle time exceeds :data:`SLA_MINUTES`.
    """
    if df is None:
        df, _ = load_event_log()

    cycle_times = calculate_cycle_times(df)
    bottlenecks = detect_bottlenecks(df)

    avg_ct = cycle_times["total_cycle_time_minutes"].mean()
    median_ct = cycle_times["total_cycle_time_minutes"].median()
    slowest = bottlenecks.iloc[0]["activity"]
    pct_over_sla = (
        (cycle_times["total_cycle_time_minutes"] > SLA_MINUTES).sum()
        / len(cycle_times)
        * 100
    )

    return {
        "avg_cycle_time_minutes": round(avg_ct, 2),
        "median_cycle_time_minutes": round(median_ct, 2),
        "slowest_activity": slowest,
        "pct_cases_over_sla": round(pct_over_sla, 2),
    }


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading event log …")
    raw_df, _ = load_event_log()

    summary = get_process_summary(raw_df)
    print("\n📊 Process Summary")
    for k, v in summary.items():
        print(f"   {k}: {v}")

    bottlenecks = detect_bottlenecks(raw_df)
    report_path = output_dir / "bottleneck_report.csv"
    bottlenecks.to_csv(report_path, index=False)
    print(f"\n✅ Bottleneck report saved to {report_path}")
    print(bottlenecks.to_string(index=False))
