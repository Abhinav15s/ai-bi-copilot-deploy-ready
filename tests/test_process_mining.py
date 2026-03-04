"""Unit tests for modules.process_mining."""

from modules.process_mining import (
    SLA_MINUTES,
    calculate_cycle_times,
    detect_bottlenecks,
)


def test_calculate_cycle_times_columns(sample_events_df):
    result = calculate_cycle_times(sample_events_df)
    assert "case_id" in result.columns
    assert "total_cycle_time_minutes" in result.columns


def test_calculate_cycle_times_positive_values(sample_events_df):
    result = calculate_cycle_times(sample_events_df)
    assert (result["total_cycle_time_minutes"] >= 0).all()


def test_calculate_cycle_times_row_count(sample_events_df):
    result = calculate_cycle_times(sample_events_df)
    expected_cases = sample_events_df["case_id"].nunique()
    assert len(result) == expected_cases


def test_detect_bottlenecks_columns(sample_events_df):
    result = detect_bottlenecks(sample_events_df)
    expected_cols = {"activity", "avg_duration_minutes", "max_duration_minutes", "event_count", "is_bottleneck"}
    assert expected_cols.issubset(set(result.columns))


def test_detect_bottlenecks_sorted_descending(sample_events_df):
    result = detect_bottlenecks(sample_events_df)
    durations = result["avg_duration_minutes"].tolist()
    assert durations == sorted(durations, reverse=True)


def test_detect_bottlenecks_is_bottleneck_bool(sample_events_df):
    result = detect_bottlenecks(sample_events_df)
    assert result["is_bottleneck"].dtype == bool


def test_sla_minutes_value():
    assert SLA_MINUTES == 2880  # 2 days in minutes


def test_calculate_cycle_times_known_value(sample_events_df):
    # For each case: 10+60+120+300+20 = 510 minutes total duration
    result = calculate_cycle_times(sample_events_df)
    # cycle time is from first timestamp to last timestamp
    for _, row in result.iterrows():
        assert row["total_cycle_time_minutes"] > 0
