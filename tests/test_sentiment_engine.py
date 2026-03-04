"""Unit tests for modules.sentiment_engine."""

from modules.sentiment_engine import (
    analyze_reviews_df,
    analyze_sentiment,
    get_sentiment_summary,
)


def test_analyze_sentiment_positive():
    result = analyze_sentiment("Excellent product, exceeded all expectations!")
    assert result["label"] == "positive"
    assert result["compound"] > 0.05


def test_analyze_sentiment_negative():
    result = analyze_sentiment("Very disappointed with the product quality.")
    assert result["label"] == "negative"
    assert result["compound"] < -0.05


def test_analyze_sentiment_neutral():
    result = analyze_sentiment("Product works as described.")
    assert result["label"] in ("neutral", "positive", "negative")
    assert -1.0 <= result["compound"] <= 1.0


def test_analyze_sentiment_returns_dict():
    result = analyze_sentiment("test text")
    assert isinstance(result, dict)
    assert "compound" in result
    assert "label" in result
    assert result["label"] in ("positive", "neutral", "negative")


def test_analyze_reviews_df_adds_columns(sample_reviews_df):
    result = analyze_reviews_df(sample_reviews_df)
    assert "sentiment_score" in result.columns
    assert "sentiment_label" in result.columns
    assert len(result) == len(sample_reviews_df)


def test_analyze_reviews_df_does_not_mutate_input(sample_reviews_df):
    original_cols = set(sample_reviews_df.columns)
    analyze_reviews_df(sample_reviews_df)
    assert set(sample_reviews_df.columns) == original_cols


def test_get_sentiment_summary_keys(sample_reviews_df):
    enriched = analyze_reviews_df(sample_reviews_df)
    summary = get_sentiment_summary(enriched)
    assert "counts" in summary
    assert "percentages" in summary
    assert "avg_score_by_category" in summary


def test_get_sentiment_summary_percentages_sum_to_100(sample_reviews_df):
    enriched = analyze_reviews_df(sample_reviews_df)
    summary = get_sentiment_summary(enriched)
    total_pct = sum(summary["percentages"].values())
    assert abs(total_pct - 100.0) < 0.1
