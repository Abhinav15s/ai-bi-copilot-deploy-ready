"""Unit tests for modules.db (uses a temporary in-memory SQLite database)."""

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from modules.db import get_engine, run_query


@pytest.fixture()
def tmp_db(tmp_path):
    """Create a minimal SQLite database for testing."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE test_table (id INTEGER PRIMARY KEY, value TEXT)"))
        conn.execute(text("INSERT INTO test_table VALUES (1, 'hello'), (2, 'world')"))
        conn.commit()
    return db_path


def test_get_engine_returns_engine(tmp_db):
    engine = get_engine(tmp_db)
    assert engine is not None


def test_run_query_returns_dataframe(tmp_db):
    result = run_query("SELECT * FROM test_table", db_path=tmp_db)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_run_query_columns(tmp_db):
    result = run_query("SELECT * FROM test_table", db_path=tmp_db)
    assert "id" in result.columns
    assert "value" in result.columns


def test_run_query_values(tmp_db):
    result = run_query("SELECT value FROM test_table ORDER BY id", db_path=tmp_db)
    assert result["value"].tolist() == ["hello", "world"]
