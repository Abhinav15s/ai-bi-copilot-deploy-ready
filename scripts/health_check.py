"""Quick health check for local AI BI Copilot setup.

Run:
    python scripts/health_check.py

Optional live LLM ping:
    python scripts/health_check.py --live
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "business_data.db"


def _ok(label: str, details: str) -> None:
    print(f"✅ {label}: {details}")


def _warn(label: str, details: str) -> None:
    print(f"⚠️  {label}: {details}")


def _fail(label: str, details: str) -> None:
    print(f"❌ {label}: {details}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local setup checks.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also run a minimal live Groq NL-to-SQL query.",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    passed = True

    api_key = os.getenv("GROQ_API_KEY", "")
    if api_key and api_key != "your-groq-api-key-here":
        _ok("GROQ_API_KEY", "present")
    else:
        _fail("GROQ_API_KEY", "missing or placeholder in .env")
        passed = False

    if DB_PATH.exists():
        _ok("Database", str(DB_PATH))
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM transactions")
                txn_count = cursor.fetchone()[0]
            _ok("Transactions table", f"{txn_count} rows")
        except Exception as exc:  # noqa: BLE001
            _fail("Database query", str(exc))
            passed = False
    else:
        _fail("Database", "not found. Run: python data/generate_data.py")
        passed = False

    try:
        import langchain_groq  # noqa: F401
        import streamlit  # noqa: F401

        _ok("Dependencies", "streamlit and langchain-groq import successfully")
    except Exception as exc:  # noqa: BLE001
        _fail("Dependencies", str(exc))
        passed = False

    if args.live and passed:
        try:
            from modules.genai_query import ask_question

            answer = ask_question("How many transactions are there?")
            if "1200" in answer or "transaction" in answer.lower():
                _ok("Live Groq query", answer)
            else:
                _warn("Live Groq query", f"unexpected response: {answer}")
        except Exception as exc:  # noqa: BLE001
            _fail("Live Groq query", str(exc))
            passed = False

    print("\nSetup status:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
