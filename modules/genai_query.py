"""
GenAI Query layer for AI BI Copilot.

Uses LangChain + Groq to translate natural language questions into SQL,
execute them against the SQLite database, and return a plain-English answer.

Usage::

    from modules.genai_query import ask_question

    answer = ask_question("Which region had the highest revenue last quarter?")

Sample questions
----------------
- "Which region had the highest revenue last quarter?"
- "What is the average cycle time for order fulfilment?"
- "Show me the top 3 product categories with the most negative customer reviews."
- "What percentage of orders were delayed by more than 2 days?"
"""

import os
import re
from pathlib import Path

# Load .env for local development (no-op on Streamlit Cloud)
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Secrets helper: prefer st.secrets (Streamlit Cloud) → os.environ (.env)
# ---------------------------------------------------------------------------

def _get_secret(key: str, default: str = "") -> str:
    """Return *key* from st.secrets if available, else from os.environ."""
    try:
        import streamlit as st
        value = st.secrets.get(key, "")
        if value:
            return value
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get(key, default)


_DB_PATH = Path(__file__).parent.parent / "data" / "business_data.db"
_chain_cache = None  # module-level cache: stores the built LangChain chain or None


def _get_chain():
    global _chain_cache
    if _chain_cache is None:
        _chain_cache = _build_chain()
    return _chain_cache


def _extract_sql_query(raw_query: str) -> str:
    query = (raw_query or "").strip()
    if "```" in query:
        blocks = re.findall(r"```(?:sql)?\s*(.*?)```", query, flags=re.IGNORECASE | re.DOTALL)
        if blocks:
            query = blocks[0].strip()
    for marker in ("SQLQuery:", "SQL Query:", "Query:"):
        if marker in query:
            query = query.split(marker, 1)[1].strip()
    query = query.split("SQLResult:", 1)[0].strip()
    query = query.split("Answer:", 1)[0].strip()
    return query.strip("` \n;") + ";"


def _build_chain():
    """Build and return the LangChain SQL query chain.

    Returns ``None`` and prints an error message when the Groq API key is
    absent or when the database file does not exist.
    """
    api_key = _get_secret("GROQ_API_KEY")
    if not api_key or api_key == "your-groq-api-key-here":
        print("⚠️  GROQ_API_KEY is not set.  Add it to st.secrets or your .env file.")
        return None

    if not _DB_PATH.exists():
        print(f"⚠️  Database not found at {_DB_PATH}.  Run `python data/generate_data.py` first.")
        return None

    try:
        from operator import itemgetter

        from langchain.chains import create_sql_query_chain
        from langchain_community.tools import QuerySQLDatabaseTool
        from langchain_community.utilities import SQLDatabase
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import PromptTemplate
        from langchain_core.runnables import RunnableLambda, RunnablePassthrough
        from langchain_groq import ChatGroq

        db = SQLDatabase.from_uri(f"sqlite:///{_DB_PATH}")
        llm = ChatGroq(
            model=_get_secret("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0,
            api_key=api_key,
        )

        write_query = create_sql_query_chain(llm, db)
        execute_query = QuerySQLDatabaseTool(db=db)
        clean_query = RunnableLambda(_extract_sql_query)

        answer_prompt = PromptTemplate.from_template(
            """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
        )

        chain = (
            RunnablePassthrough.assign(query=write_query | clean_query).assign(
                result=itemgetter("query") | execute_query
            )
            | answer_prompt
            | llm
            | StrOutputParser()
        )
        return chain
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Failed to build LangChain chain: {exc}")
        return None


def ask_question(question: str) -> str:
    """Translate a natural language *question* into SQL and return the answer.

    Parameters
    ----------
    question:
        A business question in plain English.

    Returns
    -------
    str
        A natural language answer, or an error message when the chain cannot
        be initialised.
    """
    chain = _get_chain()
    if chain is None:
        return (
            "⚠️  The AI query layer is unavailable.  "
            "Check that GROQ_API_KEY is set and the database exists."
        )
    try:
        return chain.invoke({"question": question})
    except Exception as exc:  # noqa: BLE001
        return f"⚠️  Error answering question: {exc}"


if __name__ == "__main__":
    print("🤖 AI BI Copilot — Interactive Query CLI")
    print("   Type your business question and press Enter.")
    print("   Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("❓ Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue
        print(f"\n💬 Answer: {ask_question(user_input)}\n")
