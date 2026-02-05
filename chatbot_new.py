import pandas as pd
import sqlite3
import json
import traceback
import re
from typing import TypedDict, Optional
from datetime import datetime

from openai import OpenAI
import sqlglot
from sqlglot import exp
from langgraph.graph import StateGraph, END

# ============================================================
# CONFIG
# ============================================================

EXCEL_PATH = "C:/Users/sauban.vahora/Desktop/Chatbot/data/SGD.xlsx"
SQLITE_DB = "procurement.db"
KNOWLEDGE_JSON = "Intent_base.json"

LLM_BASE_URL = "http://45.127.102.236:8000/v1"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

DEBUG = True

client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")

def debug(msg):
    if DEBUG:
        print("[DEBUG]", msg)

# ============================================================
# SAFE JSON CONVERSION
# ============================================================

def to_json_safe(x):
    if pd.isna(x):
        return None
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, (int, float, str, bool)):
        return x
    return str(x)

def normalize(x):
    if pd.isna(x):
        return None
    if isinstance(x, str):
        return x.strip().lower()
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    return x

# ============================================================
# EXCEL LOADER
# ============================================================

def clean_column(col: str) -> str:
    col = col.strip().lower()
    col = re.sub(r'[^a-z0-9]+', '_', col)
    col = re.sub(r'_+', '_', col)
    return col.strip('_')

def excel_to_sqlite():
    print("Building SQLite database...")

    xls = pd.ExcelFile(EXCEL_PATH)
    print("Available sheets:", xls.sheet_names)

    df = None
    for sheet in xls.sheet_names:
        test_df = pd.read_excel(xls, sheet_name=sheet)
        if test_df.shape[1] > 3 and test_df.shape[0] > 1:
            df = test_df
            print(f"Using sheet: {sheet}")
            break

    if df is None:
        raise Exception("No usable sheet found")

    df.columns = [clean_column(str(c)) for c in df.columns]
    df = df.apply(lambda col: col.map(normalize))
    df = df.dropna(how="all")

    print("Detected columns:", df.columns.tolist())
    print("Shape:", df.shape)

    conn = sqlite3.connect(SQLITE_DB)
    df.to_sql("data", conn, if_exists="replace", index=False)
    conn.close()

    print("✓ SQLite ready")
    return df

# ============================================================
# KNOWLEDGE JSON
# ============================================================

def build_knowledge_json(df: pd.DataFrame):
    print("Generating knowledge base JSON...")

    kb = {
        "table": "data",
        "description": "Procurement dataset containing purchase requests, purchase orders, suppliers, and authorization status.",
        "columns": {}
    }

    for col in df.columns:

        unique_vals = df[col].dropna().unique().tolist()
        sample_vals = [to_json_safe(v) for v in unique_vals[:10]]

        readable_title = col.replace("_", " ").title()

        kb["columns"][col] = {
            "name": col,
            "title": readable_title,
            "type": str(df[col].dtype),

            "category": "unknown",

            "description": (
                f"This column represents '{readable_title}' in the procurement workflow. "
                "It is used for filtering, grouping, and answering business questions."
            ),

            "user_intent_examples": [
                f"show records by {readable_title}",
                f"filter using {readable_title}",
                f"group by {readable_title}"
            ],

            "synonyms": [
                readable_title.lower(),
                col.lower(),
                col.replace("_", " ")
            ],

            "keywords": readable_title.lower().split(),

            "normalization": "Values are stored in lowercase and trimmed for consistent matching.",

            "sample_values": sample_vals
        }

    with open(KNOWLEDGE_JSON, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

    print("✓ Knowledge JSON created")
    return kb


def load_knowledge():
    with open(KNOWLEDGE_JSON, encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# INTENT PROMPT (STAGE 1)
# ============================================================

def build_intent_prompt(user_query: str):

    return f"""
Convert the user request into structured intent JSON.

Only output valid JSON.

Schema:

{{
  "action": "lookup | aggregation | ranking",
  "target_columns": [],
  "metric_column": null,
  "aggregation": "count | sum | avg | max | min | null",
  "order": "asc | desc | null",
  "limit": null,
  "filters": []
}}

User query:
{user_query}
"""

# ============================================================
# SQL PROMPT (STAGE 2)
# ============================================================

def build_sql_prompt(intent: dict, kb):

    return f"""
Convert intent into SQLite SQL.

Intent:
{json.dumps(intent, indent=2)}

Knowledge base:
{json.dumps(kb, indent=2)}

Only output SQL SELECT query.
"""

# ============================================================
# LLM CALL
# ============================================================

def llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1024
    )
    return resp.choices[0].message.content.strip()

# ============================================================
# SQL VALIDATION
# ============================================================

def normalize_sql_quotes(sql: str) -> str:
    return re.sub(r'"([^"]*)"', r"'\1'", sql)

def validate_sql(sql: str, kb) -> bool:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        valid_cols = {c.lower() for c in kb["columns"].keys()}

        for col in tree.find_all(exp.Column):
            if col.name.lower() not in valid_cols:
                debug(f"Invalid column: {col.name}")
                return False

        return True
    except Exception as e:
        debug(f"Validation error: {e}")
        return False

# ============================================================
# EXECUTION
# ============================================================

def run_sql(sql: str):
    conn = sqlite3.connect(SQLITE_DB)
    df = pd.read_sql_query(sql, conn)
    conn.close()

    if df.empty:
        return "No matching records."

    return df.to_string(index=False)

# ============================================================
# AGENT LOOP
# ============================================================

def sql_agent(query: str, kb):

    for attempt in range(3):

        intent_json = llm(build_intent_prompt(query))
        debug(f"Intent raw: {intent_json}")

        try:
            intent = json.loads(intent_json)
        except:
            continue

        sql = llm(build_sql_prompt(intent, kb))
        sql = normalize_sql_quotes(sql)

        debug(f"Attempt {attempt+1}: {sql}")

        if validate_sql(sql, kb):
            return run_sql(sql)

    return "Failed to generate valid SQL."

# ============================================================
# LANGGRAPH
# ============================================================

class ChatState(TypedDict):
    query: str
    answer: Optional[str]

def sql_node(state: ChatState):
    kb = load_knowledge()
    state["answer"] = sql_agent(state["query"], kb)
    return state

builder = StateGraph(ChatState)
builder.add_node("sql", sql_node)
builder.set_entry_point("sql")
builder.add_edge("sql", END)

graph = builder.compile()

# ============================================================
# INIT
# ============================================================

print("\nInitializing assistant...")

df = excel_to_sqlite()
build_knowledge_json(df)

print("\nREADY — type exit to quit\n")

# ============================================================
# CHAT LOOP
# ============================================================

while True:
    q = input("You: ").strip()
    if q.lower() == "exit":
        break

    try:
        result = graph.invoke({"query": q, "answer": None})
        print("\nBot:\n", result["answer"], "\n")

    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
