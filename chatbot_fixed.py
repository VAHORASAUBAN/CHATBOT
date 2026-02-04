# import pandas as pd
# import sqlite3
# import os
# import traceback
# from typing import TypedDict, Optional

# from openai import OpenAI
# import sqlglot
# from sqlglot import exp
# from langgraph.graph import StateGraph, END

# # ============================================================
# # CONFIG
# # ============================================================

# EXCEL_PATH = "C:/Users/sauban.vahora/Desktop/Chatbot/data/SGD.xlsx"   # update if needed
# SQLITE_DB = "procurement.db"

# LLM_BASE_URL = "http://45.127.102.236:8000/v1"
# LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# DEBUG = True

# client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")

# def debug(msg):
#     if DEBUG:
#         print("[DEBUG]", msg)

# # ============================================================
# # LOAD EXCEL → SQLITE
# # ============================================================

# def normalize(x):
#     if pd.isna(x):
#         return None
#     if isinstance(x, str):
#         return x.strip().lower()
#     if isinstance(x, pd.Timestamp):
#         return x.isoformat()
#     return x

# def excel_to_sqlite():
#     print("Building SQLite database from Excel...")

#     df = pd.read_excel(EXCEL_PATH)

#     df.columns = [
#         col.strip().replace(" ", "_").replace("-", "_").replace("/", "_")
#         for col in df.columns
#     ]

#     df = df.applymap(normalize)

#     conn = sqlite3.connect(SQLITE_DB)
#     df.to_sql("data", conn, if_exists="replace", index=False)
#     conn.close()

#     print("✓ Database ready")

# # ============================================================
# # SCHEMA
# # ============================================================

# def get_schema():
#     conn = sqlite3.connect(SQLITE_DB)
#     rows = conn.execute("PRAGMA table_info(data)").fetchall()
#     conn.close()
#     return [(r[1], r[2]) for r in rows]

# # ============================================================
# # DOMAIN PROMPT
# # ============================================================
# import json

# with open("schema_contract.json") as f:
#     CONTRACT = json.load(f)

# def build_domain_prompt(user_query, error=""):

#     return f"""
# You are a SQL planner.

# Schema contract:
# {json.dumps(CONTRACT, indent=2)}

# Rules:
# - Use ONLY columns defined in schema
# - Follow business rules exactly
# - Never invent columns
# - Output SQL only

# User question:
# {user_query}

# Previous error:
# {error}

# SQL:
# """



# # ============================================================
# # LLM CALL
# # ============================================================

# def llm(prompt: str) -> str:
#     resp = client.chat.completions.create(
#         model=LLM_MODEL,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0,
#         max_tokens=1024
#     )
#     return resp.choices[0].message.content.strip()

# # ============================================================
# # SQL VALIDATION
# # ============================================================

# def validate_sql(sql: str) -> bool:
#     try:
#         tree = sqlglot.parse_one(sql, read="sqlite")

#         if not isinstance(tree, exp.Select):
#             return False

#         tables = {t.name.lower() for t in tree.find_all(exp.Table)}
#         if tables != {"data"}:
#             return False

#         forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter)
#         for node in tree.walk():
#             if isinstance(node, forbidden):
#                 return False

#         conn = sqlite3.connect(SQLITE_DB)
#         conn.execute(f"EXPLAIN QUERY PLAN {sql}")
#         conn.close()

#         return True

#     except:
#         return False

# # ============================================================
# # EXECUTE SQL
# # ============================================================

# def run_sql(sql: str):
#     conn = sqlite3.connect(SQLITE_DB)
#     df = pd.read_sql_query(sql, conn)
#     conn.close()

#     if df.empty:
#         return "No matching records."

#     return df.to_string(index=False)

# # ============================================================
# # SQL AGENT LOOP
# # ============================================================

# def sql_agent(query: str):
#     error = ""

#     for attempt in range(3):
#         prompt = build_domain_prompt(query, error)
#         sql = llm(prompt)

#         debug(f"Attempt {attempt+1}: {sql}")

#         if validate_sql(sql):
#             try:
#                 return run_sql(sql)
#             except Exception as e:
#                 error = str(e)
#         else:
#             error = "Invalid SQL"

#     return "Failed to generate valid SQL."

# # ============================================================
# # LANGGRAPH STATE
# # ============================================================

# class ChatState(TypedDict):
#     query: str
#     answer: Optional[str]

# def sql_node(state: ChatState):
#     state["answer"] = sql_agent(state["query"])
#     return state

# builder = StateGraph(ChatState)
# builder.add_node("sql", sql_node)
# builder.set_entry_point("sql")
# builder.add_edge("sql", END)

# graph = builder.compile()

# # ============================================================
# # INIT
# # ============================================================

# print("\nInitializing assistant...")

# if not os.path.exists(SQLITE_DB):
#     excel_to_sqlite()

# schema = get_schema()
# print(f"Columns loaded: {len(schema)}")

# print("\nREADY — type exit to quit\n")

# # ============================================================
# # CHAT LOOP
# # ============================================================

# while True:
#     q = input("You: ").strip()
#     if q.lower() == "exit":
#         break

#     try:
#         result = graph.invoke({"query": q, "answer": None})
#         print("\nBot:\n", result["answer"], "\n")

#     except Exception as e:
#         print("Error:", e)
#         traceback.print_exc()
import pandas as pd
import sqlite3
import os
import json
import traceback
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
KNOWLEDGE_JSON = "knowledge_base.json"

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

# ============================================================
# NORMALIZATION
# ============================================================

def normalize(x):
    if pd.isna(x):
        return None
    if isinstance(x, str):
        return x.strip().lower()
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    return x

# ============================================================
# EXCEL → SQLITE
# ============================================================
import re

def clean_column(col: str) -> str:
    col = col.strip().lower()

    # replace special characters with underscore
    col = re.sub(r'[^a-z0-9]+', '_', col)

    # collapse multiple underscores
    col = re.sub(r'_+', '_', col)

    return col.strip('_')

def excel_to_sqlite():
    print("Building SQLite database...")

    df = pd.read_excel(EXCEL_PATH)

    df.columns = [clean_column(c) for c in df.columns]

    # apply normalize column-wise (applymap deprecated)
    df = df.apply(lambda col: col.map(normalize))

    conn = sqlite3.connect(SQLITE_DB)
    df.to_sql("data", conn, if_exists="replace", index=False)
    conn.close()

    print("✓ SQLite ready")
    return df

# ============================================================
# AUTO KNOWLEDGE JSON BUILDER
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

            "category": "unknown",  # optional: identifier/date/status/amount/text

            "description": f"""
This column represents '{readable_title}' in the procurement workflow.
It stores structured information used for filtering, grouping,
and answering business questions about transactions.
""".strip(),

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


# ============================================================
# LOAD KNOWLEDGE
# ============================================================

def load_knowledge():
    with open(KNOWLEDGE_JSON, encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# PROMPT CONTRACT
# ============================================================

def build_prompt(user_query, kb, error=""):

    return f"""
You are a chatbot assistant.

Knowledge Base:
{json.dumps(kb, indent=2)}

Rules:
- Match values to sample values
- Never invent fields
- Only SELECT queries
- Output SQL only

User query:
{user_query}

Previous error:
{error}

SQL:
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
import re

def normalize_sql_quotes(sql: str) -> str:
    # convert "text" → 'text' only when it's a value, not column
    return re.sub(r'"([^"]*)"', r"'\1'", sql)

def validate_sql(sql: str, kb) -> bool:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")

        if not isinstance(tree, exp.Select):
            return False

        # canonical lowercase schema
        valid_cols = {c.lower() for c in kb["columns"].keys()}

        for col in tree.find_all(exp.Column):
            col_name = col.name.lower()

            if col_name not in valid_cols:
                debug(f"Invalid column: {col.name}")
                return False

        conn = sqlite3.connect(SQLITE_DB)
        conn.execute(f"EXPLAIN QUERY PLAN {sql}")
        conn.close()

        return True

    except Exception as e:
        debug(f"Validation error: {e}")
        return False


# ============================================================
# EXECUTE SQL
# ============================================================

def run_sql(sql: str):
    conn = sqlite3.connect(SQLITE_DB)
    df = pd.read_sql_query(sql, conn)
    conn.close()

    if df.empty:
        return "No matching records."

    return df.to_string(index=False)

# ============================================================
# SQL AGENT LOOP
# ============================================================

def sql_agent(query: str, kb):
    error = ""

    for attempt in range(3):
        prompt = build_prompt(query, kb, error)
        sql = llm(prompt)
        sql = normalize_sql_quotes(sql)

        debug(f"Attempt {attempt+1}: {sql}")

        if validate_sql(sql, kb):
            try:
                return run_sql(sql)
            except Exception as e:
                error = str(e)
        else:
            error = "Invalid SQL"

    return "Failed to generate valid SQL."


# ============================================================
# LANGGRAPH STATE
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
kb = build_knowledge_json(df)

print(f"Columns loaded: {len(df.columns)}")
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
