import pandas as pd
import sqlite3
import json
import re
import traceback
from datetime import datetime
from typing import TypedDict, Optional

import streamlit as st
import plotly.express as px

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

client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")

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

def clean_column(col: str) -> str:
    col = col.strip().lower()
    col = re.sub(r'[^a-z0-9]+', '_', col)
    col = re.sub(r'_+', '_', col)
    return col.strip('_')

# ============================================================
# EXCEL → SQLITE
# ============================================================

def excel_to_sqlite():
    xls = pd.ExcelFile(EXCEL_PATH)

    df = None
    for sheet in xls.sheet_names:
        test_df = pd.read_excel(xls, sheet_name=sheet)
        if test_df.shape[1] > 3 and test_df.shape[0] > 1:
            df = test_df
            break

    df.columns = [clean_column(str(c)) for c in df.columns]
    df = df.apply(lambda col: col.map(normalize))
    df = df.dropna(how="all")

    conn = sqlite3.connect(SQLITE_DB)
    df.to_sql("data", conn, if_exists="replace", index=False)
    conn.close()

    return df

# ============================================================
# KNOWLEDGE JSON
# ============================================================

def build_knowledge_json(df):
    kb = {"table": "data", "columns": {}}

    for col in df.columns:
        unique_vals = df[col].dropna().unique().tolist()
        kb["columns"][col] = {
            "name": col,
            "samples": unique_vals[:10]
        }

    with open(KNOWLEDGE_JSON, "w") as f:
        json.dump(kb, f, indent=2)

    return kb

def load_knowledge():
    with open(KNOWLEDGE_JSON) as f:
        return json.load(f)

# ============================================================
# LLM
# ============================================================

def llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

# ============================================================
# SQL AGENT
# ============================================================

def build_prompt(user_query, kb, error=""):
    return f"""
Convert the question into a valid SQLite SELECT query.

Schema:
{json.dumps(kb, indent=2)}

User question:
{user_query}

Previous error:
{error}

Return only SQL.
"""

def normalize_sql(sql):
    return re.sub(r'"([^"]*)"', r"'\1'", sql)

def validate_sql(sql, kb):
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")

        if not isinstance(tree, exp.Select):
            return False

        valid_cols = set(kb["columns"].keys())

        for col in tree.find_all(exp.Column):
            if col.name not in valid_cols:
                return False

        return True
    except:
        return False

def run_sql(sql):
    conn = sqlite3.connect(SQLITE_DB)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df

def sql_agent(query, kb):
    error = ""

    for _ in range(3):
        sql = normalize_sql(llm(build_prompt(query, kb, error)))

        if validate_sql(sql, kb):
            return run_sql(sql)

        error = "Invalid SQL"

    return pd.DataFrame()

# ============================================================
# ANALYTICS AGENT
# ============================================================

def analytics_summary(df):
    if df.empty:
        return "No matching data found."

    prompt = f"""
Explain the key insights from this dataset in business language:

{df.head(20).to_string()}
"""
    return llm(prompt)

# ============================================================
# VISUAL AGENT
# ============================================================

def auto_chart(df):
    if df.shape[1] < 2:
        return None

    numeric = df.select_dtypes(include="number")
    text = df.select_dtypes(exclude="number")

    if numeric.shape[1] == 1 and text.shape[1] >= 1:
        return px.bar(df, x=text.columns[0], y=numeric.columns[0])

    if numeric.shape[1] >= 2:
        return px.scatter(df, x=numeric.columns[0], y=numeric.columns[1])

    return None

# ============================================================
# LANGGRAPH STATE
# ============================================================

class ChatState(TypedDict):
    query: str
    df: Optional[pd.DataFrame]
    summary: Optional[str]

def sql_node(state):
    kb = load_knowledge()
    state["df"] = sql_agent(state["query"], kb)
    return state

def analytics_node(state):
    state["summary"] = analytics_summary(state["df"])
    return state

builder = StateGraph(ChatState)
builder.add_node("sql", sql_node)
builder.add_node("analytics", analytics_node)

builder.set_entry_point("sql")
builder.add_edge("sql", "analytics")
builder.add_edge("analytics", END)

graph = builder.compile()

# ============================================================
# STREAMLIT DASHBOARD
# ============================================================

st.title("📊 AI Procurement Analytics Dashboard")

df_init = excel_to_sqlite()
build_knowledge_json(df_init)

query = st.text_input("Ask a business question")

if query:
    result = graph.invoke({"query": query, "df": None, "summary": None})

    df = result["df"]
    summary = result["summary"]

    st.subheader("Insights")
    st.write(summary)

    st.subheader("Data")
    st.dataframe(df)

    chart = auto_chart(df)
    if chart:
        st.subheader("Visualization")
        st.plotly_chart(chart)
