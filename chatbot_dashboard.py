
import http
import streamlit as st
import pandas as pd
import sqlite3
import os
import json
import traceback
import re
from typing import TypedDict, Optional
from datetime import datetime
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
# LLM_BASE_URL = "http://164.52.192.225:8888/v1"
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

def clean_column(col: str) -> str:
    col = col.strip().lower()
    col = re.sub(r'[^a-z0-9]+', '_', col)
    col = re.sub(r'_+', '_', col)
    return col.strip('_')

def excel_to_sqlite():
    xls = pd.ExcelFile(EXCEL_PATH)
    df = None

    for sheet in xls.sheet_names:
        test_df = pd.read_excel(xls, sheet_name=sheet)
        if test_df.shape[1] > 3 and test_df.shape[0] > 1:
            df = test_df
            break

    if df is None:
        raise Exception("No usable sheet found")

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

def build_knowledge_json(df: pd.DataFrame):

    kb = {
        "table": "data",
        "columns": {}
    }

    for col in df.columns:
        unique_vals = df[col].dropna().unique().tolist()
        sample_vals = [to_json_safe(v) for v in unique_vals[:10]]

        kb["columns"][col] = {
            "name": col,
            "type": str(df[col].dtype),
            "sample_values": sample_vals
        }

    with open(KNOWLEDGE_JSON, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2)

    return kb

def load_knowledge():
    with open(KNOWLEDGE_JSON, encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# PROMPT
# ============================================================

def build_prompt(user_query, kb, memory, error=""):

    return f"""
You are both a SQL expert and a senior business analyst.

Return ONLY valid JSON in this exact structure:

{{
  "sql": "SQLite SELECT query",
  "insight": "Detailed business insight in natural language"
}}

Rules:

- SQL must be valid SQLite SELECT
- Do not include markdown
- Do not include extra text
- Insight must explain results like a human analyst
- Mention trends, rankings, comparisons, or risks
- Write insight as if explaining to a business manager
- Insight must interpret the result, not repeat the query

Conversation memory:
{memory}

Database schema:
{json.dumps(kb, indent=2)}

User question:
{user_query}

Previous error:
{error}
"""



def parse_llm_json(text):

    try:
        return json.loads(text)
    except:
        # fallback: try extracting JSON block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))

    return {"sql": "", "insight": "Could not generate insight."}


# ============================================================
# LLM
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
# VALIDATION
# ============================================================

def normalize_sql_quotes(sql: str) -> str:
    return re.sub(r'"([^"]*)"', r"'\1'", sql)

def extract_sql(text: str) -> str:
    match = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def validate_sql(sql: str, kb) -> bool:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")

        # only ensure it's SELECT
        if not isinstance(tree, exp.Select):
            return False

        return True

    except Exception as e:
        debug(e)
        return False


# ============================================================
# EXECUTE
# ============================================================

def run_sql(sql: str):

    conn = sqlite3.connect(SQLITE_DB)
    df = pd.read_sql_query(sql, conn)
    conn.close()

    return df

# ============================================================
# AUTO CHART
# ============================================================

def auto_chart(df):

    if not isinstance(df, pd.DataFrame):
        return

    if df.shape[1] < 2:
        return

    numeric_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(exclude="number").columns

    if len(numeric_cols) == 0:
        return

    try:
        if len(cat_cols) > 0:
            fig = px.bar(
                df,
                x=cat_cols[0],
                y=numeric_cols[0],
                title=f"{numeric_cols[0]} by {cat_cols[0]}"
            )
        else:
            fig = px.line(df[numeric_cols[0]], title=numeric_cols[0])

        st.plotly_chart(fig, width='stretch')

    except Exception as e:
        st.write("Chart error:", e)



# ============================================================
# MEMORY
# ============================================================

if "memory" not in st.session_state:
    st.session_state.memory = []

# ============================================================
# INIT DB
# ============================================================

if not os.path.exists(SQLITE_DB):
    df = excel_to_sqlite()
    build_knowledge_json(df)

# ============================================================
# UI
# ============================================================

st.set_page_config(layout="wide")
st.title("🚀 Procurement Assistant")

# mode = st.sidebar.radio("Mode", ["Chat", "Dashboard"])

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================
# CHAT MODE
# ============================================================

# if mode == "Chat":

user_input = st.chat_input("Ask your data question...")

if user_input:

    memory_text = "\n".join(st.session_state.memory[-5:])
    kb = load_knowledge()

    raw_output = llm(build_prompt(user_input, kb, memory_text))
    parsed = parse_llm_json(raw_output)

    sql = normalize_sql_quotes(parsed["sql"])
    insight = parsed["insight"]



    if validate_sql(sql, kb):
        df = run_sql(sql)
    else:
        df = "Invalid SQL"

    st.session_state.memory.append(user_input)
    st.session_state.messages.append((user_input, df, sql, insight))

for q, ans, sql, insight in st.session_state.messages:


    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):

        if isinstance(ans, pd.DataFrame):
            st.dataframe(ans, width='content')
            auto_chart(ans)

            st.markdown("### 💡 Key Insight")
            st.write(insight)
        else:
            st.write(ans)

        st.code(sql, language="sql")

# ============================================================
# DASHBOARD MODE
# ============================================================

# if mode == "Dashboard":

#     st.subheader("📊 Live Dashboard")

#     conn = sqlite3.connect(SQLITE_DB)
#     df = pd.read_sql_query("SELECT * FROM data LIMIT 1000", conn)
#     conn.close()

#     st.dataframe(df, use_container_width=True)

#     numeric_cols = df.select_dtypes(include="number").columns

#     if len(numeric_cols):
#         st.bar_chart(df[numeric_cols[0]])
