# import pandas as pd
# import numpy as np
# import faiss
# import requests
# from typing import List
# from openai import OpenAI

# # ============================================================
# # CONFIG
# # ============================================================

# EXCEL_PATH = "C:/Users/sauban.vahora/Desktop/Chatbot/data/SGD.xlsx"

# LLM_BASE_URL = "http://45.127.102.236:8000/v1"
# LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# EMBEDDING_URL = "http://127.0.0.1:5002/embeddings"
# EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# TOP_K = 3

# # ============================================================
# # LOAD DATA
# # ============================================================

# df = pd.read_excel(EXCEL_PATH).fillna("")

# # Numeric columns (for deterministic math)
# NUMERIC_COLS = df.select_dtypes(include="number").columns.tolist()

# # ============================================================
# # TEXT CHUNKING (SEMANTIC SIDE)
# # ============================================================

# df["text_chunk"] = df.apply(
#     lambda row: " | ".join(f"{col}: {row[col]}" for col in df.columns),
#     axis=1
# )

# # ============================================================
# # EMBEDDING CLIENT
# # ============================================================

# def get_embedding(text: str) -> List[float]:
#     payload = {
#         "model": EMBEDDING_MODEL,
#         "input": [text]
#     }
#     res = requests.post(EMBEDDING_URL, json=payload, timeout=30)
#     res.raise_for_status()
#     return res.json()["data"][0]["embedding"]

# # ============================================================
# # BUILD FAISS INDEX
# # ============================================================

# embeddings = np.array(
#     [get_embedding(text) for text in df["text_chunk"]],
#     dtype="float32"
# )

# dimension = embeddings.shape[1]
# faiss_index = faiss.IndexFlatL2(dimension)
# faiss_index.add(embeddings)

# # ============================================================
# # LLM CLIENT
# # ============================================================

# llm_client = OpenAI(
#     base_url=LLM_BASE_URL,
#     api_key="FAKE"
# )

# # ============================================================
# # INTENT ROUTER
# # ============================================================

# def is_numeric_query(query: str) -> bool:
#     keywords = [
#         "average", "avg", "sum", "total",
#         "maximum", "minimum", "highest",
#         "lowest", "greater than", "less than"
#     ]
#     q = query.lower()
#     return any(k in q for k in keywords)

# # ============================================================
# # NUMERIC ENGINE (DETERMINISTIC)
# # ============================================================

# def detect_numeric_column(query: str) -> str | None:
#     q = query.lower()
#     for col in NUMERIC_COLS:
#         if col.lower().replace("_", " ") in q:
#             return col
#     return None

# def handle_numeric_query(query: str) -> str:
#     q = query.lower()
#     col = detect_numeric_column(query)

#     if col is None:
#         return "Numeric intent detected, but no matching column found."

#     if "average" in q or "avg" in q:
#         return f"Average {col}: {df[col].mean():.2f}"

#     if "maximum" in q or "highest" in q or "max" in q:
#         return f"Maximum {col}: {df[col].max():.2f}"

#     if "minimum" in q or "lowest" in q or "min" in q:
#         return f"Minimum {col}: {df[col].min():.2f}"

#     if "sum" in q or "total" in q:
#         return f"Total {col}: {df[col].sum():.2f}"

#     return "Numeric operation not supported yet."

# # ============================================================
# # SEMANTIC ENGINE (RAG)
# # ============================================================

# def retrieve_context(query: str, k: int = TOP_K) -> str:
#     q_emb = np.array([get_embedding(query)], dtype="float32")
#     _, indices = faiss_index.search(q_emb, k)
#     return "\n".join(df.iloc[i]["text_chunk"] for i in indices[0])

# def ask_llm(context: str, question: str) -> str:
#     response = llm_client.chat.completions.create(
#         model=LLM_MODEL,
#         temperature=0,
#         messages=[
#             {
#                 "role": "system",
#                 "content": (
#                     "You are a factual assistant. "
#                     "Answer ONLY using the provided context. "
#                     "If the answer is not present, say so clearly."
#                 )
#             },
#             {
#                 "role": "user",
#                 "content": f"Context:\n{context}\n\nQuestion:\n{question}"
#             }
#         ]
#     )
#     return response.choices[0].message.content

# # ============================================================
# # RESPONSE MERGER (SINGLE ENTRY POINT)
# # ============================================================

# def chat(query: str) -> str:
#     if is_numeric_query(query):
#         return handle_numeric_query(query)

#     context = retrieve_context(query)
#     return ask_llm(context, query)

# # ============================================================
# # CLI LOOP (OPTIONAL)
# # ============================================================

# if __name__ == "__main__":
#     print("✅ Hybrid Chatbot Ready (type 'exit' to quit)")
#     while True:
#         q = input("\nUser: ")
#         if q.lower() in {"exit", "quit"}:
#             break
#         print("Bot:", chat(q))

import pandas as pd
import numpy as np
import faiss
import requests
import json
import sqlite3
import functools
from typing import List
from openai import OpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================================
# CONNECTION POOLING FOR FASTER HTTP REQUESTS
# ============================================================

session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
session.mount('http://', adapter)
session.mount('https://', adapter)

# ============================================================
# RESPONSE CACHING (INSTANT ANSWERS FOR REPEATED QUERIES)
# ============================================================

query_cache = {}

# ============================================================
# CONFIG
# ============================================================

EXCEL_PATH = "C:/Users/sauban.vahora/Desktop/Chatbot/data/SGD.xlsx"

DB_PATH = "chatbot.db"
TABLE_NAME = "data"

LLM_BASE_URL = "http://45.127.102.236:5002/v1"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

EMBEDDING_URL = "http://45.127.102.236:5002/embeddings"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

TOP_K = 3

# ============================================================
# LOAD EXCEL → SQLITE (NUMERIC SOURCE OF TRUTH)
# ============================================================

df = pd.read_excel(EXCEL_PATH).fillna("")

# Fix Excel date / timestamp nonsense
for col in df.columns:
    if "date" in col.lower() or "time" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

# Discover numeric columns from SQL (authoritative)
cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
schema = cursor.fetchall()

NUMERIC_COLS = [
    col[1] for col in schema
    if col[2].upper() in ("INTEGER", "REAL", "FLOAT", "DOUBLE", "NUMERIC")
]

# ============================================================
# TEXT CHUNKS FOR SEMANTIC SEARCH
# ============================================================

df["text_chunk"] = df.apply(
    lambda row: " | ".join(f"{col}: {row[col]}" for col in df.columns),
    axis=1
)

# ============================================================
# EMBEDDING CLIENT WITH CACHING (AVOID REDUNDANT REQUESTS)
# ============================================================

@functools.lru_cache(maxsize=1000)
def get_embedding(text: str) -> tuple:
    """Cache embeddings to avoid re-requesting same texts"""
    payload = {
        "model": EMBEDDING_MODEL,
        "input": [text]
    }
    res = session.post(EMBEDDING_URL, json=payload, timeout=30)
    res.raise_for_status()
    return tuple(res.json()["data"][0]["embedding"])

def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Batch multiple embeddings in one request (more efficient)"""
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts
    }
    res = session.post(EMBEDDING_URL, json=payload, timeout=60)
    res.raise_for_status()
    return [item["embedding"] for item in res.json()["data"]]

# ============================================================
# BUILD FAISS INDEX (HNSW IS 2-3X FASTER THAN FLATL2)
# ============================================================

embeddings_list = []
for i in range(0, len(df), 32):
    batch = df["text_chunk"].tolist()[i:i+32]
    embeddings_list.extend(get_embeddings_batch(batch))

embeddings = np.array(embeddings_list, dtype="float32")

# Use HNSW (Hierarchical Navigable Small World) for fast approximate nearest neighbor search
faiss_index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
faiss_index.add(embeddings)

# ============================================================
# LLM CLIENT
# ============================================================

llm_client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key="FAKE"
)

# ============================================================
# FAST NUMERIC KEYWORD PRE-CHECK (SKIP EXPENSIVE LLM CALL)
# ============================================================

def has_numeric_keywords(query: str) -> bool:
    """Quick keyword check before calling expensive LLM"""
    keywords = {
        "average", "avg", "sum", "total", "maximum", "minimum",
        "highest", "lowest", "max", "min", "count", "many"
    }
    q_lower = query.lower()
    return any(k in q_lower for k in keywords)

# ============================================================
# LLM → NUMERIC INTENT & COLUMN MAPPING
# ============================================================

def analyze_numeric_intent(query: str) -> dict | None:
    # Skip expensive LLM call if no numeric keywords
    if not has_numeric_keywords(query):
        return None
    
    prompt = f"""
You are a data query analyzer.

Available numeric columns:
{", ".join(NUMERIC_COLS)}

User query:
"{query}"

Return ONLY valid JSON.

If numeric intent exists:
{{
  "operation": "avg | max | min | sum",
  "column": "<best matching column>"
}}

If not numeric:
{{ "numeric": false }}
"""

    res = llm_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        parsed = json.loads(res.choices[0].message.content)
        return parsed if "operation" in parsed else None
    except Exception:
        return None

# ============================================================
# SQL NUMERIC ENGINE (DETERMINISTIC)
# ============================================================

def run_numeric_query(intent: dict) -> str:
    col = intent["column"]
    op = intent["operation"].lower()

    sql_map = {
        "avg": f"SELECT AVG({col}) FROM {TABLE_NAME}",
        "max": f"SELECT MAX({col}) FROM {TABLE_NAME}",
        "min": f"SELECT MIN({col}) FROM {TABLE_NAME}",
        "sum": f"SELECT SUM({col}) FROM {TABLE_NAME}",
    }

    sql = sql_map.get(op)
    if not sql:
        return "Numeric operation not supported."

    cursor.execute(sql)
    value = cursor.fetchone()[0]

    return f"{op.upper()}({col}) = {value}"

# ============================================================
# SEMANTIC ENGINE (RAG) - WITH TOKEN LIMITING
# ============================================================

def retrieve_context(query: str, max_tokens: int = 500) -> str:
    """Retrieve top-K chunks but limit total tokens for faster LLM processing"""
    q_emb = np.array([get_embedding(query)], dtype="float32")
    _, indices = faiss_index.search(q_emb, TOP_K)
    
    context_parts = []
    token_count = 0
    
    for i in indices[0]:
        chunk = df.iloc[i]["text_chunk"]
        tokens = len(chunk.split())  # Approximate token count
        
        if token_count + tokens > max_tokens:
            break
        
        context_parts.append(chunk)
        token_count += tokens
    
    return "\n".join(context_parts) if context_parts else df.iloc[indices[0][0]]["text_chunk"]

def ask_llm(context: str, question: str) -> str:
    res = llm_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer strictly using the provided context. "
                    "If the answer is not present, say so."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ]
    )
    return res.choices[0].message.content

# ============================================================
# SINGLE CHAT ENTRY POINT (WITH RESPONSE CACHING)
# ============================================================

def chat(query: str) -> str:
    # Check cache first for instant response
    normalized = query.lower().strip()
    if normalized in query_cache:
        return query_cache[normalized]
    
    numeric_intent = analyze_numeric_intent(query)

    if numeric_intent:
        result = run_numeric_query(numeric_intent)
        query_cache[normalized] = result
        return result

    context = retrieve_context(query)
    result = ask_llm(context, query)
    query_cache[normalized] = result
    return result

# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    print("✅ SQL + FAISS + LLM Hybrid Chatbot Ready")
    print("Type 'exit' to quit")

    while True:
        q = input("\nUser: ")
        if q.lower() in {"exit", "quit"}:
            break
        print("Bot:", chat(q))
    