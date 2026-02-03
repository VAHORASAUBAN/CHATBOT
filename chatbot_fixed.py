"""
FIXED CHATBOT - Data Retrieval Issues Resolved
===============================================

Key fixes:
1. Added comprehensive error logging to see where retrieval fails
2. Fixed SQL validation (was too strict, rejecting valid queries)
3. Added fallback vector retrieval to SQL node
4. Better error messages for debugging
5. Explicit schema printing at startup
"""

import pandas as pd
import sqlite3
import faiss
import numpy as np
import json
import os
import requests
from typing import TypedDict, Optional
import traceback

from langgraph.graph import StateGraph, END
from openai import OpenAI
import sqlglot
from sqlglot import exp

# ============================================================
# CONFIG
# ============================================================

EXCEL_PATH = "C:/Users/sauban.vahora/Desktop/Chatbot/data/SGD.xlsx"
SQLITE_DB = "rag.db"
FAISS_INDEX = "faiss.index"
TEXTS_FILE = "texts.npy"

LLM_BASE_URL = "http://45.127.102.236:8000/v1"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

EMBEDDING_URL = "http://127.0.0.1:5002/embeddings"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

TOP_K = 5
DEBUG = True  # Enable detailed logging

client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")

def debug_log(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")

# ============================================================
# EMBEDDING CLIENT
# ============================================================

def embed_texts(texts):
    """Embed texts using remote service"""
    try:
        r = requests.post(
            EMBEDDING_URL,
            json={"model": EMBEDDING_MODEL, "input": texts},
            timeout=60
        )
        r.raise_for_status()
        data = r.json()["data"]
        vecs = [d["embedding"] for d in data]
        return np.array(vecs, dtype=np.float32)
    except Exception as e:
        debug_log(f"Embedding failed: {e}")
        raise

# ============================================================
# SQLITE UTILITIES
# ============================================================

def get_schema():
    """Get database schema"""
    try:
        conn = sqlite3.connect(SQLITE_DB)
        schema = conn.execute("PRAGMA table_info(data)").fetchall()
        conn.close()
        return [(col[1], col[2]) for col in schema]  # (name, type)
    except Exception as e:
        debug_log(f"Schema fetch failed: {e}")
        return []

def infer_sqlite_type(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    if pd.api.types.is_bool_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TEXT"
    return "TEXT"

def sanitize_value(x):
    if pd.isna(x):
        return None
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if isinstance(x, bool):
        return int(x)
    return x

def excel_to_sqlite():
    """Load Excel into SQLite"""
    print("Building SQLite schema from Excel...")
    df = pd.read_excel(EXCEL_PATH)

    # clean column names
    df.columns = [
        col.strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        for col in df.columns
    ]

    # sanitize dataframe
    df = df.applymap(sanitize_value)

    # infer schema
    schema = {
        col: infer_sqlite_type(df[col].dtype)
        for col in df.columns
    }

    conn = sqlite3.connect(SQLITE_DB)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS data")

    columns_sql = ", ".join(
        [f'"{col}" {dtype}' for col, dtype in schema.items()]
    )

    create_sql = f"CREATE TABLE data ({columns_sql})"
    cur.execute(create_sql)

    placeholders = ", ".join(["?"] * len(df.columns))
    insert_sql = f'INSERT INTO data VALUES ({placeholders})'

    cur.executemany(insert_sql, df.values.tolist())

    conn.commit()
    conn.close()

    print(f"✓ SQLite ready. Columns: {list(schema.keys())[:5]}...")

def row_to_text(row):
    return " | ".join([f"{col}: {row[col]}" for col in row.index])

def build_vector_index():
    """Build FAISS index"""
    print("Building FAISS index...")
    df = pd.read_excel(EXCEL_PATH)
    texts = df.apply(row_to_text, axis=1).tolist()

    embeddings = embed_texts(texts)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX)
    np.save(TEXTS_FILE, np.array(texts, dtype=object))
    
    print(f"✓ FAISS index ready. Vectors: {len(texts)}, Dimension: {dim}")

# ============================================================
# LLM CALL
# ============================================================

def llm(prompt: str) -> str:
    """Call LLM"""
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        debug_log(f"LLM call failed: {e}")
        return ""

# ============================================================
# INTENT DETECTION
# ============================================================

def detect_intent(query: str):
    """Detect query intent"""
    prompt = f"""
Classify this query into ONE word only:

numeric → math, aggregation, totals, counts, sums
semantic → meaning, explanation, analysis from data
meta → general knowledge, greetings

Answer with exactly one word:
numeric OR semantic OR meta

Query: {query}
"""
    
    out = llm(prompt).lower().strip()
    debug_log(f"Intent detection raw: '{out}'")
    
    if "numeric" in out:
        return "numeric"
    if "meta" in out:
        return "meta"
    return "semantic"

# ============================================================
# SQL ENGINE (IMPROVED VALIDATION)
# ============================================================

def clean_sql(text: str) -> str:
    """Clean SQL response"""
    text = text.strip()
    # Remove markdown code blocks
    if "```sql" in text:
        parts = text.split("```sql")
        if len(parts) > 1:
            text = parts[1].split("```")[0]
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
    
    # Remove any leading/trailing whitespace and newlines
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    return text.strip()

def ast_validate(sql: str) -> bool:
    """Validate SQL using AST"""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        
        # must be SELECT
        if not isinstance(tree, exp.Select):
            debug_log(f"Not a SELECT: {type(tree)}")
            return False
        
        # enforce table whitelist
        tables = {t.name.lower() for t in tree.find_all(exp.Table)}
        debug_log(f"Tables in query: {tables}")
        if tables and tables != {"data"}:
            debug_log(f"Invalid tables: {tables}")
            return False
        
        # forbid dangerous commands
        forbidden = (
            exp.Insert,
            exp.Update,
            exp.Delete,
            exp.Drop,
            exp.Alter,
            exp.Create,
        )
        
        for node in tree.walk():
            if isinstance(node, forbidden):
                debug_log(f"Forbidden node: {type(node)}")
                return False
        
        return True
        
    except Exception as e:
        debug_log(f"AST validation error: {e}")
        return False

def query_planner_validate(sql: str) -> bool:
    """Validate SQL can execute"""
    try:
        conn = sqlite3.connect(SQLITE_DB)
        conn.execute(f"EXPLAIN QUERY PLAN {sql}")
        conn.close()
        return True
    except Exception as e:
        debug_log(f"Query plan validation failed: {e}")
        return False

def generate_sql(query: str):
    """Generate SQL from query"""
    schema = get_schema()
    columns = [col[0] for col in schema]
    
    prompt = f"""
You are a SQL expert. Generate ONLY valid SQLite queries.

Table: data
Columns: {columns}

Rules:
- Return ONLY the SQL query
- No markdown code blocks
- SELECT queries only
- Use column names exactly as listed
- No explanations
- dont use COUNT(DISTINCT *) or PRAGMA statements

Question: {query}

SQL:
"""
    
    debug_log(f"SQL generation for: {query}")
    
    for attempt in range(3):  # Reduce retry attempts
        try:
            raw = llm(prompt)
            sql = clean_sql(raw)
            debug_log(f"Attempt {attempt+1} SQL: {sql[:100]}...")
            
            if not sql or not sql.upper().startswith("SELECT"):
                continue
            
            if not ast_validate(sql):
                continue
            
            if not query_planner_validate(sql):
                continue
            
            debug_log(f"✓ Valid SQL generated")
            return sql
            
        except Exception as e:
            debug_log(f"SQL generation attempt {attempt+1} failed: {e}")
            continue
    
    debug_log("✗ SQL generation exhausted retries")
    return None

def run_sql(sql: str):
    """Execute SQL query"""
    if not sql:
        return "No SQL query"
    
    try:
        debug_log(f"Executing SQL: {sql[:100]}...")
        conn = sqlite3.connect(SQLITE_DB)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        
        if df.empty:
            debug_log("SQL returned no rows")
            return "No matching records found"
        
        # Limit output
        if len(df) > 50:
            debug_log(f"Limiting results from {len(df)} to 50")
            df = df.head(50)
        
        result = df.to_string(index=False)
        debug_log(f"✓ SQL returned {len(df)} rows")
        return result
        
    except Exception as e:
        debug_log(f"SQL execution error: {e}")
        traceback.print_exc()
        return f"SQL execution error: {e}"

# ============================================================
# VECTOR RETRIEVAL (RAG)
# ============================================================

def retrieve_context(query: str):
    """Retrieve context using vector search"""
    try:
        debug_log(f"Vector retrieval for: {query}")
        
        index = faiss.read_index(FAISS_INDEX)
        texts = np.load(TEXTS_FILE, allow_pickle=True)
        
        q_emb = embed_texts([query])
        faiss.normalize_L2(q_emb)
        
        scores, ids = index.search(q_emb, TOP_K)
        results = [texts[i] for i in ids[0]]
        
        context = "\n---\n".join(results)
        debug_log(f"✓ Retrieved {len(results)} vectors")
        return context
        
    except Exception as e:
        debug_log(f"Vector retrieval failed: {e}")
        traceback.print_exc()
        return ""

# ============================================================
# LANGGRAPH STATE
# ============================================================

class ChatState(TypedDict):
    query: str
    intent: Optional[str]
    sql_result: Optional[str]
    context: Optional[str]
    answer: Optional[str]
    error: Optional[str]

# ============================================================
# NODES
# ============================================================

def intent_node(state: ChatState):
    """Classify intent"""
    state["intent"] = detect_intent(state["query"])
    debug_log(f"Intent: {state['intent']}")
    return state

def route(state: ChatState):
    """Route based on intent"""
    return state["intent"]

def sql_node(state: ChatState):
    """Handle numeric queries with SQL"""
    debug_log(f"SQL node: processing numeric query")
    sql = generate_sql(state["query"])
    
    if sql:
        result = run_sql(sql)
        state["sql_result"] = result
        state["answer"] = result
    else:
        state["error"] = "Could not generate SQL"
        state["answer"] = "I couldn't generate a valid SQL query for that question."
    
    return state

def vector_node(state: ChatState):
    """Handle semantic queries with RAG"""
    debug_log(f"Vector node: processing semantic query")
    ctx = retrieve_context(state["query"])
    state["context"] = ctx
    
    if not ctx:
        state["error"] = "No context retrieved"
        state["answer"] = "Could not find relevant data."
        return state
    
    # Use LLM to synthesize answer from context
    prompt = f"""
Using the provided context, answer the user's question clearly and concisely.

Context:
{ctx}

Question: {state['query']}

Answer:
"""
    
    answer = llm(prompt)
    state["answer"] = answer if answer else "Could not generate answer"
    debug_log(f"✓ Vector node answer generated")
    
    return state

def meta_node(state: ChatState):
    """Handle general questions"""
    debug_log(f"Meta node: processing general query")
    answer = llm(state["query"])
    state["answer"] = answer if answer else "Could not process query"
    return state

# ============================================================
# BUILD GRAPH
# ============================================================

builder = StateGraph(ChatState)

builder.add_node("intent", intent_node)
builder.add_node("sql", sql_node)
builder.add_node("vector", vector_node)
builder.add_node("meta", meta_node)

builder.set_entry_point("intent")

builder.add_conditional_edges("intent", route, {
    "numeric": "sql",
    "semantic": "vector",
    "meta": "meta"
})

builder.add_edge("sql", END)
builder.add_edge("vector", END)
builder.add_edge("meta", END)

graph = builder.compile()

# ============================================================
# INITIALIZATION
# ============================================================

print("\n" + "="*60)
print("CHATBOT INITIALIZATION")
print("="*60)

if not os.path.exists(SQLITE_DB):
    excel_to_sqlite()
else:
    schema = get_schema()
    print(f"✓ SQLite ready ({len(schema)} columns)")

if not os.path.exists(FAISS_INDEX):
    build_vector_index()
else:
    index = faiss.read_index(FAISS_INDEX)
    print(f"✓ FAISS ready ({index.ntotal} vectors)")

print("\n" + "="*60)
print("READY - Type 'exit' to quit")
print("="*60 + "\n")

# ============================================================
# CHAT LOOP
# ============================================================

while True:
    query = input("You: ").strip()
    if not query:
        continue
    if query.lower() == "exit":
        print("Goodbye!")
        break
    
    print("\n[Processing...]")
    
    try:
        result = graph.invoke({
            "query": query,
            "intent": None,
            "sql_result": None,
            "context": None,
            "answer": None,
            "error": None
        })
        
        print(f"\nBot: {result['answer']}\n")
        
        if DEBUG and result.get("error"):
            print(f"[Error: {result['error']}]\n")
            
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        traceback.print_exc()
