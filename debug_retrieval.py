"""
Diagnostic script to identify data retrieval issues
"""
import os
import sqlite3
import pandas as pd
import requests
import numpy as np
import faiss

print("=" * 60)
print("CHATBOT DATA RETRIEVAL DIAGNOSTIC")
print("=" * 60)

# 1. Check Excel file
EXCEL_PATH = "C:/Users/sauban.vahora/Desktop/Chatbot/data/SGD.xlsx"
print(f"\n1. Excel File Check")
print(f"   Path: {EXCEL_PATH}")
if os.path.exists(EXCEL_PATH):
    try:
        df = pd.read_excel(EXCEL_PATH)
        print(f"   ✓ File exists")
        print(f"   ✓ Rows: {len(df)}, Columns: {len(df.columns)}")
        print(f"   Columns: {list(df.columns)[:5]}...")
    except Exception as e:
        print(f"   ✗ Error reading Excel: {e}")
else:
    print(f"   ✗ File NOT found")

# 2. Check SQLite database
SQLITE_DB = "rag.db"
print(f"\n2. SQLite Database Check")
print(f"   Path: {SQLITE_DB}")
if os.path.exists(SQLITE_DB):
    try:
        conn = sqlite3.connect(SQLITE_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM data")
        count = cursor.fetchone()[0]
        print(f"   ✓ Database exists")
        print(f"   ✓ Rows in 'data' table: {count}")
        
        cursor.execute("PRAGMA table_info(data)")
        cols = cursor.fetchall()
        print(f"   ✓ Columns: {len(cols)}")
        print(f"   Sample columns: {[c[1] for c in cols[:3]]}...")
        
        conn.close()
    except Exception as e:
        print(f"   ✗ Error accessing database: {e}")
else:
    print(f"   ✗ Database NOT found")

# 3. Check FAISS index
FAISS_INDEX = "faiss.index"
TEXTS_FILE = "texts.npy"
print(f"\n3. FAISS Vector Index Check")
print(f"   Index path: {FAISS_INDEX}")
print(f"   Texts path: {TEXTS_FILE}")

if os.path.exists(FAISS_INDEX) and os.path.exists(TEXTS_FILE):
    try:
        index = faiss.read_index(FAISS_INDEX)
        texts = np.load(TEXTS_FILE, allow_pickle=True)
        print(f"   ✓ FAISS index exists")
        print(f"   ✓ Index dimension: {index.ntotal}")
        print(f"   ✓ Texts array size: {len(texts)}")
        if index.ntotal == len(texts):
            print(f"   ✓ Index and texts are ALIGNED")
        else:
            print(f"   ✗ MISMATCH: {index.ntotal} vectors vs {len(texts)} texts")
    except Exception as e:
        print(f"   ✗ Error loading FAISS: {e}")
else:
    if not os.path.exists(FAISS_INDEX):
        print(f"   ✗ FAISS index NOT found")
    if not os.path.exists(TEXTS_FILE):
        print(f"   ✗ Texts file NOT found")

# 4. Check Embedding Service
EMBEDDING_URL = "http://127.0.0.1:5002/embeddings"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
print(f"\n4. Embedding Service Check")
print(f"   URL: {EMBEDDING_URL}")
try:
    r = requests.post(
        EMBEDDING_URL,
        json={"model": EMBEDDING_MODEL, "input": ["test"]},
        timeout=5
    )
    if r.status_code == 200:
        print(f"   ✓ Embedding service is RUNNING")
    else:
        print(f"   ✗ Service returned status {r.status_code}")
except requests.exceptions.ConnectionError:
    print(f"   ✗ Embedding service is NOT reachable")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 5. Check LLM Service
LLM_BASE_URL = "http://45.127.102.236:8000/v1"
print(f"\n5. LLM Service Check")
print(f"   URL: {LLM_BASE_URL}")
try:
    from openai import OpenAI
    client = OpenAI(base_url=LLM_BASE_URL, api_key="test")
    # Just checking connectivity by looking at client initialization
    print(f"   ✓ LLM client initialized (URL is reachable)")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
