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
import sqlite3
import requests
import json
from typing import List

from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# =====================================================
# CONFIG
# =====================================================

EXCEL_PATH = "C:/Users/sauban.vahora/Desktop/Chatbot/data/SGD.xlsx"
SQLITE_DB_PATH = "sgd.db"
SQLITE_TABLE = "sgd_data"

LLM_BASE_URL = "http://45.127.102.236:8000/v1"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

EMBEDDING_URL = "http://127.0.0.1:5002/embeddings"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

FAISS_INDEX_PATH = "faiss_index_sgd"
TOP_K = 5

# =====================================================
# CUSTOM EMBEDDINGS
# =====================================================

class CustomEmbedding(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        r = requests.post(
            EMBEDDING_URL,
            json={"model": EMBEDDING_MODEL, "input": texts}
        )
        r.raise_for_status()
        return [x["embedding"] for x in r.json()["data"]]

    def embed_query(self, text: str) -> List[float]:
        r = requests.post(
            EMBEDDING_URL,
            json={"model": EMBEDDING_MODEL, "input": text}
        )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]

# =====================================================
# CUSTOM LLM
# =====================================================

class CustomLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom_llama"

    def _call(self, prompt: str, stop=None) -> str:
        r = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a precise data assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0,
                "max_tokens": 512
            }
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

# =====================================================
# LOAD DATA
# =====================================================

def load_excel():
    return pd.read_excel(EXCEL_PATH).fillna("")

# =====================================================
# SQLITE SETUP
# =====================================================

def setup_sqlite(df: pd.DataFrame):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df.to_sql(SQLITE_TABLE, conn, if_exists="replace", index=False)
    conn.close()

# =====================================================
# FAISS SETUP
# =====================================================

def setup_faiss(df: pd.DataFrame):
    docs = []
    for _, row in df.iterrows():
        text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append(Document(page_content=text))

    embeddings = CustomEmbedding()

    try:
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except:
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(FAISS_INDEX_PATH)
        return vs

# =====================================================
# INTENT CLASSIFICATION
# =====================================================

def classify_intent(query: str, llm: CustomLLM) -> str:
    prompt = f"""
Classify the question into exactly ONE category:

numeric  : calculations, counts, totals, averages
semantic : explanations, trends, summaries
general  : greetings, help, meta questions

Return ONLY one word.

Question:
{query}
"""
    return llm._call(prompt).strip().lower()

# =====================================================
# NUMERIC QUERY → SQL
# =====================================================

def generate_sql(query: str, df: pd.DataFrame, llm: CustomLLM) -> str:
    prompt = f"""
You are generating a SQLite SQL query.

Table name: {SQLITE_TABLE}
Columns: {list(df.columns)}

The query must:
- Use aggregation when required (AVG, SUM, COUNT, MIN, MAX)
- Return accurate value
- Be valid SQLite

Return SQL ONLY.

Question:
{query}
"""
    return llm._call(prompt).strip()

def execute_sql(sql: str):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(sql)
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

# =====================================================
# SEMANTIC (RAG)
# =====================================================

def build_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer using the context below.
If the answer is not present, say so.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

# =====================================================
# GENERAL (LLM ONLY)
# =====================================================

def handle_general(query: str, llm: CustomLLM):
    prompt = f"""
Respond to the user. Do NOT use any dataset.

Question:
{query}
"""
    return llm._call(prompt)

# =====================================================
# MAIN
# =====================================================

def main():
    print("📄 Loading dataset...")
    df = load_excel()

    print("🗄️ Setting up SQLite...")
    setup_sqlite(df)

    print("🧠 Setting up FAISS...")
    vectorstore = setup_faiss(df)

    llm = CustomLLM()
    rag_chain = build_rag_chain(vectorstore, llm)

    print("\n🤖 Hybrid SQL + RAG Chatbot Ready (type 'exit')\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        intent = classify_intent(query, llm)

        if intent == "numeric":
            sql = generate_sql(query, df, llm)
            result = execute_sql(sql)
            print("\nBot:", result, "\n")
            continue

        if intent == "semantic":
            print("\nBot:", rag_chain.invoke(query), "\n")
            continue

        print("\nBot:", handle_general(query, llm), "\n")

# =====================================================
# ENTRY
# =====================================================

if __name__ == "__main__":
    main()