# import pandas as pd
# import sqlite3
# import requests
# import json
# from typing import List, TypedDict, Optional, Literal
# from datetime import datetime
# from langchain_core.documents import Document
# from langchain_community.vectorstores import FAISS
# from langchain_core.embeddings import Embeddings
# from langchain_core.language_models.llms import LLM
# from langchain_core.prompts import PromptTemplate
# # from langchain.chains import RetrievalQA
# from langgraph.graph import StateGraph, END



# # =====================================================
# # CONFIG
# # =====================================================

# EXCEL_PATH = "C:/Users/sauban.vahora/Desktop/Chatbot/data/SGD.xlsx"
# SQLITE_DB_PATH = "sgd.db"
# SQLITE_TABLE = "sgd_data"

# LLM_BASE_URL = "http://45.127.102.236:8000/v1"
# LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# EMBEDDING_URL = "http://127.0.0.1:5002/embeddings"
# EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# FAISS_INDEX_PATH = "faiss_index_sgd"
# TOP_K = 15  # Increased for broader context retrieval
# MMR_LAMBDA_MULT = 0.6  # For diversity in results

# # =====================================================
# # STATE MACHINE STATE SCHEMA
# # =====================================================

# class ChatState(TypedDict):
#     """State schema for chatbot workflow"""
#     query: str
#     intent: Literal["numeric", "comparison", "filter", "semantic", "general"]
#     sql: Optional[str]
#     sql_result: Optional[any]
#     semantic_result: Optional[str]
#     final_answer: Optional[str]
#     error: Optional[str]
#     metadata: dict  # For tracking and debugging

# # =====================================================
# # CUSTOM EMBEDDINGS
# # =====================================================

# class CustomEmbedding(Embeddings):
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         r = requests.post(
#             EMBEDDING_URL,
#             json={"model": EMBEDDING_MODEL, "input": texts}
#         )
#         r.raise_for_status()
#         return [x["embedding"] for x in r.json()["data"]]

#     def embed_query(self, text: str) -> List[float]:
#         r = requests.post(
#             EMBEDDING_URL,
#             json={"model": EMBEDDING_MODEL, "input": text}
#         )
#         r.raise_for_status()
#         return r.json()["data"][0]["embedding"]

# # =====================================================
# # CUSTOM LLM
# # =====================================================

# class CustomLLM(LLM):
#     @property
#     def _llm_type(self) -> str:
#         return "custom_llama"

#     def _call(self, prompt: str, stop=None) -> str:
#         r = requests.post(
#             f"{LLM_BASE_URL}/chat/completions",
#             json={
#                 "model": LLM_MODEL,
#                 "messages": [
#                     {"role": "system", "content": "You are a highly reliable assistant helping users understand documents. "
#                     "Provide concise answers and be helpful."   
#                     "Answer only using provided context. If the answer is not present, say so clearly."
#                     },
#                     {"role": "user", "content": prompt}
#                 ],
#                 "temperature": 0,
#                 "max_tokens": 1024  # Increased from 512 for more detailed responses
#             }
#         )
#         r.raise_for_status()
#         return r.json()["choices"][0]["message"]["content"]

# # =====================================================
# # LOAD DATA
# # =====================================================

# def load_excel():
#     return pd.read_excel(EXCEL_PATH).fillna("")

# # =====================================================
# # SQLITE SETUP
# # =====================================================

# def setup_sqlite(df: pd.DataFrame):
#     conn = sqlite3.connect(SQLITE_DB_PATH)
#     df.to_sql(SQLITE_TABLE, conn, if_exists="replace", index=False)
#     conn.close()

# # =====================================================
# # FAISS SETUP
# # =====================================================

# def setup_faiss(df: pd.DataFrame):
#     """Setup FAISS with enhanced chunking strategy"""
#     docs = []
    
#     # Improved chunking: create semantic chunks with metadata
#     for idx, row in df.iterrows():
#         # Full row as primary chunk
#         full_text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
#         doc = Document(
#             page_content=full_text,
#             metadata={"row_id": idx, "chunk_type": "full_row"}
#         )
#         docs.append(doc)
        
#         # Create sub-chunks for important fields (top 5 columns)
#         if len(df.columns) > 5:
#             important_cols = df.columns[:5]
#             summary_text = "\n".join([f"{col}: {row[col]}" for col in important_cols])
#             doc_summary = Document(
#                 page_content=summary_text,
#                 metadata={"row_id": idx, "chunk_type": "summary"}
#             )
#             docs.append(doc_summary)

#     embeddings = CustomEmbedding()

#     try:
#         return FAISS.load_local(
#             FAISS_INDEX_PATH,
#             embeddings,
#             allow_dangerous_deserialization=True
#         )
#     except:
#         vs = FAISS.from_documents(docs, embeddings)
#         vs.save_local(FAISS_INDEX_PATH)
#         return vs

# # =====================================================
# # INTENT CLASSIFICATION
# # =====================================================

# def classify_intent(query: str, llm: CustomLLM) -> str:
#     """Classify query intent with more granular categories"""
#     prompt = f"""
# Classify the question into exactly ONE category:

# numeric     : calculations, counts, totals, averages, aggregations
# comparison  : compare, difference between, versus
# filter      : show/list with conditions, where, find rows and columns
# semantic    : explanations, trends, summaries, analysis
# general     : greetings, help, meta questions

# Return ONLY one word.

# Question:
# {query}
# """
#     result = llm._call(prompt).strip().lower()
#     # Validate result is one of the categories
#     valid_intents = ["numeric", "comparison", "filter", "semantic", "general"]
#     return result if result in valid_intents else "semantic"

# # =====================================================
# # NUMERIC QUERY → SQL
# # =====================================================

# def generate_sql(query: str, df: pd.DataFrame, llm: CustomLLM) -> str:
#     """Generate SQL with support for complex queries"""
#     prompt = f"""
# You are a SQL execution assistant specialized in numerical operations.
# Table name: {SQLITE_TABLE}
# Columns: {list(df.columns)}

# Generate a valid SQLite query for this question:
# {query}

# Return ONLY the SQL query with NO markdown, NO code blocks."""
#     sql = llm._call(prompt).strip()
#     # Clean up markdown formatting
#     sql = sql.replace("```sql", "").replace("```", "").strip()
#     sql = "\n".join(line for line in sql.split("\n") if line.strip())
#     return sql

# def execute_sql(sql: str, fetch_all: bool = False) -> any:
#     """Execute SQL query with flexible result retrieval"""
#     try:
#         sql = sql.strip()
#         sql = sql.replace("```sql", "").replace("```", "").strip()
#         sql = "\n".join(line for line in sql.split("\n") if line.strip())
        
#         if not sql.upper().startswith("SELECT"):
#             return None
        
#         conn = sqlite3.connect(SQLITE_DB_PATH)
#         conn.row_factory = sqlite3.Row
#         cursor = conn.cursor()
#         cursor.execute(sql)
        
#         if fetch_all:
#             results = cursor.fetchall()
#             conn.close()
#             return results if results else []
#         else:
#             result = cursor.fetchone()
#             conn.close()
#             return result[0] if result else None
#     except Exception as e:
#         return None

# # =====================================================
# # SEMANTIC (RAG)
# # =====================================================

# def build_rag_chain(vectorstore, llm):
#     """Build RAG chain with MMR retrieval for diversity"""
#     # Use MMR (Maximal Marginal Relevance) for better diversity in results
#     retriever = vectorstore.as_retriever(
#         search_type="mmr",
#         search_kwargs={
#             "k": TOP_K,
#             "lambda_mult": MMR_LAMBDA_MULT,  # Balance relevance vs diversity
#             "fetch_k": TOP_K * 2  # Fetch more candidates to rerank
#         }
#     )

#     prompt = PromptTemplate(
#         input_variables=["context", "question"],
#         template="""
# Answer comprehensively using ALL relevant context provided below.
# Extract and synthesize information from multiple sources.
# Include details, numbers, and relationships found in the context.

# Context:
# {context}

# Question:
# {question}

# Comprehensive Answer:
# """
#     )

#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type="stuff",
#         chain_type_kwargs={"prompt": prompt}
#     )

# # =====================================================
# # GENERAL (LLM ONLY)
# # =====================================================

# def handle_general(query: str, llm: CustomLLM) -> str:
#     """Handle general queries"""
#     prompt = f"""
# Respond to the user helpfully and clearly. Do NOT use any dataset.
# Be concise but informative.

# Question:
# {query}
# """
#     return llm._call(prompt)

# # =====================================================
# # LANGGRAPH NODES
# # =====================================================

# def node_classify_intent(state: ChatState, llm: CustomLLM, df: pd.DataFrame) -> ChatState:
#     """Node: Classify query intent"""
#     prompt = f"""
# Classify the question into exactly ONE category:

# numeric     : calculations, counts, totals, averages
# comparison  : compare, difference between, versus
# filter      : show/list with conditions, where, find rows and columns
# semantic    : explanations, trends, summaries, analysis
# general     : greetings, help, meta questions

# Return ONLY one word.

# Question:
# {state['query']}
# """
#     result = llm._call(prompt).strip().lower()
#     valid_intents = ["numeric", "comparison", "filter", "semantic", "general"]
#     intent = result if result in valid_intents else "semantic"
    
#     state["intent"] = intent
#     state["metadata"]["intent_classified_at"] = datetime.now().isoformat()
#     return state


# def node_handle_numeric(state: ChatState, llm: CustomLLM, df: pd.DataFrame) -> ChatState:
#     """Node: Handle numeric queries with SQL"""
    
#     sql = generate_sql(state["query"], df, llm)
#     state["sql"] = sql
    
#     if sql and "SELECT" in sql.upper():
#         result = execute_sql(sql)
#         state["sql_result"] = result
#         state["final_answer"] = str(result) if result is not None else "No results found"
#         state["metadata"]["route"] = "numeric_sql"
#     else:
#         state["error"] = "Could not generate valid SQL"
#         state["metadata"]["route"] = "numeric_failed"
    
#     return state


# def node_handle_filter(state: ChatState, llm: CustomLLM, df: pd.DataFrame) -> ChatState:
#     """Node: Handle filter queries"""
#     sql = generate_sql(state["query"], df, llm)
#     state["sql"] = sql
    
#     if sql and "SELECT" in sql.upper():
#         results = execute_sql(sql, fetch_all=True)
#         if isinstance(results, list):
#             state["sql_result"] = results
#             result_text = f"Found {len(results)} matching records:\n"
#             for r in results[:5]:
#                 result_text += f"  {r}\n"
#             if len(results) > 5:
#                 result_text += f"... and {len(results) - 5} more records"
#             state["final_answer"] = result_text
#             state["metadata"]["route"] = "filter_success"
#         else:
#             state["final_answer"] = "No matching records found"
#             state["metadata"]["route"] = "filter_empty"
#     else:
#         state["error"] = "Could not generate valid SQL for filter"
#         state["metadata"]["route"] = "filter_failed"
    
#     return state


# def node_handle_semantic(state: ChatState, vectorstore, llm: CustomLLM) -> ChatState:
#     """Node: Handle semantic queries with RAG"""
#     try:
#         retriever = vectorstore.as_retriever(
#             search_type="mmr",
#             search_kwargs={
#                 "k": TOP_K,
#                 "lambda_mult": MMR_LAMBDA_MULT,
#                 "fetch_k": TOP_K * 2
#             }
#         )
#         docs = retriever.get_relevant_documents(state["query"])
#         context = "\n".join([d.page_content for d in docs])
#         state["semantic_result"] = context
        
#         # Use LLM to generate answer from context
#         prompt = f"""
# Answer comprehensively using the context below.
# Extract and synthesize all relevant information.

# Context:
# {context}

# Question:
# {state['query']}

# Answer:"""
#         answer = llm._call(prompt)
#         state["final_answer"] = answer
#         state["metadata"]["route"] = "semantic_rag"
#     except Exception as e:
#         state["error"] = f"Semantic search failed: {str(e)}"
#         state["metadata"]["route"] = "semantic_failed"
    
#     return state


# def node_handle_general(state: ChatState, llm: CustomLLM) -> ChatState:
#     """Node: Handle general queries"""
#     prompt = f"""
# Respond helpfully and clearly. Do NOT use any dataset.
# Be concise but informative.

# Question:
# {state['query']}
# """
#     answer = llm._call(prompt)
#     state["final_answer"] = answer
#     state["metadata"]["route"] = "general"
#     return state


# def router(state: ChatState) -> Literal["numeric", "filter", "semantic", "comparison", "general"]:
#     """Router to determine next node based on intent"""
#     return state["intent"]

# # =====================================================
# # BUILD LANGGRAPH WORKFLOW
# # =====================================================

# def build_chatbot_graph(df: pd.DataFrame, vectorstore, llm: CustomLLM) -> StateGraph:
#     """Build the LangGraph state machine for chatbot workflow"""
    
#     # Create graph
#     workflow = StateGraph(ChatState)
    
#     # Add nodes
#     workflow.add_node(
#         "classify_intent",
#         lambda state: node_classify_intent(state, llm, df)
#     )
#     workflow.add_node(
#         "numeric",
#         lambda state: node_handle_numeric(state, llm, df)
#     )
#     workflow.add_node(
#         "filter",
#         lambda state: node_handle_filter(state, llm, df)
#     )
#     workflow.add_node(
#         "semantic",
#         lambda state: node_handle_semantic(state, vectorstore, llm)
#     )
#     workflow.add_node(
#         "comparison",
#         lambda state: node_handle_semantic(state, vectorstore, llm)
#     )
#     workflow.add_node(
#         "general",
#         lambda state: node_handle_general(state, llm)
#     )
    
#     # Set entry point
#     workflow.set_entry_point("classify_intent")
    
#     # Add conditional edges from classify_intent
#     workflow.add_conditional_edges(
#         "classify_intent",
#         router,
#         {
#             "numeric": "numeric",
#             "filter": "filter",
#             "semantic": "semantic",
#             "comparison": "comparison",
#             "general": "general"
#         }
#     )
    
#     # All nodes end the workflow
#     workflow.add_edge("numeric", END)
#     workflow.add_edge("filter", END)
#     workflow.add_edge("semantic", END)
#     workflow.add_edge("comparison", END)
#     workflow.add_edge("general", END)
    
#     return workflow.compile()

# def main():
#     print("📄 Loading dataset...")
#     df = load_excel()
#     print(f"   ✓ Loaded {len(df)} rows, {len(df.columns)} columns")

#     print("🗄️ Setting up SQLite...")
#     setup_sqlite(df)
#     print("   ✓ SQLite ready")

#     print("🧠 Setting up FAISS with enhanced chunking...")
#     vectorstore = setup_faiss(df)
#     print(f"   ✓ FAISS ready (TOP_K={TOP_K}, MMR enabled)")

#     llm = CustomLLM()
    
#     print("🔗 Building LangGraph state machine...")
#     graph = build_chatbot_graph(df, vectorstore, llm)
#     print("   ✓ State machine compiled")

#     print("\n🤖 Hybrid SQL + RAG Chatbot Ready (type 'exit' to quit)")
#     print("📊 Now powered by: LangGraph state machine, MMR retrieval, multi-intent routing\n")

#     while True:
#         query = input("You: ").strip()
#         if not query:
#             continue
#         if query.lower() in ["exit", "quit"]:
#             print("Goodbye!")
#             break

#         # Initialize state
#         initial_state: ChatState = {
#             "query": query,
#             "intent": "semantic",
#             "sql": None,
#             "sql_result": None,
#             "semantic_result": None,
#             "final_answer": None,
#             "error": None,
#             "metadata": {"start_time": datetime.now().isoformat()}
#         }
        
#         try:
#             # Execute workflow
#             result = graph.invoke(initial_state)
            
#             # Display result
#             if result.get("error"):
#                 print(f"\n⚠️  {result['error']}")
#             else:
#                 print(f"\n[Intent: {result['intent']}]")
#                 if result.get("final_answer"):
#                     print(f"Bot: {result['final_answer']}")
            
#             print(f"[Route: {result['metadata'].get('route', 'unknown')}]\n")
            
#         except Exception as e:
#             print(f"\n❌ Error: {str(e)}\n")

# # =====================================================
# # ENTRY
# # =====================================================

# if __name__ == "__main__":
#     main()

# import pandas as pd
# import sqlite3
# import faiss
# import numpy as np
# import json
# import os
# import requests
# from typing import TypedDict, Optional

# from langgraph.graph import StateGraph, END
# from openai import OpenAI

# # ============================================================
# # CONFIG
# # ============================================================

# EXCEL_PATH = "C:/Users/sauban.vahora/Desktop/Chatbot/data/SGD.xlsx"
# SQLITE_DB = "rag.db"
# FAISS_INDEX = "faiss.index"
# TEXTS_FILE = "texts.npy"

# LLM_BASE_URL = "http://45.127.102.236:8000/v1"
# LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# EMBEDDING_URL = "http://127.0.0.1:5002/embeddings"
# EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# TOP_K = 5

# client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")

# # ============================================================
# # EMBEDDING CLIENT
# # ============================================================

# def embed_texts(texts):
#     r = requests.post(
#         EMBEDDING_URL,
#         json={"model": EMBEDDING_MODEL, "input": texts},
#         timeout=60
#     )
#     data = r.json()["data"]
#     vecs = [d["embedding"] for d in data]
#     return np.array(vecs, dtype=np.float32)

# # ============================================================
# # SMART EXCEL → SQLITE INGESTION (NaT SAFE)
# # ============================================================

# def infer_sqlite_type(dtype):
#     if pd.api.types.is_integer_dtype(dtype):
#         return "INTEGER"
#     if pd.api.types.is_float_dtype(dtype):
#         return "REAL"
#     if pd.api.types.is_bool_dtype(dtype):
#         return "INTEGER"
#     if pd.api.types.is_datetime64_any_dtype(dtype):
#         return "TEXT"
#     return "TEXT"


# def sanitize_value(x):
#     if pd.isna(x):
#         return None

#     if isinstance(x, pd.Timestamp):
#         return x.isoformat()

#     if isinstance(x, bool):
#         return int(x)

#     return x


# def excel_to_sqlite():
#     print("Building SQLite schema from Excel...")

#     df = pd.read_excel(EXCEL_PATH)

#     # clean column names
#     df.columns = [
#         col.strip()
#         .replace(" ", "_")
#         .replace("-", "_")
#         .replace("/", "_")
#         for col in df.columns
#     ]

#     # sanitize dataframe
#     df = df.applymap(sanitize_value)

#     # infer schema
#     schema = {
#         col: infer_sqlite_type(df[col].dtype)
#         for col in df.columns
#     }

#     conn = sqlite3.connect(SQLITE_DB)
#     cur = conn.cursor()

#     cur.execute("DROP TABLE IF EXISTS data")

#     columns_sql = ", ".join(
#         [f'"{col}" {dtype}' for col, dtype in schema.items()]
#     )

#     create_sql = f"CREATE TABLE data ({columns_sql})"
#     cur.execute(create_sql)

#     placeholders = ", ".join(["?"] * len(df.columns))
#     insert_sql = f'INSERT INTO data VALUES ({placeholders})'

#     cur.executemany(insert_sql, df.values.tolist())

#     conn.commit()
#     conn.close()

#     print("SQLite ready with sanitized data.")


# def row_to_text(row):
#     return " | ".join([f"{col}: {row[col]}" for col in row.index])

# def build_vector_index():
#     df = pd.read_excel(EXCEL_PATH)
#     texts = df.apply(row_to_text, axis=1).tolist()

#     embeddings = embed_texts(texts)
#     faiss.normalize_L2(embeddings)

#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dim)
#     index.add(embeddings)

#     faiss.write_index(index, FAISS_INDEX)
#     np.save(TEXTS_FILE, np.array(texts, dtype=object))

# # ============================================================
# # LLM CALL
# # ============================================================

# def llm(prompt: str) -> str:
#     resp = client.chat.completions.create(
#         model=LLM_MODEL,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0
#     )
#     return resp.choices[0].message.content.strip()

# # ============================================================
# # INTENT DETECTION
# # ============================================================

# # ============================================================
# # INTENT DETECTION (NO JSON)
# # ============================================================

# def detect_intent(query: str):
#     prompt = f"""
# Classify this query into ONE word only:

# numeric → math, aggregation, totals, counts
# semantic → meaning or explanation from data
# meta → general knowledge

# Answer with exactly one word:
# numeric OR semantic OR meta

# Query:
# {query}
# """

#     out = llm(prompt).lower().strip()

#     # hard guard against model rambling
#     if "numeric" in out:
#         return "numeric"
#     if "meta" in out:
#         return "meta"
#     return "semantic"


# # ============================================================
# # SQL ENGINE (AST + QUERY PLANNER SAFE)
# # ============================================================

# import sqlglot
# from sqlglot import exp


# def clean_sql(text: str) -> str:
#     text = text.strip()

#     if "```" in text:
#         parts = text.split("```")
#         if len(parts) >= 2:
#             text = parts[1]

#     return text.strip()


# def ast_validate(sql: str) -> bool:
#     try:
#         tree = sqlglot.parse_one(sql)

#         # must be SELECT
#         if not isinstance(tree, exp.Select):
#             return False

#         # enforce table whitelist
#         tables = {t.name.lower() for t in tree.find_all(exp.Table)}
#         if tables != {"data"}:
#             return False

#         # forbid sub-commands
#         forbidden = (
#             exp.Insert,
#             exp.Update,
#             exp.Delete,
#             exp.Drop,
#             exp.Alter,
#             exp.Create,
#         )

#         for node in tree.walk():
#             if isinstance(node, forbidden):
#                 return False

#         return True

#     except Exception:
#         return False


# def query_planner_validate(sql: str) -> bool:
#     try:
#         conn = sqlite3.connect(SQLITE_DB)
#         conn.execute(f"EXPLAIN QUERY PLAN {sql}")
#         conn.close()
#         return True
#     except Exception:
#         return False


# def generate_sql(query: str):
#     conn = sqlite3.connect(SQLITE_DB)
#     schema = conn.execute("PRAGMA table_info(data)").fetchall()
#     conn.close()

#     columns = [col[1] for col in schema]

#     prompt = f"""
# You are a SQLite expert.

# Table name: data
# Columns: {columns}

# Rules:
# - SELECT only
# - Use only table: data
# - No PRAGMA
# - No markdown
# - No explanation
# - Output SQL only

# User question:
# {query}
# """

#     for _ in range(5):  # retry loop
#         raw = llm(prompt)
#         sql = clean_sql(raw)

#         if not ast_validate(sql):
#             continue

#         if not query_planner_validate(sql):
#             continue

#         return sql

#     return "SELECT 'SQL generation failed' AS error"


# def run_sql(sql: str):
#     try:
#         conn = sqlite3.connect(SQLITE_DB)
#         df = pd.read_sql_query(sql, conn)
#         conn.close()

#         if df.empty:
#             return "No rows found."

#         # limit massive outputs
#         if len(df) > 50:
#             df = df.head(50)

#         return df.to_string(index=False)

#     except Exception as e:
#         return f"SQL execution error: {e}"



# # ============================================================
# # VECTOR RETRIEVAL
# # ============================================================

# def retrieve_context(query: str):
#     index = faiss.read_index(FAISS_INDEX)
#     texts = np.load(TEXTS_FILE, allow_pickle=True)

#     q_emb = embed_texts([query])
#     faiss.normalize_L2(q_emb)

#     scores, ids = index.search(q_emb, TOP_K)
#     results = [texts[i] for i in ids[0]]

#     return "\n".join(results)

# # ============================================================
# # LANGGRAPH STATE
# # ============================================================

# class ChatState(TypedDict):
#     query: str
#     intent: Optional[str]
#     sql_result: Optional[str]
#     context: Optional[str]
#     answer: Optional[str]

# # ============================================================
# # NODES
# # ============================================================

# def intent_node(state: ChatState):
#     state["intent"] = detect_intent(state["query"])
#     return state

# def route(state: ChatState):
#     return state["intent"]

# def sql_node(state: ChatState):
#     sql = generate_sql(state["query"])
#     result = run_sql(sql)
#     state["sql_result"] = result
#     return state

# def vector_node(state: ChatState):
#     ctx = retrieve_context(state["query"])
#     state["context"] = ctx
#     return state

# def meta_node(state: ChatState):
#     state["answer"] = llm(state["query"])
#     return state

# def merge_node(state: ChatState):
#     prompt = f"""
# User query:
# {state['query']}

# SQL result:
# {state.get('sql_result')}

# Context:
# {state.get('context')}

# Answer clearly.
# """
#     state["answer"] = llm(prompt)
#     return state

# # ============================================================
# # BUILD GRAPH
# # ============================================================

# builder = StateGraph(ChatState)

# builder.add_node("intent", intent_node)
# builder.add_node("sql", sql_node)
# builder.add_node("vector", vector_node)
# builder.add_node("meta", meta_node)
# builder.add_node("merge", merge_node)

# builder.set_entry_point("intent")

# builder.add_conditional_edges("intent", route, {
#     "numeric": "sql",
#     "semantic": "vector",
#     "meta": "meta"
# })

# builder.add_edge("sql", "merge")
# builder.add_edge("vector", "merge")
# builder.add_edge("meta", END)
# builder.add_edge("merge", END)

# graph = builder.compile()

# # ============================================================
# # INITIAL BUILD
# # ============================================================

# if not os.path.exists(SQLITE_DB):
#     excel_to_sqlite()

# if not os.path.exists(FAISS_INDEX):
#     build_vector_index()

# # ============================================================
# # CHAT LOOP
# # ============================================================

# print("\nHybrid RAG Chatbot Ready. Type 'exit' to quit.\n")

# while True:
#     q = input("You: ")
#     if q.lower() == "exit":
#         break

#     result = graph.invoke({
#         "query": q,
#         "intent": None,
#         "sql_result": None,
#         "context": None,
#         "answer": None
#     })

#     print("\nBot:", result["answer"], "\n")
import pandas as pd
import sqlite3
import requests
import json
import re

# =====================================================
# CONFIG
# =====================================================

EXCEL_PATH = "C:/Users/sauban.vahora/Desktop/Chatbot/data/SGD.xlsx"
DB_PATH = "database.db"

LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LLM_BASE = "http://45.127.102.236:8000/v1"

TABLE_NAME = "sgd_data"

# =====================================================
# STEP 1 — LOAD EXCEL → SQLITE
# =====================================================

def load_excel_to_sqlite():
    df = pd.read_excel(EXCEL_PATH)
    df.columns = [c.replace(" ", "_").replace("-", "_") for c in df.columns]

    conn = sqlite3.connect(DB_PATH)
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.close()

    print("Excel loaded into SQLite successfully.")

# =====================================================
# STEP 2 — LLM CALL
# =====================================================

def call_llm(prompt):
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert SQL generator."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    response = requests.post(
        f"{LLM_BASE}/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=120
    )

    return response.json()["choices"][0]["message"]["content"]

# =====================================================
# STEP 3 — NLP → SQL
# =====================================================

def generate_sql(user_query):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"PRAGMA table_info({TABLE_NAME});")
    columns = [col[1] for col in cursor.fetchall()]
    conn.close()

    schema = ", ".join(columns)

    prompt = f"""
Convert the user question into a valid SQL query.

Table name: {TABLE_NAME}
Columns: {schema}

Rules:
- Only output SQL
- No explanation
- Use correct column names
- SQLite syntax only

User question:
{user_query}
"""

    sql = call_llm(prompt)

    # clean markdown formatting if any
    sql = re.sub(r"```sql|```", "", sql).strip()

    return sql

# =====================================================
# STEP 4 — EXECUTE SQL
# =====================================================

def execute_sql(sql):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(sql, conn)
    except Exception as e:
        conn.close()
        return None, str(e)

    conn.close()
    return df, None

# =====================================================
# STEP 5 — FINAL RESPONSE GENERATION
# =====================================================

def generate_answer(user_query, df):
    if df.empty:
        return "No results found."

    data_preview = df.head(20).to_string(index=False)

    prompt = f"""
User question:
{user_query}

SQL result:
{data_preview}

Provide a concise answer based on the SQL result."""

    return call_llm(prompt)

# =====================================================
# CHAT LOOP
# =====================================================

def chat():
    print("RAG SQL Chatbot ready. Type 'exit' to quit.\n")

    while True:
        user_query = input("You: ")

        if user_query.lower() == "exit":
            break

        sql = generate_sql(user_query)
        print("\nGenerated SQL:")
        print(sql)

        df, error = execute_sql(sql)

        if error:
            print("\nSQL Error:", error)
            continue

        answer = generate_answer(user_query, df)
        print("\nBot:", answer)
        print("-" * 60)

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    load_excel_to_sqlite()
    chat()
