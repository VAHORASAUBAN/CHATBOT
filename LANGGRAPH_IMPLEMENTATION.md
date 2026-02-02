# LangGraph State Machine Implementation

## Overview
The chatbot has been refactored to use **LangGraph**, a framework for building stateful, multi-step applications with language models. This provides better efficiency, modularity, and maintainability.

## Key Changes

### 1. **State Schema** (ChatState)
Defined a TypedDict that represents the complete state throughout the workflow:
```python
class ChatState(TypedDict):
    query: str                           # User query
    intent: str                          # Classified intent
    sql: Optional[str]                   # Generated SQL
    sql_result: Optional[any]            # SQL execution result
    semantic_result: Optional[str]       # RAG retrieved context
    final_answer: Optional[str]          # Final response
    error: Optional[str]                 # Any errors
    metadata: dict                       # Tracking metadata
```

### 2. **Node-Based Architecture**
Instead of if-else logic, each task is a dedicated node:

| Node | Purpose |
|------|---------|
| `classify_intent` | Determines query type (entry point) |
| `numeric` | Handles calculations using SQL |
| `filter` | Handles filtered data retrieval |
| `semantic` | Handles RAG-based semantic search |
| `comparison` | Handles comparison queries with RAG |
| `general` | Handles off-topic questions |

### 3. **Conditional Routing**
Uses a `router()` function to dynamically route based on intent:
```python
classify_intent → [numeric|filter|semantic|comparison|general] → END
```

### 4. **Benefits**

#### **Efficiency**
- ✅ State passed through workflow (no redundant computations)
- ✅ Lazy evaluation - only relevant nodes execute
- ✅ Better resource management

#### **Maintainability**
- ✅ Each node is independent and testable
- ✅ Easy to add new nodes/intents
- ✅ Clear state flow visualization
- ✅ Built-in debugging (metadata tracking)

#### **Scalability**
- ✅ Can add human-in-the-loop nodes
- ✅ Supports conditional retry logic
- ✅ Can add memory/persistence layers

### 5. **State Tracking & Metadata**
Each query execution includes metadata:
- `start_time` - When query processing started
- `intent_classified_at` - Classification timestamp
- `route` - Which execution path was taken (e.g., "numeric_sql", "semantic_rag")

### 6. **Error Handling**
Errors are captured in state rather than thrown, providing:
- Graceful degradation
- Better logging
- Fallback options

## Workflow Example

```
User: "How many unique suppliers?"
    ↓
[classify_intent] → Intent: "numeric"
    ↓
[numeric node] → 
    - Generate SQL
    - Execute SQL
    - Store result in state
    ↓
[END] → Return final state with answer
```

## Usage
The chatbot maintains the same user interface but with improved internal architecture:

```bash
You: How many unique suppliers?
[Intent: numeric]
Bot: 45
[Route: numeric_sql]
```

## Dependencies
Install LangGraph:
```bash
pip install langgraph
```

## Migration Notes
- Old `classify_intent()` function still exists but is now deprecated
- Old `build_rag_chain()` is kept for reference but not used
- All functionality preserved; only internal architecture changed

## Future Enhancements
- Add caching nodes for repeated queries
- Implement multi-turn conversation state
- Add human-in-the-loop for uncertain cases
- Parallel node execution for independent operations
- Persistence of conversation history
