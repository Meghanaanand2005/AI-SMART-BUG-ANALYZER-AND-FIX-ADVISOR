# AI Smart Bug Analyzer and Fix Advisor (AIBAFA) 🚀

AIBAFA is an intelligent, full-stack multi-agent bug analysis dashboard and log ingestion engine. Powered by a high-performance FastAPI backend and an embedded localized Retrieval-Augmented Generation (RAG) pipeline, the system parses error streams, conducts semantic similarity evaluations on stack traces, and enforces automated duplicate detection agents to streamline debugging and fix resolution.

---

## 🚀 Quickstart: Install, Launch, & Access

Run these commands sequentially in your terminal to set up your environment, spin up the backend server, and open up your interactive analytical workspace.

### 1. Install Dependencies
Ensure you have Python 3.10 or higher configured. Run this command in your project root directory (C:\Users\Admin\Desktop\AIBAFA):

```bash
pip install fastapi uvicorn pydantic langchain-core langchain-huggingface langchain-chroma sentence-transformers python-multipart
```

### 2. Launch the Analysis Engine
Kick off the local application server with active hot-reloading using Uvicorn:

```bash
uvicorn main:app --reload
```

### 3. Access the Dashboard
Once the terminal logs confirm the server is running successfully, open your web browser and jump directly to your interface:
👉 http://127.0.0.1:8000

---

## 🛠️ Tech Stack Matrix

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend Framework** | FastAPI / Uvicorn | High-concurrency ASGI web server and API endpoints |
| **Vector Database** | Chroma DB | Embedded vector store for tracking historical bugs and fixes |
| **Embedding Engine** | Hugging Face (all-MiniLM-L6-v2) | Localized log embedding translation without external API calls |
| **Agent Framework** | LangChain Core | Orchestrating multi-agent bug evaluation and fix suggestions |
| **Data Enforcement** | Pydantic v2 | Strict type-safety, log schema validation, and serialization |

---

## ⚡ Core Architecture Features

* **Multi-Agent Bug De-Duplication:** Evaluates incoming error logs using semantic comparison heuristics. Cleanses and flags duplicate incoming tickets against historical records using a 78% cosine similarity threshold.
* **Localized Fix Advisor RAG:** Leverages LangChain's Chroma integration alongside native Hugging Face embeddings to index past resolutions and serve context-aware fix recommendations entirely on local infrastructure.
* **Bulk Log Ingestion Loop:** Accepts multi-row CSV telemetry or bulk log streams equipped with row-by-row lookback caching to catch intra-file duplicate errors instantly.
* **Automated Integrity Diagnostics:** Features built-in system verification checks to validate embedding alignment and ensure database connectivity on system startup.

---

## ⚙️ Engineering & Platform Safeguards

### Cross-Platform Large Log Ingestion
The file processing architecture contains defensive error-handling abstractions designed to bypass file stream limitations unique to Windows environments.

```python
# Prevents catastrophic OverflowErrors inside underlying C layers on Windows hosts
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2147483647) # Hard ceiling capped at a safe 2 GB fallback limit
```

### Security and Resource Guards
* **In-Flight Data Constraints:** To defend against buffer injection or memory crashes from massive raw log files, the stream handler blocks any upload scaling past 10 MB.
* **Non-Blocking Main Loop:** CPU-intensive vector evaluations and text embeddings are isolated to keep the core async event cycle completely responsive under load.

---

## 📂 Repository Workspace Mapping

* main.py — Core operational application containing backend routes, bug analysis agents, and vector store configurations.
* requirements.txt — Manifest detailing specified engine packages.
* .gitignore — Filters out cache states, localized Chroma DB storage binaries, and localized Python virtual environments.
* README.md — Complete system operational architecture guide.