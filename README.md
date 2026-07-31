# AISBAFA - AI Smart Bug Analyzer and Fix Advisor

**AISBAFA** is an enterprise-grade AI-powered log parsing, bug analysis, and automated remediation engine built on a unified FastAPI backend (`main.py`). It combines custom log/CSV parsing, localized vector search via ChromaDB, and a multi-agent AI pipeline to triage, diagnose, and auto-fix software bugs from system logs and crash traces.

---

## 🏗️ Architecture & Core Components (in `main.py`)

`main.py` serves as the core orchestrator containing all schemas, ingestion logic, vector handlers, multi-agent engines, and API routes:

1. **Pydantic Data Models & Schemas**
   * `LogEntry` & `BugReportPayload`: Input definitions for log streams, CSV rows, and crash reports.
   * `TriageResult`, `RootCauseAnalysis`, and `FixSuggestion`: Structured outputs for severity scores, stack trace isolates, and code patch diffs.

2. **Log & CSV Parsing Engine**
   * **Custom Chunking:** Streaming parser for `.log`, `.txt`, `.json`, and `.csv` files.
   * **Stack Trace Isolator:** Regex extraction to isolate multi-line exceptions while filtering noisy debug logs.

3. **Localized RAG & Vector Engine (ChromaDB)**
   * Embeds error messages and stack traces into vector representations.
   * Performs semantic similarity searches to identify and prevent duplicate bug submissions.

4. **Multi-Agent Pipeline**
   * **Triage Agent:** Evaluates bug impact and assigns severity (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`).
   * **Root Cause Agent:** Cross-references incoming stack traces against ChromaDB history to detect systemic failures.
   * **Fix Advisor Agent:** Generates actionable code patches, resolution steps, and preventative measures.

---

## 🛠️ Tech Stack & Dependencies

* **Framework:** FastAPI / Uvicorn
* **Data Validation:** Pydantic v2
* **Vector Store & Embeddings:** ChromaDB / Sentence-Transformers
* **Multi-Agent & LLM Integration:** LangChain / OpenAI / Local LLMs
* **Data Processing:** Pandas, NumPy, Re

---

## ⚡ Quick Start Guide

### 1. Environment Setup
```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install fastapi uvicorn pydantic chromadb sentence-transformers langchain pandas numpy python-multipart
```

### 3. Run the Application
```powershell
uvicorn main:app --reload
```

---

## 🌐 Application Access

* **Main App & Dashboard Output:** 👉 **`http://127.0.0.1:8000`**
* **Interactive API Testing UI (Optional):** `http://127.0.0.1:8000/docs`

---

## 🛰️ API Routes Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Renders the main dashboard output and application interface |
| `POST` | `/api/v1/ingest-file` | Batch uploads and parses `.csv`, `.log`, `.json`, or `.txt` files |
| `POST` | `/api/v1/analyze-bug` | Runs the 3-stage agent pipeline (Triage ➔ Root Cause ➔ Fix Advisor) |
| `POST` | `/api/v1/deduplicate` | Queries ChromaDB to check if an incoming trace is a duplicate |
| `GET` | `/api/v1/defects` | Fetches historical defect knowledge base and bug frequency metrics |