# AI Smart Bug Analyser and Fix Advisor 🚀

An offline, privacy-safe Defect Management Engine built to automate software bug classification, diagnostics, and remediation profiling using an optimized Retrieval-Augmented Generation (RAG) pipeline.

## 🏗️ Project Architecture
The system isolates unstructured text blocks from open-source repositories using **Pandas**, chunks data dynamically via **LangChain**, and runs fully offline semantic searches utilizing a local **SentenceTransformer** embedding matrix stored inside **ChromaDB**.

## 🛠️ Technology Stack
* **Framework:** FastAPI (Python 3.10+)
* **Data Processing:** Pandas DataFrames
* **Chunking Matrix:** LangChain Text Splitters
* **Vector Engine:** ChromaDB & SentenceTransformers (`all-MiniLM-L6-v2`)

---

## 🚀 Quick Start Guide

### 1. Installation & Environment Setup
Clone this repository to your local engine and configure the dependency landscape:
```bash
# Clone the repository
git clone https://github.com/Meghanaanand2005/AI-SMART-BUG-ANALYZER-AND-FIX-ADVISOR.git
cd Meghanaanand2005

# Setup and activate virtual workspace environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install core framework components
pip install fastapi uvicorn pandas langchain-text-splitters sentence-transformers chromadb

#Launch Local Engine Server
uvicorn main:app --reload

Once initialized, access the interactive dashboard interface control panel at:
👉 http://127.0.0.1:8000/docs
