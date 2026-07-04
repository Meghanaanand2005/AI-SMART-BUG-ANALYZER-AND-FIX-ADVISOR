import os
import uuid
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Query
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# 1. Initialize FastAPI Application Configuration
app = FastAPI(
    title="AI Smart Bug Analyser and Fix Advisor",
    description="Offline Local RAG Pipeline & Automated Defect Management Engine"
)

# 2. Configure Local Persistence Directories (No Internet / Privacy-Safe)
LOCAL_MODEL_PATH = "./local_ai_model"
CHROMA_DATA_PATH = "./chroma_db"

# 3. Setup Context-Aware Text Splitter Pipeline
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# 4. Initialize Local AI Embedding and Storage Engines
embedding_model = SentenceTransformer(LOCAL_MODEL_PATH)
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = chroma_client.get_or_create_collection(name="bug_knowledge_base")

# 5. Data Structure Input Container Schema for Text Pasting
class BugTextInput(BaseModel):
    text_content: str


# --- TERMINAL ORCHESTRATION LINKS ---

@app.on_event("startup")
def print_terminal_link():
    """Generates an immediate clickable hyperlink block directly in your terminal console."""
    print("\n" + "="*65)
    print("🚀  AI SMART BUG ANALYSER CONTROL PANEL READY!")
    print("👉  Hold 'Ctrl' and click the link below to open the dashboard:")
    print("\n    http://127.0.0.1:8000/docs")
    print("="*65 + "\n")


# --- CORE API ENDPOINTS ---

@app.get("/")
def read_root():
    return {
        "status": "Online",
        "message": "AI Core operational. Access /docs for the interactive panel link."
    }

# Bulk Seed Kaggle Datasets via CSV File Stream Processing (OPTIMIZED BATCH SPEED)
@app.post("/api/seed-kaggle")
async def seed_kaggle_csv(
    file: UploadFile = File(...), 
    text_column_name: str = Query("description", description="The exact header name of the column containing the bug text")
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if text_column_name not in df.columns:
            return {
                "status": "Error", 
                "message": f"Column '{text_column_name}' not found. Headers: {list(df.columns)}"
            }
        
        # Pull out records and cap at 100 rows to keep processing super fast (under 10 seconds)
        bug_reports = df[text_column_name].dropna().astype(str).tolist()[:100]
        
        all_chunks = []
        all_embeddings = []
        all_metadatas = []
        all_ids = []
        
        print("⚡ Extracting and chunking text segments...")
        for index, report in enumerate(bug_reports):
            chunks = text_splitter.split_text(report)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({"source_origin": "kaggle_seeding", "row_index": index, "chunk_index": i})
                all_ids.append(str(uuid.uuid4()))
        
        if not all_chunks:
            return {"status": "Error", "message": "No text data extracted from the document configuration."}
            
        print(f"🧠 Generating vector matrices for {len(all_chunks)} text segments locally...")
        # Batch execution passes all vectors to the AI core at the same time
        all_embeddings = embedding_model.encode(all_chunks, batch_size=32, show_progress_bar=False).tolist()
        
        print("💾 Saving structural representations to local disk index space...")
        collection.add(documents=all_chunks, embeddings=all_embeddings, metadatas=all_metadatas, ids=all_ids)
            
        return {
            "status": "Success - Kaggle Repository Processed Seamlessly",
            "total_rows_evaluated": len(bug_reports),
            "total_semantic_chunks_indexed": len(all_chunks),
            "current_database_total_count": collection.count()
        }
    except Exception as e:
        return {"status": "Error", "message": str(e)}

# Bug Submission Module — Direct Text Paste Endpoint
@app.post("/api/submit-text")
async def submit_text(input_data: BugTextInput):
    try:
        raw_text = input_data.text_content
        if not raw_text.strip():
            return {"status": "Error", "message": "Text content cannot be empty"}
        
        chunks = text_splitter.split_text(raw_text)
        if not chunks:
            return {"status": "Error", "message": "No meaningful chunks processed"}
            
        embeddings = embedding_model.encode(chunks).tolist()
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source_type": "direct_paste", "chunk_index": i} for i in range(len(chunks))]
        
        collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
        
        return {
            "status": "Success - Text Vectorized and Indexed",
            "total_chunks_stored": len(chunks),
            "stored_db_total_count": collection.count()
        }
    except Exception as e:
        return {"status": "Error", "message": str(e)}

# Bug Submission Module — Ingestion File Upload Endpoint
@app.post("/api/submit-file")
async def submit_file(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        raw_text = file_content.decode("utf-8", errors="ignore")
        
        chunks = text_splitter.split_text(raw_text)
        if not chunks:
            return {"status": "Error", "message": "File is empty or unreadable"}
            
        embeddings = embedding_model.encode(chunks).tolist()
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source_file": file.filename, "chunk_index": i} for i in range(len(chunks))]
        
        collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
        
        return {
            "status": "Success - File Vectorized and Indexed",
            "file_processed": file.filename,
            "total_chunks_stored": len(chunks),
            "stored_db_total_count": collection.count()
        }
    except Exception as e:
        return {"status": "Error", "message": str(e)}

# Semantic Vector Optimization Query Endpoint
@app.get("/api/search-history")
def search_history(query: str):
    try:
        query_vector = embedding_model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_vector, n_results=2)
        
        formatted_matches = []
        if results and results['documents']:
            for i in range(len(results['documents'][0])):
                formatted_matches.append({
                    "similarity_distance_score": results['distances'][0][i] if 'distances' in results else None,
                    "text_segment": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if 'metadatas' in results else {}
                })
                
        return {
            "search_query": query,
            "total_matches_found": len(formatted_matches),
            "matches": formatted_matches
        }
    except Exception as e:
        return {"status": "Error", "message": str(e)}