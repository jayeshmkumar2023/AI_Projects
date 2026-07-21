# AuditChatBot API: System Architecture & Design Guide

This document provides a comprehensive technical overview of the AuditChatBot API architecture. Use this guide to understand the system design, components, configurations, and key engineering decisions—structured for technical interviews.

---

## High-Level Architecture Diagram

```text
               +---------------------------------------------------+
               |              API / PRESENTATION LAYER             |
               |       FastAPI Router | JWT Auth | Swagger UI      |
               +-------------------------+-------------------------+
                                         |
                       +-----------------+-----------------+
                       | (Async upload)                    | (Sync / Chat)
                       v                                   v
        +--------------+--------------+             +------+------+
        |      INGESTION PIPELINE     |             |     RAG     |
        |   FastAPI BackgroundTasks   |             |   RETRIEVAL |
        +--------------+--------------+             +------+------+
                       |                                   |
           +-----------+-----------+                       | 1. Similarity Search
           |                       |                       v
           v                       v               +-------+-------+
    +------+------+         +------+------+        |  VECTOR STORE |
    | OBJECT STORE |        |  RELATIONAL |        |     Qdrant    |
    |    MinIO    |        |   DATABASE  |        |   Vector DB   |
    | (Raw files)  |        |  PostgreSQL |        +-------+-------+
    +--------------+        |  (metadata) |                |
                            +------+------+                | 2. Fetch Parent IDs
                                   ^                       v
                                   |               +-------+-------+
                                   +---------------+  RE-RANKING   |
                                     (Fetch Parent |   FlashRank   |
                                       Text Chunks)| (Select Top-4)|
                                                   +-------+-------+
                                                           |
                                                           | 3. Send Context
                                                           v
                                                   +-------+-------+
                                                   |   LLM LAYER   |
                                                   | Groq | OpenAI |
                                                   +---------------+
```

---

## Layer-by-Layer Architectural Details

### 1. Presentation & API Layer (FastAPI)
* **Technologies:** FastAPI, Uvicorn, Pydantic, Python-multipart.
* **Role:** Exposes high-performance, asynchronous RESTful endpoints for document upload, sync, chat sessions, user management, and health checks.
* **Interview Talking Points:** 
  > *"I chose FastAPI for its high performance (comparable to Go/Node.js) and native support for async/await workflows. It utilizes Pydantic for robust type declaration and data validation, and generates OpenAPI documentation automatically out-of-the-box."*

### 2. Asynchronous Ingestion Pipeline (MinIO + BackgroundTasks)
* **Technologies:** MinIO SDK (`minio`), FastAPI `BackgroundTasks`.
* **Role:** Persists raw files and handles document extraction/indexing workflows asynchronously.
* **Ingestion Flow:**
  1. **Upload Persistence:** The user uploads a file (`.pdf`/`.docx`) via `/upload`. The raw file is immediately streamed into a **MinIO Object Storage** bucket.
  2. **Job Registration:** The API creates a `DocumentJob` record in PostgreSQL with a state of `PENDING`.
  3. **Async Task Offloading:** The API spawns a background thread using FastAPI's `BackgroundTasks` and immediately returns the `job_id` with a `200 OK` response to the client.
  4. **Background Indexing:** The background worker updates the job status to `PROCESSING`, reads/parses the file using `PyPDFLoader`/`Docx2txtLoader`, generates text chunks, indexes child vectors in Qdrant, and saves parent text chunks in PostgreSQL.
* **Interview Talking Points:**
  > *"Synchronous parsing of large PDFs (100+ pages) blocks the main server thread, causing gateway timeouts. I decoupled this by saving files to MinIO object storage and offloading the heavy extraction, chunking, and embedding processes to an asynchronous worker pool, allowing the front-end to poll for progress using a job ID."*

### 3. Vector Database Layer (Qdrant)
* **Technologies:** `qdrant-client`, `langchain-qdrant`, `sentence-transformers` (`all-miniLM-L6-v2`).
* **Role:** Manages high-dimensional semantic search vectors.
* **Configuration:**
  * Uses the HuggingFace Embeddings model `all-miniLM-L6-v2` mapping chunks into **384-dimensional dense vectors**.
  * Measures similarity using **Cosine Distance**.
  * Multi-tenancy isolation is enforced using Qdrant’s metadata filters (filtering queries strictly by the authenticated `user_id`).
* **Interview Talking Points:**
  > *"The previous architecture used local FAISS files saved on disk. This is a massive anti-pattern in production because multiple containerized API workers cannot share or write to a single local file without data corruption. I migrated to Qdrant, a dedicated, highly scalable Rust-based vector database. This allowed us to scale horizontally, run fast index searches out of memory (HNSW), and safely isolate user data using payload-based metadata filters."*

### 4. Database & Persistence Layer (PostgreSQL & SQLAlchemy)
* **Technologies:** PostgreSQL, SQLAlchemy ORM, `psycopg2-binary`.
* **Role:** Transactional database storing structured application metadata and the RAG parent text chunks.
* **Schema Design:**
  * `users`: Stores emails, roles (`admin`, `user`), and hashed credentials.
  * `chat_sessions` & `chat_messages`: Tracks conversational history and LLM source attribution.
  * `documents`: Tracks uploaded filenames, sizes, and owning user IDs.
  * `document_jobs`: Logs status (`PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`) and errors for document ingestion.
  * `parent_chunks`: Maps chunked document passages (`Text`) back to their parent document records.
* **Interview Talking Points:**
  > *"We use PostgreSQL for strong consistency and relational integrity. It tracks user profiles, chat threads, ingestion jobs, and holds the textual content of our document parent chunks, acting as the primary source of truth."*

### 5. RAG Retrieval & Re-ranking Pipeline (FlashRank)
This is the core RAG intelligence layer. It uses a **Parent-Document Retrieval (PDR)** strategy combined with a **Cross-Encoder Re-ranker**.

#### A. Ingestion Phase: Hierarchical Chunking
* Documents are split into two levels:
  1. **Parent Chunks:** Large sections of the document (~2,000 characters) to preserve rich context and surrounding paragraphs.
  2. **Child Chunks:** Smaller sub-slices (~400 characters) created out of each parent.
* **Storage:** We index only the **child chunks** in Qdrant as vectors (maximizing semantic search accuracy), while writing the **parent chunks** directly to PostgreSQL.

#### B. Retrieval Phase: Child-Parent Resolution
* When a user asks a question, we query Qdrant using the query vector.
* Qdrant performs a similarity search over the **child chunks** and returns the top-15 matches.
* The API extracts the `parent_id` metadata from these top matches and does a bulk SQL lookup to fetch the corresponding **parent chunks** (2,000 chars each).

#### C. Re-ranking Phase: FlashRank
* **Why Reranking?** Cosine similarity alone often returns irrelevant documents at the top of the list.
* **How it works:** We feed the top-15 parent chunks and the query into **FlashRank** (a lightweight local Cross-Encoder re-ranker running ONNX models).
* FlashRank scores the chunks for actual relevance. We take the top-4 highest-scoring chunks and construct our context block for the LLM.
* **Interview Talking Points:**
  > *"For the RAG strategy, I implemented a Parent-Document Retriever. Small chunks (child) are better for vector matching, but poor for LLM synthesis because they lack context. Conversely, large chunks (parent) are excellent for the LLM but introduce noise during vector search. We get the best of both worlds by searching small chunks, retrieving the larger parent blocks, and using FlashRank—a local, zero-cost ONNX re-ranker—to order the most relevant passages before sending them to the LLM."*

### 6. Model & Orchestration Layer (LangChain + LLMs)
* **Technologies:** `langchain`, `langchain-groq` (Llama 3.3 70B), `langchain-openai` (GPT-4o-mini).
* **Role:** Performs conversational query condensation and final answer synthesis.
* **Conversational Memory & Condensation:**
  * It maintains the last 8 messages.
  * If history exists, it uses the LLM to rewrite the follow-up question into a **standalone query** incorporating history before querying the vector database.
* **Answer Generation & Fallback Heuristic:**
  * The system prompt instructs the model to act as a compliance auditor, cite sources in a strict `Sources: [doc1.docx]` format.
  * **Fallback Heuristic:** If the LLM omits the source citation but successfully answers the question, the backend automatically performs heuristic fallback attribution to verify and attach the retrieved document names to the API response, ensuring reliable citations in the UI.

### 7. Security Layer (JWT + BCrypt)
* **Technologies:** PyJWT, BCrypt.
* **Role:** Secures the APIs using standard OAuth2 Bearer Token workflows.
* **Enforcement:**
  * Direct route dependencies (`get_current_user`, `get_current_admin`) verify the token signatures and retrieve user records from the database on every protected endpoint call.
