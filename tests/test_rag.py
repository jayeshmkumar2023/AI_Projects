import requests
import time
import zipfile
import io
import os

BASE_URL = "http://localhost:8000"

def create_dummy_docx(text: str) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as docx:
        doc_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:body>
                <w:p>
                    <w:r>
                        <w:t>{text}</w:t>
                    </w:r>
                </w:p>
            </w:body>
        </w:document>"""
        docx.writestr("word/document.xml", doc_xml)
    return zip_buffer.getvalue()

def run_test():
    print("=== Starting Production RAG Integration Test ===")
    
    # 1. Login to get JWT Token
    print("\n1. Logging in as admin...")
    login_res = requests.post(
        f"{BASE_URL}/auth/login",
        json={"email": "admin@auditing.com", "password": "admin123"}
    )
    if login_res.status_code != 200:
        print(f"Login failed: {login_res.text}")
        return
        
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("Login successful! Token acquired.")

    # 2. Programmatically create a dummy docx file
    filename = "test_audit_report_2026.docx"
    content_text = "Audit report for FY2026. The unique validation key is KEY-XYZ-9988. All compliance audits must follow standard 42."
    file_bytes = create_dummy_docx(content_text)
    
    # 3. Upload file (starts background task)
    print(f"\n2. Uploading {filename} asynchronously...")
    files = {"file": (filename, io.BytesIO(file_bytes), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    upload_res = requests.post(f"{BASE_URL}/upload", headers=headers, files=files)
    
    if upload_res.status_code != 200:
        print(f"Upload failed: {upload_res.text}")
        return
        
    upload_data = upload_res.json()
    job_id = upload_data["job_id"]
    status = upload_data["status"]
    print(f"Upload completed. Created Job ID: {job_id}, Initial Status: {status}")

    # 4. Poll job status
    print("\n3. Polling job status...")
    max_attempts = 15
    for attempt in range(max_attempts):
        job_res = requests.get(f"{BASE_URL}/documents/jobs/{job_id}", headers=headers)
        if job_res.status_code != 200:
            print(f"Failed to fetch job status: {job_res.text}")
            return
            
        job_data = job_res.json()
        status = job_data["status"]
        print(f"Attempt {attempt+1}/{max_attempts}: Status is '{status}'")
        
        if status == "COMPLETED":
            print("Document ingestion completed successfully in the background!")
            break
        elif status == "FAILED":
            print(f"Job failed with error: {job_data.get('error_message')}")
            return
            
        time.sleep(2)
    else:
        print("Polling timed out! Ingestion did not complete in time.")
        return

    # 5. Query chat (utilizing Qdrant retrieval + FlashRank reranking)
    print("\n4. Querying chat with document-specific question...")
    chat_payload = {
        "question": "What is the unique validation key mentioned in the audit report?",
        "provider": "groq",
        "k": 4
    }
    chat_res = requests.post(f"{BASE_URL}/chat", headers=headers, json=chat_payload)
    if chat_res.status_code != 200:
        # Fallback to OpenAI if Groq API limit/key issue occurs
        print("Groq query failed or returned error. Retrying with OpenAI...")
        chat_payload["provider"] = "openai"
        chat_res = requests.post(f"{BASE_URL}/chat", headers=headers, json=chat_payload)

    if chat_res.status_code != 200:
        print(f"Chat request failed: {chat_res.text}")
        return
        
    chat_data = chat_res.json()
    print("\n=== Chat Response ===")
    print(f"Answer: {chat_data['answer']}")
    print(f"Sources: {chat_data['sources']}")
    print("=====================")
    
    # 6. Verify source attribution
    sources = chat_data.get("sources", [])
    if any(s.get("source") == filename for s in sources):
        print("\nSUCCESS: Document was retrieved, reranked, and correctly cited in the source references!")
    else:
        print("\nWARNING: Answer returned but the document name was not found in sources. Check context length or LLM response.")

if __name__ == "__main__":
    run_test()
