from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn
from document_processor import process_documents, DEFAULT_VECTOR_STORE_DIR
from query_processor import query_documents

app = FastAPI(title="RAG API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    vector_store_dir: Optional[str] = Form(DEFAULT_VECTOR_STORE_DIR)
):
    """
    문서들을 업로드하고 벡터 스토어에 저장하는 엔드포인트
    
    Args:
        files: 업로드할 파일 목록
        vector_store_dir: 벡터 스토어를 저장할 디렉토리 경로 (기본값: "vector_store")
    """
    try:
        result = await process_documents(files, vector_store_dir)
        return {"message": "문서가 성공적으로 처리되었습니다.", "details": result}
    except Exception as e:
        return {"error": str(e)}

@app.post("/query")
async def query_docs(
    query: str = Form(...),
    vector_store_dir: Optional[str] = Form(DEFAULT_VECTOR_STORE_DIR),
    k: Optional[int] = Form(2)
):
    """
    벡터 스토어에서 문서를 검색하는 엔드포인트
    
    Args:
        query: 검색 쿼리
        vector_store_dir: 벡터 스토어 디렉토리 경로 (기본값: "vector_store")
        k: 검색할 문서 수 (기본값: 2)
    """
    try:
        result = await query_documents(query, vector_store_dir, k)
        return {"results": result}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    