from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn
from document_processor import process_documents, DEFAULT_VECTOR_STORE_DIR
from query_processor import query_documents
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from fastapi import HTTPException

app = FastAPI(title="RAG API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서의 접근 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 헤더 허용
    expose_headers=["*"]  # 모든 헤더 노출
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
async def query_endpoint(query: str, vector_store_dir: str = DEFAULT_VECTOR_STORE_DIR, k: int = 5):
    """
    문서에 대한 질문을 처리하고 응답을 반환합니다.
    
    - **query**: 검색할 질문
    - **vector_store_dir**: 벡터 스토어 디렉토리 경로 (기본값: "vector_store")
    - **k**: 검색할 문서 수 (기본값: 5)
    """
    try:
        result = await query_documents(query, vector_store_dir=vector_store_dir, k=k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list/{vector_store_dir}")
async def list_documents(vector_store_dir: str = DEFAULT_VECTOR_STORE_DIR):
    """
    벡터 스토어에 저장된 모든 문서를 리스트로 보여주는 엔드포인트
    
    Args:
        vector_store_dir: 벡터 스토어 디렉토리 경로 (기본값: "vector_store")
    """
    try:
        if not os.path.exists(vector_store_dir):
            raise ValueError(f"벡터 스토어가 존재하지 않습니다: {vector_store_dir}")
        
        # 벡터 스토어 로드
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory=vector_store_dir,
            embedding_function=embeddings
        )
        
        # 모든 문서 가져오기
        documents = vectorstore.get()
        
        # 결과 포맷팅
        results = []
        for i, (doc, metadata) in enumerate(zip(documents['documents'], documents['metadatas'])):
            source = metadata.get('source', '')
            filename = os.path.basename(source) if source else '알 수 없는 파일'
            sheet = metadata.get('sheet', '')
            row = metadata.get('row', '')
            
            results.append({
                "id": i,
                "content": doc,
                "source": filename,
                "sheet": sheet,
                "row": row,
                "metadata": metadata
            })
        
        return {
            "total_documents": len(results),
            "documents": results,
            "raw_data": documents  # 원본 데이터도 함께 반환
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    