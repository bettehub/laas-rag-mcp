from fastapi import FastAPI, UploadFile
from fastmcp import FastMCP
from typing import List
from document_processor import process_documents
from query_processor import query_documents
import uvicorn

# MCP 서버용 FastAPI 앱 생성
app = FastAPI(title="RAG MCP Tool Server")

# FastMCP 인스턴스 등록
mcp = FastMCP("rag-mcp", app=app)

# MCP 툴: 문서 벡터화
@mcp.tool()
async def vectorize_documents(
    files: List[UploadFile],
    vector_store_dir: str = "vector_store"
):
    return await process_documents(files, vector_store_dir)

# MCP 툴: 문서 검색
@mcp.tool()
async def query_documents_tool(
    query: str,
    vector_store_dir: str = "vector_store",
    k: int = 2
):
    return await query_documents(query, vector_store_dir, k)

# MCP 서버 실행
if __name__ == "__main__":
    print("🚀 MCP 서버 실행 중...")
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8000)
