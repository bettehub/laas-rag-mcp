from mcp.server.fastmcp import FastMCP
from fastapi import UploadFile
from typing import List
from document_processor import process_documents
from query_processor import query_documents

mcp = FastMCP("rag-mcp")

@mcp.tool()
async def vectorize_documents(files: List[UploadFile], vector_store_dir: str = "vector_store"):
    return await process_documents(files, vector_store_dir)

@mcp.tool()
async def query_documents_tool(query: str, vector_store_dir: str = "vector_store", k: int = 2):
    return await query_documents(query, vector_store_dir, k)

if __name__ == "__main__":
    mcp.run()