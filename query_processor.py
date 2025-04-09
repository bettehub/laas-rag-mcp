from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os

# 기본 벡터 스토어 디렉토리 설정
DEFAULT_VECTOR_STORE_DIR = "vector_store"

async def query_documents(query: str, vector_store_dir: str = DEFAULT_VECTOR_STORE_DIR, k: int = 2):
    """
    벡터 스토어에서 관련 문서를 검색하고 LLM을 통해 응답 생성
    
    Args:
        query: 검색 쿼리
        vector_store_dir: 벡터 스토어 디렉토리 경로 (기본값: "vector_store")
        k: 검색할 문서 수 (기본값: 2)
    """
    if not os.path.exists(vector_store_dir):
        raise ValueError(f"벡터 스토어가 존재하지 않습니다: {vector_store_dir}. 먼저 문서를 업로드해주세요.")
    
    # 벡터 스토어 로드
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=vector_store_dir,
        embedding_function=embeddings
    )
    
    # LLM 설정
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0
    )
    
    # 검색 및 QA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": k}
        ),
        return_source_documents=True
    )
    
    # 질문에 대한 응답 생성
    result = qa_chain.invoke({"query": query})
    
    # 결과 포맷팅
    results = []
    for doc in result["source_documents"]:
        source = doc.metadata.get('source', '')
        filename = os.path.basename(source) if source else '알 수 없는 파일'
        
        results.append({
            "content": doc.page_content,
            "source": filename
        })
    
    return {
        "query": query,
        "answer": result["result"],
        "results": results,
        "vector_store_dir": vector_store_dir
    } 