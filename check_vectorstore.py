from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os

def check_vectorstore(vector_store_dir: str = "vector_store_xlsx"):
    """
    벡터스토어의 내용을 확인합니다.
    """
    if not os.path.exists(vector_store_dir):
        print(f"벡터 스토어가 존재하지 않습니다: {vector_store_dir}")
        return
    
    # 벡터 스토어 로드
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=vector_store_dir,
        embedding_function=embeddings
    )
    
    # 모든 문서 가져오기
    documents = vectorstore.get()
    
    print(f"\n=== 벡터스토어 정보 ===")
    print(f"총 문서 수: {len(documents['documents'])}")
    print(f"컬렉션 이름: {vectorstore._collection.name}")
    
    print("\n=== 문서 샘플 (처음 5개) ===")
    for i, (doc, metadata) in enumerate(zip(documents['documents'][:5], documents['metadatas'][:5])):
        print(f"\n문서 {i+1}:")
        print(f"시트: {metadata.get('sheet', 'N/A')}")
        print(f"행: {metadata.get('row', 'N/A')}")
        print(f"내용: {doc[:200]}...")  # 처음 200자만 출력
    
    print("\n=== 메타데이터 샘플 ===")
    for i, metadata in enumerate(documents['metadatas'][:5]):
        print(f"\n메타데이터 {i+1}:")
        for key, value in metadata.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    check_vectorstore() 