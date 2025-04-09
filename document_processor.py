from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
from typing import List
from fastapi import UploadFile
import re

# 기본 벡터 스토어 디렉토리 설정
DEFAULT_VECTOR_STORE_DIR = "vector_store"

def clean_text(text: str) -> str:
    """
    PDF에서 추출된 텍스트를 정리합니다.
    """
    # 연속된 공백과 줄바꿈을 하나의 공백으로 변환
    text = re.sub(r'\s+', ' ', text)
    # 특수문자 제거 (마크다운 문법 등)
    text = re.sub(r'[*●]', '', text)
    return text.strip()

async def process_documents(files: List[UploadFile], vector_store_dir: str = DEFAULT_VECTOR_STORE_DIR):
    """
    업로드된 문서들을 처리하고 벡터 스토어에 저장
    
    Args:
        files: 업로드된 파일 목록
        vector_store_dir: 벡터 스토어를 저장할 디렉토리 경로 (기본값: "vector_store")
    """
    documents = []
    
    for file in files:
        file_path = file.filename
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # 임시 파일로 저장
        temp_path = f"/tmp/{file_path}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        try:
            # 파일 타입에 따라 적절한 로더 선택
            if file_extension == '.pdf':
                loader = PyPDFLoader(temp_path)
            elif file_extension == '.csv':
                loader = CSVLoader(temp_path)
            else:
                raise ValueError(f"지원하지 않는 파일 형식입니다: {file_extension}")
            
            # 문서 로드
            docs = loader.load()
            
            # 텍스트 정리
            for doc in docs:
                doc.page_content = clean_text(doc.page_content)
            
            documents.extend(docs)
            
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    if not documents:
        raise ValueError("처리할 문서가 없습니다.")
    
    # 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    
    # 벡터 스토어 생성 및 저장
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=vector_store_dir
    )
    
    return {"message": f"{len(documents)}개의 문서가 성공적으로 처리되었습니다. (총 {len(splits)}개의 청크로 분할됨)", "vector_store_dir": vector_store_dir}
