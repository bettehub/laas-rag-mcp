from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
from typing import List
from fastapi import UploadFile
import re
import pandas as pd
from langchain.docstore.document import Document

# 기본 벡터 스토어 디렉토리 설정
DEFAULT_VECTOR_STORE_DIR = "vector_store_xlsx"

def clean_text(text: str) -> str:
    """
    PDF에서 추출된 텍스트를 정리합니다.
    """
    # 연속된 공백과 줄바꿈을 하나의 공백으로 변환
    text = re.sub(r'\s+', ' ', text)
    # 특수문자 제거 (마크다운 문법 등)
    text = re.sub(r'[*●]', '', text)
    return text.strip()

def process_excel_file(file_path: str) -> List[Document]:
    """
    Excel 파일을 처리하여 Document 객체 리스트를 반환합니다.
    """
    documents = []
    
    # Excel 파일 읽기
    df = pd.read_excel(file_path, sheet_name=None)
    
    # 각 시트 처리
    for sheet_name, sheet_data in df.items():
        # 헤더 정보 추출
        headers = sheet_data.columns.tolist()
        
        # 시트의 모든 데이터를 하나의 문서로 구성
        content_parts = []
        
        # 각 행 처리
        for idx, row in sheet_data.iterrows():
            row_parts = []
            for col_idx, col in enumerate(headers):
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    # 숫자인 경우 천단위 구분자 추가
                    if isinstance(val, (int, float)):
                        val = f"{val:,.0f}"
                    
                    # 컬럼 이름이 Unnamed인 경우 처리
                    if 'Unnamed' in str(col):
                        # 이전 컬럼이 있는 경우
                        if col_idx > 0 and 'Unnamed' not in str(headers[col_idx-1]):
                            row_parts.append(f"{headers[col_idx-1]}: {val}")
                    else:
                        row_parts.append(f"{col}: {val}")
            
            if row_parts:
                content_parts.append(f"행 {idx+1}: {' | '.join(row_parts)}")
        
        if content_parts:
            doc = Document(
                page_content=f"[시트: {sheet_name}]\n" + "\n".join(content_parts),
                metadata={
                    "source": file_path,
                    "sheet": sheet_name
                }
            )
            documents.append(doc)
    
    return documents

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
                docs = loader.load()
            elif file_extension == '.csv':
                loader = CSVLoader(temp_path)
                docs = loader.load()
            elif file_extension in ['.xlsx', '.xls']:
                docs = process_excel_file(temp_path)
            else:
                raise ValueError(f"지원하지 않는 파일 형식입니다: {file_extension}")
            
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
        chunk_size=4000,
        chunk_overlap=500,
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
