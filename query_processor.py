from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from typing import List
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain

# 기본 벡터 스토어 디렉토리 설정
DEFAULT_VECTOR_STORE_DIR = "vector_store_xlsx"

FINANCIAL_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""
당신은 재무제표 분석 전문가입니다. 다음 재무제표 데이터를 분석하여 질문에 답변해주세요.

재무제표 데이터:
{context}

질문: {question}

답변 시 다음을 고려해주세요:
1. 제공된 데이터만을 기반으로 답변하세요.
2. 숫자 데이터는 단위(백만원, 억원 등)를 명확히 표시하세요.
3. 기간별 비교가 필요한 경우 해당 기간의 데이터를 모두 포함하세요.
4. 증가/감소 추세가 있는 경우 퍼센트로 표시하세요.
5. 여러 기간의 데이터를 비교해야 하는 경우 표 형식으로 작성하세요.
   - 표는 | 기호를 사용하여 작성
   - 각 열은 정렬하여 가독성을 높임
   - 헤더와 데이터를 구분하는 구분선 추가
   - 숫자는 천단위 구분자(,) 사용
   - 단위는 각 열 헤더에 명시
6. 데이터의 추세나 패턴을 보여주기 위해 ASCII 차트를 생성하세요.
   - 차트는 +, -, |, * 등의 문자를 사용하여 작성
   - x축과 y축을 명확히 표시
   - 데이터 포인트는 *로 표시
   - 추세선은 -로 표시

예시 표 형식:
| 기간 | 영업이익(백만원) | 전년대비(%) |
|------|-----------------|-------------|
| 2023년 1분기 | 1,234,567 | 15.5 |
| 2023년 2분기 | 1,345,678 | 9.0 |

예시 ASCII 차트:
    영업이익 추이 (백만원)
    |
1,400,000 |          *
    |         *
1,300,000 |        *
    |       *
1,200,000 |      *
    |     *
1,100,000 |    *
    |   *
1,000,000 |  *
    |________________
     1Q   2Q   3Q   4Q
"""
)

async def query_documents(query: str, vector_store_dir: str = DEFAULT_VECTOR_STORE_DIR, k: int = 5):
    """
    벡터 스토어에서 관련 문서를 검색하고 LLM을 통해 응답 생성
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
        model_name="gpt-4o",
        temperature=0
    )
    
    # 검색 및 QA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={
                "k": k
            }
        ),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": FINANCIAL_PROMPT_TEMPLATE
        }
    )
    
    # 질문에 대한 응답 생성
    result = qa_chain.invoke({"query": query})
    
    # 결과 포맷팅
    results = []
    for doc in result["source_documents"]:
        source = doc.metadata.get('source', '')
        filename = os.path.basename(source) if source else '알 수 없는 파일'
        sheet = doc.metadata.get('sheet', '')
        
        results.append({
            "content": doc.page_content,
            "source": filename,
            "sheet": sheet
        })
    
    # 응답 형식 수정
    response = {
        "answer": result['result'],
        "source_documents": result.get("source_documents", [])
    }
    
    return {
        "query": query,
        "answer": response["answer"],
        "results": results,
        "vector_store_dir": vector_store_dir
    } 