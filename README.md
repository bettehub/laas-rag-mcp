# RAG API

이 프로젝트는 문서 기반 질의응답 시스템을 구현한 FastAPI 기반의 API 서버입니다.

## 기능

1. 문서 업로드 및 벡터 스토어 저장

   - PDF 및 CSV 파일 지원
   - 문서 자동 분할 및 임베딩
   - Chroma 벡터 스토어에 저장

2. 문서 검색
   - 자연어 쿼리 기반 검색
   - 유사도 기반 문서 검색

## 설치 방법

1. 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

2. 환경 변수 설정:

```bash
export OPENAI_API_KEY="your-api-key"
```

## 실행 방법

```bash
python main.py
```

서버가 http://localhost:8000 에서 실행됩니다.

## API 엔드포인트

1. 문서 업로드

   - POST /upload
   - multipart/form-data 형식으로 파일 업로드
   - 지원 형식: PDF, CSV
   - 파라미터:
     - `files`: 업로드할 파일 목록 (필수)
     - `vector_store_dir`: 벡터 스토어를 저장할 디렉토리 경로 (선택, 기본값: "vector_store")

2. 문서 검색
   - POST /query
   - form-data 형식으로 파라미터 전달
   - 파라미터:
     - `query`: 검색 쿼리 (필수)
     - `vector_store_dir`: 벡터 스토어 디렉토리 경로 (선택, 기본값: "vector_store")
     - `k`: 검색할 문서 수 (선택, 기본값: 2)

## API 문서

FastAPI의 자동 생성 문서는 다음 URL에서 확인할 수 있습니다:

- http://localhost:8000/docs
- http://localhost:8000/redoc

## 벡터 스토어 이전

벡터 스토어에 저장된 파일을 다른 프로젝트에서 재사용하려면 다음과 같이 하면 됩니다:

1. 원하는 벡터 스토어 디렉토리를 다른 프로젝트의 동일한 경로로 복사합니다.
2. 다른 프로젝트에서도 다음 패키지들이 설치되어 있어야 합니다:
   - langchain-chroma
   - langchain-openai
   - 기타 필요한 의존성 패키지들
3. 동일한 임베딩 모델(OpenAIEmbeddings)을 사용해야 합니다.
4. 필요한 환경 변수(예: OpenAI API 키)가 올바르게 설정되어 있어야 합니다.

벡터 스토어를 다른 프로젝트로 이전할 때는 단순히 해당 디렉토리를 복사하는 것만으로도 충분합니다. 이렇게 하면 문서의 임베딩과 메타데이터가 모두 보존되어 새로운 프로젝트에서도 동일하게 사용할 수 있습니다.

## 여러 벡터 스토어 사용하기

이 프로젝트는 여러 개의 벡터 스토어를 동시에 사용할 수 있도록 설계되었습니다. 각 벡터 스토어는 서로 다른 디렉토리에 저장되며, API 호출 시 `vector_store_dir` 파라미터를 통해 원하는 벡터 스토어를 지정할 수 있습니다.

예를 들어, 서로 다른 프로젝트나 문서 세트에 대해 별도의 벡터 스토어를 만들고 관리할 수 있습니다:

```
project1_docs -> vector_store_project1
project2_docs -> vector_store_project2
research_papers -> vector_store_research
```

이렇게 하면 각 문서 세트를 독립적으로 관리하고 검색할 수 있습니다.
