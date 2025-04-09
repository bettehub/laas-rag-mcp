FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 프로젝트 파일 복사
COPY . .

# 의존성 설치 (✔️ RUN 하나만!)
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# STDIO 실행 명령 (Smithery용)
CMD ["python", "mcp_server.py"]
