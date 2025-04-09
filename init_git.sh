#!/bin/bash

# 사용법: ./init_git.sh <repo-name> [private|public]

REPO_NAME=$1
VISIBILITY=$2

if [ -z "$REPO_NAME" ]; then
  echo "Error: 생성할 GitHub 레포지토리 이름을 입력해주세요."
  echo "사용법: ./init_git.sh <repo-name> [private|public]"
  exit 1
fi

if [ -z "$VISIBILITY" ]; then
  VISIBILITY="private"
fi

# GitHub 인증 확인
if ! gh auth status &>/dev/null; then
  echo "GitHub 인증이 필요합니다. 로그인 절차를 시작합니다..."
  gh auth login
fi

echo "GitHub 원격 레포지토리를 생성합니다: $REPO_NAME ($VISIBILITY)"
gh repo create "$REPO_NAME" --"$VISIBILITY" --source=. --remote=origin --push=false

echo ".gitignore 파일을 생성합니다."
cat <<EOF > .gitignore
/vector_store*
/venv
DS_Store
__pycache__
EOF

echo "Git 저장소를 초기화합니다."
git init

echo "변경 사항을 스테이징합니다."
git add .

echo "첫 커밋을 생성합니다."
git commit -m "Initial commit"

echo "원격 저장소를 연결합니다."
git remote add origin "https://github.com/$(gh api user | jq -r .login)/$REPO_NAME.git"

echo "main 브랜치로 푸시합니다."
git branch -M main
git push -u origin main

echo "레포지토리 생성 및 초기화가 완료되었습니다."
