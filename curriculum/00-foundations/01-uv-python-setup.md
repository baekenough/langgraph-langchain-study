# Phase 01: UV로 Python 환경 구성

> **예상 소요시간**: 45분  
> **난이도**: ★★☆☆☆  
> **선행 페이즈**: [Phase 00: 오리엔테이션](./00-orientation.md)

---

## 🎯 학습 목표

- UV가 무엇이고 왜 pip/poetry 대신 사용하는지 이해한다
- `uv init`, `uv add`, `uv run`, `uv sync` 등 핵심 명령어를 능숙하게 사용한다
- `pyproject.toml`, `.python-version`, `uv.lock` 파일의 역할을 이해한다
- 이 프로젝트의 의존성을 UV로 설치하고 실행 환경을 준비한다

---

## 📚 핵심 개념

### UV란?

UV는 Astral 팀이 Rust로 개발한 **차세대 Python 패키지 관리자**입니다. 2024년 등장하여 빠르게 Python 커뮤니티의 표준으로 자리잡고 있습니다.

**UV가 pip/poetry보다 나은 이유:**

| 비교 항목 | pip | poetry | UV |
|----------|-----|--------|-----|
| 속도 | 느림 | 보통 | **10~100배 빠름** |
| Python 버전 관리 | ❌ (pyenv 별도) | ❌ | ✅ 내장 |
| Lock 파일 | ❌ | ✅ | ✅ |
| 가상환경 생성 | 별도 venv 필요 | 자동 | 자동 |
| 단일 바이너리 | ❌ | ❌ | ✅ |

> **요약**: UV 하나로 pyenv + pip + virtualenv + poetry를 대체할 수 있습니다.

### 핵심 파일 3가지

UV 프로젝트는 세 파일로 환경을 정의합니다:

**`.python-version`** — 프로젝트가 사용할 Python 버전을 고정합니다.
```
3.12
```

**`pyproject.toml`** — 프로젝트 메타데이터와 의존성을 정의합니다.
```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "langchain>=0.3",
    "langgraph>=0.2",
]
```

**`uv.lock`** — 실제 설치된 모든 패키지의 정확한 버전과 해시를 기록합니다. (자동 생성, 커밋 권장)

---

## 💻 코드 예제

### UV 설치 (macOS/Linux)

```bash
# UV 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 설치 확인
uv --version
# uv 0.4.x (또는 이상)
```

> **Windows**: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

### 프로젝트 초기화

```bash
# 새 프로젝트 초기화 (기존 디렉토리에서)
cd /Users/sangyi/workspace/study/langraph_langchain
uv init --python 3.12
```

이 명령은 `pyproject.toml`, `.python-version`, `hello.py`(샘플)를 생성합니다.

### 의존성 추가

```bash
# LangChain 생태계 설치 (ChatOpenAI로 OpenRouter 채팅 + OpenAI 임베딩 모두 커버)
uv add langchain langchain-core langchain-openai
uv add langchain-community langchain-chroma langchain-text-splitters

# LangGraph 관련
uv add langgraph langgraph-checkpoint-sqlite

# 유틸리티
uv add langsmith python-dotenv pydantic

# 웹 검색 도구
uv add tavily-python

# 문서 처리
uv add beautifulsoup4 pypdf

# FastAPI (배포용)
uv add fastapi uvicorn

# 개발 도구 (개발 의존성)
uv add --dev ipykernel jupyter pytest pytest-asyncio ruff
```

### 주요 UV 명령어

```bash
# 의존성 제거
uv remove some-package

# 가상환경 직접 생성
uv venv

# 모든 의존성 설치 (lock 파일 기준)
uv sync

# 개발 의존성 포함 설치
uv sync --dev

# 특정 스크립트 실행 (가상환경 자동 활성화)
uv run python my_script.py

# Jupyter 실행
uv run jupyter notebook

# Python 인터프리터 직접 실행
uv run python

# 현재 환경 확인
uv pip list

# UV 자체 업데이트
uv self update
```

### 가상환경 활성화 (직접 사용 시)

```bash
# 가상환경 생성 (UV가 자동으로 .venv 생성)
uv venv

# 활성화 (macOS/Linux)
source .venv/bin/activate

# 활성화 (Windows)
.venv\Scripts\activate

# 비활성화
deactivate
```

> **팁**: `uv run` 명령을 사용하면 가상환경을 직접 활성화하지 않아도 됩니다.

### pyproject.toml 구조 이해

```toml
[project]
name = "langchain-langgraph-study"
version = "0.1.0"
description = "LangChain + LangGraph 학습 프로젝트"
requires-python = ">=3.12"
dependencies = [
    # 버전을 너무 빡빡하게 고정하지 않음 — UV가 최적 버전 해결
    "langchain>=0.3",
    "langchain-openai>=0.2",
    "langgraph>=0.2",
]

[dependency-groups]
dev = [
    "ipykernel>=6",
    "jupyter>=1",
    "pytest>=8",
    "pytest-asyncio>=0.23",
    "ruff>=0.6",
]

[tool.ruff]
line-length = 88
target-version = "py312"

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

### 이 프로젝트 환경 설정 (실습)

```bash
# 프로젝트 루트로 이동
cd /Users/sangyi/workspace/study/langraph_langchain

# 모든 의존성 설치 (pyproject.toml 기준)
uv sync --dev

# 설치 확인
uv run python -c "import langchain; print(langchain.__version__)"
uv run python -c "import langgraph; print(langgraph.__version__)"
```

---

## ✏️ 실습 과제

1. **UV 설치 확인**: `uv --version`을 실행하여 설치를 확인하세요.

2. **의존성 설치**: `uv sync --dev`를 실행하여 모든 의존성을 설치하세요.

3. **설치 확인**: 다음 명령으로 핵심 패키지가 올바르게 설치되었는지 확인하세요:
   ```bash
   uv run python -c "
   import langchain
   import langgraph
   import langsmith
   print('langchain:', langchain.__version__)
   print('langgraph:', langgraph.__version__)
   print('All OK!')
   "
   ```

4. **`uv.lock` 파일 열기**: 생성된 `uv.lock` 파일을 열어 내용을 살펴보세요. 어떤 정보가 기록되어 있나요?

---

## ⚠️ 흔한 함정

- **`pip install` 혼용 금지**: UV 프로젝트에서 `pip install`을 직접 사용하면 lock 파일과 실제 환경이 불일치합니다. 항상 `uv add`를 사용하세요.

- **가상환경 경로 혼동**: UV는 기본적으로 `.venv` 폴더를 프로젝트 루트에 만듭니다. 다른 경로의 Python을 사용하지 않도록 주의하세요.

- **`uv.lock` 삭제 금지**: `uv.lock`는 팀원 간 환경 일치를 보장합니다. `.gitignore`에 추가하지 마세요.

- **Python 버전 불일치**: `.python-version` 파일에 `3.12`가 있어도, 해당 버전이 설치되지 않으면 UV가 자동으로 설치합니다. 처음 `uv sync` 시 시간이 걸릴 수 있습니다.

---

## ✅ 셀프 체크

- [ ] `uv --version`이 정상 출력된다
- [ ] `uv sync --dev`로 의존성을 설치했다
- [ ] `.venv` 폴더가 프로젝트 루트에 생성되었다
- [ ] `uv run python -c "import langchain"` 이 에러 없이 실행된다
- [ ] `pyproject.toml`, `.python-version`, `uv.lock` 각 파일의 역할을 설명할 수 있다

---

## 🔗 참고 자료

- [UV 공식 문서](https://docs.astral.sh/uv/)
- [UV GitHub](https://github.com/astral-sh/uv)
- [pyproject.toml 명세 (PEP 517/518)](https://packaging.python.org/en/latest/specifications/pyproject-toml/)
- [UV vs pip vs poetry 비교](https://docs.astral.sh/uv/getting-started/features/)

---

← 이전: [Phase 00: 오리엔테이션](./00-orientation.md) | 다음: [Phase 02: VSCode 개발 환경](./02-vscode-setup.md) →
